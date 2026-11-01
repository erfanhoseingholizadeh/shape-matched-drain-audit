# Earlier negative-target audit: scores a checkpoint on films edited with the tube and with its control; defines the tube's uniform fill (imported by
# the fresh-composite audit).
# Run: python3 scoring/negative_target_audit.py --help

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from cxr_shortcuts.data.datasets import NIH_14_LABELS, PNEUMOTHORAX_LABEL_INDEX
from cxr_shortcuts.data.image_io import default_image_loader
from cxr_shortcuts.data.lung_masks import load_chexmask_lateral_masks, load_chexmask_lung_masks
from cxr_shortcuts.data.manifests import select_rca_eligible_negatives, verify_manifest_file
from cxr_shortcuts.data.transforms import to_model_tensor
from cxr_shortcuts.edits.drain.control_edit import apply_control_edit
from cxr_shortcuts.edits.drain.geometry import NoValidPlacementError, placement_lung_side
from cxr_shortcuts.edits.drain.insertion import insert_drain
from cxr_shortcuts.edits.drain.synthetic_tube import generate_synthetic_tube_patch
from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_ci_indices, gate_criterion_ci
from cxr_shortcuts.eval.interventional import mean_absolute_perturbation, mean_probability_shift
from cxr_shortcuts.models.backbones import build_backbone
from cxr_shortcuts.models.checkpoint import load_checkpoint
from cxr_shortcuts.runners.audit import CONTROL_VALIDITY_RATIO_THRESHOLD, PERTURBATION_MAD_FLOOR, DegenerateInsertionError
from cxr_shortcuts.runners.composite_and_score import TubeGenerationRanges, _draw_chord_orientation_deg, _draw_tube_params

FILL_INTENSITY = 190
STRIPE_INTENSITY = FILL_INTENSITY
BLEND_MODE = "log_additive"
IMAGE_SIZE = 224

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def build_control_source_assignment(image_ids: list[str], patient_ids: list[str], seed: int) -> dict:
    n = len(image_ids)
    order = np.random.default_rng(seed).permutation(n)
    shuffled_ids = [image_ids[i] for i in order]
    shuffled_patients = [patient_ids[i] for i in order]
    assignment = {}
    for i in range(n):
        target_id, target_patient = image_ids[i], patient_ids[i]
        j = i
        for _ in range(n):
            j = (j + 1) % n
            if shuffled_ids[j] != target_id and shuffled_patients[j] != target_patient:
                assignment[target_id] = (shuffled_ids[j], shuffled_patients[j])
                break
        else:
            raise RuntimeError(f"no patient-disjoint control-source candidate found for {target_id!r}")
    return assignment

def composite_pilot_pool(
    target_rows: pd.DataFrame,
    lung_masks: dict,
    lateral_masks: dict,
    image_loader,
    control_source_assignment: dict,
    tube_ranges: TubeGenerationRanges,
    rng: np.random.Generator,
) -> dict:
    before, insertion_imgs, control_imgs, masks, patients, control_source_patients, ids = [], [], [], [], [], [], []
    n_dropped_placement = 0
    n_dropped_straddle = 0
    for _, row in target_rows.iterrows():
        image_id, patient_id = row["image_id"], row["patient_id"]
        control_source_id, control_source_patient = control_source_assignment[image_id]
        target_image = image_loader(image_id)
        control_source_image = image_loader(control_source_id)
        lung_mask = lung_masks[image_id]
        left_mask, right_mask = lateral_masks[image_id]

        background_level = float(target_image[lung_mask].mean())
        chord = _draw_chord_orientation_deg(left_mask, right_mask, rng)
        caliber_px, length_px, curvature = _draw_tube_params(tube_ranges, rng)
        patch, mask, donor_bg = generate_synthetic_tube_patch(
            length_px=length_px, caliber_px=caliber_px, rng=rng, curvature=curvature,
            background_level=background_level, chord_orientation_deg=chord,
            fill_intensity=FILL_INTENSITY, stripe_intensity=STRIPE_INTENSITY,
        )
        try:
            insertion = insert_drain(
                target_image, lung_mask, patch, mask, rng, blend_mode=BLEND_MODE,
                donor_background_level=donor_bg, target_patient_id=patient_id,
                control_source_patient_id=control_source_patient,
            )
        except NoValidPlacementError:
            n_dropped_placement += 1
            continue

        realized_side = placement_lung_side(mask, insertion.offset, left_mask, right_mask)
        if realized_side is None:
            n_dropped_straddle += 1
            continue

        control = apply_control_edit(target_image, control_source_image, insertion, blend_mode=BLEND_MODE)

        before.append(target_image)
        insertion_imgs.append(insertion.image)
        control_imgs.append(control.image)
        masks.append(insertion.placement_mask)
        patients.append(patient_id)
        control_source_patients.append(control_source_patient)
        ids.append(image_id)

    log(f"composited: n={len(before)} dropped_placement={n_dropped_placement} dropped_straddle={n_dropped_straddle}")
    return {
        "before": before, "insertion": insertion_imgs, "control": control_imgs, "masks": masks,
        "patients": patients, "control_source_patients": control_source_patients, "ids": ids,
        "n_dropped_placement": n_dropped_placement, "n_dropped_straddle": n_dropped_straddle,
    }

def to_model_input(image: np.ndarray) -> torch.Tensor:
    return to_model_tensor(image, size=IMAGE_SIZE)

def build_model_for_scoring(ckpt_path: Path, device: str, multitask: bool) -> torch.nn.Module:
    if multitask:
        model = build_backbone("densenet121", num_classes=len(NIH_14_LABELS), pretrained=False)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model.to(device)
    model = build_backbone("densenet121", num_classes=1, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(ckpt_path, model, optimizer)
    return model

@torch.no_grad()
def predict_logits(model: torch.nn.Module, images: torch.Tensor, multitask: bool = False) -> np.ndarray:
    model.eval()
    params = list(model.parameters())
    if params:
        images = images.to(params[0].device)
    raw = model(images)
    logits = raw[:, PNEUMOTHORAX_LABEL_INDEX] if multitask else raw.squeeze(-1)
    return logits.detach().cpu().numpy().astype(np.float64)

@torch.no_grad()
def predict_probabilities(model: torch.nn.Module, images: torch.Tensor, multitask: bool = False) -> np.ndarray:
    logits = predict_logits(model, images, multitask=multitask)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)

def per_image_pilot_csv_path(output_json: Path) -> Path:
    return output_json.with_name(output_json.stem + "_per_image_pilot.csv")

def per_image_val_csv_path(output_json: Path) -> Path:
    return output_json.with_name(output_json.stem + "_per_image_val.csv")

def classifier_validity_auc(
    checkpoint_path: Path, manifest: pd.DataFrame, nih_root: Path, device: str, per_image_val_csv: Path, seed_idx: int,
    multitask: bool,
) -> dict:
    from cxr_shortcuts.data.datasets import NIHPneumothoraxDataset
    from torch.utils.data import DataLoader

    label_columns = list(NIH_14_LABELS) if multitask else None
    dataset = NIHPneumothoraxDataset(manifest, image_root=nih_root, split="val", image_size=IMAGE_SIZE, label_columns=label_columns) \
        if multitask else NIHPneumothoraxDataset(manifest, image_root=nih_root, split="val", image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16)

    model = build_model_for_scoring(checkpoint_path, device, multitask)
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            raw = model(images)
            if multitask:
                logits = raw[:, PNEUMOTHORAX_LABEL_INDEX]
                lbl = labels.numpy()[:, PNEUMOTHORAX_LABEL_INDEX]
            else:
                logits = raw.squeeze(-1)
                lbl = labels.numpy()
            all_logits.append(logits.detach().cpu().numpy().astype(np.float64))
            all_labels.append(lbl)
    logits_arr = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    scores = 1.0 / (1.0 + np.exp(-logits_arr))
    auc = float(roc_auc_score(labels, scores))

    patient_ids = dataset.rows["patient_id"].astype(str).to_numpy()
    image_ids = dataset.rows["image_id"].astype(str).to_numpy()

    write_header = not per_image_val_csv.exists()
    with open(per_image_val_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["seed", "image_id", "patient_id", "pneumothorax_label", "prob", "logit"])
        for img_id, pid, label, prob, logit in zip(image_ids, patient_ids, labels, scores, logits_arr):
            writer.writerow([seed_idx, img_id, pid, float(label), float(prob), float(logit)])

    def statistic_fn(indices: np.ndarray) -> float:
        return float(roc_auc_score(labels[indices], scores[indices]))

    ci_lo, ci_hi = cluster_bootstrap_ci_indices(patient_ids, statistic_fn, n_resamples=2000, seed=0)
    return {"auc": auc, "ci_low": ci_lo, "ci_high": ci_hi, "n": len(labels), "n_patients": len(set(patient_ids))}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--rca-csv", required=True, type=Path)
    parser.add_argument("--nih-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, action="append",
                         help="repeatable: one checkpoint per classifier training seed")
    parser.add_argument("--multitask", action="store_true", default=False,
                         help="Phase A checkpoints: 14-label multi-task head, bare state_dict. Default False is the original pre-Phase-A code path, unchanged.")
    parser.add_argument("--n-target", type=int, default=500)
    parser.add_argument("--composite-seed", type=int, default=4000)
    parser.add_argument("--control-source-seed", type=int, default=4001)
    parser.add_argument("--sample-seed", type=int, default=4002)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    log(f"torch device available: cuda={torch.cuda.is_available()} multitask={args.multitask}")
    verify_manifest_file(args.split_manifest)
    manifest = pd.read_csv(args.split_manifest, dtype={"patient_id": str})
    rca_df = pd.read_csv(args.rca_csv)
    image_loader = default_image_loader(args.nih_root)

    per_image_pilot_csv = per_image_pilot_csv_path(args.output_json)
    per_image_val_csv = per_image_val_csv_path(args.output_json)
    log(f"per-image sidecars (always written, not optional): {per_image_pilot_csv} / {per_image_val_csv}")

    for i, ckpt in enumerate(args.checkpoint):
        result = classifier_validity_auc(ckpt, manifest, args.nih_root, args.device, per_image_val_csv, i, args.multitask)
        chance_excluded = result["ci_low"] > 0.5
        log(f"classifier_validity seed={i}: auc={result['auc']:.4f} ci=({result['ci_low']:.4f},{result['ci_high']:.4f}) "
            f"n={result['n']} n_patients={result['n_patients']} chance_excluded={chance_excluded}")
        with open(args.output_json, "a") as f:
            f.write(json.dumps({"item": "classifier_validity", "seed": i, **result, "chance_excluded": chance_excluded}) + "\n")

    dev_pool = manifest[manifest["gate_subset"] == "dev"]
    eligible, rca_report = select_rca_eligible_negatives(
        dev_pool, rca_df, min_rca=0.7, target_split="test", required_count=args.n_target,
    )
    log(f"RCA-eligible dev-subset negative pool: {rca_report}")
    eligible_ids = sorted(eligible["image_id"].unique())
    order = np.random.default_rng(args.sample_seed).permutation(len(eligible_ids))
    sampled_ids = [eligible_ids[i] for i in order[: args.n_target]]

    from cxr_shortcuts.eval.population_log import log_population
    log_population(args.output_json.parent, f"{args.output_json.stem}_sampled_target_images", sampled_ids, "image")
    target_rows = eligible.set_index("image_id").loc[sampled_ids].reset_index().rename(columns={"index": "image_id"})
    log(f"sampled {len(target_rows)} target images from {len(eligible_ids)} eligible")

    needed_ids = target_rows["image_id"].unique()
    relevant_rca = rca_df[rca_df["Image Index"].isin(needed_ids)]
    lung_masks = load_chexmask_lung_masks(relevant_rca)
    lateral_masks = load_chexmask_lateral_masks(relevant_rca)

    all_ids = list(target_rows["image_id"])
    all_patients = list(target_rows["patient_id"])
    control_source_assignment = build_control_source_assignment(all_ids, all_patients, seed=args.control_source_seed)

    rng = np.random.default_rng(args.composite_seed)
    tube_ranges = TubeGenerationRanges()
    composite = composite_pilot_pool(
        target_rows, lung_masks, lateral_masks, image_loader, control_source_assignment, tube_ranges, rng,
    )
    with open(args.output_json, "a") as f:
        f.write(json.dumps({
            "item": "composite_drops", "n_kept": len(composite["before"]),
            "n_dropped_placement": composite["n_dropped_placement"],
            "n_dropped_straddle": composite["n_dropped_straddle"],
            "n_target_sampled": len(target_rows),
        }) + "\n")

    insertion_mad_pop = mean_absolute_perturbation(composite["before"], composite["insertion"], composite["masks"])
    control_mad_pop = mean_absolute_perturbation(composite["before"], composite["control"], composite["masks"])
    ratio = float("inf") if control_mad_pop == 0 else insertion_mad_pop / control_mad_pop
    control_validity_flag = ratio > CONTROL_VALIDITY_RATIO_THRESHOLD
    log(f"perturbation MAD (population mean): insertion={insertion_mad_pop:.4f} control={control_mad_pop:.4f} "
        f"ratio={ratio:.2f} control_validity_flag={control_validity_flag}")
    if insertion_mad_pop < PERTURBATION_MAD_FLOOR:
        raise DegenerateInsertionError(
            f"mean insertion perturbation ({insertion_mad_pop:.4f}) is below PERTURBATION_MAD_FLOOR "
            f"({PERTURBATION_MAD_FLOOR}) -- halting before any Delta-p is reported, per the pre-registered "
            f"population-level insertion floor (fatal, not a per-image drop)."
        )
    control_floor_flag = control_mad_pop < PERTURBATION_MAD_FLOOR

    def per_image_mad(before_list, edited_list, mask_list):
        return [
            float(np.abs(e.astype(np.float64) - b.astype(np.float64))[m].mean())
            for b, e, m in zip(before_list, edited_list, mask_list)
        ]
    insertion_mad_per_image = per_image_mad(composite["before"], composite["insertion"], composite["masks"])
    control_mad_per_image = per_image_mad(composite["before"], composite["control"], composite["masks"])
    n_insertion_below_floor = int(sum(1 for v in insertion_mad_per_image if v < PERTURBATION_MAD_FLOOR))
    n_control_below_floor = int(sum(1 for v in control_mad_per_image if v < PERTURBATION_MAD_FLOOR))
    log(f"descriptive per-image floor violations (NOT pre-registered): "
        f"insertion={n_insertion_below_floor}/{len(insertion_mad_per_image)} "
        f"control={n_control_below_floor}/{len(control_mad_per_image)}")

    with open(args.output_json, "a") as f:
        f.write(json.dumps({
            "item": "perturbation_floor",
            "insertion_mad_population_mean": insertion_mad_pop, "control_mad_population_mean": control_mad_pop,
            "perturbation_mad_ratio": ratio, "control_validity_flag": control_validity_flag,
            "control_perturbation_floor_flag": control_floor_flag,
            "n_insertion_below_floor_per_image_descriptive": n_insertion_below_floor,
            "n_control_below_floor_per_image_descriptive": n_control_below_floor,
            "n_pairs": len(insertion_mad_per_image),
        }) + "\n")

    patient_ids = np.array(composite["patients"])
    donor_cluster_ids = np.array(composite["control_source_patients"])

    for seed_idx, ckpt in enumerate(args.checkpoint):
        model = build_model_for_scoring(ckpt, args.device, args.multitask)

        before_batch = torch.stack([to_model_input(img) for img in composite["before"]])
        insertion_batch = torch.stack([to_model_input(img) for img in composite["insertion"]])
        control_batch = torch.stack([to_model_input(img) for img in composite["control"]])

        probs_before = predict_probabilities(model, before_batch, multitask=args.multitask)
        probs_insertion = predict_probabilities(model, insertion_batch, multitask=args.multitask)
        probs_control = predict_probabilities(model, control_batch, multitask=args.multitask)

        logits_before = predict_logits(model, before_batch, multitask=args.multitask)
        logits_insertion = predict_logits(model, insertion_batch, multitask=args.multitask)
        logits_control = predict_logits(model, control_batch, multitask=args.multitask)

        delta_p_insertion = probs_insertion - probs_before
        delta_p_control = probs_control - probs_before
        mean_insertion = mean_probability_shift(probs_before, probs_insertion)
        mean_control = mean_probability_shift(probs_before, probs_control)
        difference = mean_insertion - mean_control

        write_header = not per_image_pilot_csv.exists()
        with open(per_image_pilot_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "seed", "image_id", "patient_id", "control_source_patient_id",
                    "prob_before", "prob_insertion", "prob_control",
                    "logit_before", "logit_insertion", "logit_control",
                    "delta_p_insertion", "delta_p_control", "delta_logit_insertion", "delta_logit_control",
                ])
            for row_i in range(len(composite["ids"])):
                writer.writerow([
                    seed_idx, composite["ids"][row_i], composite["patients"][row_i],
                    composite["control_source_patients"][row_i],
                    float(probs_before[row_i]), float(probs_insertion[row_i]), float(probs_control[row_i]),
                    float(logits_before[row_i]), float(logits_insertion[row_i]), float(logits_control[row_i]),
                    float(delta_p_insertion[row_i]), float(delta_p_control[row_i]),
                    float(logits_insertion[row_i] - logits_before[row_i]),
                    float(logits_control[row_i] - logits_before[row_i]),
                ])

        def statistic_fn(indices: np.ndarray) -> float:
            return float(delta_p_insertion[indices].mean() - delta_p_control[indices].mean())

        donor_ci = cluster_bootstrap_ci_indices(donor_cluster_ids, statistic_fn, n_resamples=2000, seed=0)
        patient_ci = cluster_bootstrap_ci_indices(patient_ids, statistic_fn, n_resamples=2000, seed=0)
        widest_ci = gate_criterion_ci(donor_cluster_ids, patient_ids, statistic_fn, n_resamples=2000, seed=0)

        def logit_statistic_fn(indices: np.ndarray) -> float:
            return float((logits_insertion[indices] - logits_before[indices]).mean() - (logits_control[indices] - logits_before[indices]).mean())

        logit_patient_ci = cluster_bootstrap_ci_indices(patient_ids, logit_statistic_fn, n_resamples=2000, seed=0)

        def insertion_vs_baseline_logit_fn(indices: np.ndarray) -> float:
            return float((logits_insertion[indices] - logits_before[indices]).mean())

        insertion_vs_baseline_ci = cluster_bootstrap_ci_indices(patient_ids, insertion_vs_baseline_logit_fn, n_resamples=2000, seed=0)

        quartiles_insertion = np.quantile(delta_p_insertion, [0, 0.25, 0.5, 0.75, 1.0]).tolist()
        quartiles_control = np.quantile(delta_p_control, [0, 0.25, 0.5, 0.75, 1.0]).tolist()
        hist_insertion, hist_edges = np.histogram(delta_p_insertion, bins=10, range=(-1, 1))
        hist_control, _ = np.histogram(delta_p_control, bins=hist_edges)

        mean_delta_logit_insertion_vs_baseline = float((logits_insertion - logits_before).mean())
        mean_delta_logit_insertion_minus_control = float((logits_insertion - logits_before).mean() - (logits_control - logits_before).mean())

        log(f"seed={seed_idx}: mean_insertion_dp={mean_insertion:.4f} mean_control_dp={mean_control:.4f} "
            f"dp_difference={difference:.4f} patient_ci={patient_ci} widest_ci={widest_ci}")
        log(f"seed={seed_idx}: mean_delta_logit_insertion_vs_baseline={mean_delta_logit_insertion_vs_baseline:.4f} "
            f"CI={insertion_vs_baseline_ci} mean_delta_logit_insertion_minus_control={mean_delta_logit_insertion_minus_control:.4f} "
            f"CI={logit_patient_ci}")

        record = {
            "item": "delta_p_result", "seed": seed_idx,
            "mean_insertion_delta_p": mean_insertion, "mean_control_delta_p": mean_control,
            "difference": difference,
            "donor_analog_ci": list(donor_ci), "patient_ci": list(patient_ci), "widest_ci": list(widest_ci),
            "mean_delta_logit_insertion_vs_baseline": mean_delta_logit_insertion_vs_baseline,
            "insertion_vs_baseline_logit_patient_ci": list(insertion_vs_baseline_ci),
            "mean_delta_logit_insertion_minus_control": mean_delta_logit_insertion_minus_control,
            "insertion_minus_control_logit_patient_ci": list(logit_patient_ci),
            "n_donor_clusters": int(len(set(donor_cluster_ids))), "n_patient_clusters": int(len(set(patient_ids))),
            "n_pairs": len(delta_p_insertion),
            "quartiles_insertion_delta_p": quartiles_insertion, "quartiles_control_delta_p": quartiles_control,
            "histogram_bin_edges": hist_edges.tolist(),
            "histogram_insertion_delta_p": hist_insertion.tolist(), "histogram_control_delta_p": hist_control.tolist(),
        }
        with open(args.output_json, "a") as f:
            f.write(json.dumps(record) + "\n")

    log("DONE")

if __name__ == "__main__":
    main()
