# Earlier negative-target audit: scores a checkpoint on films edited with the tube and with an equal-area blob (imported by the fresh-composite
# audit).
# Run: python3 scoring/blob_control_audit.py --help

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
from cxr_shortcuts.edits.drain.geometry import NoValidPlacementError, fits_inside_lung, place_mask, placement_lung_side
from cxr_shortcuts.edits.drain.insertion import insert_drain, log_additive_blend_patch
from cxr_shortcuts.edits.drain.synthetic_tube import generate_synthetic_tube_patch
from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_ci_indices, gate_criterion_ci
from cxr_shortcuts.eval.interventional import mean_probability_shift
from cxr_shortcuts.models.backbones import build_backbone
from cxr_shortcuts.models.checkpoint import load_checkpoint
from cxr_shortcuts.runners.audit import PERTURBATION_MAD_FLOOR, DegenerateInsertionError
from cxr_shortcuts.runners.composite_and_score import TubeGenerationRanges, _draw_chord_orientation_deg, _draw_tube_params

FILL_INTENSITY = 190
STRIPE_INTENSITY = FILL_INTENSITY
BLEND_MODE = "log_additive"
IMAGE_SIZE = 224
BLOB_CANVAS_MARGIN = 8

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def generate_equal_area_blob(area: int, fill_intensity: float, background_level: float) -> tuple[np.ndarray, np.ndarray]:
    if area <= 0:
        raise ValueError(f"area must be positive, got {area!r}")
    radius = np.sqrt(area / np.pi)
    canvas_size = int(np.ceil(2 * radius)) + 2 * BLOB_CANVAS_MARGIN
    yy, xx = np.indices((canvas_size, canvas_size))
    center = (canvas_size - 1) / 2.0
    dist2 = (yy - center) ** 2 + (xx - center) ** 2
    order = np.argsort(dist2, axis=None, kind="stable")
    mask_flat = np.zeros(canvas_size * canvas_size, dtype=bool)
    mask_flat[order[:area]] = True
    mask = mask_flat.reshape(canvas_size, canvas_size)
    if int(mask.sum()) != area:
        raise AssertionError(f"blob mask area {int(mask.sum())} != requested {area}")
    patch = np.full((canvas_size, canvas_size), background_level, dtype=np.float64)
    patch[mask] = fill_intensity
    patch = np.clip(patch, 0, 255).astype(np.uint8)
    return patch, mask

def blob_offset_from_tube_centroid(tube_mask: np.ndarray, tube_offset: tuple[int, int], blob_mask: np.ndarray) -> tuple[int, int]:
    rows, cols = np.nonzero(tube_mask)
    centroid_row = tube_offset[0] + rows.mean()
    centroid_col = tube_offset[1] + cols.mean()
    bh, bw = blob_mask.shape
    blob_center_row = (bh - 1) / 2.0
    blob_center_col = (bw - 1) / 2.0
    return int(round(centroid_row - blob_center_row)), int(round(centroid_col - blob_center_col))

def composite_pilot_pool(
    target_rows: pd.DataFrame,
    lung_masks: dict,
    lateral_masks: dict,
    image_loader,
    tube_ranges: TubeGenerationRanges,
    rng: np.random.Generator,
) -> dict:
    before, insertion_imgs, blob_imgs = [], [], []
    masks, patients, ids = [], [], []
    insertion_mad, blob_mad = [], []
    blob_lung_contained = []
    n_dropped_placement = 0
    n_dropped_straddle = 0
    n_dropped_blob_oob = 0
    for _, row in target_rows.iterrows():
        image_id, patient_id = row["image_id"], row["patient_id"]
        target_image = image_loader(image_id)
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
            )
        except NoValidPlacementError:
            n_dropped_placement += 1
            continue

        realized_side = placement_lung_side(mask, insertion.offset, left_mask, right_mask)
        if realized_side is None:
            n_dropped_straddle += 1
            continue

        tube_area = int(mask.sum())
        blob_patch, blob_mask = generate_equal_area_blob(tube_area, FILL_INTENSITY, donor_bg)
        blob_offset = blob_offset_from_tube_centroid(mask, insertion.offset, blob_mask)
        bh, bw = blob_mask.shape
        img_h, img_w = target_image.shape
        if (
            blob_offset[0] < 0 or blob_offset[1] < 0
            or blob_offset[0] + bh > img_h or blob_offset[1] + bw > img_w
        ):
            n_dropped_blob_oob += 1
            continue

        blob_image = log_additive_blend_patch(target_image, blob_patch, blob_mask, blob_offset, donor_bg)

        blob_placed = place_mask(blob_mask, target_image.shape, blob_offset)
        contained = fits_inside_lung(blob_placed, lung_mask)

        before.append(target_image)
        insertion_imgs.append(insertion.image)
        blob_imgs.append(blob_image)
        masks.append(insertion.placement_mask)
        patients.append(patient_id)
        ids.append(image_id)
        blob_lung_contained.append(bool(contained))

        m = insertion.placement_mask
        insertion_mad.append(float(np.abs(insertion.image.astype(np.float64) - target_image.astype(np.float64))[m].mean()))
        blob_mad.append(float(np.abs(blob_image.astype(np.float64) - target_image.astype(np.float64))[blob_placed].mean()))

    log(f"composited: n={len(before)} dropped_placement={n_dropped_placement} dropped_straddle={n_dropped_straddle} dropped_blob_oob={n_dropped_blob_oob}")
    return {
        "before": before, "insertion": insertion_imgs, "blob": blob_imgs,
        "masks": masks, "patients": patients, "ids": ids,
        "n_dropped_placement": n_dropped_placement, "n_dropped_straddle": n_dropped_straddle, "n_dropped_blob_oob": n_dropped_blob_oob,
        "insertion_mad": insertion_mad, "blob_mad": blob_mad, "blob_lung_contained": blob_lung_contained,
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--rca-csv", required=True, type=Path)
    parser.add_argument("--nih-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, action="append")
    parser.add_argument("--multitask", action="store_true", default=False)
    parser.add_argument("--n-target", type=int, default=500)
    parser.add_argument("--composite-seed", type=int, default=4000)
    parser.add_argument("--sample-seed", type=int, default=4002)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    log(f"torch device available: cuda={torch.cuda.is_available()} multitask={args.multitask}")
    verify_manifest_file(args.split_manifest)
    manifest = pd.read_csv(args.split_manifest, dtype={"patient_id": str})
    rca_df = pd.read_csv(args.rca_csv)
    image_loader = default_image_loader(args.nih_root)

    per_image_csv = args.output_json.with_name(args.output_json.stem + "_per_image.csv")
    log(f"per-image sidecar: {per_image_csv}")

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

    rng = np.random.default_rng(args.composite_seed)
    tube_ranges = TubeGenerationRanges()
    composite = composite_pilot_pool(target_rows, lung_masks, lateral_masks, image_loader, tube_ranges, rng)
    with open(args.output_json, "a") as f:
        f.write(json.dumps({
            "item": "composite_drops", "n_kept": len(composite["before"]),
            "n_dropped_placement": composite["n_dropped_placement"],
            "n_dropped_straddle": composite["n_dropped_straddle"],
            "n_dropped_blob_oob": composite["n_dropped_blob_oob"],
            "n_target_sampled": len(target_rows),
        }) + "\n")

    insertion_mad_pop = float(np.mean(composite["insertion_mad"]))
    blob_mad_pop = float(np.mean(composite["blob_mad"]))
    ratio = float("inf") if blob_mad_pop == 0 else insertion_mad_pop / blob_mad_pop
    contained_frac = float(np.mean(composite["blob_lung_contained"])) if composite["blob_lung_contained"] else float("nan")
    log(f"perturbation MAD (population mean): insertion={insertion_mad_pop:.4f} blob={blob_mad_pop:.4f} "
        f"insertion/blob={ratio:.3f} blob_lung_contained_fraction={contained_frac:.4f}")
    if insertion_mad_pop < PERTURBATION_MAD_FLOOR:
        raise DegenerateInsertionError(f"mean insertion perturbation ({insertion_mad_pop:.4f}) below PERTURBATION_MAD_FLOOR")

    with open(args.output_json, "a") as f:
        f.write(json.dumps({
            "item": "perturbation_floor_and_containment",
            "insertion_mad_population_mean": insertion_mad_pop, "blob_mad_population_mean": blob_mad_pop,
            "insertion_over_blob_ratio": ratio, "blob_lung_contained_fraction": contained_frac,
            "n_pairs": len(composite["insertion_mad"]),
        }) + "\n")

    patient_ids = np.array(composite["patients"])

    for seed_idx, ckpt in enumerate(args.checkpoint):
        model = build_model_for_scoring(ckpt, args.device, args.multitask)

        before_batch = torch.stack([to_model_input(img) for img in composite["before"]])
        insertion_batch = torch.stack([to_model_input(img) for img in composite["insertion"]])
        blob_batch = torch.stack([to_model_input(img) for img in composite["blob"]])

        probs_before = predict_probabilities(model, before_batch, multitask=args.multitask)
        probs_insertion = predict_probabilities(model, insertion_batch, multitask=args.multitask)
        probs_blob = predict_probabilities(model, blob_batch, multitask=args.multitask)

        logits_before = predict_logits(model, before_batch, multitask=args.multitask)
        logits_insertion = predict_logits(model, insertion_batch, multitask=args.multitask)
        logits_blob = predict_logits(model, blob_batch, multitask=args.multitask)

        delta_p_insertion = probs_insertion - probs_before
        delta_p_blob = probs_blob - probs_before
        delta_logit_insertion = logits_insertion - logits_before
        delta_logit_blob = logits_blob - logits_before

        mean_insertion = mean_probability_shift(probs_before, probs_insertion)
        mean_blob = mean_probability_shift(probs_before, probs_blob)

        write_header = not per_image_csv.exists()
        with open(per_image_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "seed", "image_id", "patient_id",
                    "prob_before", "prob_insertion", "prob_blob",
                    "logit_before", "logit_insertion", "logit_blob",
                    "delta_p_insertion", "delta_p_blob",
                    "delta_logit_insertion", "delta_logit_blob",
                    "insertion_mad", "blob_mad", "blob_lung_contained",
                ])
            for i in range(len(composite["ids"])):
                writer.writerow([
                    seed_idx, composite["ids"][i], composite["patients"][i],
                    float(probs_before[i]), float(probs_insertion[i]), float(probs_blob[i]),
                    float(logits_before[i]), float(logits_insertion[i]), float(logits_blob[i]),
                    float(delta_p_insertion[i]), float(delta_p_blob[i]),
                    float(delta_logit_insertion[i]), float(delta_logit_blob[i]),
                    composite["insertion_mad"][i], composite["blob_mad"][i], composite["blob_lung_contained"][i],
                ])

        def ci_for(a: np.ndarray, b: np.ndarray) -> dict:
            def statistic_fn(indices: np.ndarray) -> float:
                return float(a[indices].mean() - b[indices].mean())
            patient_ci = cluster_bootstrap_ci_indices(patient_ids, statistic_fn, n_resamples=2000, seed=0)
            return {"patient_ci": list(patient_ci)}

        insertion_minus_blob_dp = ci_for(delta_p_insertion, delta_p_blob)
        insertion_minus_blob_dlogit = ci_for(delta_logit_insertion, delta_logit_blob)

        log(f"seed={seed_idx}: mean_dp insertion={mean_insertion:.4f} blob={mean_blob:.4f}")
        log(f"seed={seed_idx}: insertion_minus_blob (delta_p) patient_ci={insertion_minus_blob_dp['patient_ci']}")
        log(f"seed={seed_idx}: insertion_minus_blob (delta_logit) patient_ci={insertion_minus_blob_dlogit['patient_ci']}")
        log(f"seed={seed_idx}: mean_delta_logit_insertion_minus_blob={float(delta_logit_insertion.mean() - delta_logit_blob.mean()):.4f}")

        record = {
            "item": "delta_p_result", "seed": seed_idx,
            "mean_insertion_delta_p": mean_insertion, "mean_blob_delta_p": mean_blob,
            "insertion_minus_blob_delta_p_ci": insertion_minus_blob_dp,
            "insertion_minus_blob_delta_logit_ci": insertion_minus_blob_dlogit,
            "mean_delta_logit_insertion_minus_blob": float(delta_logit_insertion.mean() - delta_logit_blob.mean()),
            "n_patient_clusters": int(len(set(patient_ids))), "n_pairs": len(delta_p_insertion),
        }
        with open(args.output_json, "a") as f:
            f.write(json.dumps(record) + "\n")

    log("DONE")

if __name__ == "__main__":
    main()
