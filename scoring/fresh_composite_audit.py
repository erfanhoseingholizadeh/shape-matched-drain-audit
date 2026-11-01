# Builds the arms for the negative targets (one placement per draw) and scores them with the checkpoints.
# Run: python3 scoring/fresh_composite_audit.py --help

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from cxr_shortcuts.data.datasets import NIH_14_LABELS, PNEUMOTHORAX_LABEL_INDEX
from cxr_shortcuts.data.image_io import default_image_loader
from cxr_shortcuts.data.lung_masks import load_chexmask_lateral_masks, load_chexmask_lung_masks
from cxr_shortcuts.data.reserved_guard import assert_not_reserved, reserved_ids_path
from cxr_shortcuts.data.transforms import to_model_tensor
from cxr_shortcuts.edits.drain.control_edit import apply_control_edit
from cxr_shortcuts.edits.drain.geometry import NoValidPlacementError, fits_inside_lung, place_mask, placement_lung_side
from cxr_shortcuts.edits.drain.insertion import insert_drain, log_additive_blend_patch
from cxr_shortcuts.edits.drain.synthetic_tube import generate_synthetic_tube_patch
from cxr_shortcuts.eval.population_log import ids_digest, log_population
from cxr_shortcuts.models.backbones import build_backbone
from cxr_shortcuts.runners.audit import CONTROL_VALIDITY_RATIO_THRESHOLD, PERTURBATION_MAD_FLOOR, DegenerateInsertionError
from cxr_shortcuts.runners.composite_and_score import TubeGenerationRanges, _draw_chord_orientation_deg, _draw_tube_params
from scoring.blob_control_audit import blob_offset_from_tube_centroid, generate_equal_area_blob
from scoring.negative_target_audit import BLEND_MODE, FILL_INTENSITY, STRIPE_INTENSITY, build_control_source_assignment

COMPOSITE_SEED = 6400
CONTROL_SOURCE_SEED = 6401
IMAGE_SIZE = 224
POOL_289_DIGEST = "0e2aa448e8efcc3e8971664337ba3affbef3763a7b1ea8c2a5d639f7d2e604c5"
ARMS = ("before", "insertion", "control", "blob")

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def manifest_digest(file_hashes: dict) -> str:
    return hashlib.sha256("\n".join(f"{k} {file_hashes[k]}" for k in sorted(file_hashes)).encode()).hexdigest()

def composite_targets(pool_ids, patient_of, lung_masks, lateral_masks, image_loader, assignment, out_dir: Path,
                      seed: int = COMPOSITE_SEED) -> dict:
    rng = np.random.default_rng(seed)
    tube_ranges = TubeGenerationRanges()
    out_dir.mkdir(parents=True, exist_ok=True)
    drops = {"placement": [], "straddle": [], "blob_oob": []}
    kept, file_hashes, meta = [], {}, {}
    mad = {"insertion": [], "control": [], "blob": []}
    before_l, ins_l, ctl_l, mask_l, blob_l, blobmask_l = [], [], [], [], [], []
    for image_id in pool_ids:
        patient_id = patient_of[image_id]
        control_source_id, control_source_patient = assignment[image_id]
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
            drops["placement"].append(image_id)
            continue
        if placement_lung_side(mask, insertion.offset, left_mask, right_mask) is None:
            drops["straddle"].append(image_id)
            continue
        tube_area = int(mask.sum())
        blob_patch, blob_mask = generate_equal_area_blob(tube_area, FILL_INTENSITY, donor_bg)
        blob_offset = blob_offset_from_tube_centroid(mask, insertion.offset, blob_mask)
        bh, bw = blob_mask.shape
        img_h, img_w = target_image.shape
        if blob_offset[0] < 0 or blob_offset[1] < 0 or blob_offset[0] + bh > img_h or blob_offset[1] + bw > img_w:
            drops["blob_oob"].append(image_id)
            continue
        blob_image = log_additive_blend_patch(target_image, blob_patch, blob_mask, blob_offset, donor_bg)
        blob_placed = place_mask(blob_mask, target_image.shape, blob_offset)
        control = apply_control_edit(target_image, control_source_image, insertion, blend_mode=BLEND_MODE)

        arrays = {"before": target_image, "insertion": insertion.image, "control": control.image, "blob": blob_image,
                  "insertion_mask": insertion.placement_mask, "blob_mask": blob_placed}
        fname = f"{Path(image_id).stem}.npz"
        np.savez(out_dir / fname, **arrays)
        file_hashes[fname] = sha256_file(out_dir / fname)
        kept.append(image_id)
        meta[image_id] = {"file": fname, "patient_id": patient_id, "control_source_id": control_source_id,
                          "control_source_patient_id": control_source_patient}
        m = insertion.placement_mask
        mad["insertion"].append(float(np.abs(insertion.image.astype(np.float64) - target_image.astype(np.float64))[m].mean()))
        mad["control"].append(float(np.abs(control.image.astype(np.float64) - target_image.astype(np.float64))[m].mean()))
        mad["blob"].append(float(np.abs(blob_image.astype(np.float64) - target_image.astype(np.float64))[blob_placed].mean()))
    log(f"composited: kept={len(kept)} of {len(pool_ids)} dropped_placement={len(drops['placement'])} "
        f"dropped_straddle={len(drops['straddle'])} dropped_blob_oob={len(drops['blob_oob'])}")
    return {"kept": kept, "file_hashes": file_hashes, "meta": meta, "drops": drops, "mad": mad}

def floor_checks(mad: dict) -> dict:
    ins, ctl, blob = (float(np.mean(mad[k])) for k in ("insertion", "control", "blob"))
    ratio = float("inf") if ctl == 0 else ins / ctl
    if ins < PERTURBATION_MAD_FLOOR:
        raise DegenerateInsertionError(
            f"mean insertion perturbation ({ins:.4f}) is below PERTURBATION_MAD_FLOOR ({PERTURBATION_MAD_FLOOR})")
    return {"insertion_mad_population_mean": ins, "control_mad_population_mean": ctl, "blob_mad_population_mean": blob,
            "perturbation_mad_ratio": ratio, "control_validity_flag": bool(ratio > CONTROL_VALIDITY_RATIO_THRESHOLD),
            "control_perturbation_floor_flag": bool(ctl < PERTURBATION_MAD_FLOOR),
            "perturbation_mad_floor": PERTURBATION_MAD_FLOOR, "n_pairs": len(mad["insertion"])}

def load_verified_composite(comp_dir: Path, expected_digest: str | None) -> dict:
    manifest = json.loads((comp_dir / "composite_manifest.json").read_text())
    hashes = manifest["file_hashes"]
    if manifest_digest(hashes) != manifest["manifest_sha256"]:
        raise SystemExit("composite manifest's own digest does not match its file list")
    if expected_digest is not None and manifest["manifest_sha256"] != expected_digest:
        raise SystemExit(f"composite manifest digest {manifest['manifest_sha256']} != expected {expected_digest}")
    for fname, h in hashes.items():
        if sha256_file(comp_dir / fname) != h:
            raise SystemExit(f"sha256 mismatch for persisted composite file {fname}")
    log(f"verified {len(hashes)} persisted composite files against manifest {manifest['manifest_sha256']}")
    arrays = {a: [] for a in ARMS}
    for image_id in manifest["kept"]:
        with np.load(comp_dir / manifest["meta"][image_id]["file"]) as z:
            for a in ARMS:
                arrays[a].append(z[a])
    return {"manifest": manifest, "arrays": arrays}

BACKBONE = "densenet121"

def build_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    model = build_backbone(BACKBONE, num_classes=len(NIH_14_LABELS), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    return model.to(device).eval()

@torch.no_grad()
def logits_for(model, images: list, device: str, batch_size: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([to_model_tensor(im, size=IMAGE_SIZE) for im in images[i:i + batch_size]]).to(device)
        out.append(model(batch)[:, PNEUMOTHORAX_LABEL_INDEX].detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out)

def score_checkpoints(entries: list, composite: dict, device: str, csv_path: Path) -> None:
    man = composite["manifest"]
    ids = man["kept"]
    header = ["label", "image_id", "patient_id", "control_source_patient_id",
              "logit_before", "logit_insertion", "logit_control", "logit_blob",
              "delta_logit_insertion", "delta_logit_control", "delta_logit_blob"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for label, ckpt in entries:
            model = build_model(ckpt, device)
            lg = {a: logits_for(model, composite["arrays"][a], device) for a in ARMS}
            log(f"scored {label}: mean delta-logit insertion={np.mean(lg['insertion'] - lg['before']):.4f} "
                f"control={np.mean(lg['control'] - lg['before']):.4f} blob={np.mean(lg['blob'] - lg['before']):.4f}")
            for i, image_id in enumerate(ids):
                m = man["meta"][image_id]
                w.writerow([label, image_id, m["patient_id"], m["control_source_patient_id"],
                            lg["before"][i], lg["insertion"][i], lg["control"][i], lg["blob"][i],
                            lg["insertion"][i] - lg["before"][i], lg["control"][i] - lg["before"][i],
                            lg["blob"][i] - lg["before"][i]])
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

def checkpoint_entries(manifest_path: Path, which: str) -> list:
    m = json.loads(manifest_path.read_text())
    if which == "original":
        return [(f"original_seed{e['seed']}", e["checkpoint"]) for e in m["original"]]
    return [(f"{e['strength']}_seed{e['seed']}", e["checkpoint"]) for e in m["planted"]]

def run_composite(args) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"reserved-id file: {reserved_ids_path()} sha256={sha256_file(reserved_ids_path())}")
    pool_ids = sorted(x for x in args.pool_ids.read_text().split() if x)
    if len(pool_ids) != 289 or ids_digest(pool_ids) != POOL_289_DIGEST:
        raise SystemExit("pool id list is not the registered 289-image evaluation pool")
    n_expected = 289
    if args.smoke_n:
        pool_ids = pool_ids[: args.smoke_n]
        n_expected = len(pool_ids)
    manifest = pd.read_csv(args.split_manifest, dtype={"patient_id": str}).drop_duplicates("image_id").set_index("image_id")
    patient_of = {i: str(manifest.loc[i, "patient_id"]) for i in pool_ids}
    rca_df = pd.read_csv(args.rca_csv)
    relevant = rca_df[rca_df["Image Index"].isin(pool_ids)]
    if relevant["Image Index"].nunique() != n_expected:
        raise SystemExit(f"CheXmask rows found for {relevant['Image Index'].nunique()} of {n_expected} pool ids")

    assert_not_reserved(pool_ids, "run_composite pre-check")
    lung_masks = load_chexmask_lung_masks(relevant)
    lateral_masks = load_chexmask_lateral_masks(relevant)
    assignment = build_control_source_assignment(pool_ids, [patient_of[i] for i in pool_ids], seed=args.control_source_seed)
    assert_not_reserved([v[0] for v in assignment.values()], "run_composite control-source pre-check")
    image_loader = default_image_loader(args.nih_root)

    comp_dir = args.composite_dir
    res = composite_targets(pool_ids, patient_of, lung_masks, lateral_masks, image_loader, assignment, comp_dir,
                            seed=args.composite_seed)
    floors = floor_checks(res["mad"])
    log(f"perturbation MAD: {floors}")
    manifest_out = {
        "composite_seed": args.composite_seed, "control_source_seed": args.control_source_seed,
        "kept": res["kept"], "meta": res["meta"], "file_hashes": res["file_hashes"],
        "manifest_sha256": manifest_digest(res["file_hashes"]),
    }
    (comp_dir / "composite_manifest.json").write_text(json.dumps(manifest_out, indent=1))
    control_source_patients = sorted({res["meta"][i]["control_source_patient_id"] for i in res["kept"]})
    log_population(out_dir, "fresh_composite_pool_289", pool_ids, "image")
    log_population(out_dir, "fresh_composite_kept_images", res["kept"], "image")
    log_population(out_dir, "fresh_composite_kept_patients", [res["meta"][i]["patient_id"] for i in res["kept"]], "patient")
    log_population(out_dir, "fresh_composite_control_source_patients", control_source_patients, "patient")
    (out_dir / "composite_summary.json").write_text(json.dumps({
        "n_pool": n_expected, "smoke_n": args.smoke_n, "n_kept": len(res["kept"]),
        "n_dropped_placement": len(res["drops"]["placement"]), "n_dropped_straddle": len(res["drops"]["straddle"]),
        "n_dropped_blob_oob": len(res["drops"]["blob_oob"]), "dropped_ids": res["drops"],
        "floors": floors, "composite_manifest_sha256": manifest_out["manifest_sha256"],
        "reserved_ids_file_sha256": sha256_file(reserved_ids_path()),
    }, indent=1))
    composite = load_verified_composite(comp_dir, manifest_out["manifest_sha256"])
    entries = checkpoint_entries(args.checkpoint_manifest, "original")
    if args.smoke_n:
        entries = entries[:1]
    score_checkpoints(entries, composite, args.device, out_dir / "scores_original.csv")
    log("DONE composite+originals")

def run_score(args) -> None:
    anchor = json.loads(args.anchor_json.read_text())
    composite = load_verified_composite(args.composite_dir, anchor["composite_manifest_sha256"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = checkpoint_entries(args.checkpoint_manifest, "planted")
    if args.smoke_n:
        composite["manifest"] = {**composite["manifest"], "kept": composite["manifest"]["kept"][: args.smoke_n]}
        composite["arrays"] = {a: v[: args.smoke_n] for a, v in composite["arrays"].items()}
        entries = entries[:1]
    score_checkpoints(entries, composite, args.device, args.out_dir / "scores_planted.csv")
    log_population(args.out_dir, "scored_images", composite["manifest"]["kept"], "image")
    log_population(args.out_dir, "scored_patients", [composite["manifest"]["meta"][i]["patient_id"] for i in composite["manifest"]["kept"]], "patient")
    log("DONE planted")

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("composite")
    c.add_argument("--pool-ids", required=True, type=Path)
    c.add_argument("--split-manifest", required=True, type=Path)
    c.add_argument("--rca-csv", required=True, type=Path)
    c.add_argument("--nih-root", required=True, type=Path)
    c.add_argument("--composite-dir", required=True, type=Path)
    c.add_argument("--checkpoint-manifest", required=True, type=Path)
    c.add_argument("--out-dir", required=True, type=Path)
    c.add_argument("--device", default="cuda")
    c.add_argument("--smoke-n", type=int, default=0)
    c.add_argument("--backbone", default="densenet121")

    c.add_argument("--composite-seed", type=int, default=COMPOSITE_SEED)
    c.add_argument("--control-source-seed", type=int, default=CONTROL_SOURCE_SEED)
    s = sub.add_parser("score")
    s.add_argument("--composite-dir", required=True, type=Path)
    s.add_argument("--anchor-json", required=True, type=Path)
    s.add_argument("--checkpoint-manifest", required=True, type=Path)
    s.add_argument("--out-dir", required=True, type=Path)
    s.add_argument("--device", default="cuda")
    s.add_argument("--backbone", default="densenet121")
    s.add_argument("--smoke-n", type=int, default=0)
    args = ap.parse_args()
    global BACKBONE
    BACKBONE = args.backbone
    run_composite(args) if args.mode == "composite" else run_score(args)

if __name__ == "__main__":
    main()
