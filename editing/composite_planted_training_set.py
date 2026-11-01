# Draws the planted tube into every training film marked for insertion in the planting plan, for one planting strength.
# Run: python3 editing/composite_planted_training_set.py --strength strong [--limit N] [--n-workers N]

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
from cxr_shortcuts.data.image_io import read_image_gray
from cxr_shortcuts.data.lung_masks import load_chexmask_lateral_masks, load_chexmask_lung_masks
from cxr_shortcuts.data.manifests import save_manifest
from cxr_shortcuts.edits.drain.geometry import NoValidPlacementError
from cxr_shortcuts.eval.population_log import log_population
from cxr_shortcuts.runners.composite_and_score import TubeGenerationRanges, _composite_one_target
from editing.planting_plan import OUT_DIR as PLAN_DIR
from editing.planting_plan import STRENGTHS
from editing.planting_plan import tube_geometry_seed

PERTURBATION_MAD_FLOOR = 1.0
BLEND_MODE = "log_additive"

class DegeneratePlantedInsertionError(RuntimeError):
    pass

def composite_one(args: tuple) -> dict:
    image_id, patient_id, target_path, lung_mask, left_mask, right_mask, out_path, tube_ranges_kwargs = args
    tube_ranges = TubeGenerationRanges(**tube_ranges_kwargs)
    target_image = read_image_gray(target_path, image_id=image_id)
    rng = np.random.default_rng(tube_geometry_seed(image_id))
    try:
        composite = _composite_one_target(
            image_id, patient_id, target_image, lung_mask, left_mask, right_mask, rng, tube_ranges,
            BLEND_MODE, crop_size=128, keep_full_resolution=True,
        )
    except NoValidPlacementError:
        return {"image_id": image_id, "status": "dropped_no_placement"}
    mad = float(np.abs(composite.production_full.astype(np.float64) - target_image.astype(np.float64))[_changed_pixel_mask(composite)].mean())
    Image.fromarray(composite.production_full).save(out_path)
    return {"image_id": image_id, "status": "composited", "mad_under_mask": mad}

def _changed_pixel_mask(composite) -> np.ndarray:
    diff = np.abs(composite.production_full.astype(np.int16) - composite.real_full.astype(np.int16))
    return diff != 0

def main() -> None:
    parser = argparse.ArgumentParser(description="Composite the planted drain into training images for one strength")
    parser.add_argument("--strength", required=True, choices=STRENGTHS)
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--nih-root", required=True, type=Path)
    parser.add_argument("--chexmask-csv", required=True, type=Path)
    parser.add_argument("--output-image-dir", required=True, type=Path)
    parser.add_argument("--resolved-manifest-out", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path, help="review-style output dir for population logs and the summary")
    parser.add_argument("--planting-plan", type=Path, default=None, help="defaults to PLAN_DIR/planting_plan_<strength>.csv")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N insert=True rows (smoke test)")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    plan_path = args.planting_plan or (PLAN_DIR / f"planting_plan_{args.strength}.csv")
    plan = pd.read_csv(plan_path)
    manifest = pd.read_csv(args.manifest_path)
    rca_df = pd.read_csv(args.chexmask_csv)

    to_insert = plan[plan["insert"]].sort_values("image_id", kind="stable").reset_index(drop=True)
    if args.limit is not None:
        to_insert = to_insert.iloc[: args.limit]

    args.output_image_dir.mkdir(parents=True, exist_ok=True)
    relevant_rca_rows = rca_df[rca_df["Image Index"].isin(to_insert["image_id"])]
    lung_masks = load_chexmask_lung_masks(relevant_rca_rows)
    lateral_masks = load_chexmask_lateral_masks(relevant_rca_rows)
    tube_ranges_kwargs = {"caliber_px": TubeGenerationRanges().caliber_px, "length_px": TubeGenerationRanges().length_px, "curvature": TubeGenerationRanges().curvature}

    t0 = time.time()
    jobs = []
    for _, row in to_insert.iterrows():
        image_id = row["image_id"]
        if image_id not in lung_masks:
            continue
        out_path = args.output_image_dir / image_id
        left, right = lateral_masks[image_id]
        jobs.append((image_id, str(row["patient_id"]), str(args.nih_root / image_id), lung_masks[image_id], left, right, str(out_path), tube_ranges_kwargs))

    results = []
    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            for r in pool.map(composite_one, jobs, chunksize=8):
                results.append(r)
    else:
        for j in jobs:
            results.append(composite_one(j))

    n_no_chexmask = len(to_insert) - len(jobs)
    n_composited = sum(r["status"] == "composited" for r in results)
    n_dropped = sum(r["status"] == "dropped_no_placement" for r in results) + n_no_chexmask
    mads = [r["mad_under_mask"] for r in results if r["status"] == "composited"]
    mean_mad = float(np.mean(mads)) if mads else 0.0
    elapsed = time.time() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    composited_ids = [r["image_id"] for r in results if r["status"] == "composited"]
    dropped_ids = [r["image_id"] for r in results if r["status"] == "dropped_no_placement"]
    log_population(args.out_dir, f"train_composited_{args.strength}", composited_ids, "image")
    if dropped_ids:
        log_population(args.out_dir, f"train_dropped_no_placement_{args.strength}", dropped_ids, "image")

    summary = {
        "strength": args.strength, "requested": len(to_insert), "n_composited": n_composited,
        "n_dropped_no_placement": n_dropped, "mean_mad_under_mask": mean_mad,
        "perturbation_mad_floor": PERTURBATION_MAD_FLOOR, "elapsed_seconds": round(elapsed, 1),
        "seconds_per_image": round(elapsed / max(len(jobs), 1), 4), "n_workers": args.n_workers,
        "limit": args.limit,
    }
    (args.out_dir / f"composite_summary_{args.strength}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if mads and mean_mad < PERTURBATION_MAD_FLOOR:
        raise DegeneratePlantedInsertionError(
            f"strength={args.strength}: population-mean MAD-under-mask {mean_mad:.4f} is below "
            f"PERTURBATION_MAD_FLOOR ({PERTURBATION_MAD_FLOOR}) -- refusing to write the resolved manifest."
        )

    composited_set = set(composited_ids)

    def resolve(image_id: str) -> str:
        if image_id in composited_set:
            return str(args.output_image_dir / image_id)
        return str(args.nih_root / image_id)

    full_manifest = manifest.copy()
    full_manifest["resolved_path"] = full_manifest["image_id"].map(resolve)

    digest = save_manifest(full_manifest, args.resolved_manifest_out)
    n_train_rows = int((full_manifest["split"] == "train").sum())
    print(f"wrote {args.resolved_manifest_out} ({len(full_manifest)} rows, {n_train_rows} train, {n_composited} composited, sha256 {digest})")

if __name__ == "__main__":
    main()
