# Runs the manipulation check of a planted checkpoint: validation positives unedited against negatives with the tube.
# Run: python3 scoring/manipulation_check.py --help

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cxr_shortcuts.data.image_io import read_image_gray
from cxr_shortcuts.data.datasets import NIH_14_LABELS, PNEUMOTHORAX_LABEL_INDEX, NIHPneumothoraxDataset
from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_ci_indices
from cxr_shortcuts.eval.population_log import log_population
from cxr_shortcuts.models.backbones import build_backbone

N_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
ORACLE_TOL = 5e-3

REPRODUCTION_ORACLE = {
    (0, "00000296_000.png"): 0.4457411170005798,
    (0, "00000296_002.png"): 0.405603289604187,
    (0, "00000368_003.png"): 0.1263260394334793,
    (1, "00000296_000.png"): 0.1962870508432388,
    (1, "00000296_002.png"): 0.1132514551281929,
    (1, "00000368_003.png"): 0.1167543679475784,
    (2, "00000296_000.png"): 0.2828035950660705,
    (2, "00000296_002.png"): 0.4871299266815185,
    (2, "00000368_003.png"): 0.0696279555559158,
    (3, "00000296_000.png"): 0.1279035955667495,
    (3, "00000296_002.png"): 0.2105283886194229,
    (3, "00000368_003.png"): 0.1449986845254898,
    (4, "00000296_000.png"): 0.4433963298797607,
    (4, "00000296_002.png"): 0.2814013361930847,
    (4, "00000368_003.png"): 0.1011487618088722,
}

REPRODUCTION_ORACLE_RESNET50 = {
    (0, "00000296_000.png"): 0.2198633253574371,
    (0, "00000296_002.png"): 0.127363309264183,
    (0, "00000368_003.png"): 0.0798449888825416,
    (1, "00000296_000.png"): 0.1677702367305755,
    (1, "00000296_002.png"): 0.0253480765968561,
    (1, "00000368_003.png"): 0.5042345523834229,
    (2, "00000296_000.png"): 0.1042985171079635,
    (2, "00000296_002.png"): 0.098611943423748,
    (2, "00000368_003.png"): 0.0301201455295085,
    (3, "00000296_000.png"): 0.0512166060507297,
    (3, "00000296_002.png"): 0.1162575483322143,
    (3, "00000368_003.png"): 0.0516580753028392,
    (4, "00000296_000.png"): 0.3406935036182403,
    (4, "00000296_002.png"): 0.6920974850654602,
    (4, "00000368_003.png"): 0.4529110193252563,
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

BACKBONE = "densenet121"

def build_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    model = build_backbone(BACKBONE, num_classes=len(NIH_14_LABELS), pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    return model.to(device)

@torch.no_grad()
def predict(model: torch.nn.Module, dataloader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    batches = []
    for images, _labels in dataloader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)[:, PNEUMOTHORAX_LABEL_INDEX]
        batches.append(probs.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(batches)

def clustered_auc_ci(labels: np.ndarray, probs: np.ndarray, patient_ids: np.ndarray, seed: int = BOOTSTRAP_SEED) -> dict:
    point = float(roc_auc_score(labels, probs))

    def statistic_fn(indices: np.ndarray) -> float:
        return float(roc_auc_score(labels[indices], probs[indices]))

    lo, hi = cluster_bootstrap_ci_indices(patient_ids, statistic_fn, n_resamples=N_RESAMPLES, ci=0.95, seed=seed)
    return {"auc": point, "ci_low": lo, "ci_high": hi, "n": int(len(labels)), "n_patients": int(len(set(patient_ids)))}

def pixel_diff_oracle(conflict_set: pd.DataFrame, nih_root: Path) -> dict:
    negatives = conflict_set[conflict_set["group"] == "negative_rendered"]
    positives = conflict_set[conflict_set["group"] == "positive_unedited"]

    same_path_count = 0
    for _, row in positives.iterrows():

        if Path(row["resolved_path"]).resolve() == (nih_root / row["image_id"]).resolve():
            same_path_count += 1
    if same_path_count != len(positives):
        raise SystemExit(f"pixel_diff_oracle: {len(positives) - same_path_count} positive rows have a resolved_path "
                          f"that does not equal nih_root/image_id -- the 'unedited == positive's own edited path' "
                          f"assumption this pass's registration relies on does not hold for this conflict set.")

    diff_fracs = []
    for _, row in negatives.iterrows():
        composited = read_image_gray(row["resolved_path"], image_id=row["image_id"])
        original = read_image_gray(nih_root / row["image_id"])
        if composited.shape != original.shape or composited.dtype != original.dtype:
            raise SystemExit(f"pixel_diff_oracle: shape/dtype mismatch for {row['image_id']}: "
                              f"{composited.shape}/{composited.dtype} vs {original.shape}/{original.dtype}")
        diff_mask = composited != original
        frac = float(diff_mask.mean())
        if frac <= 0.0:
            raise SystemExit(f"pixel_diff_oracle: {row['image_id']}'s composited and original files are byte-identical "
                              f"-- the rendering did not actually change this image.")
        if frac >= 0.5:
            raise SystemExit(f"pixel_diff_oracle: {row['image_id']}'s composited and original files differ over "
                              f"{frac:.1%} of pixels -- this looks like a global artifact (re-encoding, resolution "
                              f"change), not a footprint-sized edit.")
        diff_fracs.append(frac)

    return {
        "n_positive_rows_checked": len(positives),
        "n_positive_rows_same_path": same_path_count,
        "n_negative_rows_checked": len(negatives),
        "diff_frac_min": float(np.min(diff_fracs)),
        "diff_frac_max": float(np.max(diff_fracs)),
        "diff_frac_mean": float(np.mean(diff_fracs)),
    }

def score_checkpoint(checkpoint_path: Path, manifest: pd.DataFrame, nih_root: Path, image_size: int,
                      batch_size: int, num_workers: int, device: str) -> dict:
    model = build_model(checkpoint_path, device)

    ds_edited = NIHPneumothoraxDataset(manifest, image_root=nih_root, split="conflict", image_size=image_size,
                                        label_columns=("true_label",), image_path_column="resolved_path")
    ds_unedited = NIHPneumothoraxDataset(manifest, image_root=nih_root, split="conflict", image_size=image_size,
                                          label_columns=("true_label",), image_path_column="unedited_path")

    dl_edited = DataLoader(ds_edited, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_unedited = DataLoader(ds_unedited, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    probs_edited = predict(model, dl_edited, device)
    probs_unedited = predict(model, dl_unedited, device)

    labels = manifest["true_label"].to_numpy(dtype=float)
    patient_ids = manifest["patient_id"].to_numpy()
    image_ids = manifest["image_id"].to_numpy()
    groups = manifest["group"].to_numpy()

    is_positive = groups == "positive_unedited"
    if not np.allclose(probs_edited[is_positive], probs_unedited[is_positive], atol=0, rtol=0):
        max_diff = float(np.max(np.abs(probs_edited[is_positive] - probs_unedited[is_positive])))
        raise SystemExit(f"score_checkpoint({checkpoint_path}): positive rows' edited/unedited probabilities are not "
                          f"bit-identical (max abs diff {max_diff}) -- same file should give the same score twice.")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "labels": labels, "patient_ids": patient_ids, "image_ids": image_ids, "groups": groups,
        "probs_edited": probs_edited, "probs_unedited": probs_unedited,
    }

def pool_concatenated(seed_results: list[dict], key: str) -> dict:
    labels = np.concatenate([r["labels"] for r in seed_results])
    probs = np.concatenate([r[key] for r in seed_results])
    patients = np.concatenate([r["patient_ids"] for r in seed_results])
    return clustered_auc_ci(labels, probs, patients)

def pool_mean_of_per_seed(per_seed_cis: list[dict]) -> dict:
    aucs = [c["auc"] for c in per_seed_cis]
    return {"mean_auc": float(np.mean(aucs)), "per_seed_auc": aucs}

def run_group(name: str, checkpoints: list[dict], manifest: pd.DataFrame, nih_root: Path, args,
              csv_writer=None) -> dict:
    per_seed = []
    per_seed_ci_edited = []
    per_seed_ci_unedited = []
    checkpoint_records = []

    for entry in checkpoints:
        ckpt_path = Path(entry["checkpoint"])
        digest = sha256_file(ckpt_path)
        if entry.get("sha256") and entry["sha256"] != digest:
            raise SystemExit(f"run_group({name}): {ckpt_path} sha256 {digest} != the manifest's recorded {entry['sha256']}")
        result = score_checkpoint(ckpt_path, manifest, nih_root, args.image_size, args.batch_size,
                                   args.num_workers, args.device)
        result["seed"] = entry["seed"]
        per_seed.append(result)

        if csv_writer is not None:
            for img_id, pid, grp, lbl, pe, pu in zip(result["image_ids"], result["patient_ids"], result["groups"],
                                                       result["labels"], result["probs_edited"], result["probs_unedited"]):
                csv_writer.writerow([name, entry["seed"], img_id, pid, grp, float(lbl), float(pe), float(pu)])

        ci_edited = clustered_auc_ci(result["labels"], result["probs_edited"], result["patient_ids"])
        ci_unedited = clustered_auc_ci(result["labels"], result["probs_unedited"], result["patient_ids"])
        per_seed_ci_edited.append(ci_edited)
        per_seed_ci_unedited.append(ci_unedited)

        checkpoint_records.append({**entry, "sha256": digest,
                                    "auc_edited": ci_edited, "auc_unedited": ci_unedited})
        print(f"[{name} seed={entry['seed']}] AUC_edited={ci_edited['auc']:.4f} "
              f"CI=({ci_edited['ci_low']:.4f},{ci_edited['ci_high']:.4f})  "
              f"AUC_unedited={ci_unedited['auc']:.4f} CI=({ci_unedited['ci_low']:.4f},{ci_unedited['ci_high']:.4f})",
              flush=True)

    pooled_concat_edited = pool_concatenated(per_seed, "probs_edited")
    pooled_concat_unedited = pool_concatenated(per_seed, "probs_unedited")
    pooled_mean_edited = pool_mean_of_per_seed(per_seed_ci_edited)
    pooled_mean_unedited = pool_mean_of_per_seed(per_seed_ci_unedited)

    def non_overlapping_and_less(ci_a, ci_b):
        return bool(ci_a["auc"] < ci_b["auc"] and ci_a["ci_high"] < ci_b["ci_low"])

    condition1_concat = pooled_concat_edited["ci_high"] < 0.5
    condition2_concat = non_overlapping_and_less(pooled_concat_edited, pooled_concat_unedited)

    return {
        "checkpoints": checkpoint_records,
        "pooled_concatenated": {"edited": pooled_concat_edited, "unedited": pooled_concat_unedited,
                                 "condition1_ci_high_below_0.5": condition1_concat,
                                 "condition2_edited_below_unedited_nonoverlapping": condition2_concat},
        "pooled_mean_of_per_seed": {"edited": pooled_mean_edited, "unedited": pooled_mean_unedited},
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conflict-set", type=Path, required=True)
    parser.add_argument("--nih-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", default="densenet121", help="architecture of every checkpoint in the manifest")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit-images", type=int, default=None, help="cap total conflict-set rows (smoke test only)")
    parser.add_argument("--skip-reproduction-oracle", action="store_true",
                         help="smoke test only -- the real run must not skip this")

    parser.add_argument("--strengths", default="strong,weak,zero",
                         help="comma-separated planted strengths to score (default: the manipulation-check stage's three)")
    parser.add_argument("--skip-original-group", action="store_true",
                         help="do not rescore the five original checkpoints as a group (the reproduction oracle, "
                              "which loads them, still runs unless --skip-reproduction-oracle)")
    args = parser.parse_args()
    global BACKBONE
    BACKBONE = args.backbone

    conflict_set = pd.read_csv(args.conflict_set, dtype={"patient_id": str})
    conflict_set = conflict_set.rename(columns={"true_label": "true_label"})
    if args.limit_images is not None:
        pos = conflict_set[conflict_set["group"] == "positive_unedited"].head(args.limit_images // 2)
        neg = conflict_set[conflict_set["group"] == "negative_rendered"].head(args.limit_images - len(pos))
        conflict_set = pd.concat([pos, neg], ignore_index=True)

    conflict_set["split"] = "conflict"
    conflict_set["unedited_path"] = conflict_set["image_id"].apply(lambda i: str(args.nih_root / i))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== pixel-diff oracle ===", flush=True)
    pixel_oracle = pixel_diff_oracle(conflict_set, args.nih_root)
    print(json.dumps(pixel_oracle, indent=2), flush=True)

    with open(args.checkpoint_manifest) as f:
        manifest_spec = json.load(f)

    if not args.skip_reproduction_oracle:
        print("=== reproduction oracle (original checkpoints) ===", flush=True)
        oracle_table = REPRODUCTION_ORACLE if BACKBONE == "densenet121" else REPRODUCTION_ORACLE_RESNET50
        oracle_results = {}
        for entry in manifest_spec["original"]:
            seed = entry["seed"]
            wanted_ids = {img_id for (s, img_id) in oracle_table if s == seed}
            if not wanted_ids:
                continue
            rows = conflict_set[conflict_set["image_id"].isin(wanted_ids)]
            if len(rows) == 0:
                print(f"  seed {seed}: no oracle image ids present in this (possibly --limit-images'd) conflict set, skipping")
                continue
            model = build_model(Path(entry["checkpoint"]), args.device)
            ds = NIHPneumothoraxDataset(rows, image_root=args.nih_root, split="conflict", image_size=args.image_size,
                                         label_columns=("true_label",), image_path_column="unedited_path")
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            probs = predict(model, dl, args.device)
            for img_id, p in zip(rows["image_id"].to_numpy(), probs):
                expected = oracle_table[(seed, img_id)]
                diff = abs(float(p) - expected)
                ok = diff < ORACLE_TOL
                oracle_results[f"seed{seed}_{img_id}"] = {"expected": expected, "got": float(p), "diff": diff, "ok": ok}
                print(f"  seed {seed} {img_id}: expected={expected:.6f} got={p:.6f} diff={diff:.6f} OK={ok}", flush=True)
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

        all_ok = all(r["ok"] for r in oracle_results.values())
        print(f"REPRODUCTION_ORACLE_ALL_OK={all_ok}", flush=True)
        if not all_ok:
            raise SystemExit("reproduction oracle failed -- refusing to trust any checkpoint score. "
                              "See printed per-image diffs above.")
    else:
        oracle_results = {"skipped": True}

    results = {"pixel_diff_oracle": pixel_oracle, "reproduction_oracle": oracle_results, "groups": {}}

    per_image_csv_path = args.out_dir / "manipulation_check_per_image.csv"
    csv_file = open(per_image_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["group", "seed", "image_id", "patient_id", "conflict_group", "true_label", "prob_edited", "prob_unedited"])

    if args.skip_original_group:
        print("=== original group not rescored (--skip-original-group) ===", flush=True)
    else:
        print("=== scoring original checkpoints (context only; nothing rests on this reading) ===", flush=True)
        results["groups"]["original"] = run_group("original", manifest_spec["original"], conflict_set, args.nih_root, args,
                                                   csv_writer=csv_writer)

    strengths = tuple(s for s in args.strengths.split(",") if s)
    not_scored = sorted({e["strength"] for e in manifest_spec["planted"]} - set(strengths))
    if not_scored:
        print(f"NOTE: manifest strengths not scored by this invocation: {not_scored}", flush=True)
    for strength in strengths:
        entries = [e for e in manifest_spec["planted"] if e["strength"] == strength]
        if not entries:
            raise SystemExit(f"checkpoint manifest has no 'planted' entries for strength={strength!r} -- "
                              f"a real run needs all three; a smoke-test manifest must still include at "
                              f"least one entry per strength it wants scored.")
        print(f"=== scoring planted checkpoints, strength={strength} ===", flush=True)
        results["groups"][strength] = run_group(strength, entries, conflict_set, args.nih_root, args,
                                                 csv_writer=csv_writer)

    csv_file.close()
    print(f"WROTE {per_image_csv_path}", flush=True)

    log_population(args.out_dir, "conflict_set_images", conflict_set["image_id"].tolist(), kind="image")
    log_population(args.out_dir, "conflict_set_patients", conflict_set["patient_id"].tolist(), kind="patient")

    out_json = args.out_dir / "manipulation_check.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"WROTE {out_json}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
