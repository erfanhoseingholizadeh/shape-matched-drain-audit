# Scores every validation film with the five DenseNet-121 checkpoints and writes the per-image predictions the validity recomputations read.
# Run: python3 scoring/val_full_rescan_densenet121.py --help

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from cxr_shortcuts.data.datasets import NIH_14_LABELS, PNEUMOTHORAX_LABEL_INDEX, NIHPneumothoraxDataset
from cxr_shortcuts.data.manifests import VAL_SUBSET_SEED, attach_val_subset
from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_ci_indices
from cxr_shortcuts.models.backbones import build_backbone

EXPECTED_VAL_REPORT_AUC = {
    0: {"auc": 0.8802, "ci_low": 0.8524, "ci_high": 0.9049},
    1: {"auc": 0.8786, "ci_low": 0.8483, "ci_high": 0.9024},
    2: {"auc": 0.8768, "ci_low": 0.8506, "ci_high": 0.8975},
    3: {"auc": 0.8719, "ci_low": 0.8483, "ci_high": 0.8911},
    4: {"auc": 0.8781, "ci_low": 0.8501, "ci_high": 0.9001},
}
ORACLE_TOL = 5e-3

def predict(model: torch.nn.Module, dataloader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_batches, logits_batches, labels_batches = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            probs_batches.append(probs[:, PNEUMOTHORAX_LABEL_INDEX].detach().cpu().numpy().astype(np.float64))
            logits_batches.append(logits[:, PNEUMOTHORAX_LABEL_INDEX].detach().cpu().numpy().astype(np.float64))
            labels_batches.append(labels.numpy()[:, PNEUMOTHORAX_LABEL_INDEX])
    return np.concatenate(probs_batches), np.concatenate(logits_batches), np.concatenate(labels_batches)

def clustered_auc_ci(labels: np.ndarray, probs: np.ndarray, patient_ids: np.ndarray, seed: int = 0) -> dict:
    point = float(roc_auc_score(labels, probs))

    def statistic_fn(indices: np.ndarray) -> float:
        return float(roc_auc_score(labels[indices], probs[indices]))

    lo, hi = cluster_bootstrap_ci_indices(patient_ids, statistic_fn, n_resamples=2000, ci=0.95, seed=seed)
    return {"auc": point, "ci_low": lo, "ci_high": hi}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nih-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, action="append",
                         help="repeatable: one Phase A checkpoint per seed, in seed order (0,1,2,3,4)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    tagged = attach_val_subset(manifest, seed=VAL_SUBSET_SEED)

    write_header = not args.output_csv.exists()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(args.output_csv, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["seed", "image_id", "patient_id", "val_subset", "pneumothorax_label", "prob", "logit"])

    results: dict = {}

    for seed_idx, ckpt in enumerate(args.checkpoint):
        model = build_backbone("densenet121", num_classes=len(NIH_14_LABELS), pretrained=False)
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model = model.to(args.device)

        seed_result: dict = {"checkpoint": str(ckpt)}
        subset_probs: dict = {}
        subset_labels: dict = {}
        subset_patients: dict = {}

        for subset in ("select", "report"):
            rows = tagged[(tagged["split"] == "val") & (tagged["val_subset"] == subset)].reset_index(drop=True)
            dataset = NIHPneumothoraxDataset(
                rows, image_root=args.nih_root, split="val", image_size=args.image_size, label_columns=list(NIH_14_LABELS),
            )
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            probs, logits, labels = predict(model, dataloader, args.device)
            image_ids = rows["image_id"].to_numpy()
            patient_ids = rows["patient_id"].to_numpy()

            for img_id, pid, lbl, p, z in zip(image_ids, patient_ids, labels, probs, logits):
                writer.writerow([seed_idx, img_id, pid, subset, float(lbl), float(p), float(z)])

            subset_probs[subset] = probs
            subset_labels[subset] = labels
            subset_patients[subset] = patient_ids

            if subset == "report":
                ci = clustered_auc_ci(labels, probs, patient_ids, seed=0)
                expected = EXPECTED_VAL_REPORT_AUC[seed_idx]
                oracle_match = (
                    abs(ci["auc"] - expected["auc"]) < ORACLE_TOL
                    and abs(ci["ci_low"] - expected["ci_low"]) < ORACLE_TOL
                    and abs(ci["ci_high"] - expected["ci_high"]) < ORACLE_TOL
                )
                seed_result["val_report_auc_ci"] = ci
                seed_result["oracle_expected"] = expected
                seed_result["oracle_match"] = oracle_match
                print(f"[seed {seed_idx}] val-report AUC={ci['auc']:.4f} CI=({ci['ci_low']:.4f},{ci['ci_high']:.4f}) "
                      f"vs expected AUC={expected['auc']:.4f} CI=({expected['ci_low']:.4f},{expected['ci_high']:.4f}) "
                      f"ORACLE_MATCH={oracle_match}")

        full_probs = np.concatenate([subset_probs["select"], subset_probs["report"]])
        full_labels = np.concatenate([subset_labels["select"], subset_labels["report"]])
        full_patients = np.concatenate([subset_patients["select"], subset_patients["report"]])
        full_ci = clustered_auc_ci(full_labels, full_probs, full_patients, seed=0)
        seed_result["full_val_auc_ci"] = full_ci
        seed_result["n_full_val_images"] = int(len(full_probs))
        seed_result["n_full_val_patients"] = int(len(set(full_patients)))
        print(f"[seed {seed_idx}] full-val AUC={full_ci['auc']:.4f} CI=({full_ci['ci_low']:.4f},{full_ci['ci_high']:.4f}) "
              f"n_images={seed_result['n_full_val_images']} n_patients={seed_result['n_full_val_patients']}")

        results[seed_idx] = seed_result
        csv_file.flush()

    csv_file.close()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    all_oracle_match = all(r["oracle_match"] for r in results.values())
    print(f"\nALL_ORACLE_MATCH={all_oracle_match}")
    print("DONE")

if __name__ == "__main__":
    main()
