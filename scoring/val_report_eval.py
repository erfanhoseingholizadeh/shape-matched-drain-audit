# Scores a trained checkpoint on the reporting half of validation and computes the two validity clauses (AUC, median negative probability).
# Run: python3 scoring/val_report_eval.py --help

from __future__ import annotations

import argparse
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
from cxr_shortcuts.eval.calibration import brier_score, expected_calibration_error
from cxr_shortcuts.models.backbones import build_backbone

def predict(model: torch.nn.Module, dataloader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_batches, labels_batches = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_batches.append(probs[:, PNEUMOTHORAX_LABEL_INDEX])
            labels_batches.append(labels.numpy()[:, PNEUMOTHORAX_LABEL_INDEX])
    return np.concatenate(probs_batches), np.concatenate(labels_batches)

def clustered_auc_ci(labels: np.ndarray, probs: np.ndarray, patient_ids: np.ndarray, seed: int = 0) -> dict:
    point = float(roc_auc_score(labels, probs))

    def statistic_fn(indices: np.ndarray) -> float:
        return float(roc_auc_score(labels[indices], probs[indices]))

    lo, hi = cluster_bootstrap_ci_indices(patient_ids, statistic_fn, n_resamples=2000, ci=0.95, seed=seed)
    return {"auc": point, "ci_low": lo, "ci_high": hi}

def negative_probability_distribution(probs: np.ndarray, labels: np.ndarray) -> dict:
    neg_probs = probs[labels == 0]
    deciles = np.percentile(neg_probs, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    return {
        "n_negatives": int(len(neg_probs)),
        "median": float(np.median(neg_probs)),
        "mean": float(np.mean(neg_probs)),
        "deciles_0_10_..._100": [float(x) for x in deciles],
        "count_below_1e-8": int((neg_probs < 1e-8).sum()),
        "count_below_1e-4": int((neg_probs < 1e-4).sum()),
        "count_below_1e-3": int((neg_probs < 1e-3).sum()),
        "count_below_1e-2": int((neg_probs < 1e-2).sum()),
        "count_at_or_above_1e-3": int((neg_probs >= 1e-3).sum()),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--nih-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="best_checkpoint.pt (bare state_dict)")
    parser.add_argument(
        "--backbone", default="densenet121",
        help="2026-09-14: default unchanged from every prior "
        "invocation of this script -- 'resnet50'/'vit_b_16' added for Phase A3.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    tagged = attach_val_subset(manifest, seed=VAL_SUBSET_SEED)

    model = build_backbone(args.backbone, num_classes=len(NIH_14_LABELS), pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(args.device)

    results: dict = {"checkpoint": str(args.checkpoint), "label_columns": list(NIH_14_LABELS)}

    for subset in ("select", "report"):
        rows = tagged[(tagged["split"] == "val") & (tagged["val_subset"] == subset)]
        dataset = NIHPneumothoraxDataset(
            rows, image_root=args.nih_root, split="val", image_size=args.image_size, label_columns=list(NIH_14_LABELS),
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        probs, labels = predict(model, dataloader, args.device)
        patient_ids = rows.reset_index(drop=True)["patient_id"].to_numpy()

        entry: dict = {
            "n_images": int(len(rows)),
            "n_patients": int(rows["patient_id"].nunique()),
        }
        entry.update(clustered_auc_ci(labels, probs, patient_ids))
        entry["brier"] = brier_score(probs, labels)
        entry["ece"] = expected_calibration_error(probs, labels, n_bins=10)
        entry["negative_probability_distribution"] = negative_probability_distribution(probs, labels)
        results[subset] = entry

        print(f"val-{subset}: n_images={entry['n_images']} n_patients={entry['n_patients']} "
              f"AUC={entry['auc']:.4f} CI=({entry['ci_low']:.4f},{entry['ci_high']:.4f}) "
              f"Brier={entry['brier']:.4f} ECE={entry['ece']:.4f} "
              f"median_neg_p={entry['negative_probability_distribution']['median']:.6g}")

    report_entry = results["report"]
    results["kill_criterion"] = {
        "clause_a_auc_ge_0.82": report_entry["auc"] >= 0.82,
        "clause_a_auc_value": report_entry["auc"],
        "clause_b_median_p_ge_0.001": report_entry["negative_probability_distribution"]["median"] >= 0.001,
        "clause_b_median_p_value": report_entry["negative_probability_distribution"]["median"],
        "both_clauses_pass": (
            report_entry["auc"] >= 0.82
            and report_entry["negative_probability_distribution"]["median"] >= 0.001
        ),
    }
    print(f"\nKILL CRITERION: clause (a) AUC>=0.82: {results['kill_criterion']['clause_a_auc_ge_0.82']} "
          f"({report_entry['auc']:.4f}); clause (b) median_p>=0.001: "
          f"{results['kill_criterion']['clause_b_median_p_ge_0.001']} "
          f"({report_entry['negative_probability_distribution']['median']:.6g})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")

if __name__ == "__main__":
    main()
