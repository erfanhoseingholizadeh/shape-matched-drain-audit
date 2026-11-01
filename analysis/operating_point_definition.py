# Defines the probability-scale operating points used to ask whether a logit shift changes a decision.
# Run: python3 analysis/operating_point_definition.py (no options; it starts working at once)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

VAL_FULL_CSV = Path("review/densenet121_remeasure/outputs/val_full_per_image.csv")
OUTPUT_JSON = Path("review/operating_point/operating_point_definition.json")

def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return {
        "threshold": float(thresholds[best_idx]),
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(1 - fpr[best_idx]),
        "youden_j": float(j[best_idx]),
    }

def main() -> None:
    df = pd.read_csv(VAL_FULL_CSV, dtype={"patient_id": str})
    report = df[df["val_subset"] == "report"]

    results: dict = {}
    for seed in sorted(report["seed"].unique()):
        seed_df = report[report["seed"] == seed]
        labels = seed_df["pneumothorax_label"].to_numpy()
        probs = seed_df["prob"].to_numpy()
        yj = youden_threshold(labels, probs)
        results[int(seed)] = {
            "n_images": int(len(seed_df)),
            "n_positive": int(labels.sum()),
            "n_negative": int((labels == 0).sum()),
            **yj,
        }
        print(f"[seed {seed}] n={len(seed_df)} threshold={yj['threshold']:.6f} "
              f"sensitivity={yj['sensitivity']:.4f} specificity={yj['specificity']:.4f} "
              f"youden_j={yj['youden_j']:.4f}")

    thresholds = [results[s]["threshold"] for s in results]
    print(f"\nthreshold spread across seeds: min={min(thresholds):.6f} max={max(thresholds):.6f}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps({
        "definition": "Youden point (argmax sensitivity+specificity-1), per seed, val-report only, "
                       "Phase A DenseNet-121 5 seeds. A standard data-derived ROC summary, not a "
                       "clinically validated operating point (no clinical co-author / radiologist "
                       "review exists in this project -- Resource Dependency Register).",
        "per_seed": results,
    }, indent=2))
    print(f"\nwrote {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
