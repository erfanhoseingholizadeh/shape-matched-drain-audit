# Defines operating points by achieved sensitivity on the reporting half (the 90 percent sensitivity thresholds).
# Run: python3 analysis/operating_point_sensitivity_definition.py (no options; it starts working at once)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "review/operating_point_band/outputs"

ARCHITECTURES = {
    "densenet121": REPO_ROOT / "review/densenet121_remeasure/outputs/val_full_per_image.csv",
    "resnet50": REPO_ROOT / "review/resnet50_remeasure/outputs/val_full_per_image_resnet50.csv",
}
TARGET_SENSITIVITIES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

def threshold_for_sensitivity(labels: np.ndarray, probs: np.ndarray, target: float) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    idx = int(np.searchsorted(tpr, target, side="left"))
    if idx >= len(tpr):
        idx = len(tpr) - 1
    return {
        "target_sensitivity": target,
        "threshold": float(thresholds[idx]),
        "achieved_sensitivity": float(tpr[idx]),
        "achieved_specificity": float(1 - fpr[idx]),
    }

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output: dict = {
        "definition": (
            "Operating points defined by achieved sensitivity on val-report (never val-select), "
            "per seed per architecture -- rank-based, calibration-invariant. Not a replacement for "
            "the existing Youden-point registration; reported as a robustness band, no single target "
            "singled out as a clinical recommendation (no clinical co-author / radiologist review "
            "exists in this project -- Resource Dependency Register)."
        ),
        "target_sensitivities": TARGET_SENSITIVITIES,
        "architectures": {},
    }

    for arch, csv_path in ARCHITECTURES.items():
        print("=" * 80)
        print(f"ARCHITECTURE: {arch}")
        print("=" * 80)
        df = pd.read_csv(csv_path, dtype={"patient_id": str})
        report = df[df["val_subset"] == "report"]

        arch_out: dict = {}
        for seed in sorted(report["seed"].unique()):
            seed_df = report[report["seed"] == seed]
            labels = seed_df["pneumothorax_label"].to_numpy()
            probs = seed_df["prob"].to_numpy()

            per_target = {}
            for target in TARGET_SENSITIVITIES:
                res = threshold_for_sensitivity(labels, probs, target)
                per_target[target] = res
                print(f"[{arch}][seed {seed}] target_sens={target:.2f} -> threshold={res['threshold']:.4f} "
                      f"achieved_sens={res['achieved_sensitivity']:.4f} achieved_spec={res['achieved_specificity']:.4f}")

            arch_out[int(seed)] = {
                "n_images": int(len(seed_df)),
                "n_positive": int(labels.sum()),
                "n_negative": int((labels == 0).sum()),
                "per_target_sensitivity": per_target,
            }

        output["architectures"][arch] = arch_out

    out_path = OUT_DIR / "operating_point_sensitivity_definition.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {out_path}")
    print("DONE")

if __name__ == "__main__":
    main()
