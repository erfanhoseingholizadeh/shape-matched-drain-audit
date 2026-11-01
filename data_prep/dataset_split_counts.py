# Counts images and patients of the dataset and of each split from the manifest.
# Run: python3 data_prep/dataset_split_counts.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO_ROOT / "data" / "processed" / "manifests"
OUT_PATH = REPO_ROOT / "review" / "dataset_splits" / "split_counts.json"

def main() -> None:
    df = pd.read_csv(MANIFEST_DIR / "nih_pneumothorax.csv")
    pool = pd.read_csv(MANIFEST_DIR / "realism_eval_eligible_pool.csv")
    disc = pd.read_csv(MANIFEST_DIR / "discriminator_split.csv")

    out = {
        "total_images": int(len(df)),
        "total_patients": int(df["patient_id"].nunique()),
        "train_images": int((df["split"] == "train").sum()),
        "train_patients": int(df.loc[df["split"] == "train", "patient_id"].nunique()),
        "val_images": int((df["split"] == "val").sum()),
        "val_patients": int(df.loc[df["split"] == "val", "patient_id"].nunique()),
        "test_images": int((df["split"] == "test").sum()),
        "test_patients": int(df.loc[df["split"] == "test", "patient_id"].nunique()),
        "test_gate_subset_dev_images": int((df["gate_subset"] == "dev").sum()),
        "test_gate_subset_dev_patients": int(df.loc[df["gate_subset"] == "dev", "patient_id"].nunique()),
        "test_gate_subset_gate_images": int((df["gate_subset"] == "gate").sum()),
        "test_gate_subset_gate_patients": int(df.loc[df["gate_subset"] == "gate", "patient_id"].nunique()),
        "eligible_pool_images": int(len(pool)),
        "eligible_pool_patients": int(pool["patient_id"].nunique()),
        "pneumothorax_train_positive": int(df.loc[df["split"] == "train", "pneumothorax"].sum()),
        "discriminator_arm_patient_counts": {
            k: int(v) for k, v in disc.drop_duplicates("patient_id")["discriminator_arm"].value_counts().items()
        },
        "discriminator_arm_image_counts": {
            k: int(v) for k, v in disc["discriminator_arm"].value_counts().items()
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
