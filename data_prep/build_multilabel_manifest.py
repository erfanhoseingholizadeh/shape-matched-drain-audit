# Joins the 14 NIH finding labels onto the classification manifest for the multi-task head of the original checkpoints.
# Run: python3 data_prep/build_multilabel_manifest.py (no options; it starts working at once)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cxr_shortcuts.data.datasets import NIH_14_LABELS, attach_multilabel_targets, load_nih_chestxray14_multilabel_metadata
from cxr_shortcuts.data.manifests import save_manifest

NIH_CLASSIFICATION_MANIFEST = Path("data/processed/manifests/nih_pneumothorax.csv")
NIH_METADATA_CSV = Path("data/raw/nih_chestxray14/Data_Entry_2017.csv")
OUTPUT_MANIFEST = Path("data/processed/manifests/nih_pneumothorax_multilabel.csv")

def main() -> None:
    manifest = pd.read_csv(NIH_CLASSIFICATION_MANIFEST, dtype={"patient_id": str})
    print(f"loaded classification manifest: {len(manifest)} rows")

    multilabel = load_nih_chestxray14_multilabel_metadata(NIH_METADATA_CSV)
    print(f"loaded multilabel metadata: {len(multilabel)} rows, columns={list(multilabel.columns)}")

    joined = attach_multilabel_targets(manifest, multilabel)
    assert len(joined) == len(manifest), f"join changed row count: {len(manifest)} -> {len(joined)}"
    assert joined["Pneumothorax"].notna().all(), "every classification-manifest image_id must be present in Data_Entry_2017.csv"

    mismatches = joined[joined["pneumothorax"].astype(bool) != joined["Pneumothorax"].astype(bool)]
    if len(mismatches) != 0:
        raise AssertionError(
            f"{len(mismatches)} rows where the existing single-label `pneumothorax` column "
            f"disagrees with the newly-joined `Pneumothorax` column -- first few: "
            f"{mismatches[['image_id', 'pneumothorax', 'Pneumothorax']].head().to_dict('records')}"
        )
    print("verified: joined Pneumothorax column matches the manifest's own pneumothorax column row-for-row, "
          f"across all {len(joined)} rows")

    train_rows = joined[joined["split"] == "train"]
    print(f"\ntrain-split prevalence ({len(train_rows)} images):")
    for label in NIH_14_LABELS:
        n_pos = int(train_rows[label].sum())
        prevalence = n_pos / len(train_rows)
        print(f"  {label:20s} n_pos={n_pos:6d}  prevalence={prevalence:.4f}")

    digest = save_manifest(joined, OUTPUT_MANIFEST)
    print(f"\nwrote {OUTPUT_MANIFEST} ({len(joined)} rows), sha256={digest}")

if __name__ == "__main__":
    main()
