# Builds the patient-level classification split manifest from the NIH metadata.
# Run: python3 -m cxr_shortcuts.runners.build_manifest --help

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cxr_shortcuts.data.datasets import load_neatx_drain_labels, load_nih_chestxray14_metadata
from cxr_shortcuts.data.manifests import (
    assert_no_patient_leakage,
    attach_donor_eligibility,
    attach_gate_subset,
    build_manifest,
    save_manifest,
)

SPLIT_RATIOS = (0.7, 0.15, 0.15)
SPLIT_SEED = 0
GATE_FRACTION = 0.5
GATE_SEED = 0

DEFAULT_NIH_METADATA_CSV = Path("data/raw/nih_chestxray14/Data_Entry_2017.csv")
DEFAULT_NEATX_CSV = Path("data/raw/neatx/neatx.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/manifests/nih_pneumothorax.csv")

def build_real_manifest(nih_metadata_csv: Path, neatx_csv: Path) -> pd.DataFrame:
    records = load_nih_chestxray14_metadata(nih_metadata_csv)
    neatx_labels = load_neatx_drain_labels(neatx_csv)

    manifest = build_manifest(records, ratios=SPLIT_RATIOS, seed=SPLIT_SEED)
    manifest = attach_donor_eligibility(manifest, neatx_labels)
    manifest = attach_gate_subset(manifest, fraction=GATE_FRACTION, seed=GATE_SEED)

    assert_no_patient_leakage(manifest)
    return manifest

def build_and_save_real_manifest(nih_metadata_csv: Path, neatx_csv: Path, output_path: Path) -> str:
    manifest = build_real_manifest(nih_metadata_csv, neatx_csv)
    return save_manifest(manifest, output_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save the real Stage 0 classification split manifest")
    parser.add_argument("--nih-metadata-csv", type=Path, default=DEFAULT_NIH_METADATA_CSV)
    parser.add_argument("--neatx-csv", type=Path, default=DEFAULT_NEATX_CSV)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    digest = build_and_save_real_manifest(args.nih_metadata_csv, args.neatx_csv, args.output_path)
    print(f"wrote manifest to {args.output_path} (sha256={digest})")

if __name__ == "__main__":
    main()
