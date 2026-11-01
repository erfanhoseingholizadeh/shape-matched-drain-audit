# Builds the list of reserved image ids that the loader guard refuses to read.
# Run: python3 data_prep/build_reserved_image_ids.py [--out PATH]

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = REPO_ROOT / "data/processed/manifests"
EVAL_ARMS = ("eval_paired", "eval_unpaired_reference")

def reserved_image_ids(disc: pd.DataFrame, nih: pd.DataFrame) -> list:
    patients = set(disc.loc[disc["discriminator_arm"].isin(EVAL_ARMS), "patient_id"].astype(str))
    patients |= set(nih.loc[nih["gate_subset"] == "gate", "patient_id"].astype(str))
    ids = set(nih.loc[nih["patient_id"].astype(str).isin(patients), "image_id"])
    ids |= set(disc.loc[disc["patient_id"].astype(str).isin(patients), "image_id"])
    return sorted(ids)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=MANIFESTS / "reserved_image_ids.txt")
    args = ap.parse_args()
    disc = pd.read_csv(MANIFESTS / "discriminator_split.csv", dtype={"patient_id": str})
    nih = pd.read_csv(MANIFESTS / "nih_pneumothorax.csv", dtype={"patient_id": str})
    ids = reserved_image_ids(disc, nih)
    text = "\n".join(ids) + "\n"
    args.out.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    args.out.with_name(args.out.name + ".sha256").write_text(f"{digest}  {args.out.name}\n")
    print(f"wrote {len(ids)} reserved image ids to {args.out} (sha256 {digest})")

if __name__ == "__main__":
    main()
