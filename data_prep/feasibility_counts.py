# Counts NEATX-labelled pneumothorax positives of the test split by half, drain label, image and patient, after each filter.
# Run: python3 data_prep/feasibility_counts.py   -> review/feasibility/counts.json

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent
MAN = R / "data/processed/manifests"
EVAL_ARMS = ("eval_paired", "eval_unpaired_reference")
MIN_RCA = 0.7

def tally(df: pd.DataFrame) -> dict:
    out = {}
    for name, sub in (("all", df), ("drain_present", df[df.Drain == 1]), ("drain_absent", df[df.Drain == 0])):
        out[name] = {"images": int(len(sub)), "patients": int(sub.patient_id.nunique())}
    both = set(df[df.Drain == 1].patient_id) & set(df[df.Drain == 0].patient_id)
    out["patients_with_both_labels"] = len(both)
    return out

def main() -> None:
    neatx = pd.read_csv(R / "data/raw/neatx/neatx.csv").rename(columns={"Image Index": "image_id"})[["image_id", "Drain"]]
    nih = pd.read_csv(MAN / "nih_pneumothorax.csv", dtype={"patient_id": str})
    disc = pd.read_csv(MAN / "discriminator_split.csv", dtype={"patient_id": str})
    rca = pd.read_csv(R / "data/raw/chexmask/chestxray8.csv", usecols=["Image Index", "Dice RCA (Mean)"]).drop_duplicates("Image Index")
    rca = rca.set_index("Image Index")["Dice RCA (Mean)"]
    assert neatx.image_id.is_unique
    j = neatx.merge(nih, on="image_id", how="left", validate="one_to_one")
    assert j.split.notna().all(), "a NEATX row is missing from the manifest"
    assert j.pneumothorax.astype(bool).all(), "a NEATX row is not pneumothorax-positive in the manifest"
    reserved_patients = set(disc.loc[disc.discriminator_arm.isin(EVAL_ARMS), "patient_id"])
    out = {"min_rca": MIN_RCA, "neatx_rows": int(len(j)), "neatx_by_split": {k: int(v) for k, v in j.split.value_counts().items()},
           "n_reserved_arm_patients": len(reserved_patients), "halves": {}}
    test = j[j.split == "test"].copy()
    assert set(test.gate_subset.dropna()) <= {"dev", "gate"} and test.gate_subset.notna().all()
    test["rca"] = test.image_id.map(rca)
    for half in ("dev", "gate"):
        h = test[test.gate_subset == half]
        s1 = h[~h.patient_id.isin(reserved_patients)]
        s2 = s1[s1.rca >= MIN_RCA]
        out["halves"][half] = {
            "s0_in_test_half": tally(h),
            "s1_minus_reserved_arm_patients": tally(s1),
            "s2_after_rca_filter": tally(s2),
            "s1_reserved_patients_removed": int(h.patient_id.isin(reserved_patients).sum()),
            "s2_no_chexmask_row": int(s1.rca.isna().sum()),
            "s2_below_threshold": int((s1.rca < MIN_RCA).sum()),
        }
    (R / "review/feasibility/counts.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
