# Derives the cleaned reporting population C of validation patients that were in neither selection half.
# Run: python3 data_prep/clean_population.py   (seconds)

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data/processed/manifests/nih_pneumothorax.csv"
OUT_DIR = REPO_ROOT / "review/clean_population/outputs"
VAL_SUBSET_SEED = 2000
VAL_SUBSET_FRACTION = 0.5

EXPECTED = {
    "n_val_patients": 4621,
    "n_select_int": 2310,
    "n_select_str": 2310,
    "n_select_int_and_select_str": 1165,
    "n_union_of_select_halves": 3455,
    "n_clean_patients": 1166,
    "n_never_selected_and_calibration_fit": 1145,
}

def legacy_assign_val_subset(patient_ids, fraction: float = VAL_SUBSET_FRACTION, seed: int = VAL_SUBSET_SEED) -> dict:
    unique_ids = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    n_select = int(round(len(shuffled) * fraction))
    assignment: dict = {}
    for pid in shuffled[:n_select]:
        assignment[pid] = "select"
    for pid in shuffled[n_select:]:
        assignment[pid] = "report"
    return assignment

def _halves(assignment: dict) -> tuple[set, set]:
    select = {str(p) for p, v in assignment.items() if v == "select"}
    report = {str(p) for p, v in assignment.items() if v == "report"}
    return select, report

def val_halves(manifest_path: Path = MANIFEST) -> dict:
    m_int = pd.read_csv(manifest_path)
    m_str = pd.read_csv(manifest_path, dtype={"patient_id": str})
    ids_int = m_int.loc[m_int["split"] == "val", "patient_id"].unique()
    ids_str = m_str.loc[m_str["split"] == "val", "patient_id"].unique()
    s_int, r_int = _halves(legacy_assign_val_subset(ids_int))
    s_str, r_str = _halves(legacy_assign_val_subset(ids_str))
    return {"S_int": s_int, "R_int": r_int, "S_str": s_str, "R_str": r_str, "val": {str(p) for p in ids_str}}

def clean_population(halves: dict) -> set:
    return halves["val"] - (halves["S_int"] | halves["S_str"])

def id_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()

def counts_for(manifest_path: Path, patients: set) -> dict:
    m = pd.read_csv(manifest_path, dtype={"patient_id": str})
    sub = m[(m["split"] == "val") & (m["patient_id"].isin(patients))]
    pos = sub[sub["pneumothorax"] == 1]
    return {"patients": int(sub["patient_id"].nunique()), "images": int(len(sub)), "positive_images": int(len(pos)),
            "negative_images": int(len(sub) - len(pos)), "positive_patients": int(pos["patient_id"].nunique())}

def build_report(manifest_path: Path = MANIFEST) -> dict:
    h = val_halves(manifest_path)
    c = clean_population(h)
    got = {
        "n_val_patients": len(h["val"]),
        "n_select_int": len(h["S_int"]),
        "n_select_str": len(h["S_str"]),
        "n_select_int_and_select_str": len(h["S_int"] & h["S_str"]),
        "n_union_of_select_halves": len(h["S_int"] | h["S_str"]),
        "n_clean_patients": len(c),
        "n_never_selected_and_calibration_fit": len(h["R_int"] & h["S_str"]),
    }

    assert not (c & h["S_int"]) and not (c & h["S_str"]), "clean population overlaps a selection/fitting half"
    assert c == (h["R_int"] & h["R_str"]), "clean population is not R_int intersect R_str"
    return {
        "expected_arithmetic_fixed_before_outcomes": EXPECTED,
        "realized": got,
        "arithmetic_matches": got == EXPECTED,
        "clean_population": counts_for(manifest_path, c),
        "never_selected_half_R_int": counts_for(manifest_path, h["R_int"]),
        "val_report_half_R_str": counts_for(manifest_path, h["R_str"]),
        "sha256_of_sorted_ids": {k: id_digest(h[k]) for k in ("S_int", "R_int", "S_str", "R_str")} | {"clean": id_digest(c)},
    }

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = build_report()
    (OUT_DIR / "clean_population.json").write_text(json.dumps(rep, indent=2))
    h = val_halves()
    (OUT_DIR / "clean_population_patient_ids.txt").write_text("\n".join(sorted(clean_population(h))) + "\n")
    print(json.dumps({k: rep[k] for k in ("realized", "arithmetic_matches", "clean_population", "never_selected_half_R_int")}, indent=2))
    if not rep["arithmetic_matches"]:
        print("HALT: the realized arithmetic differs from the arithmetic fixed before any outcome existed", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
