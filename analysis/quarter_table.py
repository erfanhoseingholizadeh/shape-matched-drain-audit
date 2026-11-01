# Tabulates AUC and drain composition in the four validation quarters defined by the two selection halves.
# Run: python3 analysis/quarter_table.py (no options; it starts working at once)

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from data_prep import clean_population as cp
from analysis import random_half_rate as rh

OUT_DIR = REPO_ROOT / "review/population_assignment/outputs"
NEATX_CSV = REPO_ROOT / "data/raw/neatx/neatx.csv"
SEEDS = [0, 1, 2, 3, 4]

def quarter_table(df: pd.DataFrame, quarters: dict, drain: pd.DataFrame | None) -> dict:
    out = {}
    d = df.merge(drain[["image_id", "drain"]], on="image_id", how="left") if drain is not None else df.assign(drain=np.nan)
    for name, patients in quarters.items():
        x = d[d["patient_id"].isin(patients)]
        pos = x[x["pneumothorax_label"] == 1]
        out[name] = {"n_patients": int(x["patient_id"].nunique()), "n_images": int(len(x)),
                     "n_positive_patients": int(pos["patient_id"].nunique()), "n_positive_images": int(len(pos)),
                     "auc": rh.fast_auc(x["pneumothorax_label"].to_numpy().astype(int), x["prob"].to_numpy()),
                     "drain_composition": {"present": int((pos["drain"] == True).sum()), "absent": int((pos["drain"] == False).sum()),
                                           "unlabelled": int(pos["drain"].isna().sum())}}
    return out

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h = cp.val_halves()
    quarters = {"Q1_Sint_and_Sstr": h["S_int"] & h["S_str"], "Q2_Sint_and_Rstr": h["S_int"] & h["R_str"],
                "Q3_Rint_and_Sstr": h["R_int"] & h["S_str"], "Q4_C_Rint_and_Rstr": h["R_int"] & h["R_str"]}
    neatx = pd.read_csv(NEATX_CSV)[["Image Index", "Drain"]].rename(columns={"Image Index": "image_id", "Drain": "drain"})
    rep: dict = {}
    for arch in rh.CSV:
        d = pd.read_csv(rh.CSV[arch], dtype={"patient_id": str})
        per_seed = {s: quarter_table(d[d["seed"] == s], quarters, neatx if s == 0 else None) for s in SEEDS}
        mean_auc = {q: float(np.mean([per_seed[s][q]["auc"] for s in SEEDS])) for q in quarters}
        rep[arch] = {"five_seed_mean_auc_by_quarter": mean_auc,
                     "per_seed_auc": {str(s): {q: per_seed[s][q]["auc"] for q in quarters} for s in SEEDS},
                     "counts_and_drain_composition_seed0": {q: per_seed[0][q] for q in quarters}}
        print(arch, {q: round(v, 4) for q, v in mean_auc.items()})
    (OUT_DIR / "quarter_table.json").write_text(json.dumps(rep, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
