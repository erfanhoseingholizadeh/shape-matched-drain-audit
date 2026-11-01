# Chooses the planting-plan seed of the lower near-margin setting by a fixed rule on realised planting rates.
# Run: python3 training/near_margin_seed_rule.py

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "."); sys.path.insert(0, "src")
from editing import planting_plan as P
from cxr_shortcuts.data.gate_guard import load_manifest, load_table

TARGET_DELTA, DELTA_TOL, PPOS_NOMINAL, PPOS_TOL = 0.1226, 0.004, 0.565, 0.015
CANDIDATES = range(61107, 61121)

def realized(eligible: pd.DataFrame, probs: tuple, seed: int) -> tuple:
    ordered = eligible.sort_values("image_id", kind="stable").reset_index(drop=True)
    draws = np.random.default_rng(seed).random(len(ordered))
    pos = ordered.pneumothorax.astype(bool).to_numpy()
    ins = draws < np.where(pos, probs[0], probs[1])
    return float(ins[pos].mean()), float(ins[~pos].mean())

def main() -> None:
    man = load_manifest(P.MANIFEST_PATH)
    rca = load_table(P.CHEXMASK_PATH, id_col="Image Index", id_kind="image", usecols=["Image Index", "Dice RCA (Mean)"])
    eligible, _ = P.rca_eligible_any_label(man[man.split == "train"], rca)
    for seed in CANDIDATES:
        a, b = realized(eligible, P.INSERT_PROB["e3lo"], seed)
        ok = abs((a - b) - TARGET_DELTA) <= DELTA_TOL and abs(a - PPOS_NOMINAL) <= PPOS_TOL
        print(seed, round(a, 4), round(b, 4), "delta", round(a - b, 4), "ACCEPT" if ok else "reject")
        if ok:
            return
    print("no candidate qualified: amendment withdrawn")

if __name__ == "__main__":
    main()
