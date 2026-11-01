# Estimates how often a random half of validation of the cleaned population's size would trip the validity halt.
# Run: python3 analysis/random_half_rate.py (no options; it starts working at once)

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from data_prep import clean_population as cp

OUT_DIR = REPO_ROOT / "review/population_assignment/outputs"
BAR_A = 0.82
SEED = 57
REGISTERED_DRAWS = 2000
N_DRAWS = int(os.environ.get("RANDOM_HALF_N_DRAWS", str(REGISTERED_DRAWS)))
SEEDS = [0, 1, 2, 3, 4]
CSV = {
    "densenet121": REPO_ROOT / "review/densenet121_remeasure/outputs/val_full_per_image.csv",
    "resnet50": REPO_ROOT / "review/resnet50_remeasure/outputs/val_full_per_image_resnet50.csv",
}

def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def draw_patient_masks(n_patients: int, size: int, n_draws: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    masks = np.zeros((n_draws, n_patients), dtype=bool)
    for i in range(n_draws):
        masks[i, rng.choice(n_patients, size=size, replace=False)] = True
    return masks

def subset_aucs(patient_code: np.ndarray, labels: np.ndarray, scores: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = np.empty(len(masks))
    for i, m in enumerate(masks):
        keep = m[patient_code]
        out[i] = fast_auc(labels[keep], scores[keep])
    return out

def share_at_or_below(draws: np.ndarray, value: float) -> float:
    return float(np.mean(np.asarray(draws) <= value))

def summarize_draws(aucs_by_checkpoint: dict, bar: float = BAR_A) -> dict:
    per = {}
    for name, a in aucs_by_checkpoint.items():
        a = np.asarray(a)
        per[name] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
                     "p2_5": float(np.percentile(a, 2.5)), "p50": float(np.percentile(a, 50)), "p97_5": float(np.percentile(a, 97.5)),
                     "p_below_bar": float(np.mean(a < bar))}
    stack = np.vstack([np.asarray(a) for a in aucs_by_checkpoint.values()])
    below = (stack < bar).sum(axis=0)
    return {"per_checkpoint": per, "p_any_below_bar": float(np.mean(below >= 1)), "p_at_least_two_below_bar": float(np.mean(below >= 2)),
            "n_draws": int(stack.shape[1])}

def _load(arch: str, r_int: set) -> pd.DataFrame:
    d = pd.read_csv(CSV[arch], dtype={"patient_id": str})
    return d[d["patient_id"].isin(r_int)].reset_index(drop=True)

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    halves = cp.val_halves()
    r_int = sorted(halves["R_int"])
    c_set = cp.clean_population(halves)
    assert c_set <= halves["R_int"], "C must be a subset of the never-selected half"
    code = {p: i for i, p in enumerate(r_int)}
    masks = draw_patient_masks(len(r_int), len(c_set), N_DRAWS)
    aucs: dict = {}
    actual: dict = {}
    for arch in CSV:
        df = _load(arch, halves["R_int"])
        pc = df["patient_id"].map(code).to_numpy()
        for s in SEEDS:
            sd = df["seed"].to_numpy() == s
            key = f"{arch}/{s}"
            aucs[key] = subset_aucs(pc[sd], df["pneumothorax_label"].to_numpy()[sd], df["prob"].to_numpy()[sd], masks)
            in_c = df["patient_id"].isin(c_set).to_numpy() & sd
            actual[key] = fast_auc(df["pneumothorax_label"].to_numpy()[in_c], df["prob"].to_numpy()[in_c])
    summ = summarize_draws(aucs)
    for key, v in summ["per_checkpoint"].items():
        v["actual_C_auc"] = actual[key]
        v["share_of_draws_at_or_below_actual_C"] = share_at_or_below(aucs[key], actual[key])
    per_arch = {}
    for arch in CSV:
        keys = [f"{arch}/{s}" for s in SEEDS]
        mean_draws = np.mean([aucs[k] for k in keys], axis=0)
        actual_mean = float(np.mean([actual[k] for k in keys]))
        per_arch[arch] = {"draw_mean_of_five": float(mean_draws.mean()), "draw_sd_of_five_mean": float(mean_draws.std(ddof=1)),
                          "actual_C_mean_of_five": actual_mean, "share_of_draws_at_or_below_actual_C_mean": share_at_or_below(mean_draws, actual_mean)}
    rep = {"n_draws": N_DRAWS, "registered_draws": REGISTERED_DRAWS, "seed": SEED, "n_patients_R_int": len(r_int), "subset_size": len(c_set),
           "bar": BAR_A, "summary": summ, "per_architecture_five_seed_mean": per_arch}
    suffix = "" if N_DRAWS == REGISTERED_DRAWS else "_reduced"
    (OUT_DIR / f"random_half_rate{suffix}.json").write_text(json.dumps(rep, indent=2))
    print(f"draws={N_DRAWS}  P(at least one of ten below {BAR_A}) = {summ['p_any_below_bar']:.4f}   P(at least two) = {summ['p_at_least_two_below_bar']:.4f}")
    for k, v in summ["per_checkpoint"].items():
        print(f"  {k}: draw mean {v['mean']:.4f} sd {v['sd']:.4f}  P(<0.82)={v['p_below_bar']:.3f}  actual C {v['actual_C_auc']:.4f} "
              f"(share of draws at or below it {v['share_of_draws_at_or_below_actual_C']:.4f})")
    for a, v in per_arch.items():
        print(f"  {a} five-seed mean: draw mean {v['draw_mean_of_five']:.4f} sd {v['draw_sd_of_five_mean']:.4f}; actual C {v['actual_C_mean_of_five']:.4f} "
              f"at share {v['share_of_draws_at_or_below_actual_C_mean']:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
