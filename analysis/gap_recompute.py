# Recomputes the AUC gap between drain-labelled and drain-unlabelled positives on its assigned population.
# Run: python3 analysis/gap_recompute.py (no options; it starts working at once)

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scoring import subgroup_anchor as p44
from data_prep import clean_population as cp

OUT_DIR = REPO_ROOT / "review/population_assignment/outputs"
VAL_CSV = REPO_ROOT / "review/densenet121_remeasure/outputs/val_full_per_image.csv"
NEATX_CSV = REPO_ROOT / "data/raw/neatx/neatx.csv"
RECORDED = REPO_ROOT / "review/densenet121_remeasure/outputs/subgroup_anchor.json"
SEEDS = [0, 1, 2, 3, 4]
OKAKDEN_RAYNER_GAP = 0.17

def weighted_auc(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w_pos = w * (labels == 1)
    w_neg = w * (labels == 0)
    tp, tn = w_pos.sum(), w_neg.sum()
    if tp == 0 or tn == 0:
        return float("nan")
    uniq, inv = np.unique(scores, return_inverse=True)
    gp = np.bincount(inv, weights=w_pos, minlength=len(uniq))
    gn = np.bincount(inv, weights=w_neg, minlength=len(uniq))
    below_neg = np.cumsum(gn) - gn
    return float((gp * (below_neg + 0.5 * gn)).sum() / (tp * tn))

def fast_gap_ci(df: pd.DataFrame, n_resamples: int = p44.N_RESAMPLES, ci: float = p44.CI, seed: int = p44.BOOTSTRAP_SEED) -> dict:
    arm = df["arm"].to_numpy()
    prob = df["prob"].to_numpy()
    unique_patients, code = np.unique(df["patient_id"].to_numpy(), return_inverse=True)
    n_clusters = len(unique_patients)
    drain_sub = np.flatnonzero((arm == "drain") | (arm == "negative"))
    nodrain_sub = np.flatnonzero((arm == "nodrain") | (arm == "negative"))
    d_lab, nd_lab = (arm[drain_sub] == "drain").astype(int), (arm[nodrain_sub] == "nodrain").astype(int)
    is_neg, is_drain, is_nodrain = arm == "negative", arm == "drain", arm == "nodrain"
    rng = np.random.default_rng(seed)
    est = np.empty(n_resamples)
    est_d = np.empty(n_resamples)
    est_nd = np.empty(n_resamples)
    n_degenerate = 0
    for i in range(n_resamples):
        counts = np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)
        w = counts[code]
        if w[is_neg].sum() == 0 or w[is_drain].sum() == 0 or w[is_nodrain].sum() == 0:
            est[i] = est_d[i] = est_nd[i] = np.nan
            n_degenerate += 1
            continue
        d = weighted_auc(prob[drain_sub], d_lab, w[drain_sub])
        nd = weighted_auc(prob[nodrain_sub], nd_lab, w[nodrain_sub])
        est[i], est_d[i], est_nd[i] = d - nd, d, nd

    def ci_of(arr):
        valid = arr[~np.isnan(arr)]
        a = (1 - ci) / 2
        return (float("nan"), float("nan")) if len(valid) == 0 else tuple(float(x) for x in np.quantile(valid, [a, 1 - a]))

    lo, hi = ci_of(est)
    dlo, dhi = ci_of(est_d)
    nlo, nhi = ci_of(est_nd)
    return {"n_resamples": n_resamples, "n_degenerate": n_degenerate, "diff_ci_low": lo, "diff_ci_high": hi,
            "auc_drain_ci_low": dlo, "auc_drain_ci_high": dhi, "auc_nodrain_ci_low": nlo, "auc_nodrain_ci_high": nhi}

def gap_for(merged: pd.DataFrame, seeds=SEEDS) -> dict:
    out = {}
    for s in seeds:
        tagged = p44.build_arm(merged[merged["seed"] == s].copy())
        out[str(s)] = {**p44.point_estimate(tagged), **fast_gap_ci(tagged)}
    diffs = np.array([out[str(s)]["diff"] for s in seeds])
    out["_spread"] = {"diff_mean": float(diffs.mean()), "diff_std_ddof1": float(diffs.std(ddof=1)), "ci_high_max": float(max(out[str(s)]["diff_ci_high"] for s in seeds)),
                      "ci_high_per_seed": [out[str(s)]["diff_ci_high"] for s in seeds],
                      "n_seeds_with_ci_high_below_0_17": int(sum(out[str(s)]["diff_ci_high"] < OKAKDEN_RAYNER_GAP for s in seeds))}
    return out

def oracle() -> dict:
    rec = json.loads(RECORDED.read_text())
    merged = p44.load_merged(VAL_CSV, NEATX_CSV)
    rows = []

    def chk(name, recorded, got, tol):
        rows.append({"name": name, "recorded": recorded, "recomputed": got, "abs_diff": abs(recorded - got), "ok": abs(recorded - got) <= tol})

    for label, frame in (("full_val", merged), ("val_report_secondary", merged[merged["val_subset"] == "report"])):
        got = gap_for(frame)
        for s in SEEDS:
            r, g = rec[label]["per_seed"][str(s)], got[str(s)]
            for k in ("n_drain", "n_nodrain", "n_negative", "n_drain_patients", "n_nodrain_patients", "n_negative_patients", "n_degenerate"):
                chk(f"{label}/{s}/{k}", r[k], g[k], 0)
            for k in ("auc_drain", "auc_nodrain", "diff", "diff_ci_low", "diff_ci_high", "auc_drain_ci_low", "auc_drain_ci_high", "auc_nodrain_ci_low", "auc_nodrain_ci_high"):
                chk(f"{label}/{s}/{k}", r[k], g[k], 1e-9)
        for k in ("diff_mean", "diff_std"):
            chk(f"{label}/spread/{k}", rec[label]["across_seed_spread"][k], got["_spread"]["diff_mean"] if k == "diff_mean" else got["_spread"]["diff_std_ddof1"], 1e-9)
    fails = [r for r in rows if not r["ok"]]
    return {"all_passed": not fails, "n_checks": len(rows), "n_failed": len(fails), "failures": fails[:20], "max_abs_diff": max(r["abs_diff"] for r in rows), "rows": rows}

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    orc = oracle()
    (OUT_DIR / "gap_oracle.json").write_text(json.dumps(orc, indent=2))
    print(f"oracle: {'PASSED' if orc['all_passed'] else 'FAILED'}  checks={orc['n_checks']} failed={orc['n_failed']} max_abs_diff={orc['max_abs_diff']:.3g}")
    if not orc["all_passed"]:
        print("REFUSED to compute R_int or C: the oracle failed", file=sys.stderr)
        for r in orc["failures"][:8]:
            print("  FAIL", r, file=sys.stderr)
        return 1
    h = cp.val_halves()
    merged = p44.load_merged(VAL_CSV, NEATX_CSV)
    pops = {"R_int_assigned": h["R_int"], "C_other_candidate": cp.clean_population(h), "S_int_descriptive": h["S_int"]}
    res = {"oracle_passed": True, "populations": {}}
    for name, patients in pops.items():
        frame = merged[merged["patient_id"].isin(patients)]
        res["populations"][name] = {"n_patients_in_population": len(patients), **gap_for(frame)}
        sp = res["populations"][name]["_spread"]
        print(f"{name}: mean gap {sp['diff_mean']:.4f} (sd {sp['diff_std_ddof1']:.4f}); CI upper per seed {[round(x, 4) for x in sp['ci_high_per_seed']]}; "
              f"seeds with upper bound below 0.17: {sp['n_seeds_with_ci_high_below_0_17']} of 5")
    (OUT_DIR / "gap_assigned.json").write_text(json.dumps(res, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
