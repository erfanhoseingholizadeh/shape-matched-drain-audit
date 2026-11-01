# Population-level subgroup anchor (drain-labelled against other positives) re-measured on the five checkpoints; its recorded output is read by
# validity_values.py.
# Run: python3 scoring/subgroup_anchor.py --help

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

N_RESAMPLES = 2000
CI = 0.95
BOOTSTRAP_SEED = 0

def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def load_merged(val_csv: Path, neatx_csv: Path) -> pd.DataFrame:
    val = pd.read_csv(val_csv, dtype={"patient_id": str})
    neatx = pd.read_csv(neatx_csv)[["Image Index", "Drain"]].rename(
        columns={"Image Index": "image_id", "Drain": "drain_present"}
    )
    neatx["drain_present"] = neatx["drain_present"].astype(bool)
    merged = val.merge(neatx, on="image_id", how="left")
    return merged

def build_arm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    is_pos = out["pneumothorax_label"] == 1
    is_drain = out["drain_present"] == True
    is_nodrain = out["drain_present"] == False
    out["arm"] = np.select(
        [~is_pos, is_pos & is_drain, is_pos & is_nodrain],
        ["negative", "drain", "nodrain"],
        default="excluded_unlabeled_positive",
    )
    return out

def clustered_auc_gap_ci(df: pd.DataFrame, n_resamples: int = N_RESAMPLES, ci: float = CI, seed: int = BOOTSTRAP_SEED) -> dict:
    arm = df["arm"].values
    prob = df["prob"].values
    patient_id = df["patient_id"].values

    unique_patients = np.unique(patient_id)
    n_clusters = len(unique_patients)
    patient_to_indices = {p: np.where(patient_id == p)[0] for p in unique_patients}

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples)
    auc_drain_estimates = np.empty(n_resamples)
    auc_nodrain_estimates = np.empty(n_resamples)
    n_degenerate = 0
    for i in range(n_resamples):
        sampled_patients = unique_patients[rng.integers(0, n_clusters, size=n_clusters)]
        idx = np.concatenate([patient_to_indices[p] for p in sampled_patients])
        arm_i = arm[idx]
        prob_i = prob[idx]

        neg_mask = arm_i == "negative"
        drain_mask = arm_i == "drain"
        nodrain_mask = arm_i == "nodrain"
        n_neg, n_drain, n_nodrain = neg_mask.sum(), drain_mask.sum(), nodrain_mask.sum()
        if n_neg == 0 or n_drain == 0 or n_nodrain == 0:
            estimates[i] = np.nan
            auc_drain_estimates[i] = np.nan
            auc_nodrain_estimates[i] = np.nan
            n_degenerate += 1
            continue

        drain_labels = np.concatenate([np.ones(n_drain), np.zeros(n_neg)])
        drain_scores = np.concatenate([prob_i[drain_mask], prob_i[neg_mask]])
        auc_d = rank_auc(drain_labels, drain_scores)

        nodrain_labels = np.concatenate([np.ones(n_nodrain), np.zeros(n_neg)])
        nodrain_scores = np.concatenate([prob_i[nodrain_mask], prob_i[neg_mask]])
        auc_nd = rank_auc(nodrain_labels, nodrain_scores)

        estimates[i] = auc_d - auc_nd
        auc_drain_estimates[i] = auc_d
        auc_nodrain_estimates[i] = auc_nd

    def ci_of(arr):
        valid = arr[~np.isnan(arr)]
        alpha = (1 - ci) / 2
        return (float("nan"), float("nan")) if len(valid) == 0 else tuple(float(x) for x in np.quantile(valid, [alpha, 1 - alpha]))

    diff_lo, diff_hi = ci_of(estimates)
    drain_lo, drain_hi = ci_of(auc_drain_estimates)
    nodrain_lo, nodrain_hi = ci_of(auc_nodrain_estimates)

    return {
        "n_resamples": n_resamples,
        "n_degenerate": n_degenerate,
        "diff_ci_low": diff_lo, "diff_ci_high": diff_hi,
        "auc_drain_ci_low": drain_lo, "auc_drain_ci_high": drain_hi,
        "auc_nodrain_ci_low": nodrain_lo, "auc_nodrain_ci_high": nodrain_hi,
    }

def point_estimate(df: pd.DataFrame) -> dict:
    drain = df[df["arm"] == "drain"]
    nodrain = df[df["arm"] == "nodrain"]
    negative = df[df["arm"] == "negative"]

    drain_labels = np.concatenate([np.ones(len(drain)), np.zeros(len(negative))])
    drain_scores = np.concatenate([drain["prob"].values, negative["prob"].values])
    auc_drain = float(roc_auc_score(drain_labels, drain_scores))
    auc_drain_rank = rank_auc(drain_labels, drain_scores)

    nodrain_labels = np.concatenate([np.ones(len(nodrain)), np.zeros(len(negative))])
    nodrain_scores = np.concatenate([nodrain["prob"].values, negative["prob"].values])
    auc_nodrain = float(roc_auc_score(nodrain_labels, nodrain_scores))
    auc_nodrain_rank = rank_auc(nodrain_labels, nodrain_scores)

    return {
        "n_drain": int(len(drain)), "n_nodrain": int(len(nodrain)), "n_negative": int(len(negative)),
        "n_drain_patients": int(drain["patient_id"].nunique()),
        "n_nodrain_patients": int(nodrain["patient_id"].nunique()),
        "n_negative_patients": int(negative["patient_id"].nunique()),
        "auc_drain": auc_drain, "auc_drain_rank": auc_drain_rank,
        "auc_nodrain": auc_nodrain, "auc_nodrain_rank": auc_nodrain_rank,
        "diff": auc_drain - auc_nodrain, "diff_rank": auc_drain_rank - auc_nodrain_rank,
    }

def run_population(merged: pd.DataFrame, seeds: list[int], population_name: str) -> dict:
    out = {}
    for seed in seeds:
        seed_df = merged[merged["seed"] == seed].copy()
        tagged = build_arm(seed_df)
        pe = point_estimate(tagged)
        ci = clustered_auc_gap_ci(tagged)
        result = {**pe, **ci}
        out[seed] = result
        print(f"[{population_name}][seed {seed}] auc_drain={pe['auc_drain']:.4f} auc_nodrain={pe['auc_nodrain']:.4f} "
              f"diff={pe['diff']:.4f} diff_CI=({ci['diff_ci_low']:.4f},{ci['diff_ci_high']:.4f}) "
              f"n_drain={pe['n_drain']} n_nodrain={pe['n_nodrain']} n_negative={pe['n_negative']} "
              f"n_degenerate={ci['n_degenerate']}")

    aucs_drain = np.array([out[s]["auc_drain"] for s in seeds])
    aucs_nodrain = np.array([out[s]["auc_nodrain"] for s in seeds])
    diffs = np.array([out[s]["diff"] for s in seeds])
    spread = {
        "auc_drain_mean": float(aucs_drain.mean()), "auc_drain_std": float(aucs_drain.std(ddof=1)), "auc_drain_range": float(aucs_drain.max() - aucs_drain.min()),
        "auc_nodrain_mean": float(aucs_nodrain.mean()), "auc_nodrain_std": float(aucs_nodrain.std(ddof=1)), "auc_nodrain_range": float(aucs_nodrain.max() - aucs_nodrain.min()),
        "diff_mean": float(diffs.mean()), "diff_std": float(diffs.std(ddof=1)), "diff_range": float(diffs.max() - diffs.min()),
    }
    print(f"[{population_name}][across-seed spread] auc_drain mean={spread['auc_drain_mean']:.4f} std={spread['auc_drain_std']:.4f} range={spread['auc_drain_range']:.4f} | "
          f"auc_nodrain mean={spread['auc_nodrain_mean']:.4f} std={spread['auc_nodrain_std']:.4f} range={spread['auc_nodrain_range']:.4f} | "
          f"diff mean={spread['diff_mean']:.4f} std={spread['diff_std']:.4f} range={spread['diff_range']:.4f}")
    return {"per_seed": out, "across_seed_spread": spread}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val-csv", type=Path, required=True)
    p.add_argument("--neatx-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    merged = load_merged(args.val_csv, args.neatx_csv)
    seeds = sorted(merged["seed"].unique().tolist())
    print(f"seeds present: {seeds}")

    results = {}
    full_val = merged
    results["full_val"] = run_population(full_val, seeds, "full_val")

    report_only = merged[merged["val_subset"] == "report"]
    results["val_report_secondary"] = run_population(report_only, seeds, "val_report_secondary")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print("DONE")

if __name__ == "__main__":
    main()
