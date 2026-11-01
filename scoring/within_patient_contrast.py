# Within-patient paired contrast on validation films for five checkpoints (imported by operating_point_definition.py).
# Run: python3 scoring/within_patient_contrast.py --help

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_RESAMPLES = 2000
CI = 0.95
BOOTSTRAP_SEED = 0
MIN_PATIENTS_FOR_CI = 20
ORDINARY_LOGIT_FLOOR = 0.01

def load_merged(val_csv: Path, neatx_csv: Path, nih_csv: Path) -> pd.DataFrame:
    val = pd.read_csv(val_csv, dtype={"patient_id": str})
    neatx = pd.read_csv(neatx_csv)[["Image Index", "Drain"]].rename(
        columns={"Image Index": "image_id", "Drain": "drain_present"}
    )
    neatx["drain_present"] = neatx["drain_present"].astype(bool)
    nih = pd.read_csv(nih_csv)[["Image Index", "Follow-up #"]].rename(
        columns={"Image Index": "image_id", "Follow-up #": "followup"}
    )
    merged = val.merge(neatx, on="image_id", how="inner").merge(nih, on="image_id", how="left")
    n_missing_followup = int(merged["followup"].isna().sum())
    if n_missing_followup:
        raise AssertionError(f"{n_missing_followup} NEATX-labeled rows have no NIH Follow-up # match -- unexpected, stop")
    return merged

def find_both_arm_patients(pos: pd.DataFrame) -> pd.DataFrame:
    per_patient = pos.groupby("patient_id")["drain_present"].agg(["sum", "count"])
    per_patient["n_drain"] = per_patient["sum"].astype(int)
    per_patient["n_nodrain"] = (per_patient["count"] - per_patient["sum"]).astype(int)
    both = per_patient[(per_patient["n_drain"] >= 1) & (per_patient["n_nodrain"] >= 1)]
    return both[["n_drain", "n_nodrain"]]

def per_patient_paired_difference(pos_seed: pd.DataFrame, both_patients: list[str], value_col: str) -> pd.DataFrame:
    sub = pos_seed[pos_seed["patient_id"].isin(both_patients)]
    means = sub.groupby(["patient_id", "drain_present"])[value_col].mean().unstack("drain_present")
    means = means.rename(columns={True: "drain_present_mean", False: "drain_absent_mean"})
    means["paired_diff"] = means["drain_present_mean"] - means["drain_absent_mean"]
    return means

def cluster_bootstrap_ci(values: np.ndarray, n_resamples: int = N_RESAMPLES, ci: float = CI, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        estimates[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(estimates, [alpha, 1 - alpha])
    return float(lo), float(hi)

def classify_verdict(mean_diff: float, ci_low: float, ci_high: float, sign_pos: int, sign_neg: int, n_patients: int) -> str:
    if n_patients < MIN_PATIENTS_FOR_CI:
        return "UNINTERPRETABLE-TOO-FEW-PATIENTS"
    ci_excludes_zero = not (ci_low <= 0 <= ci_high)
    if ci_excludes_zero and mean_diff < 0:
        return "NEGATIVE-CONTRADICTS-RELIANCE"
    if not ci_excludes_zero or abs(mean_diff) < ORDINARY_LOGIT_FLOOR:
        return "NULL"
    majority_sign_positive = sign_pos > sign_neg
    if mean_diff > 0 and majority_sign_positive:
        return "EVIDENCE-CONSISTENT-WITH-RELIANCE"
    return "AMBIGUOUS"

def run_population(merged: pd.DataFrame, seeds: list[int], population_name: str) -> dict:
    results: dict = {}
    both_ref = None
    for seed in seeds:
        seed_df = merged[merged["seed"] == seed].copy()
        pos = seed_df[seed_df["pneumothorax_label"] == 1]
        both = find_both_arm_patients(pos)
        if both_ref is None:
            both_ref = both
        else:
            if not both.index.equals(both_ref.index):
                raise AssertionError(f"[{population_name}] both-arm patient set differs across seeds -- should be seed-invariant")
        both_patients = list(both.index)

        logit_pd = per_patient_paired_difference(pos, both_patients, "logit")
        diffs = logit_pd["paired_diff"].values
        mean_diff = float(diffs.mean())
        ci_low, ci_high = cluster_bootstrap_ci(diffs)
        sign_pos = int((diffs > 0).sum())
        sign_neg = int((diffs < 0).sum())
        sign_zero = int((diffs == 0).sum())
        quartiles = {"q25": float(np.quantile(diffs, 0.25)), "median": float(np.median(diffs)), "q75": float(np.quantile(diffs, 0.75))}

        prob_pd = per_patient_paired_difference(pos, both_patients, "prob")
        prob_diffs = prob_pd["paired_diff"].values
        prob_mean_diff = float(prob_diffs.mean())
        prob_ci_low, prob_ci_high = cluster_bootstrap_ci(prob_diffs)

        verdict = classify_verdict(mean_diff, ci_low, ci_high, sign_pos, sign_neg, len(both_patients))

        results[seed] = {
            "n_patients": len(both_patients),
            "n_drain_images_used": int(both["n_drain"].sum()),
            "n_nodrain_images_used": int(both["n_nodrain"].sum()),
            "logit": {
                "mean_paired_diff": mean_diff, "ci_low": ci_low, "ci_high": ci_high,
                "quartiles": quartiles, "sign_positive": sign_pos, "sign_negative": sign_neg, "sign_zero": sign_zero,
                "distribution": [float(x) for x in np.sort(diffs)],
            },
            "prob": {"mean_paired_diff": prob_mean_diff, "ci_low": prob_ci_low, "ci_high": prob_ci_high},
            "verdict": verdict,
        }
        print(f"[{population_name}][seed {seed}] n_patients={len(both_patients)} mean_logit_diff={mean_diff:.4f} "
              f"CI=({ci_low:.4f},{ci_high:.4f}) sign_pos={sign_pos} sign_neg={sign_neg} verdict={verdict}")

    diffs_across = np.array([results[s]["logit"]["mean_paired_diff"] for s in seeds])
    spread = {
        "mean_of_means": float(diffs_across.mean()), "std_of_means": float(diffs_across.std(ddof=1)),
        "range_of_means": float(diffs_across.max() - diffs_across.min()),
        "n_patients_per_seed": [results[s]["n_patients"] for s in seeds],
        "verdicts": [results[s]["verdict"] for s in seeds],
    }
    print(f"[{population_name}][across-seed spread] mean_of_means={spread['mean_of_means']:.4f} "
          f"std={spread['std_of_means']:.4f} range={spread['range_of_means']:.4f} verdicts={spread['verdicts']}")
    return {"per_seed": results, "across_seed_spread": spread}

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val-csv", type=Path, required=True)
    p.add_argument("--neatx-csv", type=Path, required=True)
    p.add_argument("--nih-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    merged = load_merged(args.val_csv, args.neatx_csv, args.nih_csv)
    seeds = sorted(merged["seed"].unique().tolist())
    print(f"seeds present: {seeds}")

    output = {}
    output["full_val"] = run_population(merged, seeds, "full_val")
    report_only = merged[merged["val_subset"] == "report"]
    output["val_report_secondary"] = run_population(report_only, seeds, "val_report_secondary")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print("DONE")

if __name__ == "__main__":
    main()
