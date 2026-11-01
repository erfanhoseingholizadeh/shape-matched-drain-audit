# Recomputes the validity quantities of the original checkpoints on the cleaned and the never-selected validation populations.
# Run: python3 analysis/clean_recompute.py --help

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_ci_indices
from cxr_shortcuts.eval.calibration import brier_score, expected_calibration_error
from analysis import operating_point_definition as op46
from analysis import operating_point_sensitivity_definition as op49
from data_prep import clean_population as cp

OUT_DIR = REPO_ROOT / "review/clean_population/outputs"
NEATX_CSV = REPO_ROOT / "data/raw/neatx/neatx.csv"
SEEDS = [0, 1, 2, 3, 4]
BAR_A = 0.82
FLOOR_B = 0.001
TARGETS = op49.TARGET_SENSITIVITIES
N_RESAMPLES = int(os.environ.get("CLEAN_RECOMPUTE_N_RESAMPLES", "2000"))
REGISTERED_RESAMPLES = 2000

ARCHS = {
    "densenet121": {
        "csv": REPO_ROOT / "review/densenet121_remeasure/outputs/val_full_per_image.csv",
        "youden_json": REPO_ROOT / "review/operating_point/operating_point_definition.json",
        "youden_crossing_json": REPO_ROOT / "review/operating_point/operating_point_crossing.json",
    },
    "resnet50": {
        "csv": REPO_ROOT / "review/resnet50_remeasure/outputs/val_full_per_image_resnet50.csv",
        "youden_json": REPO_ROOT / "review/operating_point_resnet50/operating_point_definition_resnet50.json",
        "youden_crossing_json": REPO_ROOT / "review/operating_point_resnet50/operating_point_crossing_resnet50.json",
    },
}
BAND_DEF_JSON = REPO_ROOT / "review/operating_point_band/outputs/operating_point_sensitivity_definition.json"
BAND_CROSSING_JSON = REPO_ROOT / "review/operating_point_band/outputs/operating_point_sensitivity_crossing.json"
TEMP_JSON = REPO_ROOT / "review/calibration/outputs/calibration_fit.json"
AFFINE_JSON = REPO_ROOT / "review/affine_calibration/outputs/affine_calibration_fit.json"
RESNET_REACH_JSON = REPO_ROOT / "review/resnet50_remeasure/outputs/reachability_resnet50.json"
DENSENET_A2_RESULT = REPO_ROOT / "review/densenet121_remeasure/outputs/val_full_rescan.json"

def solve_uniform_shift(logits: np.ndarray, target: float) -> float:
    f = lambda s: float(np.mean(expit(logits + s) - expit(logits))) - target
    assert f(-50.0) < 0 < f(50.0), f"target {target} not bracketed"
    return float(brentq(f, -50.0, 50.0, xtol=1e-10))

def clustered_auc_ci(labels, probs, patient_ids, n_resamples: int = N_RESAMPLES, seed: int = 0) -> dict:
    point = float(roc_auc_score(labels, probs))
    lo, hi = cluster_bootstrap_ci_indices(patient_ids, lambda ix: float(roc_auc_score(labels[ix], probs[ix])),
                                          n_resamples=n_resamples, ci=0.95, seed=seed)
    return {"auc": point, "ci_low": lo, "ci_high": hi}

def classify_clause_a(auc: float, ci_high: float, bar: float = BAR_A) -> str:
    if auc >= bar:
        return "clears"
    return "contradicted_ci_wholly_below_bar" if ci_high < bar else "contradicted_point_below_bar_ci_contains_bar"

def classify_clause_b(median_neg_p: float, floor: float = FLOOR_B) -> str:
    return "clears" if median_neg_p >= floor else "contradicted"

def within_patient_secondary_decision(n_qualifying: int, floor: int = 20) -> str:
    return "supportable" if n_qualifying >= floor else "withdrawn_below_registered_floor"

def seed_statistics(sd: pd.DataFrame, calib: dict | None, bootstrap: bool = True) -> dict:
    labels = sd["pneumothorax_label"].to_numpy()
    probs = sd["prob"].to_numpy()
    z = sd["logit"].to_numpy()
    neg = labels == 0
    out: dict = {"n_images": int(len(sd)), "n_positive": int(labels.sum()), "n_negative": int(neg.sum()),
                 "n_patients": int(sd["patient_id"].nunique())}
    if bootstrap:
        out |= clustered_auc_ci(labels, probs, sd["patient_id"].to_numpy())
    else:
        out["auc"] = float(roc_auc_score(labels, probs))
    out["median_neg_p"] = float(np.median(probs[neg]))
    out["frac_neg_below_1e-3"] = float(np.mean(probs[neg] < 1e-3))
    out["youden"] = op46.youden_threshold(labels, probs)
    out["band"] = {str(t): op49.threshold_for_sensitivity(labels, probs, t) for t in TARGETS}
    if calib is not None:
        T, a, b = calib["T"], calib["a"], calib["b"]
        p_temp, p_aff = expit(z / T), expit(a * z + b)
        out["calibration"] = {
            "T": T, "a": a, "b": b,
            "ece_pre": expected_calibration_error(probs, labels, 10), "brier_pre": brier_score(probs, labels),
            "ece_temperature": expected_calibration_error(p_temp, labels, 10), "brier_temperature": brier_score(p_temp, labels),
            "ece_affine": expected_calibration_error(p_aff, labels, 10), "brier_affine": brier_score(p_aff, labels),
            "median_neg_p_temperature": float(np.median(p_temp[neg])), "median_neg_p_affine": float(np.median(p_aff[neg])),
        }
    out["reachability"] = {"shift_to_0.10": solve_uniform_shift(z[neg], 0.10), "shift_fail_floor_0.02": solve_uniform_shift(z[neg], 0.02)}
    return out

def load_calibration(arch: str) -> dict:
    t = json.loads(TEMP_JSON.read_text())[arch]
    a = json.loads(AFFINE_JSON.read_text())[arch]
    return {s: {"T": t[str(s)]["T_fitted"], "a": a[str(s)]["a"], "b": a[str(s)]["b"]} for s in SEEDS}

def crossing_counts(prob_pd: pd.DataFrame, threshold: float) -> dict:
    present_side = prob_pd["drain_present_mean"] >= threshold
    absent_side = prob_pd["drain_absent_mean"] >= threshold
    n = len(prob_pd)
    opposite = int((present_side != absent_side).sum())
    return {"n_patients": n, "n_opposite_side": opposite, "frac_opposite_side": float(opposite / n),
            "n_both_above": int((present_side & absent_side).sum()), "n_both_below": int((~present_side & ~absent_side).sum()),
            "n_absent_below_present_above": int(((~absent_side) & present_side).sum()),
            "n_absent_above_present_below": int((absent_side & (~present_side)).sum())}

def within_patient_pairs(merged_seed: pd.DataFrame) -> pd.DataFrame:
    pos = merged_seed[merged_seed["pneumothorax_label"] == 1]
    g = pos.groupby("patient_id")["drain_present"].agg(["sum", "count"])
    both = list(g[(g["sum"] >= 1) & ((g["count"] - g["sum"]) >= 1)].index)
    means = pos[pos["patient_id"].isin(both)].groupby(["patient_id", "drain_present"])["prob"].mean().unstack("drain_present")
    return means.rename(columns={True: "drain_present_mean", False: "drain_absent_mean"})

def crossing_block(merged: pd.DataFrame, youden_thr: dict, band_thr: dict) -> dict:
    res: dict = {"youden": {}, "band": {str(t): {} for t in TARGETS}}
    for s in SEEDS:
        pairs = within_patient_pairs(merged[merged["seed"] == s])
        res["youden"][str(s)] = {"threshold": youden_thr[s], **crossing_counts(pairs, youden_thr[s])}
        for t in TARGETS:
            res["band"][str(t)][str(s)] = {"threshold": band_thr[s][t], **crossing_counts(pairs, band_thr[s][t])}
    def pool(per_seed: dict) -> dict:
        n = sum(v["n_patients"] for v in per_seed.values()); opp = sum(v["n_opposite_side"] for v in per_seed.values())
        rel = sum(v["n_absent_below_present_above"] for v in per_seed.values()); rev = sum(v["n_absent_above_present_below"] for v in per_seed.values())
        rates = [v["frac_opposite_side"] for v in per_seed.values()]
        return {"n_patients_total": n, "n_opposite_side_total": opp, "frac_pooled": opp / n, "reliance_direction": rel, "reverse_direction": rev,
                "both_above": sum(v["n_both_above"] for v in per_seed.values()), "both_below": sum(v["n_both_below"] for v in per_seed.values()),
                "per_seed_rate_min": min(rates), "per_seed_rate_max": max(rates)}
    res["youden_pooled"] = pool(res["youden"])
    res["band_pooled"] = {t: pool(v) for t, v in res["band"].items()}
    return res

def load_merged(val_csv: Path) -> pd.DataFrame:
    val = pd.read_csv(val_csv, dtype={"patient_id": str})
    neatx = pd.read_csv(NEATX_CSV)[["Image Index", "Drain"]].rename(columns={"Image Index": "image_id", "Drain": "drain_present"})
    neatx["drain_present"] = neatx["drain_present"].astype(bool)
    return val.merge(neatx, on="image_id", how="inner")

class Oracle:
    def __init__(self) -> None:
        self.rows: list = []

    def check(self, name: str, recorded, got, tol: float) -> None:
        diff = abs(float(recorded) - float(got))
        self.rows.append({"name": name, "recorded": float(recorded), "recomputed": float(got), "abs_diff": diff, "tol": tol, "ok": diff <= tol})

    def exact(self, name: str, recorded, got) -> None:
        self.rows.append({"name": name, "recorded": recorded, "recomputed": got, "abs_diff": 0 if recorded == got else 1, "tol": 0, "ok": recorded == got})

    @property
    def all_passed(self) -> bool:
        return bool(self.rows) and all(r["ok"] for r in self.rows)

    def summary(self) -> dict:
        fails = [r for r in self.rows if not r["ok"]]
        return {"all_passed": self.all_passed, "n_checks": len(self.rows), "n_failed": len(fails), "failures": fails[:20],
                "max_abs_diff": max((r["abs_diff"] for r in self.rows), default=None)}

def run_oracle_checks(arch: str, stats: dict, merged: pd.DataFrame, pop_stats_pooled: dict) -> Oracle:
    o = Oracle()
    cfg = ARCHS[arch]
    yj = json.loads(cfg["youden_json"].read_text())["per_seed"]
    band = json.loads(BAND_DEF_JSON.read_text())["architectures"][arch]
    temp, aff = json.loads(TEMP_JSON.read_text())[arch], json.loads(AFFINE_JSON.read_text())[arch]
    for s in SEEDS:
        st, k = stats[s], str(s)
        o.exact(f"{arch}/{s}/n_images", yj[k]["n_images"], st["n_images"])
        o.exact(f"{arch}/{s}/n_positive", yj[k]["n_positive"], st["n_positive"])
        for f in ("threshold", "sensitivity", "specificity"):
            o.check(f"{arch}/{s}/youden_{f}", yj[k][f], st["youden"][f if f != "threshold" else "threshold"], 1e-9)
        for t in TARGETS:
            for f, g in (("threshold", "threshold"), ("achieved_sensitivity", "achieved_sensitivity")):
                o.check(f"{arch}/{s}/band{t}_{f}", band[k]["per_target_sensitivity"][str(t)][f], st["band"][str(t)][g], 1e-9)
        o.check(f"{arch}/{s}/median_neg_p", temp[k]["median_neg_p_pre"], st["median_neg_p"], 1e-9)
        c = st["calibration"]
        for name, rec in (("ece_pre", temp[k]["ece_pre"]), ("brier_pre", temp[k]["brier_pre"]), ("ece_temperature", aff[k]["ece_temperature"]),
                          ("brier_temperature", aff[k]["brier_temperature"]), ("ece_affine", aff[k]["ece_affine"]), ("brier_affine", aff[k]["brier_affine"]),
                          ("median_neg_p_temperature", temp[k]["median_neg_p_post"]), ("median_neg_p_affine", aff[k]["median_neg_p_affine"])):
            o.check(f"{arch}/{s}/{name}", rec, c[name], 1e-6 if name.startswith("median_neg_p_affine") else 1e-9)
    if arch == "densenet121":
        a2 = json.loads(DENSENET_A2_RESULT.read_text())
        for s in SEEDS:
            r = a2[str(s)]["val_report_auc_ci"]
            for f, g in (("auc", "auc"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
                o.check(f"{arch}/{s}/clause_a_{f}", r[f], stats[s][g], 1e-9)

        o.check("densenet121/0/reach_fail_floor_(0.156 to 3 s.f.)", 0.156, stats[0]["reachability"]["shift_fail_floor_0.02"], 5e-4)
        o.check("densenet121/0/reach_pass_(0.73 to 2 s.f.)", 0.73, stats[0]["reachability"]["shift_to_0.10"], 5e-3)
    else:
        r50 = json.loads(cfg["youden_json"].read_text())["per_seed"]
        for s in SEEDS:
            o.check(f"{arch}/{s}/clause_a_auc", r50[str(s)]["val_report_auc_recomputed"], stats[s]["auc"], 1e-9)
        rj = json.loads(RESNET_REACH_JSON.read_text())
        for s in SEEDS:
            o.exact(f"{arch}/{s}/reach_n_negatives", rj["per_seed"][str(s)]["n_negatives"], stats[s]["n_negative"])
            o.check(f"{arch}/{s}/reach_pass", rj["per_seed"][str(s)]["uniform_logit_shift_for_mean_dp_0.10_PASS"], stats[s]["reachability"]["shift_to_0.10"], 5e-4)
            o.check(f"{arch}/{s}/reach_fail_floor", rj["per_seed"][str(s)]["uniform_logit_shift_for_mean_dp_0.02_FAIL_floor"], stats[s]["reachability"]["shift_fail_floor_0.02"], 5e-4)
        o.check(f"{arch}/pooled/reach_pass", rj["pooled"]["uniform_logit_shift_for_mean_dp_0.10_PASS"], pop_stats_pooled["shift_to_0.10"], 5e-4)
        o.check(f"{arch}/pooled/reach_fail_floor", rj["pooled"]["uniform_logit_shift_for_mean_dp_0.02_FAIL_floor"], pop_stats_pooled["shift_fail_floor_0.02"], 5e-4)

    yc = json.loads(cfg["youden_crossing_json"].read_text())["full_val_primary_n46"]["per_seed"]
    rec_youden = {s: yj[str(s)]["threshold"] for s in SEEDS}
    rec_band = {s: {t: band[str(s)]["per_target_sensitivity"][str(t)]["threshold"] for t in TARGETS} for s in SEEDS}
    cross = crossing_block(merged, rec_youden, rec_band)
    bc = json.loads(BAND_CROSSING_JSON.read_text())["architectures"][arch]["primary_n46"]["per_seed"]
    for s in SEEDS:
        o.exact(f"{arch}/{s}/crossing_youden_opposite", yc[str(s)]["n_opposite_side"], cross["youden"][str(s)]["n_opposite_side"])
        o.exact(f"{arch}/{s}/crossing_youden_n", yc[str(s)]["n_patients"], cross["youden"][str(s)]["n_patients"])
        for t in TARGETS:
            o.exact(f"{arch}/{s}/crossing_band{t}_opposite", bc[str(s)][str(t)]["n_opposite_side"], cross["band"][str(t)][str(s)]["n_opposite_side"])
    return o

def select_population(df: pd.DataFrame, mode: str, halves: dict) -> pd.DataFrame:
    if mode == "oracle":
        return df[df["val_subset"] == "report"]
    if mode == "clean":
        return df[df["patient_id"].isin(cp.clean_population(halves))]
    if mode == "sensitivity":
        return df[df["patient_id"].isin(halves["R_int"])]
    raise ValueError(mode)

def out_path(mode: str, arch: str) -> Path:
    suffix = "" if N_RESAMPLES == REGISTERED_RESAMPLES else "_reduced"
    return OUT_DIR / f"{mode}_{arch}{suffix}.json"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["oracle", "clean", "sensitivity"])
    ap.add_argument("architecture", choices=list(ARCHS))
    args = ap.parse_args()
    arch, mode = args.architecture, args.mode
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if mode != "oracle":
        gate = OUT_DIR / f"oracle_{arch}.json"
        if not gate.exists() or not json.loads(gate.read_text())["oracle"]["all_passed"]:
            print(f"REFUSED: {gate.name} missing or failed; run `oracle {arch}` first and resolve every mismatch", file=sys.stderr)
            return 2
    halves = cp.val_halves()
    df = pd.read_csv(ARCHS[arch]["csv"], dtype={"patient_id": str})
    pop = select_population(df, mode, halves)
    calib = load_calibration(arch) if mode != "sensitivity" else {s: None for s in SEEDS}
    stats = {s: seed_statistics(pop[pop["seed"] == s], calib[s], bootstrap=True) for s in SEEDS}
    neg_logits = np.concatenate([pop[(pop["seed"] == s) & (pop["pneumothorax_label"] == 0)]["logit"].to_numpy() for s in SEEDS])
    pooled_reach = {"n_negatives": int(len(neg_logits)), "shift_to_0.10": solve_uniform_shift(neg_logits, 0.10),
                    "shift_fail_floor_0.02": solve_uniform_shift(neg_logits, 0.02)}
    aucs = np.array([stats[s]["auc"] for s in SEEDS])
    result: dict = {"mode": mode, "architecture": arch, "n_resamples": N_RESAMPLES, "registered_resamples": REGISTERED_RESAMPLES,
                    "population_patients": int(pop["patient_id"].nunique()), "population_images_per_seed": int(len(pop) / len(SEEDS)),
                    "per_seed": {str(s): stats[s] for s in SEEDS},
                    "summary": {"auc_mean": float(aucs.mean()), "auc_std_ddof1": float(aucs.std(ddof=1)), "auc_min": float(aucs.min()),
                                "reachability_pooled": pooled_reach}}
    if mode == "sensitivity":
        result["summary"]["clause_a_note"] = "uncalibrated discrimination clause only; no other quantity is reported on this half"
        out_path(mode, arch).write_text(json.dumps(result, indent=2))
        print(f"[{arch}] sensitivity (never-selected half): mean AUC {aucs.mean():.4f}  per seed {[round(a, 4) for a in aucs]}")
        return 0

    merged = load_merged(ARCHS[arch]["csv"])
    if mode == "oracle":
        o = run_oracle_checks(arch, stats, merged, pooled_reach)
        result["oracle"] = o.summary()
        result["oracle_rows"] = o.rows
        ok = o.all_passed
        print(f"[{arch}] oracle: {'PASSED' if ok else 'FAILED'}  checks={len(o.rows)} failed={result['oracle']['n_failed']} max_abs_diff={result['oracle']['max_abs_diff']:.3g}")
        for r in result["oracle"]["failures"][:8]:
            print("   FAIL", r)
        out_path(mode, arch).write_text(json.dumps(result, indent=2))
        return 0 if ok else 1

    youden_thr = {s: stats[s]["youden"]["threshold"] for s in SEEDS}
    band_thr = {s: {t: stats[s]["band"][str(t)]["threshold"] for t in TARGETS} for s in SEEDS}
    result["crossing_on_primary_n46"] = crossing_block(merged, youden_thr, band_thr)
    clause_a = {str(s): classify_clause_a(stats[s]["auc"], stats[s]["ci_high"]) for s in SEEDS}
    clause_b = {str(s): classify_clause_b(stats[s]["median_neg_p"]) for s in SEEDS}
    result["clauses"] = {"clause_a": clause_a, "clause_b": clause_b,
                         "halt_fired": any(v != "clears" for v in list(clause_a.values()) + list(clause_b.values()))}
    out_path(mode, arch).write_text(json.dumps(result, indent=2))
    print(f"[{arch}] clean: mean AUC {aucs.mean():.4f} min {aucs.min():.4f}  clause_a={clause_a}  clause_b={clause_b}  halt_fired={result['clauses']['halt_fired']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
