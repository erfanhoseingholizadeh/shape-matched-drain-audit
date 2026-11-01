# Recomputes each validation-derived quantity on the population it is assigned to, one architecture per call.
# Run: python3 analysis/assigned_recompute.py --help

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
from scoring import within_patient_contrast as wp
from data_prep import clean_population as cp
from analysis import clean_recompute as rc
from analysis import population_assignment as pa

OUT_DIR = REPO_ROOT / "review/population_assignment/outputs"
OUT56 = REPO_ROOT / "review/clean_population/outputs"
NIH_CSV = REPO_ROOT / "data/raw/nih_chestxray14/Data_Entry_2017.csv"
RECORDED_WP = {"densenet121": REPO_ROOT / "review/densenet121_remeasure/outputs/within_patient_contrast.json",
               "resnet50": REPO_ROOT / "review/resnet50_remeasure/outputs/within_patient_contrast_resnet50.json"}
SEEDS = rc.SEEDS
GRID = [round(x, 2) for x in np.arange(0.02, 0.96, 0.02)]

def verify_against_recorded(new: dict, recorded: dict, tol: float = 1e-12) -> list:
    bad: list = []

    def walk(path: str, a, b) -> None:
        if isinstance(b, dict):
            for k in b:
                walk(f"{path}.{k}" if path else str(k), a.get(k) if isinstance(a, dict) else None, b[k])
        elif isinstance(b, (int, float)) and not isinstance(b, bool):
            if a is None or abs(float(a) - float(b)) > tol:
                bad.append(f"{path}: new {a} vs recorded {b}")
        elif a != b:
            bad.append(f"{path}: new {a} vs recorded {b}")

    walk("", new, recorded)
    return bad

def population_parts(patients, halves: dict) -> dict:
    p = {str(x) for x in patients}
    return {"S_int_part": p & halves["S_int"], "R_int_part": p & halves["R_int"], "C_part": p & halves["R_int"] & halves["R_str"], "R_str_part": p & halves["R_str"]}

def qualifier_counts(frame: pd.DataFrame, pops: dict) -> dict:
    pos = frame[frame["pneumothorax_label"] == 1]
    g = pos.groupby("patient_id")["drain_present"].agg(["sum", "count"])
    qual = set(g[(g["sum"] >= 1) & ((g["count"] - g["sum"]) >= 1)].index.astype(str))
    return {name: len(qual & {str(x) for x in members}) for name, members in pops.items()}

def assert_qualifier_counts_match(counts: dict) -> None:
    for k, want in pa.QUALIFYING_PATIENTS.items():
        assert counts[k] == want, f"registered qualifying-patient count for {k} is {want}, recomputed {counts[k]}: stop"

def crossing_curve(pairs_by_seed: list, grid) -> dict:
    out = {}
    for t in grid:
        n = opp = 0
        for pr in pairs_by_seed:
            side_p, side_a = pr["drain_present_mean"] >= t, pr["drain_absent_mean"] >= t
            n += len(pr)
            opp += int((side_p != side_a).sum())
        out[t] = {"n_opposite": opp, "n_pairs": n, "rate": opp / n if n else float("nan")}
    return out

def _wp_run(frame: pd.DataFrame, name: str) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        return wp.run_population(frame, SEEDS, name)

def _wp_oracle(arch: str, merged_wp: pd.DataFrame) -> dict:
    rec = json.loads(RECORDED_WP[arch].read_text())
    rows = []
    for label, frame in (("full_val", merged_wp), ("val_report_secondary", merged_wp[merged_wp["val_subset"] == "report"])):
        got = _wp_run(frame, label)
        for s in SEEDS:
            r, g = rec[label]["per_seed"][str(s)], got["per_seed"][s]
            rows.append(("n_patients", r["n_patients"] == g["n_patients"]))
            for k in ("mean_paired_diff", "ci_low", "ci_high"):
                rows.append((f"{label}/{s}/{k}", abs(r["logit"][k] - g["logit"][k]) <= 1e-9))
            for k in ("sign_positive", "sign_negative", "sign_zero"):
                rows.append((f"{label}/{s}/{k}", r["logit"][k] == g["logit"][k]))
    fails = [n for n, ok in rows if not ok]
    return {"all_passed": not fails, "n_checks": len(rows), "failures": fails[:10]}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("architecture", choices=list(rc.ARCHS))
    arch = ap.parse_args().architecture
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    halves = cp.val_halves()
    clean = cp.clean_population(halves)
    df = pd.read_csv(rc.ARCHS[arch]["csv"], dtype={"patient_id": str})
    pop = df[df["patient_id"].isin(halves["R_int"])]
    calib = rc.load_calibration(arch)
    stats = {s: rc.seed_statistics(pop[pop["seed"] == s], calib[s], bootstrap=False) for s in SEEDS}
    sens = json.loads((OUT56 / f"sensitivity_{arch}.json").read_text())
    mism: list = []
    for s in SEEDS:
        rec = {k: v for k, v in sens["per_seed"][str(s)].items() if k not in ("ci_low", "ci_high")}
        mism += [f"seed {s} {m}" for m in verify_against_recorded(stats[s], rec, 1e-12)]
    neg_logits = np.concatenate([pop[(pop["seed"] == s) & (pop["pneumothorax_label"] == 0)]["logit"].to_numpy() for s in SEEDS])
    pooled = {"n_negatives": int(len(neg_logits)), "shift_to_0.10": rc.solve_uniform_shift(neg_logits, 0.10), "shift_fail_floor_0.02": rc.solve_uniform_shift(neg_logits, 0.02)}
    mism += verify_against_recorded({"reachability_pooled": pooled}, {"reachability_pooled": sens["summary"]["reachability_pooled"]}, 1e-9)
    if mism:
        print("REFUSED: the recompute does not reproduce the clean-population stage's sensitivity output on the same population:", *mism[:10], sep="\n  ", file=sys.stderr)
        return 1
    for s in SEEDS:
        stats[s]["ci_low"], stats[s]["ci_high"] = sens["per_seed"][str(s)]["ci_low"], sens["per_seed"][str(s)]["ci_high"]
        stats[s]["ci_source"] = "sensitivity_%s.json (rc.clustered_auc_ci, 2000 resamples, seed 0)" % arch
    aucs = np.array([stats[s]["auc"] for s in SEEDS])
    result: dict = {"architecture": arch, "assigned_population": "R_int", "n_patients": int(pop["patient_id"].nunique()), "reproduces_sensitivity_output": True,
                    "per_seed": {str(s): stats[s] for s in SEEDS}, "reachability_pooled": pooled,
                    "summary": {"auc_mean": float(aucs.mean()), "auc_std_ddof1": float(aucs.std(ddof=1)), "auc_min": float(aucs.min()),
                                "clause_a": {str(s): rc.classify_clause_a(stats[s]["auc"], stats[s]["ci_high"]) for s in SEEDS},
                                "clause_b": {str(s): rc.classify_clause_b(stats[s]["median_neg_p"]) for s in SEEDS},
                                "youden_achieved_sensitivity": [stats[s]["youden"]["sensitivity"] for s in SEEDS],
                                "youden_achieved_specificity": [stats[s]["youden"]["specificity"] for s in SEEDS]}}

    wp_merged = wp.load_merged(rc.ARCHS[arch]["csv"], rc.NEATX_CSV, NIH_CSV)
    pops = {"V": halves["val"], "R_str": halves["R_str"], "R_int": halves["R_int"], "C": clean}
    counts = qualifier_counts(wp_merged[wp_merged["seed"] == 0], pops)
    assert_qualifier_counts_match(counts)
    orc = _wp_oracle(arch, wp_merged)
    result["within_patient_oracle"] = orc
    result["qualifying_patients"] = counts
    if not orc["all_passed"]:
        print("REFUSED: the within-patient oracle failed:", orc["failures"], file=sys.stderr)
        (OUT_DIR / f"assigned_{arch}_REFUSED.json").write_text(json.dumps(result, indent=2, default=float))
        return 1
    qual46 = set(wp_merged[(wp_merged["seed"] == 0) & (wp_merged["pneumothorax_label"] == 1)].groupby("patient_id")["drain_present"].agg(["sum", "count"]).pipe(
        lambda g: g[(g["sum"] >= 1) & ((g["count"] - g["sum"]) >= 1)]).index.astype(str))
    parts = population_parts(qual46, halves)
    contrast = {"V_all_46_recorded_primary": _wp_run(wp_merged, "V"), "R_str_recorded_secondary": _wp_run(wp_merged[wp_merged["val_subset"] == "report"], "R_str")}
    for name, key in (("R_int_part_assigned", "R_int_part"), ("S_int_part", "S_int_part"), ("C_part", "C_part")):
        sub = wp_merged[wp_merged["patient_id"].isin(parts[key])]
        contrast[name] = {**_wp_run(sub, name), "n_patients": len(parts[key]),
                          "below_registered_floor_report_only_sign_count_and_distribution": len(parts[key]) < wp.MIN_PATIENTS_FOR_CI}
    result["within_patient_primary_contrast"] = contrast

    youden_thr = {s: stats[s]["youden"]["threshold"] for s in SEEDS}
    band_thr = {s: {t: stats[s]["band"][str(t)]["threshold"] for t in rc.TARGETS} for s in SEEDS}
    result["crossing_R_int_thresholds"] = {"all_46": rc.crossing_block(wp_merged.drop(columns=["followup"]), youden_thr, band_thr),
                                           "R_int_part": rc.crossing_block(wp_merged[wp_merged["patient_id"].isin(parts["R_int_part"])].drop(columns=["followup"]), youden_thr, band_thr),
                                           "S_int_part": rc.crossing_block(wp_merged[wp_merged["patient_id"].isin(parts["S_int_part"])].drop(columns=["followup"]), youden_thr, band_thr)}
    pairs = [rc.within_patient_pairs(wp_merged[wp_merged["seed"] == s].drop(columns=["followup"])) for s in SEEDS]
    result["crossing_threshold_sweep_all_46_pooled"] = {str(t): v for t, v in crossing_curve(pairs, GRID).items()}
    (OUT_DIR / f"assigned_{arch}.json").write_text(json.dumps(result, indent=2, default=float))
    s = result["summary"]
    print(f"[{arch}] R_int: n_patients={result['n_patients']} mean AUC {s['auc_mean']:.4f} min {s['auc_min']:.4f} clause_a={s['clause_a']} clause_b={s['clause_b']}")
    print(f"[{arch}] qualifying patients {counts}; within-patient oracle passed ({orc['n_checks']} checks)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
