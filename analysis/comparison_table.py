# Tabulates each recomputed quantity beside the same quantity on the other candidate population.
# Run: python3 analysis/comparison_table.py (no options; it starts working at once)

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

OUT56 = REPO_ROOT / "review/clean_population/outputs"
OUT57 = REPO_ROOT / "review/population_assignment/outputs"
ARCHS = ("densenet121", "resnet50")
SEEDS = [0, 1, 2, 3, 4]
TARGETS = ["0.7", "0.75", "0.8", "0.85", "0.9", "0.95"]

def summ(values) -> dict:
    v = [float(x) for x in values]
    return {"min": min(v), "max": max(v), "mean": float(np.mean(v)), "per_seed": v}

def pooled_row(p: dict) -> dict:
    return {"rate": p["n_opposite_side_total"] / p["n_patients_total"], "n_opposite": p["n_opposite_side_total"], "n_pairs": p["n_patients_total"],
            "direction": f"{p['reliance_direction']}:{p['reverse_direction']}", "both_above": p["both_above"], "both_below": p["both_below"],
            "per_seed_rate_min": p["per_seed_rate_min"], "per_seed_rate_max": p["per_seed_rate_max"]}

def monotone_nonincreasing(xs) -> bool:
    return all(b <= a + 1e-15 for a, b in zip(xs, xs[1:]))

def wording_rule_outcome(resnet_pooled_rate: float, densenet_min: float, densenet_max: float) -> str:
    return "inside" if densenet_min <= resnet_pooled_rate <= densenet_max else "outside"

def _stat_block(per_seed: dict) -> dict:
    ps = [per_seed[str(s)] for s in SEEDS]
    out = {
        "clause_a_auc": summ([p["auc"] for p in ps]), "clause_a_ci_low": [p["ci_low"] for p in ps], "clause_a_ci_high": [p["ci_high"] for p in ps],
        "n_seeds_point_estimate_at_or_above_0_82": int(sum(p["auc"] >= 0.82 for p in ps)),
        "n_seeds_ci_excludes_0_82_from_above": int(sum(p["ci_low"] > 0.82 for p in ps)),
        "clause_b_median_neg_p": summ([p["median_neg_p"] for p in ps]),
        "youden_threshold": summ([p["youden"]["threshold"] for p in ps]), "youden_achieved_sensitivity": summ([p["youden"]["sensitivity"] for p in ps]),
        "youden_achieved_specificity": summ([p["youden"]["specificity"] for p in ps]),
        "band_threshold_mean_over_seeds": {t: float(np.mean([p["band"][t]["threshold"] for p in ps])) for t in TARGETS},
        "band_achieved_specificity_mean_over_seeds": {t: float(np.mean([p["band"][t]["achieved_specificity"] for p in ps])) for t in TARGETS},
        "reachability_to_0_10": summ([p["reachability"]["shift_to_0.10"] for p in ps]), "reachability_fail_floor_0_02": summ([p["reachability"]["shift_fail_floor_0.02"] for p in ps]),
        "n_positive_images": ps[0]["n_positive"], "n_negative_images": ps[0]["n_negative"], "n_patients": ps[0]["n_patients"],
    }
    if "calibration" in ps[0] and ps[0]["calibration"]:
        c = [p["calibration"] for p in ps]
        for k in ("ece_pre", "brier_pre", "ece_temperature", "brier_temperature", "ece_affine", "brier_affine", "median_neg_p_temperature", "median_neg_p_affine"):
            out[k] = summ([x[k] for x in c])
    return out

def _crossing_summary(block: dict) -> dict:
    return {"youden_pooled": pooled_row(block["youden_pooled"]), "band_pooled": {t: pooled_row(block["band_pooled"][t]) for t in TARGETS},
            "band_rate_monotone_nonincreasing_in_target_sensitivity": monotone_nonincreasing([block["band_pooled"][t]["frac_pooled"] for t in TARGETS]),
            "reliance_exceeds_reverse_at_every_band_point": all(block["band_pooled"][t]["reliance_direction"] > block["band_pooled"][t]["reverse_direction"] for t in TARGETS),
            "youden_reliance_exceeds_reverse": block["youden_pooled"]["reliance_direction"] > block["youden_pooled"]["reverse_direction"]}

def build() -> dict:
    from analysis import clean_recompute as rc

    out: dict = {}
    for arch in ARCHS:
        orc = json.loads((OUT56 / f"oracle_{arch}.json").read_text())
        cln = json.loads((OUT56 / f"clean_{arch}.json").read_text())
        asg = json.loads((OUT57 / f"assigned_{arch}.json").read_text())
        merged = rc.load_merged(rc.ARCHS[arch]["csv"])
        y_thr = {s: orc["per_seed"][str(s)]["youden"]["threshold"] for s in SEEDS}
        b_thr = {s: {float(t): orc["per_seed"][str(s)]["band"][t]["threshold"] for t in TARGETS} for s in SEEDS}
        b_thr = {s: {t: b_thr[s][t] for t in rc.TARGETS} for s in SEEDS}
        r_str_cross = rc.crossing_block(merged, y_thr, b_thr)
        out[arch] = {
            "R_str_contaminated_predecessor": {**_stat_block(orc["per_seed"]), "crossing_n46": _crossing_summary(r_str_cross)},
            "C_other_candidate": {**_stat_block(cln["per_seed"]), "crossing_n46": _crossing_summary(cln["crossing_on_primary_n46"])},
            "R_int_assigned": {**_stat_block(asg["per_seed"]), "crossing_n46_all": _crossing_summary(asg["crossing_R_int_thresholds"]["all_46"]),
                               "crossing_R_int_part": _crossing_summary(asg["crossing_R_int_thresholds"]["R_int_part"]),
                               "crossing_S_int_part": _crossing_summary(asg["crossing_R_int_thresholds"]["S_int_part"]),
                               "reachability_pooled": asg["reachability_pooled"]},
            "reachability_pooled_R_str_C": {"R_str": orc["summary"]["reachability_pooled"], "C": cln["summary"]["reachability_pooled"]},
            "qualifying_patients": asg["qualifying_patients"],
            "within_patient_primary": {k: {"mean_of_means": v["across_seed_spread"]["mean_of_means"], "std_of_means": v["across_seed_spread"]["std_of_means"],
                                           "n_patients_per_seed": v["across_seed_spread"]["n_patients_per_seed"],
                                           "per_seed_mean": [v["per_seed"][str(s)]["logit"]["mean_paired_diff"] for s in SEEDS],
                                           "per_seed_ci_low": [v["per_seed"][str(s)]["logit"]["ci_low"] for s in SEEDS], "per_seed_ci_high": [v["per_seed"][str(s)]["logit"]["ci_high"] for s in SEEDS],
                                           "sign_positive": [v["per_seed"][str(s)]["logit"]["sign_positive"] for s in SEEDS], "sign_negative": [v["per_seed"][str(s)]["logit"]["sign_negative"] for s in SEEDS],
                                           "verdicts": v["across_seed_spread"]["verdicts"],
                                           **({"below_registered_floor_report_only_sign_count_and_distribution": v["below_registered_floor_report_only_sign_count_and_distribution"]}
                                              if "below_registered_floor_report_only_sign_count_and_distribution" in v else {})}
                                       for k, v in asg["within_patient_primary_contrast"].items()},
        }

    for pop, key in (("R_str_contaminated_predecessor", "crossing_n46"), ("C_other_candidate", "crossing_n46"), ("R_int_assigned", "crossing_n46_all")):
        d = out["densenet121"][pop][key]["youden_pooled"]
        r = out["resnet50"][pop][key]["youden_pooled"]
        out.setdefault("wording_rule_resnet50_remeasure", {})[pop] = {"resnet_pooled_rate": r["rate"], "densenet_per_seed_min": d["per_seed_rate_min"], "densenet_per_seed_max": d["per_seed_rate_max"],
                                                          "outcome": wording_rule_outcome(r["rate"], d["per_seed_rate_min"], d["per_seed_rate_max"])}
    out["threshold_sweep_all_46_pooled"] = {arch: json.loads((OUT57 / f"assigned_{arch}.json").read_text())["crossing_threshold_sweep_all_46_pooled"] for arch in ARCHS}
    return out

def main() -> int:
    res = build()
    (OUT57 / "comparison.json").write_text(json.dumps(res, indent=2))
    for arch in ARCHS:
        print("==", arch)
        for pop in ("R_str_contaminated_predecessor", "C_other_candidate", "R_int_assigned"):
            b = res[arch][pop]
            cr = b.get("crossing_n46") or b.get("crossing_n46_all")
            print(f"  {pop}: clause a mean {b['clause_a_auc']['mean']:.4f} min {b['clause_a_auc']['min']:.4f} ({b['n_seeds_point_estimate_at_or_above_0_82']}/5 at/above .82, "
                  f"{b['n_seeds_ci_excludes_0_82_from_above']}/5 CI above); (b) median neg p {b['clause_b_median_neg_p']['min']:.4f}..{b['clause_b_median_neg_p']['max']:.4f}; "
                  f"Youden thr {b['youden_threshold']['min']:.3f}..{b['youden_threshold']['max']:.3f} sens {b['youden_achieved_sensitivity']['min']:.2f}..{b['youden_achieved_sensitivity']['max']:.2f}; "
                  f"crossing Youden {cr['youden_pooled']['n_opposite']}/{cr['youden_pooled']['n_pairs']} ({cr['youden_pooled']['rate']:.3f}, {cr['youden_pooled']['direction']}, both-above {cr['youden_pooled']['both_above']})")
    print("wording rule:", json.dumps(res["wording_rule_resnet50_remeasure"], indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
