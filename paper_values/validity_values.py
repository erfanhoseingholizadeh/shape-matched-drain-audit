# Collects the validity values the paper cites from the recorded outputs into one flat JSON.
# Run: python3 paper_values/validity_values.py (no options; it starts working at once)

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
O56 = REPO_ROOT / "review/clean_population/outputs"
O57 = REPO_ROOT / "review/population_assignment/outputs"
REC44_GAP = REPO_ROOT / "review/densenet121_remeasure/outputs/subgroup_anchor.json"
ARCHS = ("densenet121", "resnet50")
SEEDS = [0, 1, 2, 3, 4]
BAND_LO, BAND_HI = 0.70, 0.95

def minmax(xs) -> tuple:
    return (min(xs), max(xs))

def ratio_range(num, den) -> tuple:
    r = [a / b for a, b in zip(num, den)]
    return (min(r), max(r))

def sd_ddof1(xs) -> float:
    return float(np.std(np.asarray(xs, dtype=float), ddof=1))

def assert_addressable(vals: dict) -> None:
    bad = [k for k in vals if "." in k]
    assert not bad, f"keys containing a dot cannot be addressed by a dotted path: {bad[:5]}"

def _load(p: Path) -> dict:
    return json.loads(p.read_text())

def build() -> dict:
    v: dict = {}
    asg = {a: _load(O57 / f"assigned_{a}.json") for a in ARCHS}
    cln = {a: _load(O56 / f"clean_{a}.json") for a in ARCHS}
    orc = {a: _load(O56 / f"oracle_{a}.json") for a in ARCHS}
    pops = {"R_int": {a: asg[a]["per_seed"] for a in ARCHS}, "C": {a: cln[a]["per_seed"] for a in ARCHS}, "R_str": {a: orc[a]["per_seed"] for a in ARCHS}}

    for pop, by_arch in pops.items():
        for a, ps in by_arch.items():
            aucs = [ps[str(s)]["auc"] for s in SEEDS]
            v[f"a_{pop}_{a}_mean"], v[f"a_{pop}_{a}_std"] = float(np.mean(aucs)), sd_ddof1(aucs)
            v[f"a_{pop}_{a}_min"] = min(aucs)
            v[f"a_{pop}_{a}_n_at_or_above_bar"] = int(sum(x >= 0.82 for x in aucs))
            v[f"b_{pop}_{a}_median_neg_p_min"] = min(ps[str(s)]["median_neg_p"] for s in SEEDS)
            v[f"b_{pop}_{a}_median_neg_p_max"] = max(ps[str(s)]["median_neg_p"] for s in SEEDS)
    for pop in pops:
        v[f"a_{pop}_n_point_estimates_at_or_above_bar"] = sum(v[f"a_{pop}_{a}_n_at_or_above_bar"] for a in ARCHS)
    v["a_arch_diff_R_int"] = abs(v["a_R_int_densenet121_mean"] - v["a_R_int_resnet50_mean"])
    v["a_arch_diff_R_str"] = abs(v["a_R_str_densenet121_mean"] - v["a_R_str_resnet50_mean"])
    v["a_arch_diff_over_densenet_sd_R_int"] = v["a_arch_diff_R_int"] / v["a_R_int_densenet121_std"]
    v["a_arch_diff_over_densenet_sd_R_str"] = v["a_arch_diff_R_str"] / v["a_R_str_densenet121_std"]

    iw = _load(O57 / "interval_widths.json")
    for pop in ("C", "R_int"):
        p = iw["populations"][pop]
        v[f"a_{pop}_n_intervals_excluding_bar"] = p["usability"]["n_intervals_excluding_bar"]
        v[f"a_{pop}_width_min"], v[f"a_{pop}_width_max"], v[f"a_{pop}_width_mean"] = p["width_min"], p["width_max"], p["width_mean"]
        v[f"a_{pop}_reach_toward_bar_min"], v[f"a_{pop}_reach_toward_bar_max"] = p["reach_toward_bar_min"], p["reach_toward_bar_max"]
        v[f"a_{pop}_n_point_estimates_below_bar"] = p["n_below_bar_point_estimate"]
    v["a_C_n_point_estimates_below_bar"] = iw["populations"]["C"]["n_below_bar_point_estimate"]

    rh = _load(O57 / "random_half_rate.json")
    v["sim_n_draws"], v["sim_p_any_below_bar"] = rh["n_draws"], rh["summary"]["p_any_below_bar"]
    v["sim_p_at_least_two_below_bar"] = rh["summary"]["p_at_least_two_below_bar"]
    for a in ARCHS:
        v[f"sim_tail_share_{a}"] = rh["per_architecture_five_seed_mean"][a]["share_of_draws_at_or_below_actual_C_mean"]
    sm = _load(O57 / "split_mechanism.json")
    k = sm["part1_manifest_only"]["integer_id_comparison_clean_vs_rest_of_R_int"]
    v["mech_ks_p"], v["mech_ks_stat"] = k["ks_p"], k["ks_stat"]
    v["mech_n_alt_seeds"] = sm["part2_alternative_seed_null"]["n_alternative_seeds"]
    for a in ARCHS:
        pa_ = sm["part2_alternative_seed_null"]["per_architecture"][a]
        v[f"mech_share_alt_seeds_at_or_below_actual_{a}"] = pa_["share_of_alt_seeds_at_or_below_actual"]
        v[f"mech_n_alt_seeds_at_or_below_actual_{a}"] = int(round(pa_["share_of_alt_seeds_at_or_below_actual"] * v["mech_n_alt_seeds"]))
        v[f"mech_sd_ratio_{a}"] = pa_["sd_ratio_mechanism_over_uniform"]
    qt = _load(O57 / "quarter_table.json")
    for a in ARCHS:
        q = qt[a]["five_seed_mean_auc_by_quarter"]
        v[f"quarter_C_{a}"] = q["Q4_C_Rint_and_Rstr"]
        others = [q["Q1_Sint_and_Sstr"], q["Q2_Sint_and_Rstr"], q["Q3_Rint_and_Sstr"]]
        v[f"quarter_others_min_{a}"], v[f"quarter_others_max_{a}"] = min(others), max(others)

    rn = _load(O57 / "retraining_argument_numbers.json")
    for a in ARCHS:
        v[f"selected_minus_complement_{a}"] = rn["selected_minus_never_selected_half"][a]["mean"]
    v["selected_minus_complement_retired"] = rn["selected_minus_never_selected_half"]["retired_never_selected"]["mean"]

    for a in ARCHS:
        ys = [asg[a]["per_seed"][str(s)]["youden"]["sensitivity"] for s in SEEDS]
        v[f"youden_sens_R_int_{a}_min"], v[f"youden_sens_R_int_{a}_max"] = minmax(ys)
        v[f"youden_n_seeds_below_band_R_int_{a}"] = int(sum(x < BAND_LO for x in ys))
        v[f"youden_n_seeds_above_band_R_int_{a}"] = int(sum(x > BAND_HI for x in ys))
        yo = [orc[a]["per_seed"][str(s)]["youden"]["sensitivity"] for s in SEEDS]
        v[f"youden_sens_R_str_{a}_min"], v[f"youden_sens_R_str_{a}_max"] = minmax(yo)
    v["reach_pass_R_int_densenet121_seed0"] = asg["densenet121"]["per_seed"]["0"]["reachability"]["shift_to_0.10"]
    v["reach_fail_R_int_densenet121_seed0"] = asg["densenet121"]["per_seed"]["0"]["reachability"]["shift_fail_floor_0.02"]
    v["reach_pass_R_str_densenet121_seed0"] = orc["densenet121"]["per_seed"]["0"]["reachability"]["shift_to_0.10"]
    rel = []
    for a in ARCHS:
        for s in SEEDS:
            for key in ("shift_to_0.10", "shift_fail_floor_0.02"):
                r0, r1 = orc[a]["per_seed"][str(s)]["reachability"][key], asg[a]["per_seed"][str(s)]["reachability"][key]
                rel.append(abs(r1 - r0) / r0)
    v["reach_max_rel_change_R_str_to_R_int"] = max(rel)
    v["reach_max_rel_change_R_str_to_R_int_pct"] = 100.0 * max(rel)

    for pop in ("C", "R_str"):
        cal = [pops[pop][a][str(s)]["calibration"] for a in ARCHS for s in SEEDS]
        e_pre, e_t, e_a = [c["ece_pre"] for c in cal], [c["ece_temperature"] for c in cal], [c["ece_affine"] for c in cal]
        b_pre, b_t, b_a = [c["brier_pre"] for c in cal], [c["brier_temperature"] for c in cal], [c["brier_affine"] for c in cal]
        v[f"ece_affine_{pop}_min"], v[f"ece_affine_{pop}_max"] = minmax(e_a)
        v[f"brier_affine_{pop}_min"], v[f"brier_affine_{pop}_max"] = minmax(b_a)
        v[f"ratio_ece_pre_over_affine_{pop}_min"], v[f"ratio_ece_pre_over_affine_{pop}_max"] = ratio_range(e_pre, e_a)
        v[f"ratio_ece_temp_over_affine_{pop}_min"], v[f"ratio_ece_temp_over_affine_{pop}_max"] = ratio_range(e_t, e_a)
        v[f"ratio_brier_pre_over_affine_{pop}_min"], v[f"ratio_brier_pre_over_affine_{pop}_max"] = ratio_range(b_pre, b_a)
        v[f"n_checkpoints_ece_worse_after_temperature_{pop}"] = int(sum(t > p for t, p in zip(e_t, e_pre)))
        v[f"brier_temp_vs_pre_max_rel_change_{pop}"] = max(abs(t - p) / p for t, p in zip(b_t, b_pre))

    gp = _load(O57 / "gap_assigned.json")["populations"]
    rec = _load(REC44_GAP)
    for name, key in (("R_int", "R_int_assigned"), ("C", "C_other_candidate"), ("S_int", "S_int_descriptive")):
        p = gp[key]
        v[f"gap_{name}_mean"], v[f"gap_{name}_std"] = p["_spread"]["diff_mean"], p["_spread"]["diff_std_ddof1"]
        v[f"gap_{name}_ci_high_min"], v[f"gap_{name}_ci_high_max"] = minmax(p["_spread"]["ci_high_per_seed"])
        v[f"gap_{name}_ci_low_min"], v[f"gap_{name}_ci_low_max"] = minmax([p[str(s)]["diff_ci_low"] for s in SEEDS])
        v[f"gap_{name}_n_seeds_ci_high_below_0_17"] = p["_spread"]["n_seeds_with_ci_high_below_0_17"]
        v[f"gap_{name}_n_drain_images"], v[f"gap_{name}_n_nodrain_images"] = p["0"]["n_drain"], p["0"]["n_nodrain"]
        v[f"gap_{name}_n_drain_patients"], v[f"gap_{name}_n_nodrain_patients"] = p["0"]["n_drain_patients"], p["0"]["n_nodrain_patients"]
    v["gap_V_mean"] = rec["full_val"]["across_seed_spread"]["diff_mean"]
    v["gap_V_ci_high_max"] = max(rec["full_val"]["per_seed"][str(s)]["diff_ci_high"] for s in SEEDS)
    v["gap_R_str_mean"] = rec["val_report_secondary"]["across_seed_spread"]["diff_mean"]

    cmpj = _load(O57 / "comparison.json")
    q = cmpj["densenet121"]["qualifying_patients"]
    v["qualifying_V"], v["qualifying_R_str"], v["qualifying_R_int"], v["qualifying_C"] = q["V"], q["R_str"], q["R_int"], q["C"]
    for a in ARCHS:
        w = cmpj[a]["within_patient_primary"]
        v[f"wp_V_mean_{a}"] = w["V_all_46_recorded_primary"]["mean_of_means"]
        v[f"wp_S_int_part_mean_{a}"] = w["S_int_part"]["mean_of_means"]
        v[f"wp_R_int_part_n_{a}"] = w["R_int_part_assigned"]["n_patients_per_seed"][0]
        v[f"wp_S_int_part_n_{a}"] = w["S_int_part"]["n_patients_per_seed"][0]
        v[f"wp_R_int_part_sign_positive_min_{a}"], v[f"wp_R_int_part_sign_positive_max_{a}"] = minmax(w["R_int_part_assigned"]["sign_positive"])
        v[f"wp_R_int_part_sign_negative_max_{a}"] = max(w["R_int_part_assigned"]["sign_negative"])
        v[f"wp_S_int_part_sign_positive_min_{a}"], v[f"wp_S_int_part_sign_positive_max_{a}"] = minmax(w["S_int_part"]["sign_positive"])
    assert_addressable(v)
    return v

def main() -> int:
    v = build()
    (O57 / "manuscript_values.json").write_text(json.dumps(v, indent=2, sort_keys=True))
    print(f"wrote {len(v)} values to review/population_assignment/outputs/manuscript_values.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
