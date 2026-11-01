# Checks whether the cleaned population is a structurally non-random part of validation.
# Run: python3 analysis/split_mechanism_check.py (no options; it starts working at once)

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from data_prep import clean_population as cp
from analysis import random_half_rate as rh

OUT_DIR = REPO_ROOT / "review/population_assignment/outputs"
REGISTERED_SEEDS = 1000
N_SEEDS = int(os.environ.get("SPLIT_CHECK_N_SEEDS", str(REGISTERED_SEEDS)))
SEEDS = [0, 1, 2, 3, 4]
ARCHS = tuple(rh.CSV)

def alt_halves(ids_int, ids_str, seed: int) -> dict:
    s_int, r_int = cp._halves(cp.legacy_assign_val_subset(ids_int, seed=seed))
    s_str, r_str = cp._halves(cp.legacy_assign_val_subset(ids_str, seed=seed))
    return {"S_int": s_int, "R_int": r_int, "S_str": s_str, "R_str": r_str}

def compare_id_magnitude(ids_a, ids_b) -> dict:
    a, b = np.asarray(ids_a, dtype=float), np.asarray(ids_b, dtype=float)
    ks = ks_2samp(a, b)
    q = np.arange(10, 100, 10)
    return {"n_a": int(len(a)), "n_b": int(len(b)), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "deciles_a": [float(x) for x in np.percentile(a, q)], "deciles_b": [float(x) for x in np.percentile(b, q)]}

def count_rank_agreements(int_ids) -> int:
    ids = sorted(int(i) for i in int_ids)
    by_str = sorted(str(i) for i in ids)
    str_rank = {s: r for r, s in enumerate(by_str)}
    return sum(1 for r, i in enumerate(ids) if str_rank[str(i)] == r)

def deficit(patient_code, labels, scores, quarter_mask, half_mask) -> float:
    return rh.fast_auc(labels[quarter_mask[patient_code]], scores[quarter_mask[patient_code]]) - \
        rh.fast_auc(labels[half_mask[patient_code]], scores[half_mask[patient_code]])

def tail_share(draws, actual: float) -> float:
    return float(np.mean(np.asarray(draws) <= actual))

def _mask(patients: set, code: dict, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[[code[p] for p in patients]] = True
    return m

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = cp.MANIFEST
    m_int = pd.read_csv(manifest)
    m_str = pd.read_csv(manifest, dtype={"patient_id": str})
    ids_int = m_int.loc[m_int["split"] == "val", "patient_id"].unique()
    ids_str = m_str.loc[m_str["split"] == "val", "patient_id"].unique()
    h = alt_halves(ids_int, ids_str, seed=cp.VAL_SUBSET_SEED)
    clean = h["R_int"] & h["R_str"]
    rest = h["R_int"] - clean
    part1 = {"n_clean": len(clean), "n_rest_of_R_int": len(rest),
             "integer_id_comparison_clean_vs_rest_of_R_int": compare_id_magnitude([int(p) for p in clean], [int(p) for p in rest]),
             "n_ids_with_equal_integer_and_string_rank": count_rank_agreements(ids_int),
             "n_val_ids": int(len(ids_int))}

    val_patients = sorted(str(p) for p in ids_str)
    code = {p: i for i, p in enumerate(val_patients)}
    nval = len(val_patients)
    data = {}
    for arch in ARCHS:
        d = pd.read_csv(rh.CSV[arch], dtype={"patient_id": str})
        d = d[d["patient_id"].isin(code)]
        pc = d["patient_id"].map(code).to_numpy()
        for s in SEEDS:
            sd = d["seed"].to_numpy() == s
            data[(arch, s)] = (pc[sd], d["pneumothorax_label"].to_numpy()[sd].astype(int), d["prob"].to_numpy()[sd])

    def five_seed_deficit(halves: dict) -> dict:
        c_mask = _mask(halves["R_int"] & halves["R_str"], code, nval)
        r_mask = _mask(halves["R_int"], code, nval)
        return {arch: float(np.mean([deficit(*data[(arch, s)], quarter_mask=c_mask, half_mask=r_mask) for s in SEEDS])) for arch in ARCHS}

    actual = five_seed_deficit(h)
    seeds = [s for s in range(1, N_SEEDS + 2) if s != cp.VAL_SUBSET_SEED][:N_SEEDS]
    dd = {arch: [] for arch in ARCHS}
    overlap = []
    for s in seeds:
        hs = alt_halves(ids_int, ids_str, seed=s)
        overlap.append(len(hs["S_int"] & hs["S_str"]))
        for arch, v in five_seed_deficit(hs).items():
            dd[arch].append(v)
    rh_path = OUT_DIR / "random_half_rate.json"
    uniform_sd = json.loads(rh_path.read_text())["per_architecture_five_seed_mean"] if rh_path.exists() else {}
    part2 = {"n_alternative_seeds": len(seeds), "registered_seeds": REGISTERED_SEEDS,
             "select_half_overlap_mean": float(np.mean(overlap)), "select_half_overlap_sd": float(np.std(overlap, ddof=1)),
             "per_architecture": {arch: {"actual_deficit_seed_2000": actual[arch], "alt_deficit_mean": float(np.mean(dd[arch])),
                                         "alt_deficit_sd": float(np.std(dd[arch], ddof=1)),
                                         "share_of_alt_seeds_at_or_below_actual": tail_share(dd[arch], actual[arch]),
                                         "uniform_random_half_sd_of_five_seed_mean": uniform_sd.get(arch, {}).get("draw_sd_of_five_mean"),
                                         "sd_ratio_mechanism_over_uniform": (float(np.std(dd[arch], ddof=1)) / uniform_sd[arch]["draw_sd_of_five_mean"]) if arch in uniform_sd else None}
                                  for arch in ARCHS}}
    rep = {"part1_manifest_only": part1, "part2_alternative_seed_null": part2}
    suffix = "" if N_SEEDS == REGISTERED_SEEDS else "_reduced"
    (OUT_DIR / f"split_mechanism{suffix}.json").write_text(json.dumps(rep, indent=2))
    k = part1["integer_id_comparison_clean_vs_rest_of_R_int"]
    print(f"Part 1: KS on integer id, C vs rest of R_int: stat {k['ks_stat']:.4f} p {k['ks_p']:.4f}; medians {k['median_a']:.0f} vs {k['median_b']:.0f}; "
          f"ids with equal int/str rank: {part1['n_ids_with_equal_integer_and_string_rank']} of {part1['n_val_ids']}")
    for arch, v in part2["per_architecture"].items():
        print(f"Part 2 {arch}: actual deficit {v['actual_deficit_seed_2000']:+.4f}; alt-seed deficit mean {v['alt_deficit_mean']:+.4f} sd {v['alt_deficit_sd']:.4f} "
              f"(uniform-half sd {v['uniform_random_half_sd_of_five_seed_mean']}); share at or below actual {v['share_of_alt_seeds_at_or_below_actual']:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
