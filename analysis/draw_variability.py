# Measures how much the paired gap varies from one random draw of placements to the next.
# Run: python3 analysis/draw_variability.py --draws-dir DIR --fresh_audit-scores CSV --a2-json JSON --out OUT.json

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.audit_analysis import boot

CEILING = 0.156
N_DRAWS_REGISTERED = 10
COMPOSITE_SEEDS = [6410 + k for k in range(10)]
CONTROL_SOURCE_SEEDS = [6420 + k for k in range(10)]

def draw_gap_stats(df: pd.DataFrame) -> dict:
    gap = df["delta_logit_insertion"] - df["delta_logit_control"]
    ci, se = boot(df["patient_id"], gap)
    per_ckpt = {l: float(g.mean()) for l, g in gap.groupby(df["label"])}
    n_inside = sum(abs(v) < CEILING for v in per_ckpt.values())
    return {"gap": float(gap.mean()), "ci": ci, "se": se, "W": se ** 2, "n_kept": int(df["image_id"].nunique()),
            "n_patient_clusters": int(df["patient_id"].nunique()), "n_checkpoints": len(per_ckpt),
            "per_checkpoint_gap": per_ckpt,
            "confirm": bool(ci[0] > -CEILING and ci[1] < CEILING and n_inside >= math.ceil(0.8 * len(per_ckpt)))}

def rubin_combine(g, W, n_clusters_mean: float, B_override: float | None = None) -> dict:
    g = np.asarray(g, float)
    W = np.asarray(W, float)
    m = len(g)
    gbar = float(g.mean())
    Wbar = float(W.mean())
    B = float(g.var(ddof=1)) if B_override is None else float(B_override)
    T = Wbar + (1 + 1 / m) * B
    nu_com = n_clusters_mean - 1
    if B <= 0:
        nu = float(nu_com)
        gamma = 0.0
    else:
        gamma = (1 + 1 / m) * B / T
        nu_m = (m - 1) / gamma ** 2
        nu_obs = ((nu_com + 1) / (nu_com + 3)) * nu_com * (1 - gamma)
        nu = nu_m * nu_obs / (nu_m + nu_obs)
    half = float(stats.t.ppf(0.975, nu) * math.sqrt(T))
    return {"m": m, "gbar": gbar, "W_bar": Wbar, "B": B, "T": T, "gamma": gamma, "nu": float(nu), "half_width": half,
            "ci": [gbar - half, gbar + half], "inside_ceiling": bool(gbar - half > -CEILING and gbar + half < CEILING)}

def sd_chi2_interval(s: float, m: int) -> dict:
    df = m - 1
    lo_q, hi_q = stats.chi2.ppf(0.025, df), stats.chi2.ppf(0.975, df)
    return {"s": s, "df": df, "ci": [s * math.sqrt(df / hi_q), s * math.sqrt(df / lo_q)],
            "factor_lo": math.sqrt(df / hi_q), "factor_hi": math.sqrt(df / lo_q), "B_upper_multiplier": df / lo_q}

def place(value: float, draws: np.ndarray) -> dict:
    d = np.sort(np.asarray(draws, float))
    s = float(d.std(ddof=1))
    return {"value": float(value), "n_below": int((d < value).sum()), "n_above": int((d > value).sum()),
            "inside_min_max": bool(d.min() <= value <= d.max()), "z_from_mean": float((value - d.mean()) / s) if s > 0 else None}

def analyze(draw_frames: dict, fresh_audit_df: pd.DataFrame, a2_gap: float) -> dict:
    per = {k: draw_gap_stats(df) for k, df in sorted(draw_frames.items())}
    g = np.array([per[k]["gap"] for k in per])
    W = np.array([per[k]["W"] for k in per])
    ncl = float(np.mean([per[k]["n_patient_clusters"] for k in per]))
    primary = rubin_combine(g, W, ncl)
    chi = sd_chi2_interval(float(g.std(ddof=1)), len(g))
    pess = rubin_combine(g, W, ncl, B_override=primary["B"] * chi["B_upper_multiplier"])
    p64 = draw_gap_stats(fresh_audit_df)
    sens_g = np.append(g, p64["gap"])
    sens_W = np.append(W, p64["W"])
    sens = rubin_combine(sens_g, sens_W, float(np.mean([per[k]["n_patient_clusters"] for k in per] + [p64["n_patient_clusters"]])))
    return {"n_draws_usable": len(per), "per_draw": per, "across_draw": {"mean": float(g.mean()), "sd": float(g.std(ddof=1)),
            "min": float(g.min()), "max": float(g.max())}, "sd_chi2_interval": chi,
            "primary_rubin": primary, "pessimistic_rubin": pess,
            "sensitivity_with_draw_6400_m_plus_1": sens,
            "draw_6400": {"stats": p64, "placement": place(p64["gap"], g)},
            "a2_draw_289": {"gap": a2_gap, "placement": place(a2_gap, g),
                            "note": "not identically distributed: A2 used reserved-arm control sources for 123 targets"},
            "ceiling": CEILING,
            "upper_bound_over_draw6400_gap": primary["ci"][1] / p64["gap"], "ceiling_over_draw6400_gap": CEILING / p64["gap"]}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws-dir", required=True, type=Path, help="contains draw0..draw9/scores_original.csv")
    ap.add_argument("--fresh_audit-scores", required=True, type=Path)
    ap.add_argument("--a2-json", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    frames = {}
    for k in range(N_DRAWS_REGISTERED):
        f = a.draws_dir / f"draw{k}" / "scores_original.csv"
        if f.exists():
            frames[k] = pd.read_csv(f, dtype={"patient_id": str})
    a2 = json.loads(a.a2_json.read_text())["paired_gap_5seed"]["pooled_mean"]
    res = analyze(frames, pd.read_csv(a.fresh_audit_scores, dtype={"patient_id": str}), a2)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_draws_usable", "across_draw", "sd_chi2_interval", "primary_rubin", "pessimistic_rubin")}, indent=1))

if __name__ == "__main__":
    main()
