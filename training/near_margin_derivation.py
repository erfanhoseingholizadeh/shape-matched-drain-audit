# Derives the two near-margin planting strengths by interpolating the recorded dose-response.
# Run: python3 training/near_margin_derivation.py --out review/further_runs/e3/derivation.json

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.stats import norm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cxr_shortcuts.eval.population_log import log_population

ROOT = Path(__file__).resolve().parent.parent
TT = [f"logit_tube_tissue_{k}" for k in range(8)]
DELTA = {"zero": 0.0, "weak": 0.10, "mid30": 0.30, "mid50": 0.50, "strong": 0.80}
MARGIN = 0.156
TARGET_EXPECTED = {"e3lo": 0.10, "e3hi": 0.22}
STRATA = ("dev_half", "R_int_C")

def per_checkpoint_primary() -> dict:
    files = sorted(glob.glob(str(ROOT / "review/positive_audit/run/score_dn_p[123]_planted/scores_densenet121.csv")))
    df = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in files])
    df = df[df.stratum.isin(STRATA)]
    out = {}
    for lab, d in df.groupby("label"):
        base = d.logit_before.to_numpy()
        out[lab] = float(((d.logit_tube.to_numpy() - base) - (d[TT].to_numpy() - base[:, None]).mean(1)).mean())
        assert len(d) == 134
        log_population(ROOT / "review/further_runs/e3", "e3_derivation_target_images", d.image_id, "image")
        log_population(ROOT / "review/further_runs/e3", "e3_derivation_target_patients", d.patient_id, "patient")
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pc = per_checkpoint_primary()
    by = {s: [v for k, v in sorted(pc.items()) if k.rsplit("_seed", 1)[0] == s] for s in DELTA}
    assert all(len(v) == 3 for v in by.values())
    d = np.array([DELTA[s] for s in DELTA]); y = np.array([np.mean(by[s]) for s in DELTA]); sd = {s: float(np.std(by[s], ddof=1)) for s in DELTA}

    rec = json.loads((ROOT / "review/positive_audit/positive_analysis.json").read_text())["densenet121"]["planted_descriptive"]["planted_densenet121_planted_control"]
    check = {s: abs(float(np.mean(by[s])) - rec[s]["boot"]["primary"]["mean"]) for s in DELTA}
    assert max(check.values()) < 1e-9, check
    f = PchipInterpolator(d, y)
    q3 = np.poly1d(np.polyfit(d[:3], y[:3], 2)); c4 = np.poly1d(np.polyfit(d[:4], y[:4], 3))
    sd_of = lambda x: float(np.interp(x, [0.10, 0.30], [sd["weak"], sd["mid30"]]))
    out = {"margin": MARGIN, "ladder": {s: {"delta": DELTA[s], "pooled_primary": float(np.mean(by[s])), "per_seed": by[s], "seed_sd": sd[s]} for s in DELTA},
           "n_targets": 134, "strata": list(STRATA), "interpolant": "PCHIP (monotone cubic) through the five points", "strengths": {}}
    for name, t in TARGET_EXPECTED.items():
        dl_raw = brentq(lambda x: float(f(x)) - t, 0.10, 0.30)
        dl = round(dl_raw, 2)
        alt = {"quadratic_3pt": brentq(lambda x: q3(x) - t, 0.10, 0.30), "cubic_4pt": brentq(lambda x: c4(x) - t, 0.10, 0.30),
               "linear_weak_to_mid30 (not used)": 0.10 + (t - y[1]) / (y[2] - y[1]) * 0.20}
        exp = float(f(dl)); disagree = float(max(abs(q3(dl) - exp), abs(c4(dl) - exp)))
        se = float(np.sqrt(sd_of(dl) ** 2 / 3 + disagree ** 2))
        side = norm.cdf((MARGIN - exp) / se) if name == "e3lo" else 1 - norm.cdf((MARGIN - exp) / se)
        out["strengths"][name] = {"target_expected_primary": t, "delta_raw_pchip": dl_raw, "delta": dl, "p_plant_positive": round(0.5 + dl / 2, 4), "p_plant_negative": round(0.5 - dl / 2, 4),
                                  "expected_primary_at_delta": exp, "alternative_interpolants_delta": alt, "interpolant_disagreement_at_delta": disagree, "seed_sd_interpolated": sd_of(dl),
                                  "se_of_three_seed_mean": se, "probability_lands_on_intended_side_of_margin": float(side)}
    lo, hi = out["strengths"]["e3lo"], out["strengths"]["e3hi"]
    out["verdict_flip_note"] = ("with a 90% interval half-width of about 0.042 at 134 targets (weak: -0.011 to 0.073), the registered EQUIVALENT verdict is lost when the pooled mean passes about "
                                f"{MARGIN - 0.042:.3f}, below the margin itself; the intended contrasts are {lo['expected_primary_at_delta']:.3f} and {hi['expected_primary_at_delta']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"ladder": {s: round(v["pooled_primary"], 4) for s, v in out["ladder"].items()}, "seed_sd": {s: round(v["seed_sd"], 4) for s, v in out["ladder"].items()}}, indent=0))
    for k, v in out["strengths"].items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "alternative_interpolants_delta"}, {kk: round(vv, 4) for kk, vv in v["alternative_interpolants_delta"].items()})

if __name__ == "__main__":
    main()
