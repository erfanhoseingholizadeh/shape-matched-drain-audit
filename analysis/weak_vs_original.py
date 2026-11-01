# Descriptive comparison of the weak planted checkpoints with the originals; its recorded output is read by heldout_values.py.
# Run: python3 analysis/weak_vs_original.py --out review/weak_vs_original/weak_vs_original.json

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))
from analysis.heldout_analysis import B_RESAMPLES, M, contrast_arrays, effects

GATE_DIR = ROOT / "review/heldout/confirmatory/cluster_outputs"

def primary_per_target(df: pd.DataFrame, labels: list):
    ids, patients, views, R, T, B, D = effects(df, labels)
    return ids, patients, contrast_arrays(R, T, B, D)["primary"].mean(axis=1)

def joint_bootstrap(patients, sets: dict, n_resamples: int = B_RESAMPLES, seed: int = 0) -> dict:
    patients = np.asarray(patients)
    uniq = np.unique(patients)
    groups = [np.where(patients == p)[0] for p in uniq]
    rng = np.random.default_rng(seed)
    m = next(iter(sets.values())).shape[1]
    reps = {k: np.empty(n_resamples) for k in sets}
    for i in range(n_resamples):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(uniq), size=len(uniq))])
        pick = rng.integers(0, m, size=(len(idx), m))
        for k, v in sets.items():
            reps[k][i] = np.take_along_axis(v[idx], pick, axis=1).mean(1).mean()
    return reps

def summ(point: float, r: np.ndarray) -> dict:
    return {"mean": float(point), "ci90": [float(np.percentile(r, 5)), float(np.percentile(r, 95))], "ci95": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = sorted(GATE_DIR.glob("gate_s_dn_*/scores_densenet121.csv"))
    df = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in files], ignore_index=True)
    originals = sorted(df[df.family == "original"].label.unique())
    planted = sorted(df[df.family != "original"].label.unique())
    strengths = sorted({l.rsplit("_seed", 1)[0] for l in planted})
    assert len(originals) == 5 and "weak" in strengths
    weak = [l for l in planted if l.rsplit("_seed", 1)[0] == "weak"]
    assert len(weak) == 3
    ids, patients, orig = primary_per_target(df, originals)
    ids_w, patients_w, wk = primary_per_target(df, weak)
    assert ids == ids_w and (patients == patients_w).all()
    out: dict = {"n_targets": len(ids), "n_patients": int(len(set(patients))), "originals": originals, "weak": weak, "resamples": B_RESAMPLES, "seed": 0,
                 "status": "DESCRIPTIVE, UNREGISTERED; no detection threshold claimed"}
    sets = {"orig": orig, "weak": wk, "zero": primary_per_target(df, [l for l in planted if l.startswith("zero_")])[2]}
    for s in strengths:
        sets.setdefault(s, primary_per_target(df, [l for l in planted if l.rsplit("_seed", 1)[0] == s])[2])
    reps = joint_bootstrap(patients, sets)
    pts = {k: float(v.mean()) for k, v in sets.items()}
    out["pooled_primary_by_set"] = {k: summ(pts[k], reps[k]) for k in sets}
    d = reps["weak"] - reps["orig"]
    out["weak_minus_original"] = summ(pts["weak"] - pts["orig"], d)
    out["original_minus_weak_is_negative_share_of_resamples"] = float((d > 0).mean())
    d0 = reps["weak"] - reps["zero"]
    out["weak_minus_zero"] = summ(pts["weak"] - pts["zero"], d0)
    out["original_minus_zero"] = summ(pts["orig"] - pts["zero"], reps["orig"] - reps["zero"])
    out["ordering_by_point_estimate"] = [k for k, _ in sorted(pts.items(), key=lambda kv: kv[1])]
    out["per_checkpoint_weak"] = {l: float(primary_per_target(df, [l])[2].mean()) for l in weak}
    out["per_checkpoint_original"] = {l: float(primary_per_target(df, [l])[2].mean()) for l in originals}
    out["weak_seed_min_max"] = [min(out["per_checkpoint_weak"].values()), max(out["per_checkpoint_weak"].values())]
    out["original_seed_min_max"] = [min(out["per_checkpoint_original"].values()), max(out["per_checkpoint_original"].values())]

    from scipy import stats as _st
    tt = _st.ttest_ind(list(out["per_checkpoint_weak"].values()), list(out["per_checkpoint_original"].values()), equal_var=False)
    out["checkpoint_level_welch_weak_vs_original"] = {"t": float(tt.statistic), "df": float(tt.df), "p": float(tt.pvalue), "n_weak": 3, "n_original": 5,
                                                      "note": "descriptive companion: the unit is the checkpoint, so the interval above (patients and placements only, conditional on these checkpoints) is not the same question"}

    from analysis.draw_variability import rubin_combine
    from analysis.audit_analysis import boot
    lab = "densenet121_original_seed1"
    per_draw = []
    for k in range(10):
        dd = pd.read_csv(ROOT / f"review/draw_variability/outputs/draw{k}/scores_original.csv", dtype={"patient_id": str})
        dd = dd[dd.label == lab.replace("densenet121_", "")]
        gap = dd["delta_logit_insertion"] - dd["delta_logit_control"]
        ci, se = boot(dd["patient_id"], gap)
        per_draw.append({"gap": float(gap.mean()), "ci95": ci, "W": se ** 2, "n": int(len(dd)), "n_patients": int(dd["patient_id"].nunique())})
    comb = rubin_combine([x["gap"] for x in per_draw], [x["W"] for x in per_draw], float(np.mean([x["n_patients"] for x in per_draw])))
    gate_a = json.loads((ROOT / "review/heldout/confirmatory/analysis_densenet121.json").read_text())
    g1 = gate_a["originals"]["per_checkpoint_descriptive"][lab]["primary"]
    out["seed1_negative_targets_ten_draws"] = {"per_draw": per_draw, "mean": comb["gbar"], "draw_min": min(x["gap"] for x in per_draw), "draw_max": max(x["gap"] for x in per_draw), "rubin_ci95": comb["ci"], "n_targets": per_draw[0]["n"], "n_patients": per_draw[0]["n_patients"]}
    out["seed1_gate"] = {"mean": g1["mean"], "ci95": g1["ci95"], "ci90": g1["ci90"], "n_targets": gate_a["originals"]["pooled"]["_n_targets"]}
    out["seed1_gate_minus_negative_ten_draw_mean"] = g1["mean"] - comb["gbar"]
    out["seed1_intervals_overlap"] = bool(max(comb["ci"][0], g1["ci95"][0]) <= min(comb["ci"][1], g1["ci95"][1]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("weak_minus_original", "weak_minus_zero", "original_minus_zero", "ordering_by_point_estimate", "weak_seed_min_max", "original_seed_min_max")}, indent=1))
    print(json.dumps({k: out[k] for k in ("seed1_negative_targets_ten_draws", "seed1_gate", "seed1_gate_minus_negative_ten_draw_mean", "seed1_intervals_overlap")}, indent=0)[:1800])

if __name__ == "__main__":
    main()
