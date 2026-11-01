# Analyses the audit of the original and the planted checkpoints on the negative targets.
# Run: python3 analysis/audit_analysis.py --help

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cxr_shortcuts.eval.bootstrap import cluster_bootstrap_ci_indices

MDE_FACTOR = 2.8015852
INVERT = 0.73
N_RESAMPLES = 2000
ARMS = {"insertion": "delta_logit_insertion", "control": "delta_logit_control", "blob": "delta_logit_blob"}

def boot(cluster_ids, values):
    values = np.asarray(values, float)
    reps = []

    def stat(idx):
        v = float(values[idx].mean())
        reps.append(v)
        return v

    ci = cluster_bootstrap_ci_indices(cluster_ids, stat, n_resamples=N_RESAMPLES, seed=0)
    return [float(ci[0]), float(ci[1])], float(np.std(reps, ddof=1))

def arm_stats(df: pd.DataFrame) -> dict:
    out = {}
    for arm, col in ARMS.items():
        ci, se = boot(df["patient_id"], df[col])
        out[arm] = {"mean": float(df[col].mean()), "ci": ci, "se": se}
    gap = df["delta_logit_insertion"] - df["delta_logit_control"]
    ci, se = boot(df["patient_id"], gap)
    out["gap"] = {"mean": float(gap.mean()), "ci": ci, "se": se, "mde": MDE_FACTOR * se}
    return out

def classify_pool(df: pd.DataFrame, anchor_ub: dict) -> dict:
    s = arm_stats(df)
    seeds = sorted(df["label"].unique())
    seed_gaps = {l: float((d["delta_logit_insertion"] - d["delta_logit_control"]).mean()) for l, d in df.groupby("label")}
    r = {"stats": s, "per_seed_gap": seed_gaps,
         "insertion_large": s["insertion"]["ci"][0] > anchor_ub["insertion"],
         "control_large": s["control"]["ci"][0] > anchor_ub["control"],
         "close": abs(s["gap"]["mean"]) < s["gap"]["mde"],
         "branch1": bool(s["gap"]["mean"] >= INVERT and s["gap"]["ci"][0] > 0 and all(v > 0 for v in seed_gaps.values())
                         and len(seeds) == 3)}
    if r["branch1"]:
        r["branch"] = "1"
    elif not r["insertion_large"]:
        r["branch"] = "2b"
    elif r["control_large"] and r["close"]:
        r["branch"] = "2a"
    elif not r["close"]:
        r["branch"] = "2c"
    else:
        r["branch"] = "2d"
    return r

def cmd_anchor(a) -> None:
    df = pd.read_csv(a.scores, dtype={"patient_id": str})
    summ = json.loads(Path(a.summary).read_text())
    labels = sorted(df["label"].unique())
    assert len(labels) == 5 and df.groupby("label").size().nunique() == 1, "expected 5 originals x equal n"
    pooled = arm_stats(df)
    ub = {arm: pooled[arm]["ci"][1] for arm in ARMS}
    oracle, misfire = {}, False
    for sub in combinations(labels, 3):
        c = classify_pool(df[df["label"].isin(sub)], ub)
        oracle[str(sub)] = {k: c[k] for k in ("insertion_large", "control_large", "close")} | {
            "insertion_mean": c["stats"]["insertion"]["mean"], "control_mean": c["stats"]["control"]["mean"],
            "gap_mean": c["stats"]["gap"]["mean"]}
        misfire |= bool(c["insertion_large"] or c["control_large"])
    per_seed = {l: arm_stats(d) for l, d in df.groupby("label")}
    result = {"n_kept": int(df["image_id"].nunique()), "n_rows": len(df), "n_patient_clusters": int(df["patient_id"].nunique()),
              "pooled_5seed": pooled, "anchor_ci_upper_bounds": ub, "oracle_10_subsets": oracle,
              "oracle_any_misfire": misfire, "oracle_n_close_of_10": sum(v["close"] for v in oracle.values()),
              "per_seed": per_seed, "composite_manifest_sha256": summ["composite_manifest_sha256"],
              "composite_summary": summ}
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("n_kept", "anchor_ci_upper_bounds", "oracle_any_misfire", "oracle_n_close_of_10")}, indent=1))
    print("pooled insertion", pooled["insertion"], "\ncontrol", pooled["control"], "\nblob", pooled["blob"], "\ngap", pooled["gap"])

def cmd_audit(a) -> None:
    anchor = json.loads(Path(a.anchor).read_text())
    ub = anchor["anchor_ci_upper_bounds"]
    planted = pd.read_csv(a.scores_planted, dtype={"patient_id": str})
    orig = pd.read_csv(a.scores_original, dtype={"patient_id": str})
    result = {"anchor_ci_upper_bounds": ub, "strengths": {}}
    for strength in ("strong", "weak", "zero"):
        d = planted[planted["label"].str.startswith(strength + "_")]
        assert d["label"].nunique() == 3, strength
        result["strengths"][strength] = classify_pool(d, ub)
    result["originals_5seed"] = arm_stats(orig)
    Path(a.out).write_text(json.dumps(result, indent=1))
    for s, r in result["strengths"].items():
        st = r["stats"]
        print(s, "branch", r["branch"], "ins", round(st["insertion"]["mean"], 4), st["insertion"]["ci"], "ctl",
              round(st["control"]["mean"], 4), st["control"]["ci"], "blob", round(st["blob"]["mean"], 4),
              "gap", round(st["gap"]["mean"], 4), st["gap"]["ci"], "close", r["close"], "seed gaps", r["per_seed_gap"])

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("anchor")
    p.add_argument("--scores", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--out", required=True)
    q = sub.add_parser("audit")
    q.add_argument("--scores-original", required=True)
    q.add_argument("--scores-planted", required=True)
    q.add_argument("--anchor", required=True)
    q.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_anchor(a) if a.cmd == "anchor" else cmd_audit(a)

if __name__ == "__main__":
    main()
