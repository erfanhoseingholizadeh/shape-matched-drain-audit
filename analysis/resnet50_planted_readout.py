# Reads out the ResNet-50 planted checkpoints on the negative targets and applies the positive-control criterion.
# Run: python3 analysis/resnet50_planted_readout.py --help

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from analysis import intermediate_readout as p65

CONFIRM_RESNET50 = 0.1823
INVERT_RESNET50 = 0.8418
STRENGTHS = ["zero", "mid30", "strong"]

def reaches_invert(g: dict) -> bool:
    return bool(g["gap"] >= INVERT_RESNET50 and g["ci"][0] > 0 and all(v > 0 for v in g["per_seed_gap"].values()))

def branch(manip_strong_passes: bool, strong_invert: bool) -> str:
    if not manip_strong_passes:
        return "c"
    return "a" if strong_invert else "b"

def build(manip_groups: dict, scores: dict) -> dict:
    pts = {}
    for name in ["original"] + STRENGTHS:
        eff = p65.manip_effects(manip_groups[name])
        g = p65.gap_result(scores[name])
        pts[name] = {"manip_effect": eff, "audit": g, "confirm": p65.leaves_confirm(g, ceiling=CONFIRM_RESNET50),
                     "reaches_invert": reaches_invert(g), "pooled_flags": manip_groups[name]["pooled_concatenated"]}
    pc = manip_groups["strong"]["pooled_concatenated"]
    passes = bool(pc["condition1_ci_high_below_0.5"] and pc["condition2_edited_below_unedited_nonoverlapping"])
    return {"points": pts, "manipulation_check_passes_at_strong": passes, "branch": branch(passes, pts["strong"]["reaches_invert"]),
            "constants": {"confirm": CONFIRM_RESNET50, "invert": INVERT_RESNET50}}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manip", type=Path, required=True)
    ap.add_argument("--scores-original", type=Path, required=True)
    ap.add_argument("--scores-planted", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    groups = json.loads(a.manip.read_text())["groups"]
    planted = pd.read_csv(a.scores_planted, dtype={"patient_id": str})
    scores = {"original": pd.read_csv(a.scores_original, dtype={"patient_id": str})}
    for s in STRENGTHS:
        scores[s] = planted[planted["label"].str.startswith(s + "_")].copy()
        assert scores[s]["label"].nunique() == 3, s
    res = build(groups, scores)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    from cxr_shortcuts.eval.population_log import log_population
    rows = pd.concat([scores[n][["image_id", "patient_id"]] for n in scores]).drop_duplicates()
    log_population(a.out.parent, "resnet50_readout_audit_images", rows["image_id"].tolist(), "image")
    log_population(a.out.parent, "resnet50_readout_audit_patients", rows["patient_id"].tolist(), "patient")
    for n, p in res["points"].items():
        print(f"{n:9s} effect={p['manip_effect']['mean_of_per_seed']:+.4f} gap={p['audit']['gap']:+.4f} CI=({p['audit']['ci'][0]:+.4f},{p['audit']['ci'][1]:+.4f}) "
              f"leaves_CONFIRM={p['confirm']['leaves_confirm']} invert={p['reaches_invert']}")
    print("manipulation check passes at strong:", res["manipulation_check_passes_at_strong"], "| branch:", res["branch"])

if __name__ == "__main__":
    main()
