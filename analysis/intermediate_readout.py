# Reads out the planted checkpoints of the intermediate strengths against the planting log odds ratio.
# Run: python3 analysis/intermediate_readout.py --help

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from analysis.audit_analysis import INVERT, boot

CEILING = 0.156
PLANTED_ORDER = ["zero", "weak", "mid30", "mid50", "strong"]
LOG_ODDS = {"zero": 0.0, "weak": 0.4013, "mid30": 1.2381, "mid50": 2.1972, "strong": 4.3944}

def manip_effects(group: dict) -> dict:
    per_seed = [c["auc_unedited"]["auc"] - c["auc_edited"]["auc"] for c in group["checkpoints"]]
    pooled = group["pooled_concatenated"]["unedited"]["auc"] - group["pooled_concatenated"]["edited"]["auc"]
    return {"per_seed_effect": per_seed, "mean_of_per_seed": float(np.mean(per_seed)), "pooled_concatenated": float(pooled),
            "n_seeds": len(per_seed)}

def gap_result(df: pd.DataFrame) -> dict:
    gap = df["delta_logit_insertion"] - df["delta_logit_control"]
    ci, se = boot(df["patient_id"], gap)
    seed_gaps = {str(l): float(g.mean()) for l, g in gap.groupby(df["label"])}
    return {"gap": float(gap.mean()), "ci": ci, "se": se, "per_seed_gap": seed_gaps, "n_rows": int(len(df)),
            "n_images": int(df["image_id"].nunique()), "n_patients": int(df["patient_id"].nunique())}

def required_seeds_inside(n: int) -> int:
    return math.ceil(0.8 * n)

def leaves_confirm(g: dict, ceiling: float = CEILING) -> dict:
    lo, hi = g["ci"]
    ci_inside = bool(lo > -ceiling and hi < ceiling)
    inside = {s: bool(-ceiling < v < ceiling) for s, v in g["per_seed_gap"].items()}
    need = required_seeds_inside(len(inside))
    n_in = sum(inside.values())
    seeds_ok = n_in >= need
    return {"ci_inside_ceiling": ci_inside, "seeds_inside": inside, "n_seeds_inside": n_in, "n_required": need,
            "seed_clause_met": seeds_ok, "leaves_confirm": bool((not ci_inside) or (not seeds_ok)),
            "reason": [r for r, bad in (("ci_not_inside", not ci_inside), ("seed_clause_failed", not seeds_ok)) if bad]}

def reaches_invert(g: dict) -> bool:
    return bool(g["gap"] >= INVERT and g["ci"][0] > 0 and all(v > 0 for v in g["per_seed_gap"].values()))

def monotone_no_reentry(states: list) -> bool:
    seen_left = False
    for leaves in states:
        if leaves and not seen_left:
            seen_left = True
        elif (not leaves) and seen_left:
            return False
    return True

def bracket(points: list) -> dict:
    by_effect = sorted(points, key=lambda p: p["effect"])
    planted = [p for p in points if p["planted"]]
    planted_by_planting = sorted(planted, key=lambda p: PLANTED_ORDER.index(p["name"]))
    planted_by_effect = sorted(planted, key=lambda p: p["effect"])
    orderings_agree = [p["name"] for p in planted_by_planting] == [p["name"] for p in planted_by_effect]
    mono_planting = monotone_no_reentry([p["leaves"] for p in planted_by_planting])
    mono_effect = monotone_no_reentry([p["leaves"] for p in by_effect])
    out = {"order_by_effect": [(p["name"], p["effect"], p["leaves"]) for p in by_effect],
           "order_by_planting": [(p["name"], p["leaves"]) for p in planted_by_planting],
           "orderings_agree": orderings_agree, "monotone_along_planting_order": mono_planting,
           "monotone_along_effect_order": mono_effect}
    if not orderings_agree:
        out.update(claim="none", reason="ordering by manipulation-check effect disagrees with ordering by planting strength")
    elif not (mono_planting and mono_effect):
        out.update(claim="none", reason="the CONFIRM state is not monotone (a point re-enters CONFIRM after one has left it) "
                                       f"along the planting order ({mono_planting}) or the effect order ({mono_effect})")
    else:
        flips = [(a, b) for a, b in zip(by_effect, by_effect[1:]) if (not a["leaves"]) and b["leaves"]]
        if not any(p["leaves"] for p in by_effect):
            out.update(claim="no change", reason="no point leaves CONFIRM")
        elif by_effect[0]["leaves"]:
            out.update(claim="none", reason="the lowest-effect point has already left CONFIRM; no in-CONFIRM point to bracket from")
        else:
            (a, b), = flips
            out.update(claim="bracket", between=[a["name"], b["name"]], effects=[a["effect"], b["effect"]],
                       reason="adjacent points on the effect axis between which the CONFIRM state changes; no interpolation")
    return out

def build_readout(manip_groups: dict, scores_by_group: dict, names: list) -> dict:
    rows, points = {}, []
    for name in names:
        eff = manip_effects(manip_groups[name])
        g = gap_result(scores_by_group[name])
        lv = leaves_confirm(g)
        rows[name] = {"planted": name != "original", "log_odds_ratio": LOG_ODDS.get(name), "manip_effect": eff, "audit": g,
                      "confirm": lv, "reaches_invert": reaches_invert(g)}
        points.append({"name": name, "effect": eff["mean_of_per_seed"], "leaves": lv["leaves_confirm"], "planted": name != "original"})
    return {"points": rows, "bracket": bracket(points)}

def load_scores(path: Path, label_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_id": str})
    return df[df["label"].str.startswith(label_prefix)].copy()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manip-old", type=Path, required=True, help="the manipulation-check stage manipulation_check.json (originals, zero, weak, strong)")
    ap.add_argument("--manip-new", type=Path, default=None, help="manipulation_check.json of the mid30/mid50 run")
    ap.add_argument("--scores-original", type=Path, required=True, help="the fresh-audit stage job 1 scores_original.csv (draw 6400)")
    ap.add_argument("--scores-planted-old", type=Path, required=True, help="the fresh-audit stage job 2 scores_planted.csv (zero, weak, strong)")
    ap.add_argument("--scores-planted-new", type=Path, default=None, help="the new score-mode CSV (must contain mid30 and mid50)")
    ap.add_argument("--audit-json", type=Path, default=None, help="the fresh-audit stage audit.json; when given, the old strengths' pooled gaps must reproduce it")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    old = json.loads(a.manip_old.read_text())["groups"]
    manip = {k: old[k] for k in ("original", "zero", "weak", "strong")}
    scores = {"original": pd.read_csv(a.scores_original, dtype={"patient_id": str})}
    for s in ("zero", "weak", "strong"):
        scores[s] = load_scores(a.scores_planted_old, s + "_")
    names = ["original", "zero", "weak", "strong"]
    if a.manip_new is not None:
        new = json.loads(a.manip_new.read_text())["groups"]
        for s in ("mid30", "mid50"):
            manip[s] = new[s]
            scores[s] = load_scores(a.scores_planted_new, s + "_")
            assert scores[s]["label"].nunique() == 3, s
        names = ["original", "zero", "weak", "mid30", "mid50", "strong"]
    result = build_readout(manip, scores, names)
    result["includes_new_strengths"] = a.manip_new is not None
    if a.audit_json is not None:
        aud = json.loads(a.audit_json.read_text())
        rec = {}
        for s in ("zero", "weak", "strong"):
            rec[s] = abs(result["points"][s]["audit"]["gap"] - aud["strengths"][s]["stats"]["gap"]["mean"])
        rec["original"] = abs(result["points"]["original"]["audit"]["gap"] - aud["originals_5seed"]["gap"]["mean"])
        result["reproduction_vs_audit_json_abs_gap_diff"] = rec
    a.out.write_text(json.dumps(result, indent=1, default=str))

    from cxr_shortcuts.eval.population_log import log_population
    audit_rows = pd.concat([scores[n][["image_id", "patient_id"]] for n in names]).drop_duplicates()
    log_population(a.out.parent, "readout_audit_images", audit_rows["image_id"].tolist(), "image")
    log_population(a.out.parent, "readout_audit_patients", audit_rows["patient_id"].tolist(), "patient")
    for n in names:
        p = result["points"][n]
        print(f"{n:9s} effect(mean-of-seed)={p['manip_effect']['mean_of_per_seed']:+.4f} pooled-concat={p['manip_effect']['pooled_concatenated']:+.4f} "
              f"gap={p['audit']['gap']:+.4f} CI=({p['audit']['ci'][0]:+.4f},{p['audit']['ci'][1]:+.4f}) "
              f"seeds={ {k: round(v, 3) for k, v in p['audit']['per_seed_gap'].items()} } "
              f"leaves_CONFIRM={p['confirm']['leaves_confirm']} {p['confirm']['reason']} invert={p['reaches_invert']}")
    print("bracket:", json.dumps(result["bracket"], default=str))
    if "reproduction_vs_audit_json_abs_gap_diff" in result:
        print("reproduction vs the fresh-audit stage audit.json (abs gap diff):", result["reproduction_vs_audit_json_abs_gap_diff"])

if __name__ == "__main__":
    main()
