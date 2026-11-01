# Computes the 95 percent interval of the validity AUC of each original checkpoint on the cleaned population.
# Run: python3 analysis/interval_widths.py (no options; it starts working at once)

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "review/clean_population/outputs"
OUT_DIR = REPO_ROOT / "review/population_assignment/outputs"
BAR_A = 0.82
ARCHS = ("densenet121", "resnet50")
POPULATIONS = {"C": "clean", "R_int": "sensitivity"}

def relation(auc: float, lo: float, hi: float, bar: float = BAR_A) -> str:
    if lo > bar:
        return "ci_excludes_bar_from_above"
    if hi < bar:
        return "ci_excludes_bar_from_below"
    return "ci_contains_bar"

def describe_checkpoint(arch: str, seed: int, rec: dict, bar: float = BAR_A) -> dict:
    auc, lo, hi = float(rec["auc"]), float(rec["ci_low"]), float(rec["ci_high"])
    margin = auc - bar
    return {
        "architecture": arch, "seed": seed, "auc": auc, "ci_low": lo, "ci_high": hi,
        "width": hi - lo, "margin_over_bar": margin,

        "reach_toward_bar": (auc - lo) if margin >= 0 else (hi - auc),
        "relation": relation(auc, lo, hi, bar),
    }

def usability(rows: list[dict]) -> dict:
    n = len(rows)
    excl = sum(1 for r in rows if r["relation"] != "ci_contains_bar")
    return {"n_intervals_excluding_bar": excl, "n_checkpoints": n, "any_resolved": excl >= 1,
            "majority_resolved": excl >= 6 if n == 10 else excl * 2 > n, "all_resolved": excl == n}

def halt_fires_for_0_5(usability_on_C: dict) -> bool:
    return bool(usability_on_C["any_resolved"])

def build_report(in_dir: Path = DEFAULT_IN, bar: float = BAR_A) -> dict:
    pops: dict = {}
    for pop, prefix in POPULATIONS.items():
        rows: list = []
        for arch in ARCHS:
            d = json.loads((in_dir / f"{prefix}_{arch}.json").read_text())
            for s in sorted(d["per_seed"], key=int):
                rows.append(describe_checkpoint(arch, int(s), d["per_seed"][s], bar))
            n_patients = d["population_patients"]
        widths = [r["width"] for r in rows]
        pops[pop] = {
            "n_patients": n_patients, "checkpoints": rows, "usability": usability(rows),
            "width_min": min(widths), "width_max": max(widths), "width_mean": sum(widths) / len(widths),
            "reach_toward_bar_min": min(r["reach_toward_bar"] for r in rows),
            "reach_toward_bar_max": max(r["reach_toward_bar"] for r in rows),
            "margin_over_bar_min": min(r["margin_over_bar"] for r in rows),
            "margin_over_bar_max": max(r["margin_over_bar"] for r in rows),
            "n_below_bar_point_estimate": sum(1 for r in rows if r["margin_over_bar"] < 0),
        }
    return {"bar": bar, "populations": pops, "halt_fires": halt_fires_for_0_5(pops["C"]["usability"]),
            "criterion_note": "written after the C intervals were on record; nested readings, weakest one halts (module docstring)"}

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = build_report()
    (OUT_DIR / "interval_widths.json").write_text(json.dumps(rep, indent=2))
    for pop, p in rep["populations"].items():
        u = p["usability"]
        print(f"{pop}: n_patients={p['n_patients']}  intervals excluding 0.82: {u['n_intervals_excluding_bar']}/{u['n_checkpoints']}  "
              f"width {p['width_min']:.4f}..{p['width_max']:.4f} (mean {p['width_mean']:.4f})  "
              f"reach toward bar {p['reach_toward_bar_min']:.4f}..{p['reach_toward_bar_max']:.4f}  "
              f"margin over bar {p['margin_over_bar_min']:+.4f}..{p['margin_over_bar_max']:+.4f}")
    print("0.5 halt fires:", rep["halt_fires"])
    return 1 if rep["halt_fires"] else 0

if __name__ == "__main__":
    sys.exit(main())
