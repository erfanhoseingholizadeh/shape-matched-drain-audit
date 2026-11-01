# Derives the paper's margin ends, validity spans, caliber at model resolution and funnel counts from recorded aggregates.
# Run: python3 paper_values/margins_and_spans.py (no options; it starts working at once)

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "review" / "heldout_values" / "outputs" / "manuscript_values.json"
WIDTHS = ROOT / "review" / "population_assignment" / "outputs" / "interval_widths.json"
OUT = ROOT / "review" / "margins_and_spans" / "isbi_values.json"

ARCH = {"dn": "densenet121", "rn": "resnet50"}
BAR = 0.82
NATIVE_PX, MODEL_PX = 1024, 224
CALIBER_NATIVE = (10.0, 27.0)
MEAN_P_SHIFT = 0.02

def ceil3(x: float) -> float:
    return math.ceil(round(x * 1000, 9)) / 1000

def build() -> dict:
    g = json.loads(GATE.read_text())
    w = json.loads(WIDTHS.read_text())

    margin_end, per_ck, prob = {}, {}, {}
    for a, name in ARCH.items():
        lo, hi = g[f"q-gate-{a}-primary-l90"], g[f"q-gate-{a}-primary-h90"]
        end = ceil3(max(abs(lo), abs(hi)))
        assert -end <= lo and hi <= end, (a, lo, hi, end)
        assert not (-(end - 0.001) <= lo and hi <= end - 0.001), (a, lo, hi, end)
        margin_end[name] = end

        margin = g[f"q-margin-{a}"]
        cks = [(g[f"q-gate-{a}-ck{i}-primary-m"], g[f"q-gate-{a}-ck{i}-primary-l90"], g[f"q-gate-{a}-ck{i}-primary-h90"]) for i in range(5)]
        assert sum(1 for m, _, _ in cks if m < 0) == g[f"q-gate-{a}-ck-n-neg"]
        assert sum(1 for m, _, _ in cks if abs(m) < margin) == g[f"q-gate-{a}-ck-n-inside"]
        per_ck[name] = {
            "n_point_negative": sum(1 for m, _, _ in cks if m < 0),
            "n_point_beyond_margin": sum(1 for m, _, _ in cks if abs(m) >= margin),
            "n_point_beyond_above": sum(1 for m, _, _ in cks if m >= margin),
            "n_point_beyond_below": sum(1 for m, _, _ in cks if m <= -margin),
            "n_ci90_inside_margin": sum(1 for _, l, h in cks if -margin <= l and h <= margin),
            "margin": margin,
        }
        shift_margin = g[f"q-gate-{a}-margin-sens"]
        prob[name] = {
            "pooled_contrast_logits": g[f"q-gate-{a}-primary-m"],
            "margin_on_heldout_baseline": shift_margin,
            "approx_mean_probability_change": g[f"q-gate-{a}-primary-m"] * MEAN_P_SHIFT / shift_margin,
        }

    validity = {}
    for a, name in ARCH.items():
        cks = [c for c in w["populations"]["R_int"]["checkpoints"] if c["architecture"] == name]
        assert len(cks) == 5
        validity[name] = {
            "auc_min": min(c["auc"] for c in cks), "auc_max": max(c["auc"] for c in cks),
            "ci_low_min": min(c["ci_low"] for c in cks), "ci_high_max": max(c["ci_high"] for c in cks),
            "n_ci_wholly_above_bar": sum(1 for c in cks if c["ci_low"] > BAR),
            "n_point_at_or_above_bar": sum(1 for c in cks if c["auc"] >= BAR),
        }
    validity["n_ci_wholly_above_bar_both"] = sum(validity[n]["n_ci_wholly_above_bar"] for n in ARCH.values())

    lo, hi = (c * MODEL_PX / NATIVE_PX for c in CALIBER_NATIVE)
    return {
        "margin_end": margin_end,
        "validity": validity,
        "caliber_at_model_resolution": {"lo": lo, "hi": hi, "native_px": NATIVE_PX, "model_px": MODEL_PX},
        "per_checkpoint": per_ck,
        "per_checkpoint_total": {k: sum(per_ck[n][k] for n in ARCH.values()) for k in ("n_point_negative", "n_point_beyond_margin", "n_point_beyond_above", "n_point_beyond_below", "n_ci90_inside_margin")},
        "prob_scale_approximation": {**prob, "note": "approximate: pooled contrast times 0.02 over the margin recomputed on the held-out targets' own baseline; linear in the shift; response document only"},
        "sources": [str(GATE.relative_to(ROOT)), str(WIDTHS.relative_to(ROOT)), "the generator's native resolution (1024 px)"],
    }

def main() -> int:
    vals = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(vals, indent=2) + "\n")
    print(json.dumps(vals, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
