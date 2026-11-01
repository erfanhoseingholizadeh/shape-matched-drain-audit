# Works out how many placements per target to use from the recorded variance components.
# Run: python3 analysis/placements_choice.py (no options; it starts working at once)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

R = Path(__file__).resolve().parent.parent
Z90 = 1.6449
N_IMG, N_PAT = 157, 57
MARGIN = {"densenet121": 0.156, "resnet50": 0.1823}
M_GRID = (1, 2, 4, 8, 12, 16)

def power_equiv(delta: float, se: float, m: float) -> float:
    hi = (m - Z90 * se - delta) / se
    lo = (-m + Z90 * se - delta) / se
    return float(max(0.0, norm.cdf(hi) - norm.cdf(lo)))

def table(v: dict, est: float, m: float) -> dict:
    sa, sw = v["sigma_a2"], v["sigma_w2"]
    donor_k8 = v["donor_part_of_sigma_w2"]
    resid = v["residual_placement_part"]
    out = {}
    for M in M_GRID:
        row = {}
        for name, irr in (("optimistic", 0.0), ("pessimistic", resid)):
            red_single_donor = (resid - irr) + v["donor_var_single"]
            sw_m = irr + red_single_donor / M
            se = float(np.sqrt(sa / N_PAT + sw_m / N_IMG))
            row[name] = {"sigma_w2": sw_m, "se_gate": se, "power_at_zero": power_equiv(0.0, se, m), "power_at_exploratory_estimate": power_equiv(est, se, m)}
        out[str(M)] = row
    out["floor_se_target_patient_component_only"] = float(np.sqrt(sa / N_PAT))
    out["positive_audit_design_M1_K8"] = {"se_gate": float(np.sqrt(sa / N_PAT + sw / N_IMG)), "power_at_zero": power_equiv(0.0, float(np.sqrt(sa / N_PAT + sw / N_IMG)), m),
                                  "power_at_exploratory_estimate": power_equiv(est, float(np.sqrt(sa / N_PAT + sw / N_IMG)), m)}
    assert abs(sw - (resid + donor_k8)) < 1e-9
    return out

def main() -> None:
    d = json.loads((R / "review/positive_audit/positive_analysis.json").read_text())
    out = {"note": "the arithmetic behind the choice of M (placements per target), from the exploratory audit's recorded variance components.", "n_img": N_IMG, "n_pat": N_PAT, "archs": {}}
    for arch in MARGIN:
        out["archs"][arch] = {"margin": MARGIN[arch], "exploratory_estimate": d[arch]["pooled"]["primary"]["mean"],
                              "table": table(d[arch]["variance_components"], d[arch]["pooled"]["primary"]["mean"], MARGIN[arch])}
    (R / "review/heldout").mkdir(parents=True, exist_ok=True)
    (R / "review/heldout/m_choice.json").write_text(json.dumps(out, indent=1))
    for arch, a in out["archs"].items():
        print(arch, "estimate", round(a["exploratory_estimate"], 4), "margin", a["margin"], "| floor SE (patients only)", round(a["table"]["floor_se_target_patient_component_only"], 4),
              "| positive_audit M1K8 SE", round(a["table"]["positive_audit_design_M1_K8"]["se_gate"], 4), "power@est", round(a["table"]["positive_audit_design_M1_K8"]["power_at_exploratory_estimate"], 3))
        for M in M_GRID:
            r = a["table"][str(M)]
            print(f"  M={M:2d}  SE {r['optimistic']['se_gate']:.4f}..{r['pessimistic']['se_gate']:.4f}   power@0 {r['optimistic']['power_at_zero']:.4f}..{r['pessimistic']['power_at_zero']:.4f}"
                  f"   power@est {r['pessimistic']['power_at_exploratory_estimate']:.3f}..{r['optimistic']['power_at_exploratory_estimate']:.3f}")

if __name__ == "__main__":
    main()
