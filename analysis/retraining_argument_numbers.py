# Collects the numbers behind the decision not to retrain the original checkpoints.
# Run: python3 analysis/retraining_argument_numbers.py (no options; it starts working at once)

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PROV = REPO_ROOT / "review/population_claims/outputs/val_select_provenance.json"
DIAG = REPO_ROOT / "review/clean_population/outputs/clean_population_diagnostics.json"
OUT = REPO_ROOT / "review/population_assignment/outputs/retraining_argument_numbers.json"

def _gap(per_seed: dict, a: str, b: str) -> dict:
    ps = {k: v[a] - v[b] for k, v in per_seed.items()}
    return {"per_seed": ps, "mean": float(np.mean(list(ps.values())))}

def selected_minus_complement(per_seed: dict) -> dict:
    return _gap(per_seed, "auc_S_int_selected_half", "auc_R_int")

def selected_minus_rest_of_r_int(per_seed: dict) -> dict:
    return _gap(per_seed, "auc_S_int_selected_half", "auc_rest_of_R_int")

def build(prov: dict, diag: dict) -> dict:
    arch = prov["architectures"]
    return {
        "heuristic_argmax_epoch_optimism_bound": {a: arch[a]["mean_plateau_excess_on_select_set"] for a in ("densenet121", "resnet50")},
        "patient_mix_seen_minus_unseen_inside_val_report": {
            "selected_checkpoints": {a: arch[a]["mean_seen_minus_unseen"] for a in ("densenet121", "resnet50")},
            "retired_never_selected": prov["retired_instrument_control_never_selected_on_val"]["mean_seen_minus_unseen"]},
        "selected_minus_never_selected_half": {
            **{a: selected_minus_complement(diag["phaseA"][a]) for a in ("densenet121", "resnet50")},
            "retired_never_selected": selected_minus_complement(diag["retired"])},
        "comparison_as_written_selected_minus_rest_of_R_int": {a: selected_minus_rest_of_r_int(diag["phaseA"][a]) for a in ("densenet121", "resnet50")},
    }

def main() -> int:
    out = build(json.loads(PROV.read_text()), json.loads(DIAG.read_text()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v if k != "selected_minus_never_selected_half" and k != "comparison_as_written_selected_minus_rest_of_R_int"
                          else {a: round(x["mean"], 4) for a, x in v.items()}) for k, v in out.items()}, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
