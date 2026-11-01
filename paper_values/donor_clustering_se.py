# Computes the change in the held-out standard error when donor patients are clustered as well as target patients, for both architectures.
# Run: python3 paper_values/donor_clustering_se.py (no options; it starts working at once)

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "review" / "further_runs" / "gate_planted_donor.json"
OUT = ROOT / "review" / "donor_clustering_se" / "isbi_values.json"

def build() -> dict:
    d = json.loads(SRC.read_text())["1.2_donor_dependence"]
    dn, rn = d["densenet121"]["se"], d["resnet50"]["se"]
    return {
        "densenet121_se_rise_pct": 100 * (dn["two_way"] / dn["one_way_target"] - 1),
        "resnet50_se_fall_pct": 100 * (1 - rn["two_way"] / rn["one_way_target"]),
        "source": "review/further_runs/gate_planted_donor.json, 1.2_donor_dependence, se.two_way and se.one_way_target",
    }

def main() -> int:
    vals = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(vals, indent=2) + "\n")
    print(vals)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
