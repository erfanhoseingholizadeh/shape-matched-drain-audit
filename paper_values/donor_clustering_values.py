# Reads the donor-clustering summary and writes the values the paper cites for it.
# Run: python3 paper_values/donor_clustering_values.py [--write-sources]

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "review/further_runs/gate_planted_donor.json"
OUT = ROOT / "review/further_runs_summary/manuscript_values.json"
SOURCES = ROOT / "paper/sources.json"
IDS = {
    "q-w76-donor-se-change-pct": ("0.1", 0.05, "Percentage by which the DenseNet-121 original checkpoints' standard error on the gate rises when donor patients are clustered as well as target patients (two-way over one-way target standard error, minus one, times 100)."),
    "q-w76-donor-patients-two-slots": ("133", 0.5, "Donor patients serving two slots among the gate's 1,256 donor slots (once in each view stratum)."),
    "q-w76-donor-patients": ("1,123", 0.5, "Distinct donor patients among the gate's 1,256 donor slots."),
}

def build() -> dict:
    j = json.loads(SRC.read_text())
    d = j["1.2_donor_dependence"]["densenet121"]
    r = j["1.3_recorded_metadata"]
    return {"q-w76-donor-se-change-pct": 100 * (d["se"]["two_way"] / d["se"]["one_way_target"] - 1),
            "q-w76-donor-patients-two-slots": r["donor_patients_serving_two_targets"], "q-w76-donor-patients": r["distinct_donor_patients"]}

def main(argv: list) -> int:
    vals = build(); OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(vals, indent=2) + "\n"); print(vals)
    if "--write-sources" in argv:
        s = json.loads(SOURCES.read_text()); by = {e["id"]: i for i, e in enumerate(s)}
        for k, (disp, tol, note) in IDS.items():
            e = {"id": k, "value": disp, "instrument": "phaseA-densenet", "kind": "verified", "script": "paper_values/donor_clustering_values.py", "artifact": "review/further_runs_summary/manuscript_values.json",
                 "path": k, "tolerance": tol, "note": note + " Read from the recorded the further-runs stage summary by paper_values/donor_clustering_values.py."}
            if k in by: s[by[k]] = e
            else: s.append(e)
        SOURCES.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n"); print("sources upserted")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
