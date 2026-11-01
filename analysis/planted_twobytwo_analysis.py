# Analyses the negative-target two-by-two for the planted checkpoints beside the originals.
# Run: python3 analysis/planted_twobytwo_analysis.py --scores a.csv b.csv --arch densenet121 --out out.json

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent))
from scoring import twobytwo_audit as p70

def by_strength(df: pd.DataFrame) -> dict:
    s = df.label.str.extract(r"_planted_([a-z0-9]+)_seed\d+$")[0]
    if s.isna().any():
        raise SystemExit("a label does not match <arch>_planted_<strength>_seed<k>")
    return {k: df[s == k] for k in sorted(s.unique())}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, nargs="+", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--originals-json", type=Path, default=SCRIPTS_DIR.parent / "review/originals_twobytwo/twobytwo/twobytwo_analysis.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f, dtype={"patient_id": str}) for f in a.scores], ignore_index=True)
    out = {"arch": a.arch, "originals_resnet50_planted": json.loads(a.originals_json.read_text()).get(a.arch), "planted": {}}
    for strength, sub in by_strength(df).items():
        res = p70.analyze(sub)
        res["n_seeds"] = int(sub.label.nunique())
        out["planted"][strength] = res
        print(a.arch, strength, {c: round(x["mean"], 4) for c, x in res["cells"].items()}, {k: round(x["mean"], 4) for k, x in res["decomposition"].items()})
    a.out.write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
