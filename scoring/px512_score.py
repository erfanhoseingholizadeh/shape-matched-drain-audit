# Scores the persisted audit composites with the classifiers at a chosen input size.
# Run: python3 scoring/px512_score.py --size 512 --arch densenet121 --composite-dir D --checkpoints J --out-dir D [--expected-digest H] [--device cuda] [--smoke-n N]

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))
sys.path.insert(0, str(SCRIPTS.parent / "src"))

ORACLE_MAX, ORACLE_MEAN = 0.05, 0.005
EXPECTED_134_COMPOSITE = "cccb2fc8d3058f5ae1e5124dbd61294e2142454b343092fe433d275170727e64"

def oracle_224_check(new: pd.DataFrame, recorded: pd.DataFrame) -> dict:
    arms = [c for c in recorded.columns if c.startswith("logit_")]
    key = ["label", "image_id"]
    recorded = recorded[recorded["label"].isin(set(new["label"]))]
    m = new.merge(recorded, on=key, suffixes=("_new", "_rec"))
    if len(m) != len(recorded) or len(m) != len(new):
        return {"pass": False, "reason": f"row sets differ: new {len(new)}, recorded {len(recorded)}, joined {len(m)}"}
    d = np.abs(np.stack([m[f"{a}_new"].to_numpy() - m[f"{a}_rec"].to_numpy() for a in arms]))
    return {"pass": bool(d.max() <= ORACLE_MAX and d.mean() <= ORACLE_MEAN), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "rows": int(len(m)), "arms": len(arms),
            "tolerance": {"max": ORACLE_MAX, "mean": ORACLE_MEAN}}

def run(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--arch", required=True, choices=["densenet121", "resnet50"])
    ap.add_argument("--composite-dir", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-digest", default=EXPECTED_134_COMPOSITE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-n", type=int, default=0)
    ap.add_argument("--commit", default="")
    a = ap.parse_args(argv)
    from scoring import positive_audit as P71

    P71.IMAGE_SIZE = a.size
    ns = argparse.Namespace(composite_dir=a.composite_dir, checkpoints=a.checkpoints, arch=a.arch, out_dir=a.out_dir, expected_digest=a.expected_digest, device=a.device, smoke_n=a.smoke_n, commit=a.commit)
    P71.run_score(ns)
    (a.out_dir / f"size_{a.arch}.json").write_text(json.dumps({"image_size": a.size, "wrapper": "px512_score.py", "frozen_scorer": "positive_audit.py"}))
    return 0

def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "oracle":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--new", type=Path, required=True)
        ap.add_argument("--recorded", type=Path, required=True)
        a = ap.parse_args()
        read = lambda p: pd.read_csv(p, dtype={"image_id": str, "patient_id": str})
        res = oracle_224_check(read(a.new), read(a.recorded))
        print(json.dumps(res))
        return 0 if res["pass"] else 7
    return run()

if __name__ == "__main__":
    sys.exit(main())
