# Computes the margins at 512 pixels from the validity scorer's per-image probabilities.
# Run: python3 analysis/px512_margins.py --probs val_report_probs.npz --validity validity.json --out margins.json

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))
sys.path.insert(0, str(SCRIPTS.parent / "src"))

MARGIN_224 = {"densenet121": 0.156, "resnet50": 0.1823}
INVERT_224 = {"densenet121": 0.73, "resnet50": 0.8418}
EPS = 1e-12

def logit_from_prob(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))

def shifts(neg_logits: np.ndarray) -> dict:
    from analysis.reachability_constants import solve_uniform_shift_for_target_mean_dp as solve

    return {"margin_dp_0.02": float(solve(neg_logits, 0.02)), "invert_dp_0.10": float(solve(neg_logits, 0.10)), "n_negatives": int(len(neg_logits)), "median_baseline_p": float(np.median(1 / (1 + np.exp(-neg_logits))))}

def compute(probs_npz: dict, validity: dict, archs=("densenet121", "resnet50")) -> dict:
    labels = np.asarray(probs_npz["label"]).astype(int)
    names = [str(x) for x in probs_npz["checkpoints"]]
    P = np.asarray(probs_npz["probs"])
    neg = labels == 0
    out = {"definition": "uniform logit shift giving a mean probability change of 0.02 (margin) and 0.10 (INVERT) over the pooled R_str negatives of the valid E5 original checkpoints at 512 pixels (the vit-retry stage arithmetic)", "archs": {}}
    for arch in archs:
        orig = [n for n in names if n.startswith(f"{arch}/{arch}_orig_seed")]
        valid = [n for n in orig if validity["checkpoints"][n]["valid"]]
        res = {"originals_scored": len(orig), "originals_valid": len(valid), "valid_labels": valid, "margin_224_registered": MARGIN_224[arch], "invert_224_registered": INVERT_224[arch]}
        if valid:
            cols = [names.index(n) for n in valid]
            per = {n: shifts(logit_from_prob(P[neg, c])) for n, c in zip(valid, cols)}
            pooled = shifts(np.concatenate([logit_from_prob(P[neg, c]) for c in cols]))
            res.update(pooled=pooled, per_checkpoint=per, margin_512=pooled["margin_dp_0.02"], invert_512=pooled["invert_dp_0.10"])
        out["archs"][arch] = res
    return out

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", type=Path, required=True)
    ap.add_argument("--validity", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    z = np.load(a.probs, allow_pickle=False)
    res = compute({k: z[k] for k in z.files}, json.loads(a.validity.read_text()))
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {x: v.get(x) for x in ("originals_valid", "margin_512", "invert_512", "margin_224_registered")} for k, v in res["archs"].items()}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
