# Summarises the per-checkpoint validity JSONs of the ResNet-50 and ViT-B/16 runs into one table.
# Run: python3 analysis/summarize_validity_evals.py --help

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path, default=Path("."))
    args = p.parse_args()

    pattern_a = str(args.eval_dir / "val_report_eval_resnet50_seed*.json")
    pattern_b = str(args.eval_dir / "val_report_eval_vit_b16_seed*.json")
    for f in sorted(glob.glob(pattern_a)) + sorted(glob.glob(pattern_b)):
        d = json.loads(Path(f).read_text())
        r = d["report"]
        s = d["select"]
        kc = d["kill_criterion"]
        npd = r["negative_probability_distribution"]
        print(f"{Path(f).name}: AUC={r['auc']:.4f} CI=({r['ci_low']:.4f},{r['ci_high']:.4f}) "
              f"median_neg_p={npd['median']:.6g} brier={r['brier']:.4f} ece={r['ece']:.4f} "
              f"val_select_auc={s['auc']:.4f} clause_a={kc['clause_a_auc_ge_0.82']} "
              f"clause_b={kc['clause_b_median_p_ge_0.001']} both={kc['both_clauses_pass']}")

if __name__ == "__main__":
    main()
