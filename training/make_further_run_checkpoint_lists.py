# Writes the checkpoint lists that the further-seed and near-margin scoring and validity jobs read.
# Run: python3 training/make_further_run_checkpoint_lists.py --out-dir DIR

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
E3 = {"densenet121": ("e3lo", "e3hi"), "resnet50": ("e3lo", "e3hi", "weak")}
SEEDS_E3, SEEDS_E4 = (0, 1, 2), tuple(range(5, 15))
STRATA = ["dev_half", "R_int_C"]
CONFIGS = {("e3", "densenet121", "e3lo"): "e3_densenet121_e3lo.yaml", ("e3", "densenet121", "e3hi"): "e3_densenet121_e3hi.yaml", ("e3", "resnet50", "e3lo"): "e3_resnet50_e3lo.yaml",
           ("e3", "resnet50", "e3hi"): "e3_resnet50_e3hi.yaml", ("e3", "resnet50", "weak"): "e3_resnet50_weak.yaml", ("e4", "densenet121", "e4"): "e4_densenet121.yaml",
           ("e4", "resnet50", "e4"): "e4_resnet50.yaml"}

def ckpt_path(config_name: str, seed: int, size: int = 224) -> str:
    cfg = yaml.safe_load((REPO / "configs" / config_name).read_text())
    return f"{cfg['output_dir']}_seed{seed}_res{size}/best_checkpoint.pt"

def audit_entries(kind: str, arch: str) -> list:
    out = []
    for (k, a, setting), cfg in CONFIGS.items():
        if k != kind or a != arch:
            continue
        for s in (SEEDS_E3 if kind == "e3" else SEEDS_E4):
            label = f"{setting}_seed{s}" if kind == "e3" else f"e4_seed{s}"
            out.append({"label": label, "checkpoint": ckpt_path(cfg, s), "family": f"planted_{arch}_further_runs_e3" if kind == "e3" else f"e4_{arch}", "strata": list(STRATA)})
    return out

def validity_entries() -> list:
    out = []
    for kind in ("e3", "e4"):
        for arch in ("densenet121", "resnet50"):
            for e in audit_entries(kind, arch):
                out.append({"label": f"{arch}/{e['label']}", "checkpoint": e["checkpoint"], "backbone": arch, "family": e["family"]})
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("e3", "e4"):
        for arch in ("densenet121", "resnet50"):
            (a.out_dir / f"ckpts_{kind}_{arch}.json").write_text(json.dumps(audit_entries(kind, arch), indent=1))
    (a.out_dir / "ckpts_validity.json").write_text(json.dumps(validity_entries(), indent=1))
    print(f"wrote 5 files to {a.out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
