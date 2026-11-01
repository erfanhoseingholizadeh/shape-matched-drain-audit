# Writes the checkpoint lists that the 512-pixel scoring, validity and margin jobs read.
# Run: python3 training/make_px512_checkpoint_lists.py --out-dir DIR

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ARCHS = ("densenet121", "resnet50")
SEEDS = {"orig": (5, 6, 7, 8, 9), "zero": (0, 1, 2), "strong": (0, 1, 2)}
STRATA = ["dev_half", "R_int_C"]

def ckpt_path(arch: str, fam: str, seed: int, size: int = 512) -> str:
    cfg = yaml.safe_load((REPO / "configs" / f"e5_{arch}_{fam}.yaml").read_text())
    return f"{cfg['output_dir']}_seed{seed}_res{size}/best_checkpoint.pt"

def audit_entries(arch: str) -> list:
    return [{"label": f"{arch}_{fam}_seed{s}", "checkpoint": ckpt_path(arch, fam, s), "family": f"e5_{'orig' if fam == 'orig' else 'planted'}_{arch}", "strata": list(STRATA)} for fam in SEEDS for s in SEEDS[fam]]

CTL_SEEDS = (5, 6, 7, 8)

def ctl_ckpt_path(arch: str, seed: int) -> str:
    cfg = yaml.safe_load((REPO / "configs" / f"e5ctl224_{arch}.yaml").read_text())
    return f"{cfg['output_dir']}_seed{seed}_res224/best_checkpoint.pt"

def ctl_audit_entries(arch: str) -> list:
    return [{"label": f"{arch}_ctl224_seed{s}", "checkpoint": ctl_ckpt_path(arch, s), "family": f"e5ctl224_{arch}", "strata": list(STRATA)} for s in CTL_SEEDS]

def ctl_validity_entries() -> list:
    return [{"label": f"{arch}/{e['label']}", "checkpoint": e["checkpoint"], "backbone": arch, "family": e["family"]} for arch in ARCHS for e in ctl_audit_entries(arch)]

def validity_entries() -> list:
    return [{"label": f"{arch}/{e['label']}", "checkpoint": e["checkpoint"], "backbone": arch, "family": e["family"]} for arch in ARCHS for e in audit_entries(arch)]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for arch in ARCHS:
        (a.out_dir / f"e5_ckpts_audit_{arch}.json").write_text(json.dumps(audit_entries(arch), indent=1))
    (a.out_dir / "e5_ckpts_validity.json").write_text(json.dumps(validity_entries(), indent=1))
    for arch in ARCHS:
        (a.out_dir / f"e5ctl_ckpts_audit_{arch}.json").write_text(json.dumps(ctl_audit_entries(arch), indent=1))
    (a.out_dir / "e5ctl_ckpts_validity.json").write_text(json.dumps(ctl_validity_entries(), indent=1))
    print(f"wrote 6 files to {a.out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
