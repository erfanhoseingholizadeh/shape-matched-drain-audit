# Computes the validity clauses at the 512-pixel input on the reporting half.
# Run: python3 scoring/px512_validity.py score --size 512 --manifest CSV --nih-root DIR --checkpoints JSON --out-dir DIR [--device cuda] [--batch-size 32]

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))
sys.path.insert(0, str(SCRIPTS.parent / "src"))

def force_image_size(size: int):
    import cxr_shortcuts.data.datasets as D

    orig = D.NIHPneumothoraxDataset

    class Sized(orig):
        def __init__(self, *args, **kwargs):
            kwargs["image_size"] = size
            super().__init__(*args, **kwargs)

    D.NIHPneumothoraxDataset = Sized
    return orig

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--size", type=int, required=True)
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--nih-root", type=Path, required=True)
    s.add_argument("--checkpoints", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--num-workers", type=int, default=8)
    s.add_argument("--smoke-n", type=int, default=0)
    s.add_argument("--commit", default="")
    an = sub.add_parser("analyze")
    an.add_argument("--scores", type=Path, required=True)
    an.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from scoring import further_run_validity as V79

    if a.cmd == "analyze":
        return V79.run_analyze(a)
    force_image_size(a.size)
    return V79.run_score(a)

if __name__ == "__main__":
    sys.exit(main())
