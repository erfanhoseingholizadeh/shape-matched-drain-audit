# Applies the memory rule to the probe outputs and writes configs/e5_microbatch.json.
# Run: python3 training/record_px512_microbatch.py <probe output dir> [<probe output dir> ...]    (each dir holds <backbone>_<micro-batch>_<mode>.json files)

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARD_MIB = 40960
LIMIT_MIB = int(0.9 * CARD_MIB)
CANDIDATES = (16, 32, 64)
EMULATION_CAP_MIB = 40000

def choose_micro_batch(cells: list[dict], backbone: str) -> int | None:
    ok = []
    for c in cells:
        if c.get("backbone") != backbone or c.get("mode") != "plain" or c.get("micro_batch") not in CANDIDATES:
            continue
        if c.get("oom") or "peak_reserved_mib_overall" not in c:
            continue
        if c["peak_reserved_mib_overall"] <= LIMIT_MIB:
            ok.append(c["micro_batch"])
    return max(ok) if ok else None

def measurement_is_admissible(cell: dict) -> bool:
    total, cap = cell.get("card_total_mib", 0), cell.get("memory_cap_mib", 0)
    return (total <= 41000) or (0 < cap <= EMULATION_CAP_MIB)

def main() -> int:
    cells = []
    for d in sys.argv[1:]:
        for p in sorted(Path(d).glob("*.json")):
            rec = json.loads(p.read_text().strip().splitlines()[0])
            rec["_file"] = p.name
            cells.append(rec)
    bad = [c["_file"] for c in cells if c.get("mode") == "plain" and not measurement_is_admissible(c)]
    if bad:
        print("refused: these plain cells were not measured on a 40 GB card or with the 40 GB cap:", bad)
        return 2
    chosen = {a: choose_micro_batch(cells, a) for a in ("densenet121", "resnet50")}
    if None in chosen.values():
        print("no registered candidate fits for", [a for a, m in chosen.items() if m is None], "-> E5 is not launched; report this as the first headline")
        return 3
    record = {**chosen, "rule": f"largest of {list(CANDIDATES)} with no out-of-memory and peak reserved at most {LIMIT_MIB} MiB (90% of {CARD_MIB}); registration of 2026-10-14",
              "cells": [{k: v for k, v in c.items() if k != "_file"} for c in cells if c.get("mode") == "plain"],
              "information_only_cells": [{k: v for k, v in c.items() if k != "_file"} for c in cells if c.get("mode") != "plain"]}
    (ROOT / "configs" / "e5_microbatch.json").write_text(json.dumps(record, indent=2) + "\n")
    print("recorded", chosen)
    return 0

if __name__ == "__main__":
    sys.exit(main())
