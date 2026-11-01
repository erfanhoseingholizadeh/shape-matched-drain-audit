# Writes the ids of the population a job used, with a digest.
# Run: library module, imported by the scripts

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

KINDS = ("patient", "image")

def ids_digest(ids: Iterable) -> str:
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()

def log_population(out_dir, name: str, ids: Iterable, kind: str = "patient") -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"population name must be a plain file stem, got {name!r}")
    unique = sorted({str(i) for i in ids})
    pop_dir = Path(out_dir) / "populations"
    pop_dir.mkdir(parents=True, exist_ok=True)
    (pop_dir / f"{name}.ids.txt").write_text("\n".join(unique) + ("\n" if unique else ""))
    record = {"kind": kind, "n": len(unique), "sha256": ids_digest(unique)}
    index_path = pop_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[name] = record
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return record

def read_population(out_dir, name: str) -> list:
    pop_dir = Path(out_dir) / "populations"
    ids = [x for x in (pop_dir / f"{name}.ids.txt").read_text().split("\n") if x]
    rec = json.loads((pop_dir / "index.json").read_text())[name]
    if len(ids) != rec["n"] or ids_digest(ids) != rec["sha256"]:
        raise ValueError(f"logged population {name!r} does not match its recorded count/digest")
    return ids
