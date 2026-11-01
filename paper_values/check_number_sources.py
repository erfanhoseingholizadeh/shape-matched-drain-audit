# Re-reads each paper number that comes from a recorded output (docs/number_sources.csv, kind verified) and compares it with the value printed in the paper.
# Run: python3 paper_values/check_number_sources.py [--table docs/number_sources.csv] [--root .]
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_json_path(artifact: Path, dotted_path: str):
    cur = json.loads(artifact.read_text())
    for part in dotted_path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def number(s: str) -> float:
    return float(s.strip().strip("*").replace(",", "").replace("−", "-").rstrip("%"))


def check_row(row: dict, root: Path) -> str | None:
    artifact = root / row["recorded_output"]
    if not artifact.exists():
        return f"[{row['id']}] recorded output not found: {row['recorded_output']}"
    try:
        actual = load_json_path(artifact, row["key_in_output"])
    except (KeyError, IndexError, ValueError) as exc:
        return f"[{row['id']}] cannot read {row['key_in_output']} in {row['recorded_output']}: {exc!r}"
    try:
        tol = float(row["tolerance"]) if row["tolerance"] else 1e-6
        if abs(number(str(actual)) - number(row["value_as_printed"])) > tol:
            return f"[{row['id']}] printed {row['value_as_printed']!r}, recorded output says {actual!r}"
    except ValueError:
        if str(actual).strip() != row["value_as_printed"].strip():
            return f"[{row['id']}] printed {row['value_as_printed']!r}, recorded output says {actual!r}"
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="check the paper's recorded numbers against the recorded outputs")
    ap.add_argument("--table", default="docs/number_sources.csv")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    root = Path(a.root)
    with open(root / a.table, newline="") as f:
        rows = list(csv.DictReader(f))
    verified = [r for r in rows if r["kind"] == "verified"]
    problems = [p for p in (check_row(r, root) for r in verified) if p]
    print(f"{len(rows)} numbers in the table, {len(verified)} re-derivable from recorded outputs, {len(rows) - len(verified)} recorded design constants or study-record values")
    for p in problems:
        print("  -", p)
    print("all recorded numbers match" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
