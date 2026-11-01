# The results-row schema every evaluation appends to.
# Run: library module, imported by the scripts

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = (
    "run_id",
    "method",
    "seed",
    "backbone",
    "dataset",
    "split",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "config_hash",
    "git_sha",
    "timestamp",
)

@dataclasses.dataclass(frozen=True)
class ResultRow:
    run_id: str
    method: str
    seed: int
    backbone: str
    dataset: str
    split: str
    metric: str
    value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    config_hash: str = ""
    git_sha: str = ""
    timestamp: str = ""

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

def append_result(row: ResultRow, path: Path) -> None:
    path = Path(path)
    new_row = pd.DataFrame([row.as_dict()])[list(REQUIRED_COLUMNS)]

    if path.exists():
        existing = _read(path)
        if list(existing.columns) != list(REQUIRED_COLUMNS):
            raise ValueError(
                f"schema mismatch in {path}: expected columns {REQUIRED_COLUMNS}, "
                f"found {tuple(existing.columns)}"
            )
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    path.parent.mkdir(parents=True, exist_ok=True)
    _write(combined, path)

def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

def _write(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

def git_sha() -> str:
    from_env = os.environ.get("CXR_GIT_SHA", "").strip()
    if from_env:
        return from_env
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-commit"

def require_git_sha() -> str:
    sha = git_sha()
    if sha == "no-commit":
        raise RuntimeError(
            "git_sha() found no commit: set CXR_GIT_SHA at submission time "
            "(sbatch --export=ALL,CXR_GIT_SHA=$(git rev-parse HEAD) ...)"
        )
    return sha
