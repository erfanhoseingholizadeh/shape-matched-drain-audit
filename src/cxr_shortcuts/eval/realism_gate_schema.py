# Results schema of the realism-panel measurement.
# Run: library module, imported by the scripts

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import pandas as pd

from cxr_shortcuts.eval.schema import config_hash, git_sha

REQUIRED_COLUMNS = (
    "run_id",
    "seed",
    "discriminator",
    "comparison",
    "train_arm",
    "eval_arm",
    "n_patients",
    "n_images",
    "value",
    "ci_low",
    "ci_high",
    "blend_mode",
    "crop_size",
    "config_hash",
    "git_sha",
    "timestamp",
)

VALID_DISCRIMINATORS = frozenset(
    {
        "main",
        "positive_control",
        "panel_realism_shallow",
        "panel_realism_deep",
        "panel_resnet18",
        "panel_frequency",
        "panel_statistic",
    }
)
VALID_COMPARISONS = frozenset({"paired_production", "paired_crude", "unpaired_production"})
VALID_TRAIN_ARMS = frozenset({"primary", "extended"})
VALID_EVAL_ARMS = frozenset({"eval_paired", "eval_unpaired_reference"})

@dataclasses.dataclass(frozen=True)
class RealismGateResultRow:
    run_id: str
    seed: int
    discriminator: str
    comparison: str
    train_arm: str
    eval_arm: str
    n_patients: int
    n_images: int
    value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    blend_mode: str = "log_additive"
    crop_size: int = 128
    config_hash: str = ""
    git_sha: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.discriminator not in VALID_DISCRIMINATORS:
            raise ValueError(f"discriminator must be one of {sorted(VALID_DISCRIMINATORS)}, got {self.discriminator!r}")
        if self.comparison not in VALID_COMPARISONS:
            raise ValueError(f"comparison must be one of {sorted(VALID_COMPARISONS)}, got {self.comparison!r}")
        if self.train_arm not in VALID_TRAIN_ARMS:
            raise ValueError(f"train_arm must be one of {sorted(VALID_TRAIN_ARMS)}, got {self.train_arm!r}")
        if self.eval_arm not in VALID_EVAL_ARMS:
            raise ValueError(f"eval_arm must be one of {sorted(VALID_EVAL_ARMS)}, got {self.eval_arm!r}")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

def append_realism_gate_result(row: RealismGateResultRow, path: Path) -> None:
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

__all__ = [
    "REQUIRED_COLUMNS",
    "VALID_DISCRIMINATORS",
    "VALID_COMPARISONS",
    "VALID_TRAIN_ARMS",
    "VALID_EVAL_ARMS",
    "RealismGateResultRow",
    "append_realism_gate_result",
    "config_hash",
    "git_sha",
]
