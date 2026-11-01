# Reads settings.toml (with environment overrides) for the config generators and the Slurm renderer.
# Run: library module, imported by the scripts
from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {"cluster_root": "${HOME}/cxr_shortcuts", "nodelist": "", "exclude": "", "partition_gpu": "gpu", "partition_cpu_small": "cpu", "partition_cpu_mid": "cpu", "partition_cpu_large": "cpu"}


def settings_path() -> Path:
    return Path(os.environ.get("CXR_SETTINGS", ROOT / "settings.toml"))


def load() -> dict[str, str]:
    values = dict(DEFAULTS)
    path = settings_path()
    if path.exists():
        with path.open("rb") as f:
            for key, value in tomllib.load(f).items():
                if key not in DEFAULTS:
                    raise KeyError(f"unknown setting {key!r} in {path}")
                values[key] = str(value)
    for key in DEFAULTS:
        if f"CXR_{key.upper()}" in os.environ:
            values[key] = os.environ[f"CXR_{key.upper()}"]
    return {k: os.path.expandvars(os.path.expanduser(v)) for k, v in values.items()}


def get(key: str) -> str:
    values = load()
    if key not in values:
        raise KeyError(f"unknown setting {key!r}")
    return values[key]
