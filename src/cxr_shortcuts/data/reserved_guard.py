# Refuses to read any image or mask of a reserved patient.
# Run: library module, imported by the scripts

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

RESERVED_IDS_ENV = "CXR_RESERVED_IDS_FILE"
DEFAULT_RESERVED_IDS_RELPATH = Path("data/processed/manifests/reserved_image_ids.txt")
_REPO_ROOT = Path(__file__).resolve().parents[3]

_cache: dict = {}

class ReservedImageError(RuntimeError):
    pass

class ReservedGuardUnavailableError(RuntimeError):
    pass

def reserved_ids_path() -> Path:
    env = os.environ.get(RESERVED_IDS_ENV)
    return Path(env) if env else _REPO_ROOT / DEFAULT_RESERVED_IDS_RELPATH

def _reserved_ids() -> frozenset:
    path = reserved_ids_path()
    key = (str(path), path.stat().st_mtime_ns if path.exists() else None)
    if key in _cache:
        return _cache[key]
    if not path.exists():
        raise ReservedGuardUnavailableError(
            f"reserved-id file not found at {path}; refusing to read images unguarded "
            f"(build it with data_prep/build_reserved_image_ids.py or set {RESERVED_IDS_ENV})"
        )
    ids = frozenset(line.strip() for line in path.read_text().split("\n") if line.strip())
    _cache.clear()
    _cache[key] = ids
    return ids

def assert_not_reserved(image_ids: Iterable[str] | str, reader: str) -> None:
    if isinstance(image_ids, str):
        image_ids = [image_ids]
    reserved = _reserved_ids()
    hits = sorted({Path(str(i)).name for i in image_ids} & reserved)
    if hits:
        raise ReservedImageError(
            f"{reader} was asked for {len(hits)} reserved image(s), first: {hits[0]!r}; "
            f"reserved patients' images and masks are not readable"
        )
