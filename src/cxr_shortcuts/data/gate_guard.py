# The single guarded entry to the held-out half of the test split: loaders cut its rows, and each read is logged.
# Run: library module, imported by the scripts

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE = "gate"
PROTOCOL_FILE = "docs/confirmatory_protocol.md"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "processed" / "manifests" / "nih_pneumothorax.csv"
DEFAULT_LOG = REPO_ROOT / "review" / "heldout" / "confirmatory" / "gate_reads.jsonl"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_ENTRY_TOKEN = object()

class GateAccessError(RuntimeError):
    pass

def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise GateAccessError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()

def verify_protocol_commit(protocol_commit: str, repo: Path = REPO_ROOT, protocol_file: str = PROTOCOL_FILE) -> dict:
    if not isinstance(protocol_commit, str) or not _HEX40.match(protocol_commit):
        raise GateAccessError("the protocol commit must be a full 40-hex commit hash")
    if _git(repo, "cat-file", "-t", protocol_commit) != "commit":
        raise GateAccessError(f"{protocol_commit} is not a commit")
    r = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", protocol_commit, "HEAD"], capture_output=True)
    if r.returncode != 0:
        raise GateAccessError(f"{protocol_commit} is not an ancestor of HEAD")
    added = _git(repo, "log", "--diff-filter=A", "--format=%H", "--", protocol_file).split()
    if not added or added[-1] != protocol_commit:
        raise GateAccessError(f"{protocol_commit} is not the commit that first added {protocol_file} (that is {added[-1] if added else 'none'})")
    at_commit = _git(repo, "rev-parse", f"{protocol_commit}:{protocol_file}")
    at_head = _git(repo, "rev-parse", f"HEAD:{protocol_file}")
    in_tree = _git(repo, "hash-object", str(repo / protocol_file))
    if not (at_commit == at_head == in_tree):
        raise GateAccessError(f"{protocol_file} has changed since the commit that froze it (commit {at_commit}, HEAD {at_head}, working tree {in_tree})")
    return {"protocol_commit": protocol_commit, "protocol_commit_time": _git(repo, "log", "-1", "--format=%aI", protocol_commit), "protocol_blob": at_commit,
            "head": _git(repo, "rev-parse", "HEAD")}

def _ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode()).hexdigest()

def _gate_sets(manifest_path: Path) -> dict:
    m = pd.read_csv(manifest_path, dtype=str, keep_default_na=False, usecols=["image_id", "patient_id", "gate_subset"])
    g = m[m["gate_subset"] == GATE]
    return {"image": set(g["image_id"]), "patient": set(g["patient_id"])}

def _cut(df: pd.DataFrame, id_col: str, id_kind: str, manifest_path: Path, keep_gate: bool) -> pd.DataFrame:
    sets = _gate_sets(manifest_path)[id_kind]
    ids = df[id_col].astype(str)
    if id_kind == "patient":
        ids = ids.str.replace(r"\.0$", "", regex=True)
    mask = ids.isin(sets)
    return df[mask] if keep_gate else df[~mask]

def load_manifest(path, *, include_gate: bool = False, **read_csv_kwargs) -> pd.DataFrame:
    if include_gate:
        raise GateAccessError("gate-half rows are only available through confirmatory_entry(protocol_commit)")
    df = pd.read_csv(path, **read_csv_kwargs)
    if "gate_subset" not in df.columns:
        raise GateAccessError(f"{path} has no gate_subset column; use load_table(path, id_col=..., id_kind=...) for a table keyed by image or patient id")
    return df[df["gate_subset"].astype(str) != GATE]

def load_table(path, *, id_col: str, id_kind: str = "image", include_gate: bool = False, manifest_path: Path = DEFAULT_MANIFEST, **read_csv_kwargs) -> pd.DataFrame:
    if include_gate:
        raise GateAccessError("gate-half rows are only available through confirmatory_entry(protocol_commit)")
    if id_kind not in ("image", "patient"):
        raise ValueError("id_kind must be 'image' or 'patient'")
    return _cut(pd.read_csv(path, **read_csv_kwargs), id_col, id_kind, manifest_path, keep_gate=False)

def _gate_rows(token, df: pd.DataFrame, id_col: str, id_kind: str, manifest_path: Path) -> pd.DataFrame:
    if token is not _ENTRY_TOKEN:
        raise GateAccessError("a gate read outside confirmatory_entry is refused")
    return _cut(df, id_col, id_kind, manifest_path, keep_gate=True)

class ConfirmatoryEntry:

    def __init__(self, token, facts: dict, manifest_path: Path, log_path: Path, purpose: str):
        if token is not _ENTRY_TOKEN:
            raise GateAccessError("construct this through confirmatory_entry(protocol_commit)")
        self._facts, self._manifest_path, self._log_path, self.purpose = facts, manifest_path, log_path, purpose

    def _log(self, table: str, df: pd.DataFrame, id_col: str) -> None:
        caller = next((f.filename for f in inspect.stack()[1:] if f.filename != __file__), "unknown")
        rec = {"utc": datetime.now(timezone.utc).isoformat(), "protocol_commit": self._facts["protocol_commit"], "protocol_commit_time": self._facts["protocol_commit_time"],
               "head": self._facts["head"], "caller": caller, "purpose": self.purpose, "table": table, "n_rows": int(len(df)), "ids_digest": _ids_digest(df[id_col].astype(str))}
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def gate_manifest(self, path: Optional[Path] = None, **read_csv_kwargs) -> pd.DataFrame:
        path = Path(path) if path else self._manifest_path
        df = pd.read_csv(path, **read_csv_kwargs)
        out = df[df["gate_subset"].astype(str) == GATE]
        self._log(str(path), out, "image_id")
        return out

    def gate_table(self, path, *, id_col: str, id_kind: str = "image", **read_csv_kwargs) -> pd.DataFrame:
        out = _gate_rows(_ENTRY_TOKEN, pd.read_csv(path, **read_csv_kwargs), id_col, id_kind, self._manifest_path)
        self._log(str(path), out, id_col)
        return out

def confirmatory_entry(protocol_commit: str, *, purpose: str, repo: Path = REPO_ROOT, protocol_file: str = PROTOCOL_FILE,
                       manifest_path: Optional[Path] = None, log_path: Optional[Path] = None) -> ConfirmatoryEntry:
    facts = verify_protocol_commit(protocol_commit, Path(repo), protocol_file)
    return ConfirmatoryEntry(_ENTRY_TOKEN, facts, Path(manifest_path) if manifest_path else DEFAULT_MANIFEST,
                             Path(log_path) if log_path else Path(repo) / "review" / "heldout" / "confirmatory" / "gate_reads.jsonl", purpose)
