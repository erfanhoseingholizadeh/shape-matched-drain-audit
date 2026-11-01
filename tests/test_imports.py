# Tests that the package and the analysis modules import and that the stage entry points have a command line.
# Run: python3 -m pytest tests/test_imports.py
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULES = [
    "cxr_shortcuts.data.manifests", "cxr_shortcuts.data.gate_guard", "cxr_shortcuts.data.reserved_guard", "cxr_shortcuts.edits.drain.synthetic_tube",
    "cxr_shortcuts.edits.drain.insertion", "cxr_shortcuts.edits.drain.control_edit", "cxr_shortcuts.eval.bootstrap", "cxr_shortcuts.models.backbones",
    "cxr_shortcuts.methods.erm", "cxr_shortcuts.runners.train", "cxr_shortcuts.runners.composite_and_score", "cxr_shortcuts.runners.audit",
    "analysis.heldout_analysis", "analysis.positive_analysis", "analysis.patient_bootstrap", "analysis.population_assignment", "slurm.cxr_settings",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name)


@pytest.mark.parametrize("script", ["scoring/heldout_run.py", "scoring/positive_audit.py", "scoring/artifact_stats.py"])
def test_stage_script_prints_its_help(script):
    r = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "usage" in r.stdout.lower()


def test_training_module_prints_its_help():
    r = subprocess.run([sys.executable, "-m", "cxr_shortcuts.runners.train", "--help"], capture_output=True, text=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}, timeout=60)
    assert r.returncode == 0 and "usage" in r.stdout.lower()
