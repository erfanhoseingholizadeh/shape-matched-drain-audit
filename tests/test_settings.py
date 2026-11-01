# Tests of the settings loader and the Slurm template renderer.
# Run: python3 -m pytest tests/test_settings.py
import pytest

from slurm import cxr_settings
from slurm import render as render_mod


def test_defaults_without_a_file(monkeypatch, tmp_path):
    monkeypatch.setenv("CXR_SETTINGS", str(tmp_path / "absent.toml"))
    monkeypatch.setenv("HOME", "/h")
    v = cxr_settings.load()
    assert v["cluster_root"] == "/h/cxr_shortcuts" and v["nodelist"] == "" and v["exclude"] == "" and v["partition_gpu"] == "gpu"


def test_file_overrides_defaults_and_environment_overrides_the_file(monkeypatch, tmp_path):
    f = tmp_path / "s.toml"
    f.write_text('cluster_root = "/data/cxr"\nnodelist = "n1"\n')
    monkeypatch.setenv("CXR_SETTINGS", str(f))
    assert cxr_settings.get("cluster_root") == "/data/cxr"
    assert cxr_settings.get("nodelist") == "n1"
    monkeypatch.setenv("CXR_NODELIST", "n2")
    assert cxr_settings.get("nodelist") == "n2"
    assert cxr_settings.get("exclude") == ""


def test_unknown_key_in_the_file_is_refused(monkeypatch, tmp_path):
    f = tmp_path / "s.toml"
    f.write_text('cluster = "/x"\n')
    monkeypatch.setenv("CXR_SETTINGS", str(f))
    with pytest.raises(KeyError):
        cxr_settings.load()


def test_the_shipped_settings_file_parses_and_has_only_known_keys(monkeypatch):
    monkeypatch.delenv("CXR_SETTINGS", raising=False)
    for k in ("CXR_CLUSTER_ROOT", "CXR_NODELIST", "CXR_EXCLUDE"):
        monkeypatch.delenv(k, raising=False)
    assert set(cxr_settings.load()) == set(cxr_settings.DEFAULTS)


def test_render_fills_placeholders_and_drops_empty_sbatch_lines():
    text = "#!/bin/bash\n#SBATCH --nodelist=@CXR_NODELIST@\n#SBATCH --exclude=@CXR_EXCLUDE@\nB=@CXR_CLUSTER_ROOT@\ncd $B/stage_a\n"
    values = {"cluster_root": "/r", "nodelist": "", "exclude": "bad"}
    out = render_mod.render(text, values)
    assert "--nodelist" not in out
    assert "#SBATCH --exclude=bad" in out
    assert "B=/r\n" in out and "@CXR_" not in out


def test_render_refuses_an_unknown_placeholder():
    with pytest.raises(KeyError):
        render_mod.render("x=@CXR_SOMETHING_ELSE@", {"cluster_root": "/r"})


def test_every_shipped_template_renders_with_the_shipped_settings(monkeypatch):
    from pathlib import Path
    for k in ("CXR_SETTINGS", "CXR_CLUSTER_ROOT", "CXR_NODELIST", "CXR_EXCLUDE"):
        monkeypatch.delenv(k, raising=False)
    values = cxr_settings.load()
    assert values["partition_gpu"] == "gpu"
    root = Path(__file__).resolve().parent.parent
    files = [p for p in (root / "slurm").rglob("*") if p.is_file() and p.suffix in (".sbatch", ".sh", ".yaml")] + list((root / "configs").glob("*.yaml"))
    assert len(files) >= 90
    for p in files:
        out = render_mod.render(p.read_text(), values)
        assert "@CXR_" not in out, p
