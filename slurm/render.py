# Fills the @CXR_...@ placeholders of a Slurm, shell or YAML template from settings.toml; a #SBATCH line whose setting is empty is dropped.
# Run: python3 slurm/render.py TEMPLATE [--out FILE]
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import cxr_settings  # noqa: E402

TOKEN = re.compile(r"@CXR_([A-Z_]+)@")


def render(text: str, values: dict[str, str]) -> str:
    out = []
    for line in text.split("\n"):
        keys = [k.lower() for k in TOKEN.findall(line)]
        for k in keys:
            if k not in values:
                raise KeyError(f"placeholder @CXR_{k.upper()}@ has no setting")
        if line.lstrip().startswith("#SBATCH") and any(values[k] == "" for k in keys):
            continue
        out.append(TOKEN.sub(lambda m: values[m.group(1).lower()], line))
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="render a template with the values of settings.toml")
    ap.add_argument("template")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    text = render(Path(a.template).read_text(), cxr_settings.load())
    if a.out:
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
