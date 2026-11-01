#!/bin/bash
# Renders a Slurm template with the values of settings.toml and submits it. Run: slurm/submit.sh slurm/heldout/heldout.sbatch [sbatch options]
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
template="$1"
shift
rendered="$(mktemp "${TMPDIR:-/tmp}/cxr_XXXXXX.sbatch")"
python3 "$here/render.py" "$template" --out "$rendered"
sbatch "$@" "$rendered"
RC=$?
rm -f "$rendered"
exit "$RC"
