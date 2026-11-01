#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_planting_plan
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=@CXR_CLUSTER_ROOT@/planted_control/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/planted_control
export PYTHONPATH="@CXR_CLUSTER_ROOT@/planted_control/src"
VENV=@CXR_CLUSTER_ROOT@/venv312
"$VENV/bin/python3" editing/planting_plan.py
RC=$?
exit "$RC"
