#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=@CXR_CLUSTER_ROOT@/further_runs/logs/%x_%j.out
set -uo pipefail
P=@CXR_CLUSTER_ROOT@/further_runs
cd $P
export PYTHONPATH="$P/src"
export CXR_GIT_SHA="$(cat $P/COMMIT)"
echo "commit $CXR_GIT_SHA"
@CXR_CLUSTER_ROOT@/venv312/bin/python3 editing/planting_plan.py
RC=$?
exit "$RC"
