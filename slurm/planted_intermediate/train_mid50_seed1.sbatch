#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_train_mid50_seed1
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/planted_control/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/planted_control
export PYTHONPATH="@CXR_CLUSTER_ROOT@/planted_control/src"
VENV=@CXR_CLUSTER_ROOT@/venv312_gpu
export OMP_NUM_THREADS=16
"$VENV/bin/python3" -m cxr_shortcuts.runners.train \
  --config @CXR_CLUSTER_ROOT@/planted_control/configs/densenet121_planted_mid50.yaml \
  --seed 1 --image-size 224 --device cuda
RC=$?
exit "$RC"
