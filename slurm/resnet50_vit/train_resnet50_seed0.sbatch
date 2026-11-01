#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_train_resnet50_seed0
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/resnet50_vit/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/original_checkpoints
export PYTHONPATH="@CXR_CLUSTER_ROOT@/original_checkpoints/src"
VENV=@CXR_CLUSTER_ROOT@/venv312_gpu
export OMP_NUM_THREADS=16

"$VENV/bin/python3" -m cxr_shortcuts.runners.train \
  --config @CXR_CLUSTER_ROOT@/resnet50_vit/configs/resnet50_original.yaml \
  --seed 0 \
  --image-size 224 \
  --device cuda
