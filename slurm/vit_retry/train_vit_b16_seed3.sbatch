#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_train_vit_b16_seed3
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/vit_retry/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/original_checkpoints
export PYTHONPATH="@CXR_CLUSTER_ROOT@/original_checkpoints/src"
VENV=@CXR_CLUSTER_ROOT@/venv312_gpu
export OMP_NUM_THREADS=16

echo "=== nvidia-smi before ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv

"$VENV/bin/python3" -m cxr_shortcuts.runners.train \
  --config @CXR_CLUSTER_ROOT@/vit_retry/configs/vit_b16_retry.yaml \
  --seed 3 \
  --image-size 224 \
  --device cuda
RC=$?
echo "=== nvidia-smi after ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
echo "MAIN_COMMAND_EXIT_CODE=$RC"
exit "$RC"
