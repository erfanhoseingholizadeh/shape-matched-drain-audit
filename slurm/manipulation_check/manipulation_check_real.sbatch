#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_manip_real
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/planted_control/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/planted_control
export PYTHONPATH="@CXR_CLUSTER_ROOT@/planted_control/src"
VENV=@CXR_CLUSTER_ROOT@/venv312_gpu
export OMP_NUM_THREADS=8

"$VENV/bin/python3" @CXR_CLUSTER_ROOT@/manipulation_check/scoring/manipulation_check.py \
  --conflict-set @CXR_CLUSTER_ROOT@/planted_control/outputs/conflict_set.csv \
  --nih-root @CXR_CLUSTER_ROOT@/planted_control/data/raw/nih_chestxray14/images \
  --checkpoint-manifest @CXR_CLUSTER_ROOT@/manipulation_check/inputs/checkpoint_manifest.json \
  --out-dir @CXR_CLUSTER_ROOT@/manipulation_check/review/manipulation_check/outputs \
  --device cuda --batch-size 128 --num-workers 8
