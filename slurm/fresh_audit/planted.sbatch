#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_fresh_audit_job2
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/fresh_audit/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/fresh_audit
export PYTHONPATH="@CXR_CLUSTER_ROOT@/fresh_audit/src"
export CXR_RESERVED_IDS_FILE="@CXR_CLUSTER_ROOT@/fresh_audit/inputs/reserved_image_ids.txt"
VENV=@CXR_CLUSTER_ROOT@/venv312_gpu
export OMP_NUM_THREADS=8
"$VENV/bin/python3" @CXR_CLUSTER_ROOT@/fresh_audit/scoring/fresh_composite_audit.py score \
  --composite-dir @CXR_CLUSTER_ROOT@/fresh_audit/composites --anchor-json @CXR_CLUSTER_ROOT@/fresh_audit/inputs/anchor.json \
  --checkpoint-manifest @CXR_CLUSTER_ROOT@/fresh_audit/inputs/checkpoint_manifest.json \
  --out-dir @CXR_CLUSTER_ROOT@/fresh_audit/outputs/job2 --device cuda
RC=$?
exit "$RC"
