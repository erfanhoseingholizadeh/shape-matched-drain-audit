#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_fresh_audit_job1
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
"$VENV/bin/python3" @CXR_CLUSTER_ROOT@/fresh_audit/scoring/fresh_composite_audit.py composite \
  --pool-ids @CXR_CLUSTER_ROOT@/fresh_audit/inputs/evaluation_pool_289_images.ids.txt \
  --split-manifest @CXR_CLUSTER_ROOT@/original_checkpoints/data/nih_pneumothorax_multilabel.csv \
  --rca-csv @CXR_CLUSTER_ROOT@/dev_subset/data/chexmask_dev_subset.csv \
  --nih-root @CXR_CLUSTER_ROOT@/data/raw/nih_chestxray14/images \
  --composite-dir @CXR_CLUSTER_ROOT@/fresh_audit/composites --checkpoint-manifest @CXR_CLUSTER_ROOT@/fresh_audit/inputs/checkpoint_manifest.json \
  --out-dir @CXR_CLUSTER_ROOT@/fresh_audit/outputs/job1 --device cuda
RC=$?
exit "$RC"
