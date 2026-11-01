#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_valfull
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/original_checkpoints/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/original_checkpoints
export PYTHONPATH="@CXR_CLUSTER_ROOT@/original_checkpoints/src"
VENV=@CXR_CLUSTER_ROOT@/venv312_gpu
export OMP_NUM_THREADS=16
mkdir -p @CXR_CLUSTER_ROOT@/rescan/outputs

"$VENV/bin/python3" scoring/val_full_rescan_densenet121.py \
  --manifest @CXR_CLUSTER_ROOT@/original_checkpoints/data/nih_pneumothorax_multilabel.csv \
  --nih-root @CXR_CLUSTER_ROOT@/data/raw/nih_chestxray14/images \
  --checkpoint @CXR_CLUSTER_ROOT@/original_checkpoints/outputs/stage0_densenet121_nih_phaseA_seed0_res224/best_checkpoint.pt \
  --checkpoint @CXR_CLUSTER_ROOT@/original_checkpoints/outputs/stage0_densenet121_nih_phaseA_seed1_res224/best_checkpoint.pt \
  --checkpoint @CXR_CLUSTER_ROOT@/original_checkpoints/outputs/stage0_densenet121_nih_phaseA_seed2_res224/best_checkpoint.pt \
  --checkpoint @CXR_CLUSTER_ROOT@/original_checkpoints/outputs/stage0_densenet121_nih_phaseA_seed3_res224/best_checkpoint.pt \
  --checkpoint @CXR_CLUSTER_ROOT@/original_checkpoints/outputs/stage0_densenet121_nih_phaseA_seed4_res224/best_checkpoint.pt \
  --image-size 224 --batch-size 128 --num-workers 16 --device cuda \
  --output-csv @CXR_CLUSTER_ROOT@/rescan/outputs/val_full_per_image.csv \
  --output-json @CXR_CLUSTER_ROOT@/rescan/outputs/val_full_rescan.json
