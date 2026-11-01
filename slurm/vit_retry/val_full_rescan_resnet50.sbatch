#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_val_full_rescan_resnet50
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/vit_retry/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/original_checkpoints
export PYTHONPATH="@CXR_CLUSTER_ROOT@/original_checkpoints/src"
VENV=@CXR_CLUSTER_ROOT@/venv312_gpu
export OMP_NUM_THREADS=16

"$VENV/bin/python3" @CXR_CLUSTER_ROOT@/vit_retry/scoring/val_full_rescan_resnet50.py \
  --manifest @CXR_CLUSTER_ROOT@/original_checkpoints/data/nih_pneumothorax_multilabel.csv \
  --nih-root @CXR_CLUSTER_ROOT@/data/raw/nih_chestxray14/images \
  --checkpoint @CXR_CLUSTER_ROOT@/resnet50_vit/outputs/stage0_resnet50_nih_phaseA_seed0_res224/best_checkpoint.pt \
  --checkpoint @CXR_CLUSTER_ROOT@/resnet50_vit/outputs/stage0_resnet50_nih_phaseA_seed1_res224/best_checkpoint.pt \
  --checkpoint @CXR_CLUSTER_ROOT@/resnet50_vit/outputs/stage0_resnet50_nih_phaseA_seed2_res224/best_checkpoint.pt \
  --checkpoint @CXR_CLUSTER_ROOT@/resnet50_vit/outputs/stage0_resnet50_nih_phaseA_seed3_res224/best_checkpoint.pt \
  --checkpoint @CXR_CLUSTER_ROOT@/resnet50_vit/outputs/stage0_resnet50_nih_phaseA_seed4_res224/best_checkpoint.pt \
  --backbone resnet50 \
  --device cuda \
  --output-csv @CXR_CLUSTER_ROOT@/vit_retry/outputs/val_full_per_image_resnet50.csv \
  --output-json @CXR_CLUSTER_ROOT@/vit_retry/outputs/val_full_rescan_resnet50.json
RC=$?
echo "MAIN_COMMAND_EXIT_CODE=$RC"
exit "$RC"
