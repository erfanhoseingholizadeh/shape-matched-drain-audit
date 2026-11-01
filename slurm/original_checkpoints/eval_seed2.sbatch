#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_eval_seed2
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --nodelist=@CXR_NODELIST@
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/original_checkpoints/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/original_checkpoints
export PYTHONPATH="@CXR_CLUSTER_ROOT@/original_checkpoints/src"
VENV=@CXR_CLUSTER_ROOT@/venv312_gpu
export OMP_NUM_THREADS=16

"$VENV/bin/python3" scoring/val_report_eval.py \
  --manifest @CXR_CLUSTER_ROOT@/original_checkpoints/data/nih_pneumothorax_multilabel.csv \
  --nih-root @CXR_CLUSTER_ROOT@/data/raw/nih_chestxray14/images \
  --checkpoint @CXR_CLUSTER_ROOT@/original_checkpoints/outputs/stage0_densenet121_nih_phaseA_seed2_res224/best_checkpoint.pt \
  --image-size 224 --batch-size 128 --num-workers 16 --device cuda \
  --output @CXR_CLUSTER_ROOT@/seed_evals/outputs/val_report_eval_seed2.json
