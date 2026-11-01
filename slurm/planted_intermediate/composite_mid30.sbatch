#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_composite_mid30
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/planted_control/logs/%x_%j.out
set -uo pipefail
cd @CXR_CLUSTER_ROOT@/planted_control
export PYTHONPATH="@CXR_CLUSTER_ROOT@/planted_control/src"
VENV=@CXR_CLUSTER_ROOT@/venv312_panel
"$VENV/bin/python3" editing/composite_planted_training_set.py \
  --strength mid30 \
  --manifest-path data/processed/manifests/nih_pneumothorax_multilabel.csv \
  --nih-root data/raw/nih_chestxray14/images \
  --chexmask-csv data/raw/chexmask/chestxray8.csv \
  --output-image-dir data/processed/planted_insertion/mid30 \
  --resolved-manifest-out outputs/resolved_manifest_mid30.csv \
  --out-dir review/planted_control/outputs \
  --n-workers 16
RC=$?
exit "$RC"
