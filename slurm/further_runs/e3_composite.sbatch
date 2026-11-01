#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/further_runs/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
P=$B/further_runs
cd $B/planted_control
export PYTHONPATH="$P/src"
export CXR_GIT_SHA="$(cat $P/COMMIT)"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
echo "commit $CXR_GIT_SHA strength $STRENGTH"
"$B/venv312_panel/bin/python3" $P/editing/composite_planted_training_set.py \
  --strength "$STRENGTH" \
  --manifest-path data/processed/manifests/nih_pneumothorax_multilabel.csv \
  --nih-root data/raw/nih_chestxray14/images \
  --chexmask-csv data/raw/chexmask/chestxray8.csv \
  --output-image-dir data/processed/planted_insertion/$STRENGTH \
  --resolved-manifest-out outputs/resolved_manifest_$STRENGTH.csv \
  --out-dir $P/review/planted_control/outputs \
  --n-workers 16
RC=$?
exit "$RC"
