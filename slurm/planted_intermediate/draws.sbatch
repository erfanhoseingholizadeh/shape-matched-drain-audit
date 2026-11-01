#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_draws
#SBATCH --array=0-4
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/planted_intermediate/logs/%x_%A_%a.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
cd $B/planted_intermediate
export PYTHONPATH="$B/planted_intermediate/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
VENV=$B/venv312_gpu   # torch build is the same one the fresh-audit stage used; run on CPU because the GPU partition is saturated
export OMP_NUM_THREADS=16
RC=0
T=${SLURM_ARRAY_TASK_ID:-0}
if [ "$T" = "0" ]; then
"$VENV/bin/python3" $B/planted_intermediate/scoring/fresh_composite_audit.py composite \
  --pool-ids $B/planted_intermediate/inputs/evaluation_pool_289_images.ids.txt \
  --split-manifest $B/original_checkpoints/data/nih_pneumothorax_multilabel.csv \
  --rca-csv $B/dev_subset/data/chexmask_dev_subset.csv \
  --nih-root $B/data/raw/nih_chestxray14/images \
  --composite-dir $B/planted_intermediate/composites/ref6400 --checkpoint-manifest $B/planted_intermediate/inputs/checkpoint_manifest.json \
  --out-dir $B/planted_intermediate/outputs/ref6400 --device cpu --composite-seed 6400 --control-source-seed 6401
r=$?; if [ $r -ne 0 ]; then echo "=== ref6400 FAILED rc=$r"; RC=$r; fi
fi
for k in $((2*T)) $((2*T+1)); do
  echo "=== draw $k composite_seed $((6410+k)) control_source_seed $((6420+k)) $(date -Is)"
  "$VENV/bin/python3" $B/planted_intermediate/scoring/fresh_composite_audit.py composite \
    --pool-ids $B/planted_intermediate/inputs/evaluation_pool_289_images.ids.txt \
    --split-manifest $B/original_checkpoints/data/nih_pneumothorax_multilabel.csv \
    --rca-csv $B/dev_subset/data/chexmask_dev_subset.csv \
    --nih-root $B/data/raw/nih_chestxray14/images \
    --composite-dir $B/planted_intermediate/composites/draw$k --checkpoint-manifest $B/planted_intermediate/inputs/checkpoint_manifest.json \
    --out-dir $B/planted_intermediate/outputs/draw$k --device cpu \
    --composite-seed $((6410+k)) --control-source-seed $((6420+k))
  r=$?
  if [ $r -ne 0 ]; then echo "=== draw $k FAILED rc=$r"; RC=$r; fi
done
exit "$RC"
