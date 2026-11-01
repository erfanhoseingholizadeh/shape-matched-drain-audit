#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_manip_real
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/planted_intermediate_real/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
cd $B/planted_control   # the conflict set's resolved_path column is relative to the planted-control stage's working directory (as the manipulation-check stage's own sbatch ran)
export PYTHONPATH="$B/planted_intermediate_real/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
export OMP_NUM_THREADS=16
"$B/venv312_gpu/bin/python3" $B/planted_intermediate_real/scoring/manipulation_check.py \
  --conflict-set $B/planted_control/outputs/conflict_set.csv \
  --nih-root $B/planted_control/data/raw/nih_chestxray14/images \
  --checkpoint-manifest $B/planted_intermediate_real/inputs/checkpoint_manifest.json \
  --out-dir $B/planted_intermediate_real/outputs/manip \
  --device cpu --batch-size 32 --num-workers 8 \
  --strengths mid30,mid50
RC=$?
exit "$RC"
