#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --job-name=cxr_audit_real
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/planted_intermediate_real/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
cd $B/planted_intermediate_real
export PYTHONPATH="$B/planted_intermediate_real/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
export OMP_NUM_THREADS=16
"$B/venv312_gpu/bin/python3" $B/planted_intermediate_real/scoring/fresh_composite_audit.py score \
  --composite-dir $B/fresh_audit/composites --anchor-json $B/fresh_audit/inputs/anchor.json \
  --checkpoint-manifest $B/planted_intermediate_real/inputs/checkpoint_manifest.json \
  --out-dir $B/planted_intermediate_real/outputs/audit --device cpu
RC=$?
exit "$RC"
