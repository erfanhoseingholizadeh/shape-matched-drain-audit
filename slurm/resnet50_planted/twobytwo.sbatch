#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/resnet50_planted/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
cd $B/resnet50_planted
export PYTHONPATH="$B/resnet50_planted/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
export CXR_GIT_SHA="$COMMIT"
export OMP_NUM_THREADS=16
mkdir -p "$B/resnet50_planted/manifests"
cat > "$B/resnet50_planted/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${COMMIT}", "arch": "${ARCH}", "smoke_n": ${SMOKE},
 "script_sha256": "$(sha256sum $B/resnet50_planted/scoring/twobytwo_audit.py | cut -d' ' -f1)", "checkpoints_sha256": "$(sha256sum $B/resnet50_planted/inputs/ckpts_${ARCH}.json | cut -d' ' -f1)",
 "src_py_files_digest": "$(cat $B/resnet50_planted/SRC_TREE_DIGEST)"}
JSON
"$B/venv312_gpu/bin/python3" $B/resnet50_planted/scoring/twobytwo_audit.py score \
  --composite-dir $B/fresh_audit/composites --anchor-json $B/fresh_audit/inputs/anchor.json \
  --checkpoints $B/resnet50_planted/inputs/ckpts_${ARCH}.json --nih-root $B/data/raw/nih_chestxray14/images \
  --arch "$ARCH" --out-dir "$B/resnet50_planted/outputs/$OUT" --device cpu --commit "$COMMIT" --smoke-n "$SMOKE"
RC=$?
exit "$RC"
