#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/positive_audit/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
cd $B/positive_audit
export PYTHONPATH="$B/positive_audit/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
COMMIT="$(cat $B/positive_audit/COMMIT)"
export CXR_GIT_SHA="$COMMIT"
export OMP_NUM_THREADS=16
mkdir -p "$B/positive_audit/manifests"
cat > "$B/positive_audit/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${COMMIT}", "mode": "score", "arch": "${ARCH}", "smoke_n": ${SMOKE},
 "script_sha256": "$(sha256sum $B/positive_audit/scoring/positive_audit.py | cut -d' ' -f1)", "checkpoints_sha256": "$(sha256sum $B/positive_audit/inputs/${CKPTS}.json | cut -d' ' -f1)",
 "composite_manifest_sha256": "$(python3 -c "import json;print(json.load(open('$B/positive_audit/composites/$COMP/composite_manifest.json'))['manifest_sha256'])")",
 "src_py_files_digest": "$(cat $B/positive_audit/SRC_TREE_DIGEST)"}
JSON
"$B/venv312_gpu/bin/python3" $B/positive_audit/scoring/positive_audit.py score \
  --composite-dir "$B/positive_audit/composites/$COMP" --checkpoints $B/positive_audit/inputs/${CKPTS}.json --arch "$ARCH" \
  --out-dir "$B/positive_audit/outputs/$OUT" --device cpu --smoke-n "$SMOKE" --commit "$COMMIT"
RC=$?
exit "$RC"
