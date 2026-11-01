#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=@CXR_EXCLUDE@
#SBATCH --output=@CXR_CLUSTER_ROOT@/edit_artifacts/logs/%x_%j.out
set -uo pipefail
B0=@CXR_CLUSTER_ROOT@
B=$B0/edit_artifacts
P71=$B0/positive_audit
cd "$B"
export PYTHONPATH="$B/src"
export CXR_RESERVED_IDS_FILE="$B0/planted_intermediate/inputs/reserved_image_ids.txt"
COMMIT="$(cat $B/COMMIT)"
export OMP_NUM_THREADS=4
mkdir -p "$B/manifests" "$B/outputs/$OUT"
cat > "$B/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${COMMIT}", "mode": "artifact_stats", "smoke_n": ${SMOKE},
 "script_sha256": "$(sha256sum $B/scoring/artifact_stats.py | cut -d' ' -f1)", "composite_manifest_file_sha256": "$(sha256sum $P71/composites/real/composite_manifest.json | cut -d' ' -f1)",
 "masks_csv_sha256": "$(sha256sum $P71/inputs/chexmask_targets.csv | cut -d' ' -f1)", "reserved_ids_sha256": "$(sha256sum $CXR_RESERVED_IDS_FILE | cut -d' ' -f1)",
 "src_py_files_digest": "$(cat $B/SRC_TREE_DIGEST)"}
JSON
"$B0/venv312_gpu/bin/python3" $B/scoring/artifact_stats.py stats \
  --composite-dir "$P71/composites/real" --masks-csv "$P71/inputs/chexmask_targets.csv" \
  --expect-masks-sha 92c1a20e3496e154e3ba030471e13ee3a55b48de1559fdda44974dd8bbfd3f6c \
  --out-dir "$B/outputs/$OUT" --smoke-n "$SMOKE" --commit "$COMMIT"
RC=$?
exit "$RC"
