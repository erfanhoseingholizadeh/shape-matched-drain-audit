#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/heldout/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
cd $B/heldout
export PYTHONPATH="$B/heldout/src"
export CXR_RESERVED_IDS_FILE="${RESERVED_OVERRIDE:-$B/planted_intermediate/inputs/reserved_image_ids.txt}"   # gate composite: the arms-only list (the standing list reserves the gate half and refuses its images)
COMMIT="$(cat $B/heldout/COMMIT)"
export CXR_GIT_SHA="$COMMIT"
export OMP_NUM_THREADS=16
PY="$B/venv312_gpu/bin/python3"
IN="$B/heldout/${HALF}_inputs"
mkdir -p "$B/heldout/manifests"
cat > "$B/heldout/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${COMMIT}", "job": "${JOB}", "half": "${HALF}", "smoke_n": ${SMOKE},
 "script_sha256": "$(sha256sum $B/heldout/scoring/heldout_run.py | cut -d' ' -f1)", "inputs_sha256": "$(sha256sum $IN/inputs.json | cut -d' ' -f1)",
 "reserved_ids_sha256": "$(sha256sum $CXR_RESERVED_IDS_FILE | cut -d' ' -f1)", "prepare_record_sha256": "$(sha256sum $IN/prepare_record.json | cut -d' ' -f1)", "src_py_files_digest": "$(cat $B/heldout/SRC_TREE_DIGEST)", "ckpts": "${CKPTS:-}"}
JSON
case "$JOB" in
  composite)
    EXPECTED="$(python3 -c "import json;print(json.load(open('$IN/prepare_record.json'))['inputs_sha256'])")"
    "$PY" $B/heldout/scoring/heldout_run.py composite --input-json $IN/inputs.json --expected-inputs-sha256 "$EXPECTED" --rca-csv $IN/chexmask_targets.csv \
      --nih-root $B/data/raw/nih_chestxray14/images --composite-dir "$B/heldout/composites/$COMP" --out-dir "$B/heldout/outputs/${HALF}_composite_${COMP}" --smoke-n "$SMOKE" --commit "$COMMIT" ;;
  score)
    "$PY" $B/heldout/scoring/heldout_run.py score --composite-dir "$B/heldout/composites/$COMP" --checkpoints $B/heldout/inputs/gate_ckpts/${CKPTS}.json --arch "$ARCH" \
      --out-dir "$B/heldout/outputs/${HALF}_${OUT}" --device cpu --smoke-n "$SMOKE" --commit "$COMMIT" ;;
  *) echo "unknown JOB $JOB"; false ;;
esac
RC=$?
exit "$RC"
