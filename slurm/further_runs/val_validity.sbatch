#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --exclude=@CXR_EXCLUDE@
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/further_runs/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
P=$B/further_runs
source "$P/slurm/lib/gpu_occupancy.sh"
gpu_preflight; PF=$?
[ "$PF" -eq 0 ] || exit "$PF"
cd "$P"
export PYTHONPATH="$P/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
COMMIT="$(cat $P/COMMIT)"
export CXR_GIT_SHA="$COMMIT"
export OMP_NUM_THREADS=8
OUTD="$P/outputs/further_readout/$OUT"
mkdir -p "$P/manifests" "$OUTD"
cat > "$P/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "mode": "further_run_validity", "commit": "${COMMIT}", "src_py_files_digest": "$(cat $P/SRC_TREE_DIGEST)", "smoke_n": ${SMOKE_N:-0},
 "script_sha256": "$(sha256sum $P/scoring/further_run_validity.py | cut -d' ' -f1)", "checkpoints_json_sha256": "$(sha256sum $P/inputs/${CKPTS}.json | cut -d' ' -f1)",
 "reserved_ids_sha256": "$(sha256sum "$CXR_RESERVED_IDS_FILE" | cut -d' ' -f1)"}
JSON
"$B/venv312_gpu/bin/python3" "$P/scoring/further_run_validity.py" score --manifest "$B/original_checkpoints/data/nih_pneumothorax_multilabel.csv" --nih-root "$B/data/raw/nih_chestxray14/images" \
  --checkpoints "$P/inputs/${CKPTS}.json" --out-dir "$OUTD" --device cuda --smoke-n "${SMOKE_N:-0}" --commit "$COMMIT"
RC=$?
gpu_postmortem   # a failure on a card that is still occupied after our process exited becomes exit 5; success and the other codes are never touched
exit "$RC"
