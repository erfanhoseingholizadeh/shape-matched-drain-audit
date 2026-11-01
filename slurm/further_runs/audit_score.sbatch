#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --exclude=@CXR_EXCLUDE@
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/further_runs/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
P=$B/further_runs
Q=$B/positive_audit
EXPECT_COMPOSITE=cccb2fc8d3058f5ae1e5124dbd61294e2142454b343092fe433d275170727e64   # exploratory composite manifest digest (review/positive_audit/run/*/summary_*.json)
source "$P/slurm/lib/gpu_occupancy.sh"
gpu_preflight; PF=$?
[ "$PF" -eq 0 ] || exit "$PF"
cd "$Q"
export PYTHONPATH="$Q/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
COMMIT_Q="$(cat $Q/COMMIT)"
COMMIT_P="$(cat $P/COMMIT)"
export CXR_GIT_SHA="$COMMIT_Q"
export OMP_NUM_THREADS=8
OUTD="$P/outputs/further_readout/$OUT"
mkdir -p "$P/manifests" "$OUTD"
cat > "$P/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "mode": "audit_score", "arch": "${ARCH}", "smoke_n": ${SMOKE_N:-0},
 "scorer_tree_commit": "${COMMIT_Q}", "scorer_tree_src_digest": "$(cat $Q/SRC_TREE_DIGEST)", "submitting_tree_commit": "${COMMIT_P}", "submitting_tree_src_digest": "$(cat $P/SRC_TREE_DIGEST)",
 "scorer_sha256": "$(sha256sum $Q/scoring/positive_audit.py | cut -d' ' -f1)", "checkpoints_json_sha256": "$(sha256sum $P/inputs/${CKPTS}.json | cut -d' ' -f1)",
 "expected_composite_manifest_sha256": "${EXPECT_COMPOSITE}", "reserved_ids_sha256": "$(sha256sum "$CXR_RESERVED_IDS_FILE" | cut -d' ' -f1)"}
JSON
"$B/venv312_gpu/bin/python3" "$Q/scoring/positive_audit.py" score --composite-dir "$Q/composites/real" --checkpoints "$P/inputs/${CKPTS}.json" --arch "$ARCH" \
  --out-dir "$OUTD" --expected-digest "$EXPECT_COMPOSITE" --device cuda --smoke-n "${SMOKE_N:-0}" --commit "$COMMIT_Q"
RC=$?
gpu_postmortem   # a failure on a card that is still occupied after our process exited becomes exit 5; success and the other codes are never touched
exit "$RC"
