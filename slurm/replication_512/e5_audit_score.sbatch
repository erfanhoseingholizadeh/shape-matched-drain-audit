#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/replication_512_score/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
P=$B/replication_512_score   # the scoring tree (a second link to this repository, see slurm/link_stage_dirs.sh): the training tree replication_512 keeps the COMMIT and source digest its trainings recorded
EXPECT_COMPOSITE=cccb2fc8d3058f5ae1e5124dbd61294e2142454b343092fe433d275170727e64
source "$P/slurm/lib/gpu_occupancy.sh"
PF=5
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13; do gpu_preflight; PF=$?; [ "$PF" -ne 5 ] && break; sleep 10; done
if [ "$PF" -eq 5 ]; then echo "PREFLIGHT still occupied after the bounded wait; holding 600 s before exit 5"; sleep 600; fi
[ "$PF" -eq 0 ] || exit "$PF"
cd "$P"
export PYTHONPATH="$P/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
export OMP_NUM_THREADS=8
COMMIT="$(cat $P/COMMIT)"
OUT=$P/outputs/${OUTNAME:?OUTNAME}
ARCHS="${ARCHS:-densenet121 resnet50}"
mkdir -p "$OUT" "$P/manifests"
cat > "$P/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${COMMIT}", "src_py_files_digest": "$(cat $P/SRC_TREE_DIGEST)", "mode": "e5_audit_score", "size": ${SIZE:?SIZE},
 "archs": "${ARCHS}", "script_sha256": "$(sha256sum $P/scoring/px512_score.py | cut -d' ' -f1)", "frozen_scorer_sha256": "$(sha256sum $P/scoring/positive_audit.py | cut -d' ' -f1)",
 "expected_composite_sha256": "${EXPECT_COMPOSITE}", "reserved_ids_sha256": "$(sha256sum "$CXR_RESERVED_IDS_FILE" | cut -d' ' -f1)"}
JSON
RC=0
for arch in $ARCHS; do
  IN="$P/slurm/replication_512/inputs"
  if [ -f "$IN/${CKPT:?CKPT}_${arch}.json" ]; then CK="$IN/${CKPT}_${arch}.json"; else CK="$IN/${CKPT}.json"; fi   # a per-architecture list (e5_ckpts_audit, e5ctl_ckpts_audit) or a single list (the oracle)
  "$B/venv312_gpu/bin/python3" "$P/scoring/px512_score.py" --size "$SIZE" --arch "$arch" --composite-dir "$B/positive_audit/composites/real" --checkpoints "$CK" --out-dir "$OUT" \
      --expected-digest "$EXPECT_COMPOSITE" --device cuda --commit "$COMMIT"
  RC=$?
  [ "$RC" -eq 0 ] || break
done
gpu_postmortem
exit "$RC"
