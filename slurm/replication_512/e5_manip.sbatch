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
source "$P/slurm/lib/gpu_occupancy.sh"
PF=5
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13; do gpu_preflight; PF=$?; [ "$PF" -ne 5 ] && break; sleep 10; done
if [ "$PF" -eq 5 ]; then echo "PREFLIGHT still occupied after the bounded wait; holding 600 s before exit 5"; sleep 600; fi
[ "$PF" -eq 0 ] || exit "$PF"
cd "$B/planted_control"   # the conflict set's resolved_path column is relative to the planted-control stage's working directory
export PYTHONPATH="$P/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
export OMP_NUM_THREADS=8
COMMIT="$(cat $P/COMMIT)"
OUT=$P/outputs/${OUTNAME:?OUTNAME}
ARCHS="${ARCHS:-densenet121 resnet50}"
STRENGTHS="${STRENGTHS:-zero,strong}"
ORACLE="${ORACLE:-skip}"
BATCH="${BATCH:-32}"
SMOKE_N="${SMOKE_N:-0}"
IN="$P/slurm/replication_512/inputs"
mkdir -p "$OUT" "$P/manifests"
cat > "$P/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${COMMIT}", "src_py_files_digest": "$(cat $P/SRC_TREE_DIGEST)", "mode": "e5_manip", "size": ${SIZE:?SIZE}, "archs": "${ARCHS}",
 "strengths": "${STRENGTHS}", "oracle": "${ORACLE}", "batch": ${BATCH}, "smoke_n": ${SMOKE_N}, "script_sha256": "$(sha256sum $P/scoring/manipulation_check.py | cut -d' ' -f1)", "analysis_sha256": "$(sha256sum $P/scoring/px512_manipulation_check.py | cut -d' ' -f1)",
 "reserved_ids_sha256": "$(sha256sum "$CXR_RESERVED_IDS_FILE" | cut -d' ' -f1)"}
JSON
RC=0
for arch in $ARCHS; do
  if [ -f "$IN/${MANIFEST:?MANIFEST}_${arch}.json" ]; then MF="$IN/${MANIFEST}_${arch}.json"; else MF="$IN/${MANIFEST}.json"; fi
  EXTRA=""
  [ "$ORACLE" = "skip" ] && EXTRA="$EXTRA --skip-reproduction-oracle"
  [ "$SMOKE_N" != "0" ] && EXTRA="$EXTRA --limit-images $SMOKE_N"
  "$B/venv312_gpu/bin/python3" "$P/scoring/manipulation_check.py" --conflict-set "$B/planted_control/outputs/conflict_set.csv" --nih-root "$B/data/raw/nih_chestxray14/images" \
      --checkpoint-manifest "$MF" --out-dir "$OUT/$arch" --device cuda --image-size "$SIZE" --batch-size "$BATCH" --num-workers 8 --backbone "$arch" --strengths "$STRENGTHS" $EXTRA
  RC=$?
  [ "$RC" -eq 0 ] || break
  VAL=""
  [ -n "${VALIDITY:-}" ] && VAL="--validity $VALIDITY"
  "$B/venv312_gpu/bin/python3" "$P/scoring/px512_manipulation_check.py" analyze --scores-csv "$OUT/$arch/manipulation_check_per_image.csv" --arch "$arch" $VAL --out "$OUT/$arch/manipulation_readout.json"
  RC=$?
  [ "$RC" -eq 0 ] || break
done
gpu_postmortem   # a failure on a card that is still occupied after our process exited becomes exit 5; success and the other codes are never touched
exit "$RC"
