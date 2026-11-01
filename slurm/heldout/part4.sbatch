#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_CPU_LARGE@
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/heldout/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
cd $B/heldout
export PYTHONPATH="$B/heldout/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
COMMIT="$(cat $B/heldout/COMMIT)"
export CXR_GIT_SHA="$COMMIT"
export OMP_NUM_THREADS=16
PY="$B/venv312_gpu/bin/python3"
mkdir -p "$B/heldout/manifests" "$B/heldout/outputs/$OUT"
cat > "$B/heldout/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${COMMIT}", "job": "${JOB}", "smoke_n": ${SMOKE},
 "src_py_files_digest": "$(cat $B/heldout/SRC_TREE_DIGEST)", "inputs_digest": "$(cat $B/heldout/INPUTS_DIGEST)"}
JSON
case "$JOB" in
  manip)
    cd $B/planted_control   # the conflict set's resolved_path column is relative to the planted-control stage's working directory
    LIM=""; [ "$SMOKE" != "0" ] && LIM="--limit-images $SMOKE"
    "$PY" $B/heldout/scoring/manipulation_check.py --conflict-set $B/planted_control/outputs/conflict_set.csv \
      --nih-root $B/data/raw/nih_chestxray14/images --checkpoint-manifest $B/heldout/inputs/resnet50_checkpoint_manifest.json \
      --out-dir "$B/heldout/outputs/$OUT" --device cpu --batch-size 32 --num-workers 8 --backbone resnet50 --strengths strong,mid30,zero $LIM ;;
  audit_planted)
    "$PY" $B/heldout/scoring/fresh_composite_audit.py score --composite-dir $B/fresh_audit/composites --anchor-json $B/fresh_audit/inputs/anchor.json \
      --checkpoint-manifest $B/heldout/inputs/rn_planted_manifest.json --out-dir "$B/heldout/outputs/$OUT" --device cpu --backbone resnet50 --smoke-n "$SMOKE" ;;
  audit_originals)
    "$PY" $B/heldout/scoring/fresh_composite_audit.py score --composite-dir $B/fresh_audit/composites --anchor-json $B/fresh_audit/inputs/anchor.json \
      --checkpoint-manifest $B/heldout/inputs/rn_originals_as_planted_manifest.json --out-dir "$B/heldout/outputs/$OUT" --device cpu --backbone resnet50 --smoke-n "$SMOKE" ;;
  twobytwo)
    "$PY" $B/heldout/scoring/twobytwo_audit.py score --composite-dir $B/fresh_audit/composites --anchor-json $B/fresh_audit/inputs/anchor.json \
      --checkpoints $B/heldout/inputs/ckpts_resnet50_planted_289.json --nih-root $B/data/raw/nih_chestxray14/images \
      --arch resnet50 --out-dir "$B/heldout/outputs/$OUT" --device cpu --commit "$COMMIT" --smoke-n "$SMOKE" ;;
  positives)
    "$PY" $B/heldout/scoring/positive_audit.py score --composite-dir $B/positive_audit/composites/real --checkpoints $B/heldout/inputs/ckpts_resnet50_planted_positives.json \
      --arch resnet50 --out-dir "$B/heldout/outputs/$OUT" --device cpu --smoke-n "$SMOKE" --commit "$COMMIT" ;;
  *) echo "unknown JOB $JOB"; false ;;
esac
RC=$?
exit "$RC"
