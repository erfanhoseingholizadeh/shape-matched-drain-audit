#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=@CXR_CLUSTER_ROOT@/replication_512_score/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
P=$B/replication_512_score   # the scoring tree (a second link to this repository, see slurm/link_stage_dirs.sh): the training tree replication_512 keeps the COMMIT and source digest its trainings recorded
source "$P/slurm/lib/gpu_occupancy.sh"
PF=5
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13; do gpu_preflight; PF=$?; [ "$PF" -ne 5 ] && break; sleep 10; done
if [ "$PF" -eq 5 ]; then echo "PREFLIGHT still occupied after the bounded wait; holding 600 s before exit 5"; sleep 600; fi
[ "$PF" -eq 0 ] || exit "$PF"
export PYTHONPATH="$P/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
export OMP_NUM_THREADS=16
COMMIT="$(cat $P/COMMIT)"
SIZE="${SIZE:-512}"; CKPTS="${CKPTS:-e5_ckpts_validity}"   # the 224 micro-batch control (E5 addendum): SIZE=224,CKPTS=e5ctl_ckpts_validity,OUTNAME=e5ctl_validity224
OUT=$P/outputs/${OUTNAME:-e5_validity}
mkdir -p "$OUT" "$P/manifests"
cat > "$P/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${COMMIT}", "src_py_files_digest": "$(cat $P/SRC_TREE_DIGEST)", "mode": "e5_validity",
 "script_sha256": "$(sha256sum $P/scoring/px512_validity.py | cut -d' ' -f1)", "checkpoints_json_sha256": "$(sha256sum $P/slurm/replication_512/inputs/${CKPTS}.json | cut -d' ' -f1)", "image_size": ${SIZE}}
JSON
PY="$B/venv312_gpu/bin/python3"
"$PY" "$P/scoring/px512_validity.py" score --size "$SIZE" --manifest "$B/original_checkpoints/data/nih_pneumothorax_multilabel.csv" --nih-root "$B/data/raw/nih_chestxray14/images" \
    --checkpoints "$P/slurm/replication_512/inputs/${CKPTS}.json" --out-dir "$OUT" --device cuda --batch-size 32 --num-workers 12 --commit "$COMMIT"
RC=$?
if [ "$RC" -eq 0 ]; then
  "$PY" "$P/scoring/px512_validity.py" analyze --scores "$OUT/validity_scores.json" --out "$OUT/validity.json"
  RC=$?
fi
gpu_postmortem
exit "$RC"
