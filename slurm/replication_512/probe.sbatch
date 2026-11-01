#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=@CXR_CLUSTER_ROOT@/replication_512/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
P=$B/replication_512
source "$P/slurm/lib/gpu_occupancy.sh"
gpu_preflight; PF=$?
[ "$PF" -eq 0 ] || exit "$PF"
export PYTHONPATH="$P/src"
COMMIT="$(cat $P/COMMIT)"
OUT=$P/outputs/probe_${SLURM_JOB_ID}
mkdir -p "$OUT" "$P/manifests"
cat > "$P/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${COMMIT}", "src_py_files_digest": "$(cat $P/SRC_TREE_DIGEST)", "mode": "e5_memory_probe", "cap_mib": "${CAP_MIB:-none}",
 "script_sha256": "$(sha256sum $P/training/px512_memory_probe.py | cut -d' ' -f1)", "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES:-unset}"}
JSON
FAIL=0
CAPARG=""; [ -n "${CAP_MIB:-}" ] && CAPARG="--memory-cap-mib $CAP_MIB"
for cell in "densenet121 16 plain" "densenet121 32 plain" "densenet121 64 plain" "densenet121 128 plain" "resnet50 16 plain" "resnet50 32 plain" "resnet50 64 plain" "resnet50 128 plain" \
            "densenet121 64 checkpointed" "densenet121 128 checkpointed" "resnet50 64 checkpointed" "resnet50 128 checkpointed"; do
  set -- $cell
  "$B/venv312_gpu/bin/python3" "$P/training/px512_memory_probe.py" --backbone "$1" --micro-batch "$2" --mode "$3" $CAPARG --out "$OUT/${1}_${2}_${3}.json"
  rc=$?
  [ "$rc" -eq 0 ] || { echo "probe cell $cell exited $rc"; FAIL=$rc; }
done
RC=$FAIL
gpu_postmortem
exit "$RC"
