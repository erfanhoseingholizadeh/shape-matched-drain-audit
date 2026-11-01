#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/resnet50_planted/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
cd $B/planted_control   # the resolved manifests' resolved_path column is relative to the planted-control stage's working directory, as in the planted-control stage and 65
export PYTHONPATH="$B/resnet50_planted/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
export CXR_GIT_SHA="$(cat $B/resnet50_planted/COMMIT)"
export OMP_NUM_THREADS=16
CFG="$B/resnet50_planted/configs/${CONFIG}.yaml"
mkdir -p "$B/resnet50_planted/manifests"
cat > "$B/resnet50_planted/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${CXR_GIT_SHA}", "src_py_files_digest": "$(cat $B/resnet50_planted/SRC_TREE_DIGEST)",
 "config": "${CFG}", "config_sha256": "$(sha256sum "$CFG" | cut -d' ' -f1)", "seed": ${SEED}, "image_size": 224, "reserved_ids_sha256": "$(sha256sum "$CXR_RESERVED_IDS_FILE" | cut -d' ' -f1)"}
JSON
"$B/venv312_gpu/bin/python3" -m cxr_shortcuts.runners.train --config "$CFG" --seed "$SEED" --image-size 224 --device cuda
RC=$?
exit "$RC"
