#!/bin/bash
# Submit through slurm/submit.sh (it fills the placeholders from settings.toml); see README.md.
#SBATCH --partition=@CXR_PARTITION_GPU@
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=@CXR_CLUSTER_ROOT@/replication_512_ctl224/logs/%x_%j.out
set -uo pipefail
B=@CXR_CLUSTER_ROOT@
P=$B/replication_512_ctl224
source "$P/slurm/lib/gpu_occupancy.sh"
PF=5
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13; do gpu_preflight; PF=$?; [ "$PF" -ne 5 ] && break; sleep 10; done
if [ "$PF" -eq 5 ]; then echo "PREFLIGHT still occupied after the bounded wait; holding 600 s before exit 5"; sleep 600; fi
[ "$PF" -eq 0 ] || exit "$PF"
cd $B/planted_control   # the planted manifests' resolved_path column is relative to the planted-control stage's working directory, as in the planted-control, intermediate, ResNet-50 and further-run stages; the plain manifest's paths are absolute
export PYTHONPATH="$P/src"
export CXR_RESERVED_IDS_FILE="$B/planted_intermediate/inputs/reserved_image_ids.txt"
export CXR_GIT_SHA="$(cat $P/COMMIT)"
export OMP_NUM_THREADS=16
CFG="$P/configs/${CONFIG}.yaml"
mkdir -p "$P/manifests" "$P/logs"
cat > "$P/manifests/job_${SLURM_JOB_ID}.json" <<JSON
{"job_id": "${SLURM_JOB_ID}", "job_name": "${SLURM_JOB_NAME}", "node": "$(hostname)", "commit": "${CXR_GIT_SHA}", "src_py_files_digest": "$(cat $P/SRC_TREE_DIGEST)", "src_digest_recipe": "$(cat $P/SRC_TREE_DIGEST_RECIPE)",
 "config": "${CFG}", "config_sha256": "$(sha256sum "$CFG" | cut -d' ' -f1)", "seed": ${SEED}, "image_size": 224, "reserved_ids_sha256": "$(sha256sum "$CXR_RESERVED_IDS_FILE" | cut -d' ' -f1)",
 "expect_select_sha": "${EXPECT_SELECT_SHA:-none}", "sbatch_script_sha256": "$(sha256sum $P/slurm/replication_512/train_ctl224.sbatch | cut -d' ' -f1)", "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES:-unset}"}
JSON
"$B/venv312_gpu/bin/python3" -m cxr_shortcuts.runners.train --config "$CFG" --seed "$SEED" --image-size 224 --device cuda
RC=$?
gpu_postmortem
if [ "$RC" -eq 0 ] && [ -n "${EXPECT_SELECT_SHA:-}" ]; then
  OUTD=$("$B/venv312_gpu/bin/python3" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'] + '_seed' + sys.argv[2] + '_res224')" "$CFG" "$SEED")
  GOT=$("$B/venv312_gpu/bin/python3" -c "import json,sys; print(json.load(open(sys.argv[1]))['val_select_patients']['sha256'])" "$OUTD/populations/index.json" 2>/dev/null)
  echo "val_select_patients sha256 expected=${EXPECT_SELECT_SHA} got=${GOT} dir=${OUTD}"
  [ "$GOT" = "$EXPECT_SELECT_SHA" ] || RC=4
fi
exit "$RC"
