#!/bin/bash

_GPU_OCC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_USED_MIB_MAX="${GPU_USED_MIB_MAX:-$(cat "$_GPU_OCC_DIR/gpu_idle_threshold_mib.txt" 2>/dev/null | tr -d '[:space:]')}"
GPU_POSTMORTEM_SLEEP="${GPU_POSTMORTEM_SLEEP:-3}"

_gpu_query() { # one line per allocated device: index, uuid, memory.used (MiB), memory.total (MiB)
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES"
  else
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader,nounits
  fi
}

_gpu_read() {
  local tag=$1 out rc line idx uuid used total over=0 uuids=" " apps a_uuid a_pid a_used a_name
  case "$GPU_USED_MIB_MAX" in ''|*[!0-9]*) echo "$tag unreadable: no numeric threshold (GPU_USED_MIB_MAX='${GPU_USED_MIB_MAX}'; slurm/lib/gpu_idle_threshold_mib.txt missing or not a number)"; return 6;; esac
  out=$(_gpu_query 2>&1); rc=$?
  if [ "$rc" -ne 0 ] || [ -z "$out" ]; then echo "$tag unreadable node=$(hostname) cvd=${CUDA_VISIBLE_DEVICES:-unset} nvidia_smi_rc=$rc output=$(echo "$out" | head -c 300 | tr '\n' ' ')"; return 6; fi
  while IFS= read -r line; do
    IFS=',' read -r idx uuid used total <<< "$line"
    idx=$(echo "$idx" | tr -d '[:space:]'); uuid=$(echo "$uuid" | tr -d '[:space:]'); used=$(echo "$used" | tr -d '[:space:]'); total=$(echo "$total" | tr -d '[:space:]')
    case "$used" in ''|*[!0-9]*) echo "$tag unreadable node=$(hostname) cvd=${CUDA_VISIBLE_DEVICES:-unset} device=$idx used_mib='$used'"; return 6;; esac
    uuids="$uuids$uuid "
    [ "$used" -gt "$GPU_USED_MIB_MAX" ] && over=1
    echo "$tag node=$(hostname) cvd=${CUDA_VISIBLE_DEVICES:-unset} device=$idx uuid=$uuid used_mib=$used total_mib=$total threshold_mib=$GPU_USED_MIB_MAX"
  done <<< "$out"
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null)
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    IFS=',' read -r a_uuid a_pid a_used a_name <<< "$line"   # gpu_uuid, pid, used_memory, process_name (the name may hold commas; it takes the rest of the line)
    a_uuid=$(echo "$a_uuid" | tr -d '[:space:]')
    case "$uuids" in *" $a_uuid "*) echo "$tag compute_process device_uuid=$a_uuid pid=$(echo "$a_pid" | tr -d '[:space:]') used_mib=$(echo "$a_used" | tr -d '[:space:]') name=$(echo "$a_name" | sed 's/^ *//')";; esac
  done <<< "$apps"
  [ "$over" -eq 0 ] && return 0
  return 5
}

gpu_preflight() {
  _gpu_read PREFLIGHT; local rc=$?
  [ "$rc" -eq 5 ] && echo "PREFLIGHT occupied: the allocated GPU is in use by something else; exiting 5 before any training starts"
  return "$rc"
}

gpu_postmortem() { # reads the global RC; reassigns it to 5 only when RC was a failure of our own and the card is still occupied after our process has exited
  case "${RC:-0}" in 0|4|5|6) return 0;; esac
  sleep "$GPU_POSTMORTEM_SLEEP"
  _gpu_read POSTMORTEM; local rc=$?
  if [ "$rc" -eq 5 ]; then echo "POSTMORTEM occupied: the main command failed (status $RC) and the card is still above the threshold with our process gone; reclassified as exit 5"; RC=5; fi
  return 0
}
