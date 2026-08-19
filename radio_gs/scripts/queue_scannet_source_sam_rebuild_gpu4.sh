#!/usr/bin/env bash

set -euo pipefail

LOG=/tmp/scannet_source_sam_rebuild_gpu4_queue.log
while tmux has-session -t spin9_priority_gpu4 2>/dev/null \
  || tmux has-session -t spin9_full9_readout_followup 2>/dev/null; do
  echo "[$(date -Is)] waiting for SPIn GPU4 ownership to end" >> "$LOG"
  sleep 30
done

idle_checks=0
while [[ "$idle_checks" -lt 3 ]]; do
  used=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits)
  if [[ "$used" -lt 512 ]]; then
    idle_checks=$((idle_checks + 1))
  else
    idle_checks=0
  fi
  echo "[$(date -Is)] GPU4 memory=${used}MiB idle_checks=$idle_checks/3" >> "$LOG"
  sleep 10
done

echo "[$(date -Is)] GPU4 released from SPIn; starting ScanNet official-SAM rebuild" >> "$LOG"
PHYSICAL_GPUS=4 \
  bash radio_gs/scripts/run_scannet_source_sam_hierarchy_rebuild_full.sh \
  >> "$LOG" 2>&1
