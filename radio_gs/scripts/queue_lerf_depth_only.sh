#!/usr/bin/env bash

set -euo pipefail

SCENES="${1:-ramen,teatime,waldo_kitchen}"
GPU_LIST="${GPU_LIST:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-15000}"
MAX_UTIL="${MAX_UTIL:-50}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
LOG_ROOT="${LOG_ROOT:-output/radio_gs}"

declare -A CONFIGS=(
  [figurines]="radio_gs/configs/lerf_hybrid_v14_figurines_pure_frozen_depth_only.yaml"
  [ramen]="radio_gs/configs/lerf_hybrid_v14_ramen_pure_frozen_depth_only.yaml"
  [teatime]="radio_gs/configs/lerf_hybrid_v14_teatime_pure_frozen_depth_only.yaml"
  [waldo_kitchen]="radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_pure_frozen_depth_only.yaml"
)

IFS=',' read -r -a SCENE_LIST <<< "$SCENES"

for scene in "${SCENE_LIST[@]}"; do
  config="${CONFIGS[$scene]:-}"
  if [[ -z "$config" ]]; then
    echo "Unknown LERF scene: $scene" >&2
    exit 1
  fi

  log_path="$LOG_ROOT/train_${scene}_pure_frozen_depth_only_waiter.log"
  echo "[$(date '+%F %T')] Queueing $scene depth-only pure-frozen run"
  MIN_FREE_MIB="$MIN_FREE_MIB" \
  MAX_UTIL="$MAX_UTIL" \
  CHECK_INTERVAL="$CHECK_INTERVAL" \
  GPU_LIST="$GPU_LIST" \
    bash radio_gs/scripts/wait_and_train.sh "$config" "$log_path"
done
