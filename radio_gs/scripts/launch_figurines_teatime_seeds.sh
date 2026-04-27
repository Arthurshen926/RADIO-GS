#!/bin/bash
# launch_figurines_teatime_seeds.sh
# Launch all 8 figurines+teatime seed runs when GPUs become available.
# Usage: bash radio_gs/scripts/launch_figurines_teatime_seeds.sh
# Assign one GPU per run. Edit GPUS array to match available devices.

set -e

GPUS=(0 1 2 3 4 5 6 7)
CONFIGS=(
  "radio_gs/configs/generated/seeds/lerf_hybrid_v14_figurines_fdh_ws240_240ep_seed7.yaml"
  "radio_gs/configs/generated/seeds/lerf_hybrid_v14_figurines_fdh_ws240_240ep_seed123.yaml"
  "radio_gs/configs/generated/seeds/lerf_hybrid_v14_figurines_nofdh_240ep_seed7.yaml"
  "radio_gs/configs/generated/seeds/lerf_hybrid_v14_figurines_nofdh_240ep_seed123.yaml"
  "radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed7.yaml"
  "radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed123.yaml"
  "radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_nofdh_240ep_seed7.yaml"
  "radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_nofdh_240ep_seed123.yaml"
)

LOG_DIR="output/radio_gs"
mkdir -p "$LOG_DIR"

for i in "${!CONFIGS[@]}"; do
  CFG="${CONFIGS[$i]}"
  GPU="${GPUS[$i]}"
  BASENAME=$(basename "$CFG" .yaml)
  LOG="$LOG_DIR/${BASENAME}.train.log"

  echo "[$(date '+%H:%M:%S')] Launching $BASENAME on GPU $GPU -> $LOG"
  CUDA_VISIBLE_DEVICES=$GPU nohup bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/train_feature_field.py \
    --config "$CFG" > "$LOG" 2>&1 &
  echo "  PID=$!"
  sleep 2
done

echo ""
echo "All 8 runs launched. Monitor with:"
echo "  tail -f output/radio_gs/lerf_hybrid_v14_figurines_fdh_ws240_240ep_seed7.train.log"
echo "  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
