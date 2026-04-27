#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MAX_UTIL="${MAX_UTIL:-40}"
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
LOG_ROOT="${LOG_ROOT:-output/radio_gs/conservative_queue}"
MARKER_ROOT="$LOG_ROOT/markers"

mkdir -p "$LOG_ROOT" "$MARKER_ROOT"

queue_gpu_job() {
  local log_path="$1"
  shift
  nohup env \
    GPU_LIST="$GPU_LIST" \
    MIN_FREE_MIB="$MIN_FREE_MIB" \
    MAX_UTIL="$MAX_UTIL" \
    CHECK_INTERVAL="$CHECK_INTERVAL" \
    bash radio_gs/scripts/wait_and_run.sh "$log_path" "$@" \
    >/dev/null 2>&1 &
  echo $!
}

queue_train_job() {
  local marker_path="$1"
  local config_path="$2"
  local train_log_path="$3"
  nohup env \
    GPU_LIST="$GPU_LIST" \
    MIN_FREE_MIB="$MIN_FREE_MIB" \
    MAX_UTIL="$MAX_UTIL" \
    CHECK_INTERVAL="$CHECK_INTERVAL" \
    bash radio_gs/scripts/run_and_mark_success.sh "$marker_path" \
      bash radio_gs/scripts/wait_and_train.sh "$config_path" "$train_log_path" \
    >/dev/null 2>&1 &
  echo $!
}

queue_file_job() {
  local target_file="$1"
  local log_path="$2"
  shift 2
  nohup env \
    GPU_LIST="$GPU_LIST" \
    MIN_FREE_MIB="$MIN_FREE_MIB" \
    MAX_UTIL="$MAX_UTIL" \
    CHECK_INTERVAL="$CHECK_INTERVAL" \
    bash radio_gs/scripts/wait_for_file_and_run.sh "$target_file" "$log_path" "$@" \
    >/dev/null 2>&1 &
  echo $!
}

SEED_CONFIG_DIR="radio_gs/configs/generated/seeds"
SEEDS="7,123"

bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/generate_seed_configs.py \
  --seeds "$SEEDS" \
  radio_gs/configs/lerf_hybrid_v14_ramen_nofdh_240ep.yaml \
  radio_gs/configs/lerf_hybrid_v14_ramen_fdh_ws240_240ep.yaml \
  radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_nofdh_240ep.yaml \
  radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep.yaml

declare -a TRAIN_CONFIGS=(
  "$SEED_CONFIG_DIR/lerf_hybrid_v14_ramen_nofdh_240ep_seed7.yaml"
  "$SEED_CONFIG_DIR/lerf_hybrid_v14_ramen_nofdh_240ep_seed123.yaml"
  "$SEED_CONFIG_DIR/lerf_hybrid_v14_ramen_fdh_ws240_240ep_seed7.yaml"
  "$SEED_CONFIG_DIR/lerf_hybrid_v14_ramen_fdh_ws240_240ep_seed123.yaml"
  "$SEED_CONFIG_DIR/lerf_hybrid_v14_waldo_kitchen_nofdh_240ep_seed7.yaml"
  "$SEED_CONFIG_DIR/lerf_hybrid_v14_waldo_kitchen_nofdh_240ep_seed123.yaml"
  "$SEED_CONFIG_DIR/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed7.yaml"
  "$SEED_CONFIG_DIR/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed123.yaml"
)

for config in "${TRAIN_CONFIGS[@]}"; do
  stem="$(basename "$config" .yaml)"
  marker="$MARKER_ROOT/${stem}.train.done"
  train_log="$LOG_ROOT/${stem}.train.log"
  queue_train_job \
    "$marker" \
    "$config" \
    "$train_log"
done

for config in "${TRAIN_CONFIGS[@]}"; do
  stem="$(basename "$config" .yaml)"
  train_marker="$MARKER_ROOT/${stem}.train.done"
  eval_marker="$MARKER_ROOT/${stem}.eval.done"
  eval_log="$LOG_ROOT/${stem}.eval.log"
  output_dir="$(bash radio_gs/scripts/run_repo_python.sh -c 'import sys; from radio_gs.config import load_config; cfg=load_config(sys.argv[1]); print(cfg.output_dir)' "$config")"
  queue_file_job \
    "$train_marker" \
    "$eval_log" \
    bash radio_gs/scripts/wait_and_run.sh "$eval_log" \
      bash radio_gs/scripts/run_and_mark_success.sh "$eval_marker" \
        bash radio_gs/scripts/auto_eval_lerf_after_train.sh none "$config" "$output_dir"
done

ROOM0_PROFILE_DIR="output/radio_gs/profiles/room0_pure_frozen_depth_only_autoeval"
room0_profile_pid=$(queue_gpu_job \
  "$LOG_ROOT/room0_profile.log" \
  bash radio_gs/scripts/profile_command.sh \
    --output_dir "$ROOM0_PROFILE_DIR" \
    -- \
    bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/eval_rendered.py \
      --config radio_gs/configs/replica_hybrid_v14_room_0_pure_frozen_depth_only.yaml \
      --checkpoint output/radio_gs/room0_hybrid_v14_pure_frozen_depth_only/checkpoints/latest.pth \
      --depth_head_checkpoint output/radio_gs/oracle_heads/room_0_seq1_depth_head.pth \
      --output_dir output/radio_gs/room0_hybrid_v14_pure_frozen_depth_only/auto_eval_profiled/latest \
      --eval_seed 42)

cat <<EOF
Queued conservative submission jobs.
- Logs root: $LOG_ROOT
- Room0 profile waiter PID: $room0_profile_pid
- Seed configs generated under: $SEED_CONFIG_DIR
EOF
