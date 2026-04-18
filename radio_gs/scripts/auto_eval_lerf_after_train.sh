#!/usr/bin/env bash

set -euo pipefail

WAIT_TOKEN="${1:-}"
CONFIG_PATH="${2:-}"
OUTPUT_DIR="${3:-}"
EVAL_GPU="${EVAL_GPU:-}"
TEMPS="${TEMPS:-}"
EXTRA_ARGS=()

if [[ -z "$WAIT_TOKEN" || -z "$CONFIG_PATH" || -z "$OUTPUT_DIR" ]]; then
  echo "Usage: bash radio_gs/scripts/auto_eval_lerf_after_train.sh <train_pid|none> <config> <output_dir>" >&2
  exit 1
fi

if [[ -n "$TEMPS" ]]; then
  EXTRA_ARGS+=(--temps "$TEMPS")
fi

timestamp() {
  date '+%F %T'
}

BEST_CKPT="$OUTPUT_DIR/checkpoints/best.pth"
LATEST_CKPT="$OUTPUT_DIR/checkpoints/latest.pth"
ran_eval=0

run_sweep() {
  local checkpoint_path="$1"
  local output_root="$2"

  if [[ ! -f "$checkpoint_path" ]]; then
    echo "[$(timestamp)] Missing checkpoint, skipping LERF eval: $checkpoint_path"
    return 0
  fi

  ran_eval=1
  echo "[$(timestamp)] Starting LERF eval for $checkpoint_path"
  if [[ -n "$EVAL_GPU" ]]; then
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python radio_gs/scripts/auto_eval_lerf_sweep.py \
      --config "$CONFIG_PATH" \
      --checkpoint "$checkpoint_path" \
      --output_root "$output_root" \
      --gpu 0 \
      "${EXTRA_ARGS[@]}"
  else
    python radio_gs/scripts/auto_eval_lerf_sweep.py \
      --config "$CONFIG_PATH" \
      --checkpoint "$checkpoint_path" \
      --output_root "$output_root" \
      --gpu 0 \
      "${EXTRA_ARGS[@]}"
  fi
}

case "$WAIT_TOKEN" in
  none|nowait|0)
    echo "[$(timestamp)] Skipping PID wait; starting LERF evaluation immediately for $CONFIG_PATH"
    ;;
  *)
    echo "[$(timestamp)] Waiting for PID $WAIT_TOKEN before LERF eval: $CONFIG_PATH"
    while ps -p "$WAIT_TOKEN" >/dev/null 2>&1; do
      sleep 120
    done
    echo "[$(timestamp)] Training exited; starting LERF evaluation for $CONFIG_PATH"
    ;;
esac

run_sweep "$BEST_CKPT" "$OUTPUT_DIR/lerf_eval_best"
run_sweep "$LATEST_CKPT" "$OUTPUT_DIR/lerf_eval_latest"

if [[ "$ran_eval" -eq 0 ]]; then
  echo "[$(timestamp)] No LERF checkpoints were available to evaluate for $CONFIG_PATH" >&2
  exit 1
fi

echo "[$(timestamp)] LERF auto-eval complete for $CONFIG_PATH"
