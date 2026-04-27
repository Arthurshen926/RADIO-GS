#!/usr/bin/env bash

set -euo pipefail

WAIT_TOKEN="${1:-}"
EVAL_GPU="${EVAL_GPU:-}"
CONFIG="${CONFIG:-radio_gs/configs/replica_hybrid_v14_room_0_pure_frozen.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-output/radio_gs/room0_hybrid_v14_pure_frozen}"
DEPTH_HEAD="${DEPTH_HEAD:-output/radio_gs/oracle_heads/room_0_seq1_depth_head.pth}"
LOG_DIR="$OUTPUT_DIR/auto_eval"

if [[ -z "$WAIT_TOKEN" ]]; then
  echo "Usage: bash radio_gs/scripts/auto_eval_room0_pure_frozen.sh <trainer_pid|none>" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
ran_eval=0

timestamp() {
  date '+%F %T'
}

run_eval() {
  local checkpoint_path="$1"
  local label="$2"
  local log_path="$LOG_DIR/eval_${label}.log"
  local eval_output_dir="$LOG_DIR/${label}"

  if [[ ! -f "$checkpoint_path" ]]; then
    echo "[$(timestamp)] Missing ${label} checkpoint: $checkpoint_path" | tee "$log_path"
    return 0
  fi

  ran_eval=1
  mkdir -p "$eval_output_dir"
  echo "[$(timestamp)] Evaluating ${label} checkpoint: $checkpoint_path" | tee "$log_path"
  if [[ -n "$EVAL_GPU" ]]; then
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python radio_gs/scripts/eval_rendered.py \
      --config "$CONFIG" \
      --checkpoint "$checkpoint_path" \
      --depth_head_checkpoint "$DEPTH_HEAD" \
      --output_dir "$eval_output_dir" \
      2>&1 | tee -a "$log_path"
  else
    python radio_gs/scripts/eval_rendered.py \
      --config "$CONFIG" \
      --checkpoint "$checkpoint_path" \
      --depth_head_checkpoint "$DEPTH_HEAD" \
      --output_dir "$eval_output_dir" \
      2>&1 | tee -a "$log_path"
  fi
}

case "$WAIT_TOKEN" in
  none|nowait|0)
    echo "[$(timestamp)] Skipping PID wait; starting room0 evaluation immediately"
    ;;
  *)
    echo "[$(timestamp)] Waiting for room0 pure-frozen PID $WAIT_TOKEN to finish"
    while ps -p "$WAIT_TOKEN" >/dev/null 2>&1; do
      sleep 120
    done
    echo "[$(timestamp)] Training exited; starting room0 evaluation"
    ;;
esac

run_eval "$OUTPUT_DIR/checkpoints/best.pth" best
run_eval "$OUTPUT_DIR/checkpoints/latest.pth" latest

if [[ "$ran_eval" -eq 0 ]]; then
  echo "[$(timestamp)] No room0 checkpoints were available to evaluate" >&2
  exit 1
fi

echo "[$(timestamp)] Room0 pure-frozen auto-eval complete"
echo "  best log:   $LOG_DIR/eval_best.log"
echo "  latest log: $LOG_DIR/eval_latest.log"
echo "  best json:  $LOG_DIR/best/eval_rendered_results.json"
echo "  latest json:$LOG_DIR/latest/eval_rendered_results.json"
