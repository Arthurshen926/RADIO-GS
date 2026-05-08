#!/usr/bin/env bash

set -euo pipefail

WAIT_TOKEN="${1:-}"
CONFIG_PATH="${2:-}"
OUTPUT_DIR="${3:-}"
EVAL_GPU="${EVAL_GPU:-}"
EVAL_GPU_LIST="${EVAL_GPU_LIST:-}"
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

EVAL_LOCK_DIR="$OUTPUT_DIR/reports/lerf_eval.lock"

acquire_eval_lock() {
  mkdir -p "$OUTPUT_DIR/reports"
  while true; do
    if mkdir "$EVAL_LOCK_DIR" 2>/dev/null; then
      echo "$$" > "$EVAL_LOCK_DIR/pid"
      trap 'rm -rf "$EVAL_LOCK_DIR"' EXIT
      return 0
    fi

    local lock_pid=""
    if [[ -f "$EVAL_LOCK_DIR/pid" ]]; then
      lock_pid="$(cat "$EVAL_LOCK_DIR/pid" 2>/dev/null || true)"
    fi
    if [[ -n "$lock_pid" ]] && ps -p "$lock_pid" >/dev/null 2>&1; then
      echo "[$(timestamp)] LERF eval lock is active for $OUTPUT_DIR (pid=$lock_pid); skipping duplicate eval" >&2
      exit 75
    fi
    rm -rf "$EVAL_LOCK_DIR"
  done
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
  mkdir -p "$output_root"
  if [[ -n "$EVAL_GPU_LIST" ]]; then
    GPU_LIST="$EVAL_GPU_LIST" MIN_FREE_MIB="${MIN_FREE_MIB:-4096}" MAX_UTIL="${MAX_UTIL:-70}" CHECK_INTERVAL="${CHECK_INTERVAL:-120}" \
      bash radio_gs/scripts/wait_and_run.sh "$output_root/eval_wait.log" \
      bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/auto_eval_lerf_sweep.py \
        --config "$CONFIG_PATH" \
        --checkpoint "$checkpoint_path" \
        --output_root "$output_root" \
        --gpu 0 \
        "${EXTRA_ARGS[@]}"
  elif [[ -n "$EVAL_GPU" ]]; then
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/auto_eval_lerf_sweep.py \
        --config "$CONFIG_PATH" \
        --checkpoint "$checkpoint_path" \
        --output_root "$output_root" \
        --gpu 0 \
        "${EXTRA_ARGS[@]}"
  else
    bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/auto_eval_lerf_sweep.py \
      --config "$CONFIG_PATH" \
      --checkpoint "$checkpoint_path" \
      --output_root "$output_root" \
      --gpu 0 \
      "${EXTRA_ARGS[@]}"
  fi
}

recover_partial_summary() {
  local output_root="$1"
  local existing_summary="$output_root/summary.json"

  if [[ -f "$existing_summary" ]]; then
    return 0
  fi

  echo "[$(timestamp)] No summary.json found; attempting recovery from partial sweep outputs under $output_root"
  bash radio_gs/scripts/run_repo_python.sh - "$output_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = []
for temp_dir in sorted(root.glob("T*")):
    p = temp_dir / "lerf_ovs_results.json"
    if not p.exists():
        continue
    try:
        data = json.loads(p.read_text())
        rendered = data.get("rendered") or {}
        loc_acc = rendered.get("loc_acc")
        miou = rendered.get("mean_iou")
        loc_total = rendered.get("loc_total")
        temp = float(temp_dir.name[1:])
        if loc_acc is None or miou is None:
            continue
        results.append(
            {
                "temp": temp,
                "output_dir": str(temp_dir),
                "loc_acc": float(loc_acc),
                "miou": float(miou),
                "loc_total": int(loc_total) if loc_total is not None else None,
            }
        )
    except Exception:
        continue

if not results:
    sys.exit(0)

best = max(results, key=lambda item: (item["loc_acc"], item["miou"]))
summary = {
    "results": results,
    "best": best,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"Recovered {root / 'summary.json'} from {len(results)} temperature runs")
PY
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

acquire_eval_lock
run_sweep "$BEST_CKPT" "$OUTPUT_DIR/lerf_eval_best"
recover_partial_summary "$OUTPUT_DIR/lerf_eval_best"
run_sweep "$LATEST_CKPT" "$OUTPUT_DIR/lerf_eval_latest"
recover_partial_summary "$OUTPUT_DIR/lerf_eval_latest"

if [[ "$ran_eval" -eq 0 ]]; then
  echo "[$(timestamp)] No LERF checkpoints were available to evaluate for $CONFIG_PATH" >&2
  exit 1
fi

echo "[$(timestamp)] LERF auto-eval complete for $CONFIG_PATH"
