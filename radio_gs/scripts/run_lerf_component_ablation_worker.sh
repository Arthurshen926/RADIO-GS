#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash radio_gs/scripts/run_lerf_component_ablation_worker.sh <gpu_id> <config...>" >&2
  exit 1
fi

GPU_ID="$1"
shift

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_ROOT="${LOG_ROOT:-output/radio_gs/lerf_component_ablation_logs}"
mkdir -p "$LOG_ROOT"

timestamp() {
  date '+%F %T'
}

resolve_output_dir() {
  local config_path="$1"
  bash radio_gs/scripts/run_repo_python.sh -c \
    'import sys; from radio_gs.config import load_config; print(load_config(sys.argv[1]).output_dir)' \
    "$config_path"
}

for config_path in "$@"; do
  stem="$(basename "$config_path" .yaml)"
  output_dir="$(resolve_output_dir "$config_path")"
  train_log="$LOG_ROOT/${stem}.gpu${GPU_ID}.train.log"
  eval_log="$LOG_ROOT/${stem}.gpu${GPU_ID}.eval.log"
  final_report="$output_dir/reports/experiment_report.json"

  echo "[$(timestamp)] [GPU ${GPU_ID}] Config: $config_path"
  echo "[$(timestamp)] [GPU ${GPU_ID}] Output: $output_dir"

  if [[ -f "$final_report" ]] && grep -q '"final": true' "$final_report"; then
    echo "[$(timestamp)] [GPU ${GPU_ID}] Training already complete; skipping train" | tee -a "$train_log"
  else
    echo "[$(timestamp)] [GPU ${GPU_ID}] Starting train -> $train_log"
    CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
      bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/train_feature_field.py \
        --config "$config_path" 2>&1 | tee -a "$train_log"
  fi

  echo "[$(timestamp)] [GPU ${GPU_ID}] Starting eval -> $eval_log"
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    bash radio_gs/scripts/auto_eval_lerf_after_train.sh none "$config_path" "$output_dir" \
      2>&1 | tee -a "$eval_log"
done
