#!/usr/bin/env bash

set -euo pipefail

GPU="${SELECTED_GPU:-0}"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Usage: bash radio_gs/scripts/profile_command.sh [--gpu N] [--output_dir DIR] -- <command...>" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  stamp="$(date '+%Y%m%d_%H%M%S')"
  OUTPUT_DIR="output/radio_gs/profiles/$stamp"
fi

mkdir -p "$OUTPUT_DIR"
GPU_LOG="$OUTPUT_DIR/gpu_metrics.csv"
CMD_LOG="$OUTPUT_DIR/command.log"
TIME_LOG="$OUTPUT_DIR/time.log"
META_LOG="$OUTPUT_DIR/meta.txt"

printf 'gpu=%s\ncommand=%q ' "$GPU" "$1" > "$META_LOG"
printf '%q ' "$@" >> "$META_LOG"
printf '\nstart=%s\n' "$(date '+%F %T')" >> "$META_LOG"

nvidia-smi -i "$GPU" --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l 1 > "$GPU_LOG" &
SAMPLER_PID=$!
cleanup() {
  kill "$SAMPLER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ -x "/usr/bin/time" ]]; then
  /usr/bin/time -v "$@" 2> "$TIME_LOG" | tee "$CMD_LOG"
else
  TIMEFORMAT=$'real %3R\nuser %3U\nsys %3S'
  { time "$@"; } 2> "$TIME_LOG" | tee "$CMD_LOG"
fi

printf 'end=%s\n' "$(date '+%F %T')" >> "$META_LOG"
cleanup
trap - EXIT

echo "Profile saved to $OUTPUT_DIR"
