#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash radio_gs/scripts/wait_for_file_and_run.sh <target_file> <log_path> <command...>" >&2
  exit 1
fi

TARGET_FILE="$1"
LOG_PATH="$2"
shift 2
CMD=("$@")

CHECK_INTERVAL="${CHECK_INTERVAL:-120}"

mkdir -p "$(dirname "$LOG_PATH")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] Waiting for file: $TARGET_FILE" | tee -a "$LOG_PATH"
echo "[$(timestamp)] Then running: ${CMD[*]}" | tee -a "$LOG_PATH"

while [[ ! -f "$TARGET_FILE" ]]; do
  echo "[$(timestamp)] File not ready; sleeping ${CHECK_INTERVAL}s" | tee -a "$LOG_PATH"
  sleep "$CHECK_INTERVAL"
done

echo "[$(timestamp)] File ready; launching command" | tee -a "$LOG_PATH"
"${CMD[@]}" 2>&1 | tee -a "$LOG_PATH"
exit ${PIPESTATUS[0]}
