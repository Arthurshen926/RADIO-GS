#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash radio_gs/scripts/run_and_mark_success.sh <marker_path> <command...>" >&2
  exit 1
fi

MARKER_PATH="$1"
shift

mkdir -p "$(dirname "$MARKER_PATH")"
LOCK_PATH="${MARKER_PATH}.lock"

timestamp() {
  date '+%F %T'
}

if [[ -f "$MARKER_PATH" ]]; then
  echo "[$(timestamp)] Success marker already exists: $MARKER_PATH"
  exit 0
fi

while ! mkdir "$LOCK_PATH" 2>/dev/null; do
  if [[ -f "$MARKER_PATH" ]]; then
    echo "[$(timestamp)] Success marker already exists: $MARKER_PATH"
    exit 0
  fi
  sleep 5
done

trap 'rm -rf "$LOCK_PATH"' EXIT

if [[ -f "$MARKER_PATH" ]]; then
  echo "[$(timestamp)] Success marker already exists: $MARKER_PATH"
  exit 0
fi

rm -f "$MARKER_PATH"

"$@"

touch "$MARKER_PATH"
echo "[$(date '+%F %T')] Wrote success marker: $MARKER_PATH"
