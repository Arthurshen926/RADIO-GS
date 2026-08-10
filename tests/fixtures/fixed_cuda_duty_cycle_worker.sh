#!/usr/bin/env bash

# Deterministic CPU-only worker for the fixed CUDA duty-cycle integration test.
# SIGSTOP/SIGCONT may delay lines, but must never change their values/order.

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 OUTPUT COUNT STEP_SECONDS STATE_DIR" >&2
  exit 2
fi

output_path="$1"
count="$2"
step_seconds="$3"
state_dir="$4"

mkdir -p "$state_dir"
printf '%s\n' "$$" >"$state_dir/worker.pid"
trap 'printf "continued\n" >>"$state_dir/signals.log"' CONT
trap 'printf "terminated\n" >>"$state_dir/signals.log"; exit 143' TERM INT HUP

: >"$output_path"
for ((index=1; index<=count; index++)); do
  printf '%s\n' "$index" >>"$output_path"
  sleep "$step_seconds"
done

printf 'complete\n' >"$state_dir/complete"
