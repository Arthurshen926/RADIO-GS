#!/usr/bin/env bash

# Run one command while recording cgroup memory.current and fail closed above
# an explicit byte ceiling.  The guarded command receives its own process
# group; TERM is followed by KILL if it does not exit promptly.

set -euo pipefail

if [[ "${1:-}" != "--" || "$#" -lt 2 ]]; then
  echo "usage: HOST_MEMORY_LOG=<csv> $0 -- command ..." >&2
  exit 2
fi
shift

HOST_MEMORY_CURRENT_PATH="${HOST_MEMORY_CURRENT_PATH:-/sys/fs/cgroup/memory.current}"
HOST_MEMORY_MAX_BYTES="${HOST_MEMORY_MAX_BYTES:-30064771072}"
HOST_MEMORY_POLL_SECONDS="${HOST_MEMORY_POLL_SECONDS:-5}"
HOST_MEMORY_LOG="${HOST_MEMORY_LOG:?HOST_MEMORY_LOG is required}"

if [[ ! "$HOST_MEMORY_MAX_BYTES" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$HOST_MEMORY_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "host memory guard limit and poll interval must be positive integers" >&2
  exit 2
fi
if [[ ! -r "$HOST_MEMORY_CURRENT_PATH" ]]; then
  echo "host memory current path is not readable" >&2
  exit 2
fi

read_current() {
  local value
  value="$(<"$HOST_MEMORY_CURRENT_PATH")" || return 1
  value="${value//[[:space:]]/}"
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  MEMORY_CURRENT="$value"
}

mkdir -p "$(dirname "$HOST_MEMORY_LOG")"
if [[ ! -s "$HOST_MEMORY_LOG" ]]; then
  printf 'timestamp,child_pgid,current_bytes,peak_bytes,limit_bytes,event\n' \
    >>"$HOST_MEMORY_LOG"
fi

if ! read_current; then
  echo "host memory current value is invalid before launch" >&2
  exit 2
fi
peak_bytes="$MEMORY_CURRENT"
if (( MEMORY_CURRENT > HOST_MEMORY_MAX_BYTES )); then
  printf '%s,,%s,%s,%s,prelaunch_memory_above_limit\n' \
    "$(date --iso-8601=seconds)" "$MEMORY_CURRENT" "$peak_bytes" \
    "$HOST_MEMORY_MAX_BYTES" >>"$HOST_MEMORY_LOG"
  exit 89
fi

setsid "$@" &
child_pid=$!
memory_abort=0

terminate_child_group() {
  kill -TERM -- "-$child_pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    if ! kill -0 "$child_pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  kill -KILL -- "-$child_pid" 2>/dev/null || true
}

trap 'terminate_child_group; exit 130' INT TERM

printf '%s,%s,%s,%s,%s,launch\n' \
  "$(date --iso-8601=seconds)" "$child_pid" "$MEMORY_CURRENT" \
  "$peak_bytes" "$HOST_MEMORY_MAX_BYTES" >>"$HOST_MEMORY_LOG"

while kill -0 "$child_pid" 2>/dev/null; do
  if ! read_current; then
    printf '%s,%s,,%s,%s,memory_telemetry_failed\n' \
      "$(date --iso-8601=seconds)" "$child_pid" "$peak_bytes" \
      "$HOST_MEMORY_MAX_BYTES" >>"$HOST_MEMORY_LOG"
    memory_abort=1
    terminate_child_group
    break
  fi
  if (( MEMORY_CURRENT > peak_bytes )); then
    peak_bytes="$MEMORY_CURRENT"
  fi
  event="sample"
  if (( MEMORY_CURRENT > HOST_MEMORY_MAX_BYTES )); then
    event="memory_abort"
    memory_abort=1
  fi
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" "$child_pid" "$MEMORY_CURRENT" \
    "$peak_bytes" "$HOST_MEMORY_MAX_BYTES" "$event" >>"$HOST_MEMORY_LOG"
  if (( memory_abort )); then
    terminate_child_group
    break
  fi
  for _ in $(seq 1 "$HOST_MEMORY_POLL_SECONDS"); do
    kill -0 "$child_pid" 2>/dev/null || break
    sleep 1
  done
done

command_status=0
wait "$child_pid" || command_status=$?
if read_current; then
  if (( MEMORY_CURRENT > peak_bytes )); then
    peak_bytes="$MEMORY_CURRENT"
  fi
  printf '%s,%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" "$child_pid" "$MEMORY_CURRENT" \
    "$peak_bytes" "$HOST_MEMORY_MAX_BYTES" \
    "$([[ "$command_status" == 0 ]] && echo complete || echo child_exit_${command_status})" \
    >>"$HOST_MEMORY_LOG"
fi
if (( memory_abort )); then
  exit 89
fi
exit "$command_status"
