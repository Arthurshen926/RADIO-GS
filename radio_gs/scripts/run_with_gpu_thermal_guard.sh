#!/usr/bin/env bash

# Run one command on an explicitly assigned physical GPU while recording
# low-frequency telemetry and failing closed on PCIe loss, excessive
# temperature, or an unexpectedly high board-power limit.

set -euo pipefail

if [[ "${1:-}" != "--" ]]; then
  echo "usage: GPU=<physical-index> GPU_TELEMETRY_LOG=<csv> $0 -- command ..." >&2
  exit 2
fi
shift
if [[ "$#" -eq 0 ]]; then
  echo "thermal guard requires a command" >&2
  exit 2
fi

GPU="${GPU:-1}"
GPU_TELEMETRY_LOG="${GPU_TELEMETRY_LOG:?GPU_TELEMETRY_LOG is required}"
GPU_OWNER_AUDIT_LOG="${GPU_OWNER_AUDIT_LOG:-}"
GPU_OWNER_PID_NAMESPACE_MODE="${GPU_OWNER_PID_NAMESPACE_MODE:-strict}"
GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"
GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-70}"
GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"
GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES="${GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES:-1}"
GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS="${GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS:-1}"
GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-0}"
GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-0}"
GPU_PEER_INDEX="${GPU_PEER_INDEX:-}"
GPU_PEER_PAUSE_TEMP_C="${GPU_PEER_PAUSE_TEMP_C:-0}"
GPU_PEER_RESUME_TEMP_C="${GPU_PEER_RESUME_TEMP_C:-0}"
GPU_PEER_QUIET_SECONDS="${GPU_PEER_QUIET_SECONDS:-0}"
GPU_PEER_MAX_POWER_W="${GPU_PEER_MAX_POWER_W:-0}"
GPU_PEER_MAX_MEMORY_MIB="${GPU_PEER_MAX_MEMORY_MIB:-0}"
GPU_PEER_MAX_UTIL_PCT="${GPU_PEER_MAX_UTIL_PCT:-100}"
GPU_PEER_ACTIVITY_ACTION="${GPU_PEER_ACTIVITY_ACTION:-pause}"
GPU_PEER_INTERRUPT_EXIT_CODE=87

for integer_value in "$GPU" "$GPU_MAX_TEMP_C" \
  "$GPU_START_MAX_TEMP_C" "$GPU_POLL_SECONDS" \
  "$GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES" \
  "$GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS" \
  "$GPU_SOFT_PAUSE_TEMP_C" "$GPU_SOFT_RESUME_TEMP_C" \
  "$GPU_PEER_PAUSE_TEMP_C" "$GPU_PEER_RESUME_TEMP_C" \
  "$GPU_PEER_QUIET_SECONDS" "$GPU_PEER_MAX_MEMORY_MIB" \
  "$GPU_PEER_MAX_UTIL_PCT"; do
  if [[ ! "$integer_value" =~ ^[0-9]+$ ]]; then
    echo "thermal guard integer settings must be non-negative integers" >&2
    exit 2
  fi
done
if (( GPU_START_MAX_TEMP_C >= GPU_MAX_TEMP_C )); then
  echo "GPU_START_MAX_TEMP_C must be lower than GPU_MAX_TEMP_C" >&2
  exit 2
fi
if (( GPU_POLL_SECONDS < 1 )); then
  echo "GPU_POLL_SECONDS must be at least 1 second" >&2
  exit 2
fi
if (( GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES < 1 )); then
  echo "GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES must be at least 1" >&2
  exit 2
fi
if (( GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS < 1 )); then
  echo "GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS must be at least 1" >&2
  exit 2
fi
if (( GPU_SOFT_PAUSE_TEMP_C > 0 )); then
  if (( GPU_SOFT_RESUME_TEMP_C >= GPU_SOFT_PAUSE_TEMP_C \
        || GPU_SOFT_PAUSE_TEMP_C >= GPU_MAX_TEMP_C )); then
    echo "GPU soft resume must be below pause, and pause below maximum" >&2
    exit 2
  fi
elif (( GPU_SOFT_RESUME_TEMP_C != 0 )); then
  echo "GPU_SOFT_RESUME_TEMP_C requires a soft pause threshold" >&2
  exit 2
fi
if [[ ! "$GPU_PEER_MAX_POWER_W" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "GPU_PEER_MAX_POWER_W must be finite and non-negative" >&2
  exit 2
fi
if (( GPU_PEER_MAX_UTIL_PCT > 100 )); then
  echo "GPU_PEER_MAX_UTIL_PCT must be in [0,100]" >&2
  exit 2
fi
if [[ "$GPU_PEER_ACTIVITY_ACTION" != "pause" \
      && "$GPU_PEER_ACTIVITY_ACTION" != "terminate" ]]; then
  echo "GPU_PEER_ACTIVITY_ACTION must be pause or terminate" >&2
  exit 2
fi
if [[ "$GPU_OWNER_PID_NAMESPACE_MODE" != "strict" \
      && "$GPU_OWNER_PID_NAMESPACE_MODE" \
      != "exclusive-singleton-after-clear-v1" ]]; then
  echo "GPU_OWNER_PID_NAMESPACE_MODE must be strict or exclusive-singleton-after-clear-v1" >&2
  exit 2
fi
if [[ -n "$GPU_PEER_INDEX" ]]; then
  if [[ ! "$GPU_PEER_INDEX" =~ ^[0-9]+$ ]] \
    || [[ "$GPU_PEER_INDEX" == "$GPU" ]]; then
    echo "GPU_PEER_INDEX must identify another physical GPU" >&2
    exit 2
  fi
  if (( GPU_PEER_RESUME_TEMP_C >= GPU_PEER_PAUSE_TEMP_C \
        || GPU_PEER_PAUSE_TEMP_C <= 0 )); then
    echo "peer GPU resume temperature must be below a positive pause threshold" >&2
    exit 2
  fi
elif (( GPU_PEER_PAUSE_TEMP_C != 0 || GPU_PEER_RESUME_TEMP_C != 0 \
        || GPU_PEER_MAX_MEMORY_MIB != 0 || GPU_PEER_MAX_UTIL_PCT != 100 )) \
  || [[ ! "$GPU_PEER_MAX_POWER_W" =~ ^0([.]0+)?$ ]]; then
  echo "peer thresholds require GPU_PEER_INDEX" >&2
  exit 2
fi

gpu_identity="$(
  timeout --kill-after=2s 5s nvidia-smi -i "$GPU" \
    --query-gpu=pci.bus_id,uuid --format=csv,noheader,nounits
)" || {
  echo "physical GPU$GPU identity is not queryable before launch" >&2
  exit 2
}
IFS=',' read -r gpu_bus_id gpu_uuid <<<"$gpu_identity"
gpu_bus_id="$(tr -d '[:space:]' <<<"$gpu_bus_id" | sed 's/^00000000:/0000:/')"
gpu_uuid="$(tr -d '[:space:]' <<<"$gpu_uuid")"
gpu_config="/sys/bus/pci/devices/$gpu_bus_id/config"
if [[ -z "$gpu_bus_id" || -z "$gpu_uuid" || "$gpu_uuid" != GPU-* ]]; then
  echo "physical GPU$GPU returned an invalid identity" >&2
  exit 2
fi

check_pcie() {
  local prefix
  prefix="$(od -An -tx1 -N16 "$gpu_config" 2>/dev/null | tr -d ' \n')"
  [[ -n "$prefix" && ! "$prefix" =~ ^f+$ ]]
}

sample_gpu() {
  timeout --kill-after=2s 5s nvidia-smi -i "$GPU" \
    --query-gpu=temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,pstate \
    --format=csv,noheader,nounits
}

parse_sample() {
  local sample="$1"
  IFS=',' read -r sample_temp sample_power sample_limit \
    sample_util sample_memory sample_pstate <<<"$sample"
  SAMPLE_TEMP="${sample_temp// /}"
  SAMPLE_POWER="${sample_power// /}"
  SAMPLE_LIMIT="${sample_limit// /}"
  SAMPLE_UTIL="${sample_util// /}"
  SAMPLE_MEMORY="${sample_memory// /}"
  SAMPLE_PSTATE="${sample_pstate// /}"
  [[ "$SAMPLE_TEMP" =~ ^[0-9]+$ ]] \
    && [[ "$SAMPLE_LIMIT" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

sample_peer() {
  timeout --kill-after=2s 5s nvidia-smi -i "$GPU_PEER_INDEX" \
    --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu,pstate \
    --format=csv,noheader,nounits
}

parse_peer_sample() {
  local sample="$1"
  IFS=',' read -r peer_temp peer_power peer_memory peer_util peer_pstate <<<"$sample"
  PEER_TEMP="${peer_temp// /}"
  PEER_POWER="${peer_power// /}"
  PEER_MEMORY="${peer_memory// /}"
  PEER_UTIL="${peer_util// /}"
  PEER_PSTATE="${peer_pstate// /}"
  [[ "$PEER_TEMP" =~ ^[0-9]+$ ]] \
    && [[ "$PEER_POWER" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && [[ "$PEER_MEMORY" =~ ^[0-9]+$ ]] \
    && [[ "$PEER_UTIL" =~ ^[0-9]+$ ]]
}

refresh_peer() {
  PEER_TEMP=0
  PEER_POWER=0
  PEER_MEMORY=0
  PEER_UTIL=0
  PEER_PSTATE=""
  if [[ -n "$GPU_PEER_INDEX" ]]; then
    local peer_sample
    peer_sample="$(sample_peer)" || return 1
    parse_peer_sample "$peer_sample"
  fi
}

target_compute_owners() {
  local rows
  rows="$(
    timeout --kill-after=2s 5s nvidia-smi -i "$GPU" \
      --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits
  )" || return 1
  awk -F', *' -v uuid="$gpu_uuid" '$1 == uuid {print $2}' <<<"$rows" \
    | paste -sd, -
}

peer_power_exceeds_limit() {
  awk -v actual="$PEER_POWER" -v maximum="$GPU_PEER_MAX_POWER_W" \
    'BEGIN { exit !(maximum > 0 && actual > maximum) }'
}

peer_has_activity() {
  (( PEER_TEMP >= GPU_PEER_PAUSE_TEMP_C )) \
    || peer_power_exceeds_limit \
    || (( GPU_PEER_MAX_MEMORY_MIB > 0 \
          && PEER_MEMORY > GPU_PEER_MAX_MEMORY_MIB )) \
    || (( PEER_UTIL > GPU_PEER_MAX_UTIL_PCT ))
}

peer_is_quiet() {
  (( PEER_TEMP <= GPU_PEER_RESUME_TEMP_C )) \
    && ! peer_power_exceeds_limit \
    && (( GPU_PEER_MAX_MEMORY_MIB == 0 \
          || PEER_MEMORY <= GPU_PEER_MAX_MEMORY_MIB )) \
    && (( PEER_UTIL <= GPU_PEER_MAX_UTIL_PCT ))
}

if ! check_pcie; then
  echo "physical GPU$GPU PCIe configuration space is not responding" >&2
  exit 2
fi
initial_sample="$(sample_gpu)" || {
  echo "physical GPU$GPU is not queryable before launch" >&2
  exit 2
}
if ! parse_sample "$initial_sample"; then
  echo "physical GPU$GPU returned invalid telemetry" >&2
  exit 2
fi
if ! refresh_peer; then
  echo "peer GPU telemetry failed before launch" >&2
  exit 2
fi
if ! awk -v actual="$SAMPLE_LIMIT" -v maximum="$GPU_MAX_POWER_LIMIT_W" \
  'BEGIN { exit !(actual <= maximum) }'; then
  echo "GPU$GPU power limit ${SAMPLE_LIMIT}W exceeds guard ${GPU_MAX_POWER_LIMIT_W}W" >&2
  exit 2
fi

mkdir -p "$(dirname "$GPU_TELEMETRY_LOG")"
if [[ ! -s "$GPU_TELEMETRY_LOG" ]]; then
  printf 'timestamp,gpu,bus_id,temp_c,power_w,power_limit_w,util_pct,memory_mib,pstate,event\n' \
    >>"$GPU_TELEMETRY_LOG"
fi
if [[ -n "$GPU_OWNER_AUDIT_LOG" ]]; then
  mkdir -p "$(dirname "$GPU_OWNER_AUDIT_LOG")"
  if [[ ! -s "$GPU_OWNER_AUDIT_LOG" ]]; then
    printf 'timestamp,gpu_uuid,child_pgid,owner_pids,child_owner_pids,foreign_owner_pids,event\n' \
      >>"$GPU_OWNER_AUDIT_LOG"
  fi
fi

peer_quiet_samples=0
peer_quiet_required=1
if [[ -n "$GPU_PEER_INDEX" ]] && (( GPU_PEER_QUIET_SECONDS > 0 )); then
  peer_quiet_required=$((
    (GPU_PEER_QUIET_SECONDS + GPU_POLL_SECONDS - 1) / GPU_POLL_SECONDS + 1
  ))
fi
while true; do
  launch_ready=1
  if (( SAMPLE_TEMP > GPU_START_MAX_TEMP_C )); then
    launch_ready=0
  fi
  if [[ -n "$GPU_PEER_INDEX" ]]; then
    if peer_is_quiet; then
      peer_quiet_samples=$((peer_quiet_samples + 1))
    else
      peer_quiet_samples=0
    fi
    if (( peer_quiet_samples < peer_quiet_required )); then
      launch_ready=0
    fi
  fi
  if (( launch_ready )); then
    break
  fi
  cooldown_event="cooldown"
  if [[ -n "$GPU_PEER_INDEX" ]] && ! peer_is_quiet; then
    cooldown_event="peer${GPU_PEER_INDEX}_activity_t${PEER_TEMP}_p${PEER_POWER}_m${PEER_MEMORY}_u${PEER_UTIL}"
  elif [[ -n "$GPU_PEER_INDEX" ]]; then
    cooldown_event="peer${GPU_PEER_INDEX}_quiet_${peer_quiet_samples}of${peer_quiet_required}"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
    "$SAMPLE_TEMP" "$SAMPLE_POWER" "$SAMPLE_LIMIT" "$SAMPLE_UTIL" \
    "$SAMPLE_MEMORY" "$SAMPLE_PSTATE" "$cooldown_event" \
    >>"$GPU_TELEMETRY_LOG"
  sleep "$GPU_POLL_SECONDS"
  if ! check_pcie; then
    echo "GPU$GPU lost PCIe response during pre-launch cooldown" >&2
    exit 86
  fi
  initial_sample="$(sample_gpu)" || {
    echo "GPU$GPU telemetry failed during pre-launch cooldown" >&2
    exit 86
  }
  parse_sample "$initial_sample" || {
    echo "GPU$GPU returned invalid cooldown telemetry" >&2
    exit 86
  }
  refresh_peer || {
    echo "peer GPU telemetry failed during pre-launch cooldown" >&2
    exit 86
  }
done

prelaunch_owners="$(target_compute_owners)" || {
  echo "GPU$GPU compute-owner query failed immediately before launch" >&2
  exit 86
}
if [[ -n "$prelaunch_owners" ]]; then
  printf '%s,%s,%s,,,,,,,foreign_compute_owner_prelaunch_%s\n' \
    "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
    "$prelaunch_owners" >>"$GPU_TELEMETRY_LOG"
  echo "GPU$GPU acquired compute owner(s) during cooldown: $prelaunch_owners" >&2
  exit 86
fi
setsid "$@" &
child_pid=$!
thermal_abort=0
peer_activity_interrupt=0
child_paused=0
bound_host_owner_pid=""

resolve_owner_process_group() {
  local owner_pid="$1"
  local status_path local_pid namespace_line process_group
  for status_path in /proc/[0-9]*/status; do
    [[ -r "$status_path" ]] || continue
    local_pid="${status_path#/proc/}"
    local_pid="${local_pid%/status}"
    namespace_line="$(awk '/^NSpid:/ {for (i=2;i<=NF;i++) printf "%s%s", (i==2?"":" "), $i; print ""; exit}' "$status_path")"
    if [[ "$local_pid" != "$owner_pid" \
          && ! " $namespace_line " =~ [[:space:]]$owner_pid[[:space:]] ]]; then
      continue
    fi
    process_group="$(ps -o pgid= -p "$local_pid" 2>/dev/null | tr -d ' ')"
    if [[ "$process_group" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$process_group"
      return 0
    fi
  done
  return 1
}

audit_target_compute_owners() {
  local owners owner_pid owner_pgid
  local -a owner_values child_values foreign_values
  owners="$(target_compute_owners)" || return 1
  owner_values=()
  child_values=()
  foreign_values=()
  if [[ -n "$owners" ]]; then
    IFS=',' read -r -a owner_values <<<"$owners"
  fi
  OWNER_AUDIT_EVENT="runtime_owner_audit"
  for owner_pid in "${owner_values[@]}"; do
    [[ "$owner_pid" =~ ^[0-9]+$ ]] || {
      foreign_values+=("$owner_pid")
      continue
    }
    if owner_pgid="$(resolve_owner_process_group "$owner_pid")"; then
      if [[ "$owner_pgid" == "$child_pid" ]]; then
        child_values+=("$owner_pid")
      else
        foreign_values+=("$owner_pid")
      fi
    elif [[ "$GPU_OWNER_PID_NAMESPACE_MODE" \
            == "exclusive-singleton-after-clear-v1" \
            && "${#owner_values[@]}" -eq 1 \
            && ! -e "/proc/$owner_pid" \
            && ( -z "$bound_host_owner_pid" \
                 || "$bound_host_owner_pid" == "$owner_pid" ) ]]; then
      # NVIDIA reports host-namespace PIDs even when /proc is mounted in the
      # container PID namespace.  The target was atomically owner-free before
      # launch, so bind only the first singleton invisible PID and reject any
      # later PID change or additional owner.  The frozen CUDA child also
      # records this same singleton in its postchecked attestation.
      bound_host_owner_pid="$owner_pid"
      child_values+=("$owner_pid")
      OWNER_AUDIT_EVENT="runtime_owner_audit_host_pid_singleton"
    else
      foreign_values+=("$owner_pid")
    fi
  done
  OWNER_AUDIT_OWNER_PIDS="$(IFS=';'; printf '%s' "${owner_values[*]}")"
  OWNER_AUDIT_CHILD_PIDS="$(IFS=';'; printf '%s' "${child_values[*]}")"
  OWNER_AUDIT_FOREIGN_PIDS="$(IFS=';'; printf '%s' "${foreign_values[*]}")"
}

append_owner_audit() {
  local event="$1"
  [[ -n "$GPU_OWNER_AUDIT_LOG" ]] || return 0
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" "$gpu_uuid" "$child_pid" \
    "$OWNER_AUDIT_OWNER_PIDS" "$OWNER_AUDIT_CHILD_PIDS" \
    "$OWNER_AUDIT_FOREIGN_PIDS" "$event" >>"$GPU_OWNER_AUDIT_LOG"
}

OWNER_AUDIT_OWNER_PIDS=""
OWNER_AUDIT_CHILD_PIDS=""
OWNER_AUDIT_FOREIGN_PIDS=""
OWNER_AUDIT_EVENT="runtime_owner_audit"
append_owner_audit "prelaunch_owner_clear"

process_group_has_live_members() {
  ps -eo pgid=,stat= | awk -v pgid="$child_pid" '
    $1 == pgid && $2 !~ /^Z/ { found = 1 }
    END { exit !found }
  '
}

wait_until_next_poll_or_child_exit() {
  local _
  # A sparse GPU telemetry interval must not impose the same delay between
  # short jobs.  Poll only the local process table once per second and return
  # immediately when the guarded process group exits; nvidia-smi is still
  # queried no more often than GPU_POLL_SECONDS.
  for ((_=0; _<GPU_POLL_SECONDS; _++)); do
    process_group_has_live_members || return 0
    sleep 1
  done
}

pause_child_group() {
  if (( ! child_paused )) && process_group_has_live_members; then
    kill -STOP -- "-$child_pid" 2>/dev/null || return 1
    child_paused=1
  fi
}

resume_child_group() {
  if (( child_paused )); then
    kill -CONT -- "-$child_pid" 2>/dev/null || true
    child_paused=0
  fi
}

terminate_child_group() {
  local _
  if process_group_has_live_members; then
    resume_child_group
    kill -TERM -- "-$child_pid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! process_group_has_live_members; then
        child_paused=0
        return 0
      fi
      sleep 1
    done
    kill -KILL -- "-$child_pid" 2>/dev/null || true
    for _ in {1..5}; do
      if ! process_group_has_live_members; then
        child_paused=0
        return 0
      fi
      sleep 1
    done
    echo "GPU$GPU guarded process group $child_pid survived TERM/KILL" >&2
    return 1
  fi
  child_paused=0
  return 0
}

verify_target_gpu_released() {
  local _ current_uuid owners release_sample
  if ! check_pcie; then
    echo "GPU$GPU lost PCIe response during CUDA-release verification" >&2
    return 1
  fi
  current_uuid="$(
    timeout --kill-after=2s 5s nvidia-smi -i "$GPU" \
      --query-gpu=uuid --format=csv,noheader,nounits \
      | tr -d '[:space:]'
  )" || return 1
  if [[ "$current_uuid" != "$gpu_uuid" ]]; then
    echo "physical GPU$GPU UUID changed during CUDA-release verification" >&2
    return 1
  fi
  owners=""
  for _ in {1..20}; do
    owners="$(target_compute_owners)" || return 1
    if [[ -z "$owners" ]]; then
      release_sample="$(sample_gpu)" || return 1
      parse_sample "$release_sample" || return 1
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
        "$SAMPLE_TEMP" "$SAMPLE_POWER" "$SAMPLE_LIMIT" "$SAMPLE_UTIL" \
        "$SAMPLE_MEMORY" "$SAMPLE_PSTATE" \
        "cuda_release_verified_no_compute_owner" \
        >>"$GPU_TELEMETRY_LOG"
      return 0
    fi
    sleep 1
  done
  echo "GPU$GPU retained compute owner(s) after group termination: $owners" >&2
  return 1
}

trap 'terminate_child_group || true; exit 130' INT TERM

consecutive_telemetry_failures=0
consecutive_overheat_polls=0
while process_group_has_live_members; do
  if ! check_pcie; then
    printf '%s,%s,%s,,,,,,,pcie_unresponsive\n' \
      "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
      >>"$GPU_TELEMETRY_LOG"
    echo "GPU$GPU lost PCIe response; terminating guarded command" >&2
    thermal_abort=1
    terminate_child_group || true
    break
  fi
  current_sample=""
  telemetry_failure_reason=""
  if ! current_sample="$(sample_gpu)"; then
    telemetry_failure_reason="query_failed"
  elif ! parse_sample "$current_sample"; then
    telemetry_failure_reason="invalid_sample"
  fi
  if [[ -n "$telemetry_failure_reason" ]]; then
    consecutive_telemetry_failures=$((consecutive_telemetry_failures + 1))
    if ! check_pcie; then
      printf '%s,%s,%s,,,,,,,pcie_unresponsive_after_telemetry_failure\n' \
        "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
        >>"$GPU_TELEMETRY_LOG"
      echo "GPU$GPU lost PCIe response after telemetry failure; terminating guarded command" >&2
      thermal_abort=1
      terminate_child_group || true
      break
    fi
    if (( consecutive_telemetry_failures \
          < GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES )); then
      printf '%s,%s,%s,,,,,,,telemetry_retry_%s_of_%s\n' \
        "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
        "$consecutive_telemetry_failures" \
        "$GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES" >>"$GPU_TELEMETRY_LOG"
      echo "GPU$GPU telemetry failure ${consecutive_telemetry_failures}/${GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES} (${telemetry_failure_reason}); PCIe still responds, retrying" >&2
      sleep "$GPU_POLL_SECONDS"
      continue
    fi
    printf '%s,%s,%s,,,,,,,telemetry_failed_%s_of_%s\n' \
      "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
      "$consecutive_telemetry_failures" \
      "$GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES" >>"$GPU_TELEMETRY_LOG"
    echo "GPU$GPU telemetry failed ${consecutive_telemetry_failures} consecutive time(s) (${telemetry_failure_reason}); terminating guarded command" >&2
    thermal_abort=1
    terminate_child_group || true
    break
  fi
  consecutive_telemetry_failures=0
  if ! audit_target_compute_owners; then
    printf '%s,%s,%s,,,,,,,owner_audit_failed\n' \
      "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
      >>"$GPU_TELEMETRY_LOG"
    echo "GPU$GPU compute-owner audit failed; terminating guarded command" >&2
    thermal_abort=1
    terminate_child_group || true
    break
  fi
  append_owner_audit "$OWNER_AUDIT_EVENT"
  if [[ -n "$OWNER_AUDIT_FOREIGN_PIDS" ]]; then
    printf '%s,%s,%s,,,,,,,foreign_compute_owner_%s\n' \
      "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
      "$OWNER_AUDIT_FOREIGN_PIDS" >>"$GPU_TELEMETRY_LOG"
    echo "GPU$GPU has owner(s) outside guarded PGID $child_pid: $OWNER_AUDIT_FOREIGN_PIDS" >&2
    thermal_abort=1
    terminate_child_group || true
    break
  fi
  if ! refresh_peer; then
    printf '%s,%s,%s,,,,,,,peer_telemetry_failed\n' \
      "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
      >>"$GPU_TELEMETRY_LOG"
    echo "peer GPU telemetry failed; terminating guarded command" >&2
    thermal_abort=1
    terminate_child_group || true
    break
  fi
  event="sample"
  if (( SAMPLE_TEMP >= GPU_MAX_TEMP_C )); then
    consecutive_overheat_polls=$((consecutive_overheat_polls + 1))
    if (( consecutive_overheat_polls >= GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS )); then
      event="thermal_abort_${consecutive_overheat_polls}_of_${GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS}"
      thermal_abort=1
    else
      event="thermal_warning_${consecutive_overheat_polls}_of_${GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS}"
    fi
  else
    consecutive_overheat_polls=0
  fi
  pause_reason=""
  if (( GPU_SOFT_PAUSE_TEMP_C > 0 \
        && SAMPLE_TEMP >= GPU_SOFT_PAUSE_TEMP_C )); then
    pause_reason="gpu${GPU}_t${SAMPLE_TEMP}"
  fi
  if [[ -n "$GPU_PEER_INDEX" ]] && peer_has_activity; then
    peer_reason="peer${GPU_PEER_INDEX}_activity_t${PEER_TEMP}_p${PEER_POWER}_m${PEER_MEMORY}_u${PEER_UTIL}"
    if [[ "$GPU_PEER_ACTIVITY_ACTION" == "terminate" ]] \
        && (( ! thermal_abort )); then
      peer_activity_interrupt=1
      event="peer_activity_interrupt_release_cuda_${peer_reason}"
    else
      pause_reason="$peer_reason"
    fi
  fi
  if (( ! thermal_abort && ! peer_activity_interrupt )); then
    if (( ! child_paused )) && [[ -n "$pause_reason" ]]; then
      pause_child_group || {
        echo "failed to soft-pause guarded command" >&2
        thermal_abort=1
      }
      event="soft_pause_${pause_reason}"
    elif (( child_paused )); then
      resume_ready=1
      if (( GPU_SOFT_PAUSE_TEMP_C > 0 \
            && SAMPLE_TEMP > GPU_SOFT_RESUME_TEMP_C )); then
        resume_ready=0
      fi
      if [[ -n "$GPU_PEER_INDEX" ]] && ! peer_is_quiet; then
        resume_ready=0
      fi
      if (( resume_ready )); then
        resume_child_group
        event="soft_resume"
      else
        event="soft_cooldown"
      fi
    fi
  fi
  if [[ -n "$GPU_PEER_INDEX" ]]; then
    event="${event}_peer${GPU_PEER_INDEX}_t${PEER_TEMP}_p${PEER_POWER}_m${PEER_MEMORY}_u${PEER_UTIL}"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
    "$SAMPLE_TEMP" "$SAMPLE_POWER" "$SAMPLE_LIMIT" "$SAMPLE_UTIL" \
    "$SAMPLE_MEMORY" "$SAMPLE_PSTATE" "$event" \
    >>"$GPU_TELEMETRY_LOG"
  if (( thermal_abort )); then
    echo "GPU$GPU reached ${SAMPLE_TEMP}C; terminating at ${GPU_MAX_TEMP_C}C guard" >&2
    terminate_child_group || true
    break
  fi
  if (( peer_activity_interrupt )); then
    echo "peer GPU activity requested CUDA-releasing interruption; terminating guarded command" >&2
    if ! terminate_child_group; then
      thermal_abort=1
      peer_activity_interrupt=0
    fi
    break
  fi
  wait_until_next_poll_or_child_exit
done

command_status=0
wait "$child_pid" || command_status=$?
if (( thermal_abort )); then
  exit 86
fi
if (( peer_activity_interrupt )); then
  if ! verify_target_gpu_released; then
    printf '%s,%s,%s,,,,,,,cuda_release_verification_failed\n' \
      "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
      >>"$GPU_TELEMETRY_LOG"
    exit 86
  fi
  exit "$GPU_PEER_INTERRUPT_EXIT_CODE"
fi
if (( command_status == 0 )); then
  if ! verify_target_gpu_released; then
    printf '%s,%s,%s,,,,,,,cuda_release_verification_failed\n' \
      "$(date --iso-8601=seconds)" "$GPU" "$gpu_bus_id" \
      >>"$GPU_TELEMETRY_LOG"
    exit 86
  fi
  OWNER_AUDIT_OWNER_PIDS=""
  OWNER_AUDIT_CHILD_PIDS=""
  OWNER_AUDIT_FOREIGN_PIDS=""
  append_owner_audit "postexit_owner_clear"
fi
exit "$command_status"
