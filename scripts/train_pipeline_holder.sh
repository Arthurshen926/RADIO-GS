
# scripts/train_pipeline_holder.sh start --gpus 2,3
# scripts/train_pipeline_holder.sh stop --gpus 2,3
# scripts/train_pipeline_holder.sh status --gpus 2,3

# # 更低功耗
# scripts/train_pipeline_holder.sh restart --gpus 2,3 --sleep-ms 90

# # 更高利用率/更高功耗
# scripts/train_pipeline_holder.sh restart --gpus 2,3 --sleep-ms 50
# scripts/train_pipeline_holder.sh run -- <你的主线实验命令>

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT_DIR/output/gpu_placeholder"
VISIBLE_GPUS="${GPU_PLACEHOLDER_PHYSICAL_GPUS:-2,3}"
WORKER_VISIBLE_GPUS="${GPU_PLACEHOLDER_VISIBLE_GPUS:-}"
MEMORY_FRACTION="${GPU_PLACEHOLDER_MEMORY_FRACTION:-0.80}"
MATRIX_SIZE="${GPU_PLACEHOLDER_MATRIX_SIZE:-8192}"
CHUNK_MIB="${GPU_PLACEHOLDER_CHUNK_MIB:-512}"
HEARTBEAT_SEC="${GPU_PLACEHOLDER_HEARTBEAT_SEC:-60}"
SAFETY_FREE_MIB="${GPU_PLACEHOLDER_SAFETY_FREE_MIB:-3072}"
MAX_EXISTING_MEM_MIB="${GPU_PLACEHOLDER_MAX_EXISTING_MEM_MIB:-1024}"
SYNC_EVERY="${GPU_PLACEHOLDER_SYNC_EVERY:-16}"
SLEEP_MS="${GPU_PLACEHOLDER_SLEEP_MS:-70}"
PID_FILE=""
LOG_FILE=""

usage() {
  cat >&2 <<'USAGE'
usage: scripts/gpu_placeholder.sh {start|stop|restart|status|run -- <command...>} [options]

Options:
  --gpus IDS              Physical GPU ids, e.g. 2,3 or 0
  --visible-gpus IDS      Visible ids inside CUDA_VISIBLE_DEVICES; default 0,1
  --memory-fraction X     Fraction of free memory to reserve; default 0.80
  --matrix-size N         GEMM matrix size; default 8192
  --sleep-ms X            Sleep after each sync window; default 70 for ~300W on 4090
  --sync-every N          GEMM iterations per sync/sleep window; default 16
  --safety-free-mib N     Minimum free MiB left per GPU; default 3072
  --max-existing-mem-mib N
                          Skip start if target GPU already uses more than N MiB; default 1024
  --heartbeat-sec X       Worker log heartbeat interval; default 60
  --chunk-mib N           Memory reservation chunk size; default 512
USAGE
}

configure_state_paths() {
  local tag
  tag="${VISIBLE_GPUS//,/_}"
  tag="${tag// /}"
  PID_FILE="$STATE_DIR/gpu${tag}.pid"
  LOG_FILE="$STATE_DIR/gpu${tag}.log"
}

default_worker_visible_gpus() {
  local clean count ids i
  clean="${VISIBLE_GPUS// /}"
  if [[ -z "$clean" ]]; then
    echo "0"
    return 0
  fi
  IFS=',' read -r -a ids <<< "$clean"
  count="${#ids[@]}"
  local out=()
  for ((i = 0; i < count; i++)); do
    out+=("$i")
  done
  local joined
  joined="$(IFS=,; echo "${out[*]}")"
  echo "$joined"
}

normalise_worker_visible_gpus() {
  local physical_clean count max_id id
  physical_clean="${VISIBLE_GPUS// /}"
  IFS=',' read -r -a _physical_ids <<< "$physical_clean"
  count="${#_physical_ids[@]}"
  if [[ -z "${WORKER_VISIBLE_GPUS// /}" ]]; then
    WORKER_VISIBLE_GPUS="$(default_worker_visible_gpus)"
    return 0
  fi
  max_id=$((count - 1))
  IFS=',' read -r -a _worker_ids <<< "${WORKER_VISIBLE_GPUS// /}"
  for id in "${_worker_ids[@]}"; do
    if [[ -z "$id" ]]; then
      continue
    fi
    if (( id < 0 || id > max_id )); then
      echo "warning: --visible-gpus $WORKER_VISIBLE_GPUS is invalid for CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS; using $(default_worker_visible_gpus)" >&2
      WORKER_VISIBLE_GPUS="$(default_worker_visible_gpus)"
      return 0
    fi
  done
}

parse_options() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpus)
        VISIBLE_GPUS="$2"
        shift 2
        ;;
      --visible-gpus)
        WORKER_VISIBLE_GPUS="$2"
        shift 2
        ;;
      --memory-fraction)
        MEMORY_FRACTION="$2"
        shift 2
        ;;
      --matrix-size)
        MATRIX_SIZE="$2"
        shift 2
        ;;
      --sleep-ms)
        SLEEP_MS="$2"
        shift 2
        ;;
      --sync-every)
        SYNC_EVERY="$2"
        shift 2
        ;;
      --safety-free-mib)
        SAFETY_FREE_MIB="$2"
        shift 2
        ;;
      --max-existing-mem-mib)
        MAX_EXISTING_MEM_MIB="$2"
        shift 2
        ;;
      --heartbeat-sec)
        HEARTBEAT_SEC="$2"
        shift 2
        ;;
      --chunk-mib)
        CHUNK_MIB="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "unknown option: $1" >&2
        usage
        exit 2
        ;;
    esac
  done
  normalise_worker_visible_gpus
  configure_state_paths
}

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

orphan_placeholder_pids() {
  local pid env_gpus physical_env
  for pid in $(pgrep -f "radio_gs/scripts/gpu_placeholder_worker.py" 2>/dev/null || true); do
    env_gpus="$(
      tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null \
        | awk -F= '$1 == "CUDA_VISIBLE_DEVICES" {print $2; exit}'
    )"
    physical_env="$(
      tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null \
        | awk -F= '$1 == "GPU_PLACEHOLDER_PHYSICAL_GPUS" {print $2; exit}'
    )"
    if [[ "$env_gpus" == "$VISIBLE_GPUS" || "$physical_env" == "$VISIBLE_GPUS" ]]; then
      echo "$pid"
    fi
  done
}

stop_placeholder_pids() {
  local pid pgid
  for pid in "$@"; do
    [[ -n "$pid" ]] || continue
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "$pgid" ]]; then
      kill -TERM -- "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}

active_non_placeholder_gpu_processes() {
  local raw pid cmd
  while IFS= read -r raw; do
    pid="${raw%%,*}"
    pid="${pid// /}"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmd" != *"gpu_placeholder_worker.py"* ]]; then
      echo "$pid ${cmd:-unknown}"
    fi
  done < <(
    nvidia-smi -i "$VISIBLE_GPUS" --query-compute-apps=pid,process_name \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
}

existing_gpu_memory_exceeds_limit() {
  local used
  while IFS= read -r used; do
    used="${used// /}"
    [[ "$used" =~ ^[0-9]+$ ]] || continue
    if (( used > MAX_EXISTING_MEM_MIB )); then
      return 0
    fi
  done < <(
    nvidia-smi -i "$VISIBLE_GPUS" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null || true
  )
  return 1
}

start_placeholder() {
  mkdir -p "$STATE_DIR"
  if is_running; then
    echo "gpu-placeholder already running: pid=$(cat "$PID_FILE")"
    return 0
  fi
  if existing_gpu_memory_exceeds_limit; then
    echo "gpu-placeholder skipped: GPU memory already in use on $VISIBLE_GPUS (limit ${MAX_EXISTING_MEM_MIB} MiB)" | tee -a "$LOG_FILE"
    return 0
  fi
  mapfile -t active_gpu_processes < <(active_non_placeholder_gpu_processes)
  if [[ ${#active_gpu_processes[@]} -gt 0 ]]; then
    echo "gpu-placeholder skipped: non-placeholder GPU process(es) already active on $VISIBLE_GPUS" | tee -a "$LOG_FILE"
    printf '%s\n' "${active_gpu_processes[@]}" | tee -a "$LOG_FILE"
    return 0
  fi
  rm -f "$PID_FILE"
  echo "starting gpu-placeholder on physical GPUs $VISIBLE_GPUS" | tee -a "$LOG_FILE"
  (
    cd "$ROOT_DIR"
    nohup setsid env CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" GPU_PLACEHOLDER_PHYSICAL_GPUS="$VISIBLE_GPUS" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/gpu_placeholder_worker.py \
      --gpus "$WORKER_VISIBLE_GPUS" \
      --memory_fraction "$MEMORY_FRACTION" \
      --matrix_size "$MATRIX_SIZE" \
      --chunk_mib "$CHUNK_MIB" \
      --heartbeat_sec "$HEARTBEAT_SEC" \
      --safety_free_mib "$SAFETY_FREE_MIB" \
      --sync_every "$SYNC_EVERY" \
      --sleep_ms "$SLEEP_MS" \
      >>"$LOG_FILE" 2>&1 </dev/null &
    echo "$!" > "$PID_FILE"
  )
  sleep 2
  if is_running; then
    echo "gpu-placeholder started: pid=$(cat "$PID_FILE") log=$LOG_FILE"
  else
    echo "gpu-placeholder failed to start; see $LOG_FILE" >&2
    return 1
  fi
}

stop_placeholder() {
  if ! is_running; then
    mapfile -t orphan_pids < <(orphan_placeholder_pids)
    if [[ ${#orphan_pids[@]} -eq 0 ]]; then
      echo "gpu-placeholder not running"
    else
      echo "stopping orphan gpu-placeholder process(es): ${orphan_pids[*]}"
      stop_placeholder_pids "${orphan_pids[@]}"
    fi
    rm -f "$PID_FILE"
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "stopping gpu-placeholder: pid=$pid"
  stop_placeholder_pids "$pid"
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "gpu-placeholder stopped"
      return 0
    fi
    sleep 1
  done
  echo "gpu-placeholder did not stop after 30s; sending SIGKILL"
  local pgid
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [[ -n "$pgid" ]]; then
    kill -KILL -- "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  else
    kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}

status_placeholder() {
  if is_running; then
    echo "gpu-placeholder running: pid=$(cat "$PID_FILE")"
  else
    echo "gpu-placeholder not running"
  fi
  echo "log: $LOG_FILE"
  echo "physical GPUs: $VISIBLE_GPUS  matrix_size=$MATRIX_SIZE sleep_ms=$SLEEP_MS sync_every=$SYNC_EVERY"
  nvidia-smi -i "$VISIBLE_GPUS" --query-gpu=index,power.draw,power.limit,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits || true
}

run_with_placeholder_suspended() {
  stop_placeholder
  trap 'start_placeholder' EXIT
  "$@"
}

COMMAND="${1:-status}"
shift || true

RUN_COMMAND=()
if [[ "$COMMAND" == "run" ]]; then
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
      shift
      RUN_COMMAND=("$@")
      set --
      break
    fi
    case "$1" in
      --gpus|--visible-gpus|--memory-fraction|--matrix-size|--sleep-ms|--sync-every|--safety-free-mib|--max-existing-mem-mib|--heartbeat-sec|--chunk-mib)
        parse_options "$1" "$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        RUN_COMMAND=("$@")
        set --
        break
        ;;
    esac
  done
  configure_state_paths
else
  parse_options "$@"
fi

case "$COMMAND" in
  start)
    start_placeholder
    ;;
  stop)
    stop_placeholder
    ;;
  restart)
    stop_placeholder
    start_placeholder
    ;;
  status)
    status_placeholder
    ;;
  run)
    if [[ ${#RUN_COMMAND[@]} -eq 0 ]]; then
      echo "usage: $0 run [options] -- <command...>" >&2
      exit 2
    fi
    run_with_placeholder_suspended "${RUN_COMMAND[@]}"
    ;;
  *)
    usage
    exit 2
    ;;
esac
