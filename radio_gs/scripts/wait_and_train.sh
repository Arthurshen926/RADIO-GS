#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash radio_gs/scripts/wait_and_train.sh <config> [log_path]"
    exit 1
fi

CONFIG_PATH="$1"
LOG_PATH="${2:-output/radio_gs/wait_and_train.log}"
MIN_FREE_MIB="${MIN_FREE_MIB:-4096}"
MAX_UTIL="${MAX_UTIL:-70}"
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$(dirname "$LOG_PATH")"
LOCK_DIR="${GPU_LOCK_DIR:-/tmp/radio_gs_gpu_locks}"
mkdir -p "$LOCK_DIR"

IFS=',' read -r -a ALLOWED_GPUS <<< "$GPU_LIST"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

gpu_allowed() {
    local gpu="$1"
    for candidate in "${ALLOWED_GPUS[@]}"; do
        if [[ "$gpu" == "$candidate" ]]; then
            return 0
        fi
    done
    return 1
}

lock_gpu() {
    local gpu="$1"
    local lock_path="$LOCK_DIR/gpu_${gpu}.lock"
    if mkdir "$lock_path" 2>/dev/null; then
        printf '%s\n' "$$" > "$lock_path/pid"
        return 0
    fi
    return 1
}

unlock_gpu() {
    local gpu="$1"
    rm -rf "$LOCK_DIR/gpu_${gpu}.lock"
}

echo "[$(timestamp)] Waiting for a free GPU to start training" | tee -a "$LOG_PATH"
echo "[$(timestamp)] config=$CONFIG_PATH min_free=${MIN_FREE_MIB}MiB max_util=${MAX_UTIL}% gpus=$GPU_LIST" | tee -a "$LOG_PATH"

while true; do
    while IFS=',' read -r gpu_idx free_mib util_pct; do
        gpu_idx="${gpu_idx// /}"
        free_mib="${free_mib// /}"
        util_pct="${util_pct// /}"

        if ! gpu_allowed "$gpu_idx"; then
            continue
        fi

        if [[ "$free_mib" =~ ^[0-9]+$ ]] && [[ "$util_pct" =~ ^[0-9]+$ ]]; then
            if (( free_mib >= MIN_FREE_MIB && util_pct <= MAX_UTIL )); then
                if ! lock_gpu "$gpu_idx"; then
                    continue
                fi
                echo "[$(timestamp)] Launching training on GPU $gpu_idx (free=${free_mib}MiB util=${util_pct}%)" | tee -a "$LOG_PATH"
                trap 'unlock_gpu "$gpu_idx"' EXIT
                export CUDA_VISIBLE_DEVICES="$gpu_idx"
                export PYTHONUNBUFFERED=1
                python radio_gs/scripts/train_feature_field.py --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_PATH"
                exit $?
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

    echo "[$(timestamp)] No suitable GPU yet; sleeping ${CHECK_INTERVAL}s" | tee -a "$LOG_PATH"
    sleep "$CHECK_INTERVAL"
done
