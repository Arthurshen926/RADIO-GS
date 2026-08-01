#!/usr/bin/env bash

# Continue the query-free Surface attention screen from an immutable parent
# whose ten cache stages completed, without rebuilding or rewriting any cache.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-1}"
PARENT_MANIFEST="${PARENT_MANIFEST:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/surface_c1024_attention_pooling_v1_gpu1only_p4_h78_srcd44cf2c8/run_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260801/surface_c1024_attention_postcache_v1_gpu1only}"
READOUT_SEEDS="${READOUT_SEEDS:-0,1,2}"

AUTHORITY="$REPO_ROOT/radio_gs/scripts/surface_attention_postcache_continuation.py"
TRAINER="$REPO_ROOT/radio_gs/scripts/train_surface_region_summary_readout.py"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
CLOSURE_GUARD="$REPO_ROOT/radio_gs/scripts/surface_region_run_guard.py"
LOCK_SUPERVISOR="$REPO_ROOT/radio_gs/scripts/surface_gpu1_lock_supervisor.py"
RUN_REPO_PYTHON="$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
GLOBAL_GPU1_LOCK="/root/RADIO-GS/output/.physical_gpu1.lock"
GPU1_SINGLETON_PROTOCOL="linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"

MANIFEST="$OUTPUT_ROOT/run_manifest.json"
PAIRING_REPORT="$OUTPUT_ROOT/cache_pairing.json"
SCREEN_REPORT="$OUTPUT_ROOT/attention_pooling_screen.json"
CLOSURE_FINAL_REPORT="$OUTPUT_ROOT/runtime_closure_final.json"
GPU_TELEMETRY_LOG="$OUTPUT_ROOT/gpu1_telemetry.csv"
LOG_ROOT="$OUTPUT_ROOT/logs"
ATTEMPT_RECEIPT_ROOT="$OUTPUT_ROOT/stage_attempts"

# Preserve the parent canary's proven GPU1-only safety envelope.  Readout
# training has no RADIO image pacing and every stage boundary performs a CPU
# parent/closure audit, which also supplies natural cooldown time.
GPU_MAX_TEMP_C=78
GPU_START_MAX_TEMP_C=65
GPU_MAX_POWER_LIMIT_W=300.5
GPU_POLL_SECONDS=3
GPU_SOFT_PAUSE_TEMP_C=75
GPU_SOFT_RESUME_TEMP_C=70
GPU_PEER_INDEX=""
GPU_PEER_PAUSE_TEMP_C=0
GPU_PEER_RESUME_TEMP_C=0
GPU_PEER_QUIET_SECONDS=0
GPU_PEER_MAX_POWER_W=0
GPU_PEER_MAX_MEMORY_MIB=0
GPU_PEER_MAX_UTIL_PCT=100
GPU_PEER_ACTIVITY_ACTION="terminate"
GPU_OWNER_PID_NAMESPACE_MODE="exclusive-singleton-after-clear-v1"
GPU_PEER_INTERRUPT_EXIT_CODE=87

run_authority() {
  CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" "$AUTHORITY" "$@"
}

case "${1:-run}" in
  --validate-parent-only)
    run_authority validate-parent --parent-manifest "$PARENT_MANIFEST"
    exit 0
    ;;
  run) ;;
  *)
    echo "usage: $0 [run|--validate-parent-only]" >&2
    exit 2
    ;;
esac

for required in \
  "$PARENT_MANIFEST" "$AUTHORITY" "$TRAINER" "$THERMAL_GUARD" \
  "$CLOSURE_GUARD" "$LOCK_SUPERVISOR" "$RUN_REPO_PYTHON"; do
  if [[ ! -e "$required" ]]; then
    echo "missing Surface post-cache input: $required" >&2
    exit 2
  fi
done
if [[ "$GPU" != "1" ]]; then
  echo "Surface post-cache continuation is assigned to physical GPU1" >&2
  exit 2
fi
if [[ "$PARENT_MANIFEST" != /* || "$OUTPUT_ROOT" != /* ]]; then
  echo "PARENT_MANIFEST and OUTPUT_ROOT must be absolute" >&2
  exit 2
fi
if [[ "$READOUT_SEEDS" != "0,1,2" ]]; then
  echo "Surface post-cache continuation requires READOUT_SEEDS=0,1,2" >&2
  exit 2
fi

# Fail before GPU ownership if the immutable parent is incomplete or changed.
run_authority validate-parent --parent-manifest "$PARENT_MANIFEST" >/dev/null

if [[ -z "${RADIO_GS_GPU1_LOCK_FD:-}" \
      && -z "${RADIO_GS_GPU1_SINGLETON_FD:-}" ]]; then
  exec bash "$RUN_REPO_PYTHON" "$LOCK_SUPERVISOR" run -- bash "$0" "$@"
fi
if [[ -z "${RADIO_GS_GPU1_LOCK_FD:-}" \
      || -z "${RADIO_GS_GPU1_SINGLETON_FD:-}" ]]; then
  echo "physical GPU1 lock inheritance is incomplete" >&2
  exit 2
fi
if [[ "${RADIO_GS_GPU1_LOCK_PATH:-}" != "$GLOBAL_GPU1_LOCK" \
      || "${RADIO_GS_GPU1_SINGLETON_PROTOCOL:-}" != "$GPU1_SINGLETON_PROTOCOL" ]]; then
  echo "physical GPU1 supervisor contract differs" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" \
  "$LOCK_SUPERVISOR" verify-inherited \
  --fd "$RADIO_GS_GPU1_LOCK_FD" \
  --singleton-fd "$RADIO_GS_GPU1_SINGLETON_FD" >/dev/null

GPU_IDENTITY="$(
  timeout --kill-after=2s 10s nvidia-smi -i "$GPU" \
    --query-gpu=pci.bus_id,uuid --format=csv,noheader,nounits
)" || {
  echo "physical GPU1 identity is not queryable" >&2
  exit 2
}
IFS=',' read -r GPU_BUS_ID GPU_UUID <<<"$GPU_IDENTITY"
GPU_BUS_ID="$(tr -d '[:space:]' <<<"$GPU_BUS_ID" | sed 's/^00000000:/0000:/')"
GPU_UUID="$(tr -d '[:space:]' <<<"$GPU_UUID")"
if [[ -z "$GPU_BUS_ID" || -z "$GPU_UUID" ]]; then
  echo "physical GPU1 returned an incomplete identity" >&2
  exit 2
fi
GPU_CONFIG="/sys/bus/pci/devices/$GPU_BUS_ID/config"
GPU_CONFIG_PREFIX="$(od -An -tx1 -N16 "$GPU_CONFIG" 2>/dev/null | tr -d ' \n')"
if [[ -z "$GPU_CONFIG_PREFIX" || "$GPU_CONFIG_PREFIX" =~ ^f+$ ]]; then
  echo "physical GPU1 PCIe configuration space is not responding" >&2
  exit 2
fi

assert_gpu_uuid_unowned() {
  local current_uuid owners
  current_uuid="$(
    nvidia-smi -i "$GPU" --query-gpu=uuid \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  if [[ "$current_uuid" != "$GPU_UUID" ]]; then
    echo "physical GPU1 UUID changed during continuation" >&2
    return 2
  fi
  owners="$(
    nvidia-smi -i "$GPU" --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits \
      | awk -F', *' -v uuid="$GPU_UUID" '$1 == uuid {print $2}' \
      | paste -sd, -
  )"
  if [[ -n "$owners" ]]; then
    echo "physical GPU1 has unexpected compute owner(s): $owners" >&2
    return 2
  fi
}
assert_gpu_uuid_unowned

for directory in "$OUTPUT_ROOT" "$LOG_ROOT" "$ATTEMPT_RECEIPT_ROOT"; do
  if [[ -L "$directory" ]]; then
    echo "refusing symlinked continuation directory: $directory" >&2
    exit 2
  fi
done

run_authority create-manifest \
  --repo-root "$REPO_ROOT" --output-root "$OUTPUT_ROOT" \
  --runner "$0" --manifest "$MANIFEST" \
  --parent-manifest "$PARENT_MANIFEST" \
  --telemetry "$GPU_TELEMETRY_LOG" --gpu-uuid "$GPU_UUID" >/dev/null
mkdir -p "$LOG_ROOT" "$ATTEMPT_RECEIPT_ROOT" "$OUTPUT_ROOT/readouts"

verify_run_contract() {
  local phase="$1"
  run_authority verify-manifest --manifest "$MANIFEST" >/dev/null
  CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" \
    "$CLOSURE_GUARD" verify-closure \
    --manifest "$MANIFEST" --phase "$phase" >/dev/null
}

telemetry_line_count() {
  if [[ -f "$GPU_TELEMETRY_LOG" ]]; then
    wc -l <"$GPU_TELEMETRY_LOG"
  else
    printf '0\n'
  fi
}

attempt_peer_release_interrupt_count() {
  local start_line="$1" end_line="$2"
  [[ -f "$GPU_TELEMETRY_LOG" ]] || { printf '0\n'; return; }
  awk -F',' -v first="$start_line" -v last="$end_line" '
    NR > first && NR <= last && $10 ~ /^peer_activity_interrupt_release_cuda_/ {
      count += 1
    }
    END { print count + 0 }
  ' "$GPU_TELEMETRY_LOG"
}

write_attempt_receipt() {
  local receipt="$1" stage="$2" attempt_index="$3" command_status="$4"
  local result="$5" attempt_log="$6" telemetry_start="$7"
  local telemetry_end="$8" start_epoch="$9"
  shift 9
  local end_epoch="$1" terminal="$2" sidecar="$3"
  local kernel_capture_status="$4" kernel_log="$5"
  local postflight_capture_status="$6" postflight_report="$7"
  local owner_audit="$8"
  shift 8
  CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" - \
    "$receipt" "$MANIFEST" "$stage" "$attempt_index" \
    "$command_status" "$result" "$attempt_log" \
    "$GPU_TELEMETRY_LOG" "$telemetry_start" "$telemetry_end" \
    "$start_epoch" "$end_epoch" "$terminal" "$sidecar" \
    "$kernel_capture_status" "$kernel_log" \
    "$postflight_capture_status" "$postflight_report" "$owner_audit" \
    "$GPU_PEER_ACTIVITY_ACTION" "$GPU_PEER_INTERRUPT_EXIT_CODE" "$@" <<'PY'
import os
import sys
from pathlib import Path

from radio_gs.scripts.surface_region_run_guard import (
    _stable_artifact_bytes,
    kernel_fault_lines,
    telemetry_interval_record,
)
from radio_gs.utils.immutable_artifacts import file_record, write_frozen_json

(
    receipt, manifest, stage, attempt_index, command_status, result,
    attempt_log, telemetry, telemetry_start, telemetry_end, start_epoch,
    end_epoch, terminal, sidecar, kernel_capture_status, kernel_log,
    postflight_capture_status, postflight_report, owner_audit, peer_action,
    peer_interrupt_exit_code, *command,
) = sys.argv[1:]

def optional_record(raw_path):
    if not raw_path or not os.path.lexists(raw_path):
        return None
    return file_record(Path(raw_path))

kernel_encoded, _, _ = _stable_artifact_bytes(
    kernel_log, label="Surface post-cache attempt kernel journal"
)
payload = {
    "artifact_type": "surface-region-stage-attempt-v1",
    "schema_version": 1,
    "run_manifest": file_record(manifest),
    "stage": stage,
    "attempt_index": int(attempt_index),
    "command": command,
    "command_status": int(command_status),
    "result": result,
    "log": file_record(attempt_log),
    "telemetry_interval": telemetry_interval_record(
        telemetry, start_line=int(telemetry_start), end_line=int(telemetry_end)
    ),
    "kernel_journal": {
        "start_epoch": int(start_epoch),
        "end_epoch": int(end_epoch),
        "capture_status": int(kernel_capture_status),
        "fault_count": len(kernel_fault_lines(kernel_encoded)),
        "file": file_record(kernel_log),
    },
    "gpu_release_postflight": (
        None if not postflight_report else {
            "capture_status": int(postflight_capture_status),
            "report": optional_record(postflight_report),
        }
    ),
    "owner_audit": file_record(owner_audit),
    "terminal": optional_record(terminal),
    "sidecar": optional_record(sidecar),
    "peer_activity_action": peer_action,
    "peer_activity_interrupt_exit_code": int(peer_interrupt_exit_code),
}
write_frozen_json(receipt, payload)
PY
}

verify_existing_attempt_receipt() {
  local receipt="$1" stage="$2" attempt_index="$3" attempt_log="$4"
  shift 4
  local command=(
    bash "$RUN_REPO_PYTHON" "$CLOSURE_GUARD" verify-attempt
    --manifest "$MANIFEST" --receipt "$receipt" --stage "$stage"
    --index "$attempt_index" --log "$attempt_log"
    --allowed-result peer_activity_interrupted_cuda_released_retry_authorized
  )
  local argument
  for argument in "$@"; do command+=("--command-arg=$argument"); done
  CUDA_VISIBLE_DEVICES="" "${command[@]}" >/dev/null
}

run_stage() {
  local stage="$1" terminal="$2"
  shift 2
  local sidecar="${terminal}.json"
  verify_run_contract "pre_${stage}"
  if [[ -L "$terminal" || -L "$sidecar" ]]; then
    echo "refusing symlinked continuation terminal: $terminal" >&2
    return 1
  fi
  if [[ -s "$terminal" && -s "$sidecar" ]]; then
    local completed_attempt_dir completed_receipt completed_log
    completed_attempt_dir="$ATTEMPT_RECEIPT_ROOT/$stage"
    completed_receipt="$completed_attempt_dir/attempt_000001.json"
    completed_log="$LOG_ROOT/${stage}.attempt_000001.log"
    if [[ ! -s "$completed_receipt" || ! -s "$completed_log" ]]; then
      echo "continuation terminal exists without its completed receipt: $stage" >&2
      return 1
    fi
    verify_existing_attempt_receipt \
      "$completed_receipt" "$stage" 1 "$completed_log" "$@"
    return 0
  fi
  if [[ -e "$terminal" || -L "$terminal" \
        || -e "$sidecar" || -L "$sidecar" ]]; then
    echo "partial continuation terminal requires inspection: $terminal" >&2
    return 1
  fi
  local attempt_dir="$ATTEMPT_RECEIPT_ROOT/$stage"
  [[ ! -L "$attempt_dir" ]] || { echo "symlinked attempt directory" >&2; return 1; }
  mkdir -p "$attempt_dir"
  local attempt_index=1
  while true; do
    local attempt_tag receipt attempt_log kernel_log postflight_report owner_audit
    local receipt_present=0 log_present=0 kernel_present=0
    local postflight_present=0 owner_audit_present=0
    attempt_tag="$(printf '%06d' "$attempt_index")"
    receipt="$attempt_dir/attempt_${attempt_tag}.json"
    attempt_log="$LOG_ROOT/${stage}.attempt_${attempt_tag}.log"
    kernel_log="$LOG_ROOT/${stage}.attempt_${attempt_tag}.kernel.log"
    postflight_report="$LOG_ROOT/${stage}.attempt_${attempt_tag}.gpu_release_postflight.json"
    owner_audit="$attempt_dir/attempt_${attempt_tag}.owner_audit.csv"
    [[ -e "$receipt" || -L "$receipt" ]] && receipt_present=1
    [[ -e "$attempt_log" || -L "$attempt_log" ]] && log_present=1
    [[ -e "$kernel_log" || -L "$kernel_log" ]] && kernel_present=1
    [[ -e "$postflight_report" || -L "$postflight_report" ]] && postflight_present=1
    [[ -e "$owner_audit" || -L "$owner_audit" ]] && owner_audit_present=1
    if (( receipt_present && log_present )); then
      verify_existing_attempt_receipt \
        "$receipt" "$stage" "$attempt_index" "$attempt_log" "$@"
      attempt_index=$((attempt_index + 1))
      continue
    fi
    if (( receipt_present || log_present || kernel_present \
          || postflight_present || owner_audit_present )); then
      echo "half-published continuation attempt: $stage/$attempt_tag" >&2
      return 1
    fi

    verify_run_contract "pre_${stage}_attempt_${attempt_tag}"
    assert_gpu_uuid_unowned
    local telemetry_start telemetry_end start_epoch end_epoch command_status
    local kernel_capture_status postflight_capture_status peer_event_count result
    telemetry_start="$(telemetry_line_count)"
    start_epoch="$(date +%s)"
    command_status=0
    GPU="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
      GPU_TELEMETRY_LOG="$GPU_TELEMETRY_LOG" \
      GPU_MAX_TEMP_C="$GPU_MAX_TEMP_C" \
      GPU_START_MAX_TEMP_C="$GPU_START_MAX_TEMP_C" \
      GPU_MAX_POWER_LIMIT_W="$GPU_MAX_POWER_LIMIT_W" \
      GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
      GPU_SOFT_PAUSE_TEMP_C="$GPU_SOFT_PAUSE_TEMP_C" \
      GPU_SOFT_RESUME_TEMP_C="$GPU_SOFT_RESUME_TEMP_C" \
      GPU_OWNER_PID_NAMESPACE_MODE="$GPU_OWNER_PID_NAMESPACE_MODE" \
      GPU_OWNER_AUDIT_LOG="$owner_audit" \
      GPU_PEER_INDEX="$GPU_PEER_INDEX" \
      GPU_PEER_PAUSE_TEMP_C="$GPU_PEER_PAUSE_TEMP_C" \
      GPU_PEER_RESUME_TEMP_C="$GPU_PEER_RESUME_TEMP_C" \
      GPU_PEER_QUIET_SECONDS="$GPU_PEER_QUIET_SECONDS" \
      GPU_PEER_MAX_POWER_W="$GPU_PEER_MAX_POWER_W" \
      GPU_PEER_MAX_MEMORY_MIB="$GPU_PEER_MAX_MEMORY_MIB" \
      GPU_PEER_MAX_UTIL_PCT="$GPU_PEER_MAX_UTIL_PCT" \
      GPU_PEER_ACTIVITY_ACTION="$GPU_PEER_ACTIVITY_ACTION" \
      bash "$THERMAL_GUARD" -- bash "$RUN_REPO_PYTHON" "$@" \
      >"$attempt_log" 2>&1 || command_status=$?
    end_epoch="$(date +%s)"
    telemetry_end="$(telemetry_line_count)"
    kernel_capture_status=0
    CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" \
      "$CLOSURE_GUARD" capture-kernel-journal \
      --start-epoch "$start_epoch" --end-epoch "$end_epoch" \
      --gpu-bus-id "$GPU_BUS_ID" --output "$kernel_log" >/dev/null \
      || kernel_capture_status=$?
    postflight_capture_status=-1
    peer_event_count="$(attempt_peer_release_interrupt_count \
      "$telemetry_start" "$telemetry_end")"
    if (( command_status == GPU_PEER_INTERRUPT_EXIT_CODE )) \
        && (( peer_event_count == 1 )); then
      postflight_capture_status=0
      CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" \
        "$CLOSURE_GUARD" capture-gpu-release-postflight \
        --gpu "$GPU" --expected-uuid "$GPU_UUID" \
        --output "$postflight_report" >/dev/null \
        || postflight_capture_status=$?
    else
      postflight_report=""
    fi
    verify_run_contract "post_${stage}_attempt_${attempt_tag}"

    if (( kernel_capture_status != 0 )); then
      result="attempt_evidence_failed_no_retry"
    elif (( command_status == 0 )); then
      result="completed"
    elif (( command_status == GPU_PEER_INTERRUPT_EXIT_CODE )) \
        && (( peer_event_count == 1 )) \
        && (( postflight_capture_status == 0 )); then
      result="peer_activity_interrupted_cuda_released_retry_authorized"
    elif (( command_status == GPU_PEER_INTERRUPT_EXIT_CODE )); then
      result="peer_activity_interrupted_cuda_release_unverified_no_retry"
    else
      result="failed_no_retry"
    fi
    write_attempt_receipt \
      "$receipt" "$stage" "$attempt_index" "$command_status" "$result" \
      "$attempt_log" "$telemetry_start" "$telemetry_end" \
      "$start_epoch" "$end_epoch" "$terminal" "$sidecar" \
      "$kernel_capture_status" "$kernel_log" \
      "$postflight_capture_status" "$postflight_report" "$owner_audit" "$@"

    if [[ "$result" == "peer_activity_interrupted_cuda_released_retry_authorized" ]]; then
      if [[ -s "$terminal" && -s "$sidecar" ]]; then return 0; fi
      if [[ -e "$terminal" || -L "$terminal" \
            || -e "$sidecar" || -L "$sidecar" ]]; then
        echo "peer interruption left a partial terminal: $stage" >&2
        return 1
      fi
      attempt_index=$((attempt_index + 1))
      continue
    fi
    if [[ "$result" == *"no_retry" ]]; then
      echo "continuation attempt failed closed: $stage ($result)" >&2
      (( command_status != 0 )) && return "$command_status"
      return 86
    fi
    if [[ ! -s "$terminal" || ! -s "$sidecar" ]]; then
      echo "continuation stage lacks terminal: $stage" >&2
      return 1
    fi
    return 0
  done
}

verify_run_contract pre_cache_pairing
run_authority verify-pairing --manifest "$MANIFEST" \
  --output "$PAIRING_REPORT" >/dev/null
verify_run_contract post_cache_pairing

PARENT_ROOT="$(dirname "$PARENT_MANIFEST")"
TRAIN_CACHE_GLOB="$PARENT_ROOT/caches/context_c1024_geometric/train_shard*.pt"
VALIDATION_CACHE_GLOB="$PARENT_ROOT/caches/context_c1024_geometric/validation_shard*.pt"
IFS=',' read -r -a SEEDS <<<"$READOUT_SEEDS"
for pooling_mode in joint_attention_v1 core_context_separate_attention_v1; do
  for seed in "${SEEDS[@]}"; do
    model="$OUTPUT_ROOT/readouts/${pooling_mode}_seed${seed}.pt"
    run_stage "readout_${pooling_mode}_seed${seed}" "$model" \
      radio_gs/scripts/train_surface_region_summary_readout.py \
      --train-caches "$TRAIN_CACHE_GLOB" \
      --validation-caches "$VALIDATION_CACHE_GLOB" \
      --output "$model" --hidden-dim 256 --epochs 60 --patience 10 \
      --batch-size 16 --learning-rate 2e-4 --weight-decay 1e-4 \
      --token-weight 0.25 --relation-weight 0.1 \
      --reliability-attention-mode log_prior \
      --context-pooling-mode "$pooling_mode" --seed "$seed" \
      --device cuda:0 --radio-checkpoint \
      "$(CUDA_VISIBLE_DEVICES='' bash "$RUN_REPO_PYTHON" - "$MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['radio_checkpoint'])
PY
)"
  done
done

verify_run_contract pre_attention_screen_report
run_authority finalize --manifest "$MANIFEST" --pairing "$PAIRING_REPORT" \
  --output "$SCREEN_REPORT" >/dev/null
CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" \
  "$CLOSURE_GUARD" verify-closure --manifest "$MANIFEST" \
  --phase final_before_completion --full-checkpoint \
  --attempt-root "$ATTEMPT_RECEIPT_ROOT" --log-root "$LOG_ROOT" \
  --report "$CLOSURE_FINAL_REPORT" >/dev/null
date -Iseconds >"$OUTPUT_ROOT/screen.complete"
