#!/usr/bin/env bash

# Isolated query-free Surface screen: one byte-matched c1024/geometric cache,
# two attention pooling rules, and paired seeds 0/1/2.  The historical
# capacity runner is deliberately not sourced or modified.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

value_source() {
  if [[ -v "$1" && -n "${!1}" ]]; then
    printf 'environment_override\n'
  else
    printf 'balanced_default\n'
  fi
}

GPU_MAX_TEMP_C_SOURCE="$(value_source GPU_MAX_TEMP_C)"
GPU_START_MAX_TEMP_C_SOURCE="$(value_source GPU_START_MAX_TEMP_C)"
GPU_POLL_SECONDS_SOURCE="$(value_source GPU_POLL_SECONDS)"
GPU_MAX_POWER_LIMIT_W_SOURCE="$(value_source GPU_MAX_POWER_LIMIT_W)"
GPU_SOFT_PAUSE_TEMP_C_SOURCE="$(value_source GPU_SOFT_PAUSE_TEMP_C)"
GPU_SOFT_RESUME_TEMP_C_SOURCE="$(value_source GPU_SOFT_RESUME_TEMP_C)"
GPU_PEER_PAUSE_TEMP_C_SOURCE="frozen_contract"
GPU_PEER_RESUME_TEMP_C_SOURCE="frozen_contract"
GPU_PEER_QUIET_SECONDS_SOURCE="frozen_contract"
GPU_PEER_MAX_POWER_W_SOURCE="frozen_contract"
GPU_PEER_MAX_MEMORY_MIB_SOURCE="frozen_contract"
GPU_PEER_MAX_UTIL_PCT_SOURCE="frozen_contract"
GPU_PEER_ACTIVITY_ACTION_SOURCE="frozen_contract"
GPU_OWNER_PID_NAMESPACE_MODE_SOURCE="frozen_contract"
RADIO_PACING_SOURCE="$(value_source RADIO_THERMAL_PACING_SECONDS_PER_IMAGE)"
CANARY_MAX_TEMP_C_SOURCE="$(value_source SURFACE_CANARY_MAX_TEMP_C)"
EXTERNAL_REUSE_SOURCE="$(value_source SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE)"

GPU="${GPU:-1}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/pool/Datasets/ScanNet/data/tasks/scannet_frames_25k}"
TRAIN_SPLIT="${TRAIN_SPLIT:-$REPO_ROOT/paper/artifacts/scannet_surface_region_query_free_train_scenes_20260731.txt}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-$REPO_ROOT/paper/artifacts/scannet_surface_region_query_free_validation_scenes_20260731.txt}"
RADIO_REPO="${RADIO_REPO:-/root/RADIO}"
RADIO_VERSION="${RADIO_VERSION:-c-radio_v4-h}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/RADIO-GS/output/optimization_20260801/surface_c1024_attention_pooling_v1_gpu1_p6_hard78_canary74}"
EXTERNAL_CONTROL_ROOT="${EXTERNAL_CONTROL_ROOT:-/root/RADIO-GS/output/optimization_20260731/surface_fixed_teacher_replay_v2_gpu1_p8_hard75}"
READOUT_SEEDS="${READOUT_SEEDS:-0,1,2}"

THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
CLOSURE_GUARD="$REPO_ROOT/radio_gs/scripts/surface_region_run_guard.py"
LOCK_SUPERVISOR="$REPO_ROOT/radio_gs/scripts/surface_gpu1_lock_supervisor.py"
AUTHORITY="$REPO_ROOT/radio_gs/scripts/surface_attention_pooling_screen.py"
BUILDER="$REPO_ROOT/radio_gs/scripts/build_scannet_surface_region_cache.py"
GLOBAL_GPU1_LOCK="/root/RADIO-GS/output/.physical_gpu1.lock"
GPU1_SINGLETON_PROTOCOL="linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"

# Balanced GPU1-only long-run defaults.  p6 is an execution-only pause: the
# validated p8 run stayed <=71C and the more aggressive p4 observations stayed
# <=73C.  The guard now responds only to physical GPU1 with 75/70C hysteresis;
# a 74C full-shard canary remains stricter than the 78C emergency stop.
GPU_TELEMETRY_LOG="${GPU_TELEMETRY_LOG:-$OUTPUT_ROOT/gpu1_telemetry.csv}"
GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"
GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-65}"
GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-3}"
GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-75}"
GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-70}"
# Frozen neutral peer fields preserve the schema-1 receipt shape while an
# empty index guarantees the guard never samples or gates on another GPU.
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
RADIO_THERMAL_PACING_SECONDS_PER_IMAGE="${RADIO_THERMAL_PACING_SECONDS_PER_IMAGE:-6.0}"
SURFACE_CANARY_MAX_TEMP_C="${SURFACE_CANARY_MAX_TEMP_C:-74}"
SURFACE_CANARY_RESUME="${SURFACE_CANARY_RESUME:-0}"
SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE="${SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE:-0}"
ADAPTOR_BATCH_SIZE="${ADAPTOR_BATCH_SIZE:-64}"

# The fastpath is required for the four controls built by this run and their
# four c1024 counterparts.  Legacy train shards 0/1 predate the intermediate
# artifact and deliberately take the full builder path.  Set "disabled" only
# as an explicit, manifest-recorded fallback; there is no silent downgrade.
SURFACE_INTERMEDIATE_FASTPATH="${SURFACE_INTERMEDIATE_FASTPATH:-required_local_shards}"

PFIR_DEV="$REPO_ROOT/radio_gs/benchmarks/scannet_pfir/split/scannet_pfir_small_v1_dev_candidates.txt"
PFIR_TEST="$REPO_ROOT/radio_gs/benchmarks/scannet_pfir/split/scannet_pfir_small_v1_test_candidates.txt"
EXCLUDED_SCENES="scene0000_00,scene0062_00,scene0070_00,scene0097_00,scene0140_00,scene0200_00,scene0347_00,scene0400_00,scene0590_00"

MANIFEST="$OUTPUT_ROOT/run_manifest.json"
PAIRING_REPORT="$OUTPUT_ROOT/cache_pairing.json"
SCREEN_REPORT="$OUTPUT_ROOT/attention_pooling_screen.json"
CANARY_REPORT="$OUTPUT_ROOT/control_train_shard2_p6_canary.json"
CLOSURE_FINAL_REPORT="$OUTPUT_ROOT/runtime_closure_final.json"
LOG_ROOT="$OUTPUT_ROOT/logs"
CACHE_RESUME_ROOT="$OUTPUT_ROOT/cache_scene_resume"
INTERMEDIATE_ROOT="$OUTPUT_ROOT/scene_intermediate"
ATTEMPT_RECEIPT_ROOT="$OUTPUT_ROOT/stage_attempts"
REPLAY_AUTHORITY_ROOT="$OUTPUT_ROOT/teacher_replay_authorities"

run_authority() {
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" "$AUTHORITY" "$@"
}

validate_external_only() {
  run_authority validate-external \
    --external-control-root "$EXTERNAL_CONTROL_ROOT" \
    --train-split "$TRAIN_SPLIT" \
    --validation-split "$VALIDATION_SPLIT" \
    --radio-checkpoint "$RADIO_CHECKPOINT" \
    --pfir-dev "$PFIR_DEV" \
    --pfir-test "$PFIR_TEST"
}

case "${1:-run}" in
  --validate-external-only)
    validate_external_only
    exit 0
    ;;
  run)
    ;;
  *)
    echo "usage: $0 [run|--validate-external-only]" >&2
    exit 2
    ;;
esac

for required in \
  "$DATASET_ROOT" "$TRAIN_SPLIT" "$VALIDATION_SPLIT" \
  "$RADIO_REPO" "$RADIO_CHECKPOINT" "$PFIR_DEV" "$PFIR_TEST" \
  "$THERMAL_GUARD" "$CLOSURE_GUARD" "$LOCK_SUPERVISOR" \
  "$AUTHORITY" "$BUILDER" \
  "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"; do
  if [[ ! -e "$required" ]]; then
    echo "missing Surface attention-screen input: $required" >&2
    exit 2
  fi
done
if [[ "$GPU" != "1" ]]; then
  echo "the isolated Surface screen is assigned to physical GPU1; got GPU=$GPU" >&2
  exit 2
fi
if [[ "$OUTPUT_ROOT" != /* || "$EXTERNAL_CONTROL_ROOT" != /* ]]; then
  echo "OUTPUT_ROOT and EXTERNAL_CONTROL_ROOT must be absolute" >&2
  exit 2
fi
if [[ "$READOUT_SEEDS" != "0,1,2" ]]; then
  echo "the isolated Surface screen requires READOUT_SEEDS=0,1,2" >&2
  exit 2
fi
if [[ "$SURFACE_CANARY_RESUME" != "0" && "$SURFACE_CANARY_RESUME" != "1" ]]; then
  echo "SURFACE_CANARY_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "$SURFACE_INTERMEDIATE_FASTPATH" != "required_local_shards" \
      && "$SURFACE_INTERMEDIATE_FASTPATH" != "disabled" ]]; then
  echo "SURFACE_INTERMEDIATE_FASTPATH must be required_local_shards or disabled" >&2
  exit 2
fi
if [[ ! "$ADAPTOR_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ADAPTOR_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ "$SURFACE_INTERMEDIATE_FASTPATH" == "required_local_shards" ]]; then
  builder_help="$(CUDA_VISIBLE_DEVICES="" bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" "$BUILDER" --help)"
  for option in \
    --scene-intermediate-output-root \
    --scene-intermediate-manifest \
    --scene-intermediate-manifest-sha256; do
    if [[ "$builder_help" != *"$option"* ]]; then
      echo "required Surface intermediate fastpath is unavailable: $option" >&2
      exit 2
    fi
  done
fi
builder_help="$(CUDA_VISIBLE_DEVICES="" bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" "$BUILDER" --help)"
for option in \
  --teacher-replay-authority \
  --teacher-replay-authority-sha256; do
  if [[ "$builder_help" != *"$option"* ]]; then
    echo "required historical replay authority interface is unavailable: $option" >&2
    exit 2
  fi
done

# The external authority runs before GPU ownership and before any output is
# created.  A missing, moved-to-different-bytes, or protocol-ineligible shard
# is therefore a clean CPU-only failure.
validate_external_only >/dev/null
if [[ "$SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE" != "1" \
      || "$EXTERNAL_REUSE_SOURCE" != "environment_override" ]]; then
  echo "full run is fail-closed: explicitly set SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE=1" >&2
  exit 2
fi

if [[ -z "${RADIO_GS_GPU1_LOCK_FD:-}" \
      && -z "${RADIO_GS_GPU1_SINGLETON_FD:-}" ]]; then
  exec bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$LOCK_SUPERVISOR" run -- bash "$0" "$@"
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
CUDA_VISIBLE_DEVICES="" \
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
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
    echo "physical GPU1 UUID changed during the Surface run" >&2
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

for directory in \
  "$OUTPUT_ROOT" "$LOG_ROOT" "$CACHE_RESUME_ROOT" \
  "$INTERMEDIATE_ROOT" "$ATTEMPT_RECEIPT_ROOT" \
  "$REPLAY_AUTHORITY_ROOT"; do
  if [[ -L "$directory" ]]; then
    echo "refusing symlinked Surface attention run directory: $directory" >&2
    exit 2
  fi
done

THERMAL_VALUES="$(
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" - \
    "$GPU_MAX_TEMP_C" "$GPU_START_MAX_TEMP_C" \
    "$GPU_MAX_POWER_LIMIT_W" "$GPU_POLL_SECONDS" \
    "$GPU_SOFT_PAUSE_TEMP_C" "$GPU_SOFT_RESUME_TEMP_C" \
    "$GPU_PEER_INDEX" "$GPU_PEER_PAUSE_TEMP_C" \
    "$GPU_PEER_RESUME_TEMP_C" "$GPU_PEER_QUIET_SECONDS" \
    "$GPU_PEER_MAX_POWER_W" "$GPU_PEER_MAX_MEMORY_MIB" \
    "$GPU_PEER_MAX_UTIL_PCT" "$GPU_PEER_ACTIVITY_ACTION" \
    "$GPU_OWNER_PID_NAMESPACE_MODE" \
    "$RADIO_THERMAL_PACING_SECONDS_PER_IMAGE" \
    "$SURFACE_CANARY_MAX_TEMP_C" "$GPU_UUID" <<'PY'
import json
import sys

(
    hard, start, power_limit, poll, soft_pause, soft_resume,
    peer, peer_pause, peer_resume, peer_quiet, peer_power,
    peer_memory, peer_util, peer_action, owner_namespace_mode, pacing,
    canary, uuid,
) = sys.argv[1:]
print(json.dumps({
    "physical_gpu": 1,
    "gpu_uuid": uuid,
    "maximum_temperature_c": int(hard),
    "maximum_start_temperature_c": int(start),
    "maximum_power_limit_w": float(power_limit),
    "poll_seconds": int(poll),
    "soft_pause_temperature_c": int(soft_pause),
    "soft_resume_temperature_c": int(soft_resume),
    "peer_gpu": None if not peer else int(peer),
    "peer_pause_temperature_c": int(peer_pause),
    "peer_resume_temperature_c": int(peer_resume),
    "peer_quiet_seconds_before_launch": int(peer_quiet),
    "peer_max_power_w": float(peer_power),
    "peer_max_memory_mib": int(peer_memory),
    "peer_max_utilization_pct": int(peer_util),
    "peer_activity_action": peer_action,
    "owner_pid_namespace_mode": owner_namespace_mode,
    "radio_pacing_seconds_per_image": float(pacing),
    "canary_max_temp_c": int(canary),
}, sort_keys=True))
PY
)"
THERMAL_SOURCES="$(
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" - \
    "$GPU_MAX_TEMP_C_SOURCE" "$GPU_START_MAX_TEMP_C_SOURCE" \
    "$GPU_MAX_POWER_LIMIT_W_SOURCE" "$GPU_POLL_SECONDS_SOURCE" \
    "$GPU_SOFT_PAUSE_TEMP_C_SOURCE" "$GPU_SOFT_RESUME_TEMP_C_SOURCE" \
    "$GPU_PEER_PAUSE_TEMP_C_SOURCE" "$GPU_PEER_RESUME_TEMP_C_SOURCE" \
    "$GPU_PEER_QUIET_SECONDS_SOURCE" "$GPU_PEER_MAX_POWER_W_SOURCE" \
    "$GPU_PEER_MAX_MEMORY_MIB_SOURCE" "$GPU_PEER_MAX_UTIL_PCT_SOURCE" \
    "$GPU_PEER_ACTIVITY_ACTION_SOURCE" "$RADIO_PACING_SOURCE" \
    "$GPU_OWNER_PID_NAMESPACE_MODE_SOURCE" \
    "$CANARY_MAX_TEMP_C_SOURCE" <<'PY'
import json
import sys

keys = (
    "maximum_temperature_c", "maximum_start_temperature_c",
    "maximum_power_limit_w", "poll_seconds", "soft_pause_temperature_c",
    "soft_resume_temperature_c", "peer_pause_temperature_c",
    "peer_resume_temperature_c", "peer_quiet_seconds_before_launch",
    "peer_max_power_w", "peer_max_memory_mib", "peer_max_utilization_pct",
    "peer_activity_action", "radio_pacing_seconds_per_image",
    "owner_pid_namespace_mode", "canary_max_temp_c",
)
sources = dict(zip(keys, sys.argv[1:]))
sources["physical_gpu"] = "frozen_contract"
sources["gpu_uuid"] = "runtime_attested"
sources["peer_gpu"] = "frozen_contract"
print(json.dumps(sources, sort_keys=True))
PY
)"

run_authority create-manifest \
  --repo-root "$REPO_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --runner "$0" \
  --manifest "$MANIFEST" \
  --dataset-root "$DATASET_ROOT" \
  --train-split "$TRAIN_SPLIT" \
  --validation-split "$VALIDATION_SPLIT" \
  --radio-repo "$RADIO_REPO" \
  --radio-version "$RADIO_VERSION" \
  --radio-checkpoint "$RADIO_CHECKPOINT" \
  --pfir-dev "$PFIR_DEV" \
  --pfir-test "$PFIR_TEST" \
  --excluded-scene-names "$EXCLUDED_SCENES" \
  --external-control-root "$EXTERNAL_CONTROL_ROOT" \
  --telemetry "$GPU_TELEMETRY_LOG" \
  --thermal-values "$THERMAL_VALUES" \
  --thermal-sources "$THERMAL_SOURCES" \
  --intermediate-fastpath "$SURFACE_INTERMEDIATE_FASTPATH" \
  --external-reuse-authorization \
  "$SURFACE_ALLOW_VALIDATED_EXTERNAL_CONTROL_REUSE:$EXTERNAL_REUSE_SOURCE" \
  --adaptor-batch-size "$ADAPTOR_BATCH_SIZE" >/dev/null

mkdir -p "$LOG_ROOT" "$CACHE_RESUME_ROOT" "$INTERMEDIATE_ROOT" \
  "$ATTEMPT_RECEIPT_ROOT" "$REPLAY_AUTHORITY_ROOT"

verify_run_contract() {
  local phase="$1"
  run_authority verify-manifest --manifest "$MANIFEST" >/dev/null
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
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
  local start_line="$1"
  local end_line="$2"
  [[ -f "$GPU_TELEMETRY_LOG" ]] || {
    printf '0\n'
    return
  }
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
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" - \
    "$receipt" "$MANIFEST" "$stage" "$attempt_index" \
    "$command_status" "$result" "$attempt_log" \
    "$GPU_TELEMETRY_LOG" "$telemetry_start" "$telemetry_end" \
    "$start_epoch" "$end_epoch" "$terminal" "$sidecar" \
    "$kernel_capture_status" "$kernel_log" \
    "$postflight_capture_status" "$postflight_report" \
    "$owner_audit" \
    "$GPU_PEER_ACTIVITY_ACTION" "$GPU_PEER_INTERRUPT_EXIT_CODE" \
    "$@" <<'PY'
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
    kernel_log, label="Surface attention attempt kernel journal"
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
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
    "$CLOSURE_GUARD" verify-attempt
    --manifest "$MANIFEST"
    --receipt "$receipt"
    --stage "$stage"
    --index "$attempt_index"
    --log "$attempt_log"
    --allowed-result peer_activity_interrupted_cuda_released_retry_authorized
  )
  local argument
  for argument in "$@"; do
    command+=("--command-arg=$argument")
  done
  CUDA_VISIBLE_DEVICES="" "${command[@]}" >/dev/null
}

run_stage() {
  local stage="$1" terminal="$2"
  shift 2
  local sidecar="${terminal}.json"
  verify_run_contract "pre_${stage}"
  if [[ -L "$terminal" || -L "$sidecar" ]]; then
    echo "refusing symlinked Surface stage terminal: $terminal" >&2
    return 1
  fi
  if [[ -s "$terminal" && -s "$sidecar" ]]; then
    return 0
  fi
  if [[ -e "$terminal" || -L "$terminal" \
        || -e "$sidecar" || -L "$sidecar" ]]; then
    echo "partial Surface stage terminal requires inspection: $terminal" >&2
    return 1
  fi
  local attempt_dir="$ATTEMPT_RECEIPT_ROOT/$stage"
  if [[ -L "$attempt_dir" ]]; then
    echo "refusing symlinked Surface attempt directory: $attempt_dir" >&2
    return 1
  fi
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
      echo "half-published Surface attempt requires inspection: $stage/$attempt_tag" >&2
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
      bash "$THERMAL_GUARD" -- \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" "$@" \
      >"$attempt_log" 2>&1 || command_status=$?
    end_epoch="$(date +%s)"
    telemetry_end="$(telemetry_line_count)"
    kernel_capture_status=0
    CUDA_VISIBLE_DEVICES="" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$CLOSURE_GUARD" capture-kernel-journal \
      --start-epoch "$start_epoch" --end-epoch "$end_epoch" \
      --gpu-bus-id "$GPU_BUS_ID" \
      --output "$kernel_log" >/dev/null || kernel_capture_status=$?
    postflight_capture_status=-1
    peer_event_count="$(attempt_peer_release_interrupt_count \
      "$telemetry_start" "$telemetry_end")"
    if (( command_status == GPU_PEER_INTERRUPT_EXIT_CODE )) \
        && (( peer_event_count == 1 )); then
      postflight_capture_status=0
      CUDA_VISIBLE_DEVICES="" \
        bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
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
      "$postflight_capture_status" "$postflight_report" \
      "$owner_audit" "$@"

    if [[ "$result" == "peer_activity_interrupted_cuda_released_retry_authorized" ]]; then
      if [[ -s "$terminal" && -s "$sidecar" ]]; then
        return 0
      fi
      if [[ -e "$terminal" || -L "$terminal" \
            || -e "$sidecar" || -L "$sidecar" ]]; then
        echo "peer interruption left a half-published terminal: $stage" >&2
        return 1
      fi
      attempt_index=$((attempt_index + 1))
      continue
    fi
    if [[ "$result" == *"no_retry" ]]; then
      echo "Surface attempt failed closed: $stage ($result)" >&2
      if (( command_status == 0 )); then
        return 86
      fi
      return "$command_status"
    fi
    if [[ ! -s "$terminal" || ! -s "$sidecar" ]]; then
      echo "Surface stage did not produce an audited terminal: $stage" >&2
      return 1
    fi
    return 0
  done
}

COMMON_CACHE_ARGS=(
  radio_gs/scripts/build_scannet_surface_region_cache.py
  --dataset-root "$DATASET_ROOT"
  --exclude-scene-files "$PFIR_DEV,$PFIR_TEST"
  --exclude-scene-names "$EXCLUDED_SCENES"
  --frames-per-scene 8
  --regions-per-scene 12
  --region-radii 0.25,0.45,0.70
  --graph-neighbors 16
  --voxel-size 0.04
  --depth-stride 8
  --min-tokens 24
  --max-tokens 256
  --token-subsampling core_context_radial_stratified_v1
  --core-token-fraction 0.60
  --path-cost-mode appearance_boundary_geometric
  --path-affinity-floor 1e-4
  --min-visible-tokens 12
  --teacher-views 3
  --teacher-region-candidate-limit 4096
  --adaptor-batch-size "$ADAPTOR_BATCH_SIZE"
  --affinity-dim 256
  --radio-resolution 384
  --radio-thermal-pacing-seconds-per-image \
  "$RADIO_THERMAL_PACING_SECONDS_PER_IMAGE"
  --seed 0
  --device cuda:0
  --radio-repo "$RADIO_REPO"
  --radio-version "$RADIO_VERSION"
  --radio-checkpoint "$RADIO_CHECKPOINT"
)

control_path() {
  local role="$1" shard="$2"
  if [[ "$role" == "train" && "$shard" -lt 2 ]]; then
    printf '%s/caches/control_c256_geometric/train_shard%s.pt\n' \
      "$EXTERNAL_CONTROL_ROOT" "$shard"
  else
    printf '%s/caches/control_c256_geometric/%s_shard%s.pt\n' \
      "$OUTPUT_ROOT" "$role" "$shard"
  fi
}

build_local_control() {
  local role="$1" shard="$2" shard_count split output resume_dir
  local fastpath_args=()
  if [[ "$role" == "train" ]]; then
    split="$TRAIN_SPLIT"
    shard_count=4
  else
    split="$VALIDATION_SPLIT"
    shard_count=2
  fi
  output="$(control_path "$role" "$shard")"
  resume_dir="$CACHE_RESUME_ROOT/control_c256_geometric/${role}_shard${shard}"
  if [[ "$SURFACE_INTERMEDIATE_FASTPATH" == "required_local_shards" ]]; then
    fastpath_args=(
      --scene-intermediate-output-root
      "$INTERMEDIATE_ROOT/control_c256_geometric/${role}_shard${shard}"
    )
  fi
  run_stage "cache_control_c256_geometric_${role}_${shard}" "$output" \
    "${COMMON_CACHE_ARGS[@]}" \
    --split-file "$split" --split-role "$role" \
    --shard-count "$shard_count" --shard-index "$shard" --max-scenes 100 \
    --context-ratio 1.20 --token-candidate-limit 256 \
    --region-reliability-mode geometric_mean_observation_agreement \
    "${fastpath_args[@]}" \
    --resume-dir "$resume_dir" --output "$output"
  if [[ "$SURFACE_INTERMEDIATE_FASTPATH" == "required_local_shards" ]]; then
    local intermediate_manifest
    intermediate_manifest="$INTERMEDIATE_ROOT/control_c256_geometric/${role}_shard${shard}/manifest.json"
    if [[ ! -s "$intermediate_manifest" || -L "$intermediate_manifest" ]]; then
      echo "required intermediate manifest is missing: $intermediate_manifest" >&2
      return 1
    fi
  fi
}

# Reuse exact legacy train0/1 controls.  Build only the four missing controls.
build_local_control train 2
run_authority audit-canary --manifest "$MANIFEST" --output "$CANARY_REPORT" >/dev/null
if [[ "$SURFACE_CANARY_RESUME" != "1" ]]; then
  echo "Surface p6 full-shard canary passed and stopped fail-closed." >&2
  echo "Audit $CANARY_REPORT, then resume with SURFACE_CANARY_RESUME=1." >&2
  exit 3
fi
build_local_control train 3
build_local_control validation 0
build_local_control validation 1

build_c1024() {
  local role="$1" shard="$2" shard_count split output resume_dir replay
  local fastpath_args=()
  local replay_authority_args=()
  if [[ "$role" == "train" ]]; then
    split="$TRAIN_SPLIT"
    shard_count=4
  else
    split="$VALIDATION_SPLIT"
    shard_count=2
  fi
  output="$OUTPUT_ROOT/caches/context_c1024_geometric/${role}_shard${shard}.pt"
  resume_dir="$CACHE_RESUME_ROOT/context_c1024_geometric/${role}_shard${shard}"
  replay="$(control_path "$role" "$shard")"
  if [[ "$role" == "train" && "$shard" -lt 2 ]]; then
    local replay_authority replay_authority_canonical replay_authority_sha
    local replay_authority_output
    local -a replay_authority_lines
    replay_authority="$REPLAY_AUTHORITY_ROOT/train_shard${shard}.json"
    if ! replay_authority_output="$(
      run_authority legacy-replay-authority \
        --manifest "$MANIFEST" --shard "$shard" \
        --output "$replay_authority" --output-format lines
    )"; then
      echo "legacy replay authority publication failed" >&2
      return 1
    fi
    mapfile -t replay_authority_lines <<<"$replay_authority_output"
    if ! replay_authority_canonical="$(realpath -e -- "$replay_authority")"; then
      echo "legacy replay authority path cannot be canonicalized" >&2
      return 1
    fi
    if (( ${#replay_authority_lines[@]} != 2 )) \
        || [[ "${replay_authority_lines[0]}" != "$replay_authority_canonical" ]] \
        || [[ ! "${replay_authority_lines[1]}" =~ ^[0-9a-f]{64}$ ]]; then
      echo "legacy replay authority did not return one exact path/SHA binding" >&2
      return 1
    fi
    replay_authority="$replay_authority_canonical"
    replay_authority_sha="${replay_authority_lines[1]}"
    replay_authority_args=(
      --teacher-replay-authority "$replay_authority"
      --teacher-replay-authority-sha256 "$replay_authority_sha"
    )
  fi
  if [[ "$SURFACE_INTERMEDIATE_FASTPATH" == "required_local_shards" \
        && ! ( "$role" == "train" && "$shard" -lt 2 ) ]]; then
    local intermediate_manifest intermediate_sha expected_intermediate_root
    local -a intermediate_binding_lines
    expected_intermediate_root="$INTERMEDIATE_ROOT/control_c256_geometric/${role}_shard${shard}"
    mapfile -t intermediate_binding_lines < <(
      run_authority intermediate-binding \
        --control-sidecar "${replay}.json" \
        --expected-root "$expected_intermediate_root" \
        --output-format lines
    )
    if (( ${#intermediate_binding_lines[@]} != 2 )); then
      echo "control report did not return one intermediate path/SHA binding" >&2
      return 1
    fi
    intermediate_manifest="${intermediate_binding_lines[0]}"
    intermediate_sha="${intermediate_binding_lines[1]}"
    fastpath_args=(
      --scene-intermediate-manifest "$intermediate_manifest"
      --scene-intermediate-manifest-sha256 "$intermediate_sha"
    )
  fi
  run_stage "cache_context_c1024_geometric_${role}_${shard}" "$output" \
    "${COMMON_CACHE_ARGS[@]}" \
    --split-file "$split" --split-role "$role" \
    --shard-count "$shard_count" --shard-index "$shard" --max-scenes 100 \
    --context-ratio 1.20 --token-candidate-limit 1024 \
    --region-reliability-mode geometric_mean_observation_agreement \
    --teacher-replay-cache "$replay" \
    "${replay_authority_args[@]}" \
    "${fastpath_args[@]}" \
    --resume-dir "$resume_dir" --output "$output"
}

for shard in 0 1 2 3; do
  build_c1024 train "$shard"
done
for shard in 0 1; do
  build_c1024 validation "$shard"
done

verify_run_contract pre_cache_pairing
run_authority verify-pairing \
  --manifest "$MANIFEST" --output "$PAIRING_REPORT" >/dev/null
verify_run_contract post_cache_pairing

IFS=',' read -r -a SEEDS <<<"$READOUT_SEEDS"
for pooling_mode in joint_attention_v1 core_context_separate_attention_v1; do
  for seed in "${SEEDS[@]}"; do
    model="$OUTPUT_ROOT/readouts/${pooling_mode}_seed${seed}.pt"
    run_stage "readout_${pooling_mode}_seed${seed}" "$model" \
      radio_gs/scripts/train_surface_region_summary_readout.py \
      --train-caches "$OUTPUT_ROOT/caches/context_c1024_geometric/train_shard*.pt" \
      --validation-caches "$OUTPUT_ROOT/caches/context_c1024_geometric/validation_shard*.pt" \
      --output "$model" \
      --hidden-dim 256 --epochs 60 --patience 10 --batch-size 16 \
      --learning-rate 2e-4 --weight-decay 1e-4 \
      --token-weight 0.25 --relation-weight 0.1 \
      --reliability-attention-mode log_prior \
      --context-pooling-mode "$pooling_mode" \
      --seed "$seed" --device cuda:0 \
      --radio-checkpoint "$RADIO_CHECKPOINT"
  done
done

verify_run_contract pre_attention_screen_report
run_authority finalize \
  --manifest "$MANIFEST" --pairing "$PAIRING_REPORT" \
  --output "$SCREEN_REPORT" >/dev/null
CUDA_VISIBLE_DEVICES="" \
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$CLOSURE_GUARD" verify-closure \
  --manifest "$MANIFEST" --phase final_before_completion \
  --full-checkpoint --attempt-root "$ATTEMPT_RECEIPT_ROOT" \
  --log-root "$LOG_ROOT" --report "$CLOSURE_FINAL_REPORT" >/dev/null
date -Iseconds >"$OUTPUT_ROOT/screen.complete"
