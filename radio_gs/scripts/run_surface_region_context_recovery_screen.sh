#!/usr/bin/env bash

# Query-free, scene-disjoint SurfaceRegion capacity screen. A fresh
# candidate-256/geometric control owns the fixed-core official teacher
# tensors. Every treatment replays those tensors exactly while changing only
# student context capacity and/or reliability semantics.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-1}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/pool/Datasets/ScanNet/data/tasks/scannet_frames_25k}"
TRAIN_SPLIT="${TRAIN_SPLIT:-$REPO_ROOT/paper/artifacts/scannet_surface_region_query_free_train_scenes_20260731.txt}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-$REPO_ROOT/paper/artifacts/scannet_surface_region_query_free_validation_scenes_20260731.txt}"
RADIO_REPO="${RADIO_REPO:-/root/RADIO}"
RADIO_VERSION="${RADIO_VERSION:-c-radio_v4-h}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/RADIO-GS/output/optimization_20260731/surface_fixed_teacher_replay_v2_gpu1_p2_canary71_closure}"
READOUT_SEEDS="${READOUT_SEEDS:-0,1,2}"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
CLOSURE_GUARD="$REPO_ROOT/radio_gs/scripts/surface_region_run_guard.py"
LOCK_SUPERVISOR="$REPO_ROOT/radio_gs/scripts/surface_gpu1_lock_supervisor.py"
GLOBAL_GPU1_LOCK="/root/RADIO-GS/output/.physical_gpu1.lock"
GPU1_SINGLETON_PROTOCOL="linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"
GPU_TELEMETRY_LOG="${GPU_TELEMETRY_LOG:-$OUTPUT_ROOT/gpu1_telemetry.csv}"
GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-75}"
GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-52}"
GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-1}"
GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-0}"
GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-0}"
GPU_PEER_INDEX="${GPU_PEER_INDEX:-0}"
GPU_PEER_PAUSE_TEMP_C="${GPU_PEER_PAUSE_TEMP_C:-77}"
GPU_PEER_RESUME_TEMP_C="${GPU_PEER_RESUME_TEMP_C:-75}"
GPU_PEER_QUIET_SECONDS="${GPU_PEER_QUIET_SECONDS:-0}"
GPU_PEER_MAX_POWER_W="${GPU_PEER_MAX_POWER_W:-0}"
GPU_PEER_MAX_MEMORY_MIB="${GPU_PEER_MAX_MEMORY_MIB:-0}"
GPU_PEER_MAX_UTIL_PCT="${GPU_PEER_MAX_UTIL_PCT:-100}"
GPU_PEER_ACTIVITY_ACTION="${GPU_PEER_ACTIVITY_ACTION:-terminate}"
GPU_PEER_INTERRUPT_EXIT_CODE=87
ADAPTOR_BATCH_SIZE="${ADAPTOR_BATCH_SIZE:-64}"
RADIO_THERMAL_PACING_SECONDS_PER_IMAGE="${RADIO_THERMAL_PACING_SECONDS_PER_IMAGE:-2.0}"
SURFACE_CANARY_RESUME="${SURFACE_CANARY_RESUME:-0}"

PFIR_DEV="$REPO_ROOT/radio_gs/benchmarks/scannet_pfir/split/scannet_pfir_small_v1_dev_candidates.txt"
PFIR_TEST="$REPO_ROOT/radio_gs/benchmarks/scannet_pfir/split/scannet_pfir_small_v1_test_candidates.txt"
EXCLUDED_SCENES="scene0000_00,scene0062_00,scene0070_00,scene0097_00,scene0140_00,scene0200_00,scene0347_00,scene0400_00,scene0590_00"
MANIFEST="$OUTPUT_ROOT/run_manifest.json"
PAIRING_REPORT="$OUTPUT_ROOT/cache_pairing.json"
SCREEN_REPORT="$OUTPUT_ROOT/query_free_screen.json"
CANARY_REPORT="$OUTPUT_ROOT/control_train_shard0_canary.json"
CLOSURE_FINAL_REPORT="$OUTPUT_ROOT/runtime_closure_final.json"
LOG_ROOT="$OUTPUT_ROOT/logs"
CACHE_RESUME_ROOT="$OUTPUT_ROOT/cache_scene_resume"
ATTEMPT_RECEIPT_ROOT="$OUTPUT_ROOT/stage_attempts"

for required in \
  "$DATASET_ROOT" "$TRAIN_SPLIT" "$VALIDATION_SPLIT" \
  "$RADIO_REPO" "$RADIO_CHECKPOINT" "$PFIR_DEV" "$PFIR_TEST" \
  "$THERMAL_GUARD" "$CLOSURE_GUARD" \
  "$LOCK_SUPERVISOR" \
  "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"; do
  if [[ ! -e "$required" ]]; then
    echo "missing SurfaceRegion screen input: $required" >&2
    exit 2
  fi
done
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
if [[ "${RADIO_GS_GPU1_LOCK_PATH:-}" != "$GLOBAL_GPU1_LOCK" ]]; then
  echo "physical GPU1 lock path was not inherited from the supervisor" >&2
  exit 2
fi
if [[ "${RADIO_GS_GPU1_SINGLETON_PROTOCOL:-}" \
      != "$GPU1_SINGLETON_PROTOCOL" ]]; then
  echo "physical GPU1 kernel singleton protocol was not inherited" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="" \
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$LOCK_SUPERVISOR" verify-inherited \
  --fd "$RADIO_GS_GPU1_LOCK_FD" \
  --singleton-fd "$RADIO_GS_GPU1_SINGLETON_FD" >/dev/null
if [[ "$GPU" != "1" ]]; then
  echo "this frozen screen is assigned to physical GPU1; got GPU=$GPU" >&2
  exit 2
fi
if [[ "$OUTPUT_ROOT" != /* ]]; then
  echo "OUTPUT_ROOT must be an absolute path" >&2
  exit 2
fi
if [[ "$SURFACE_CANARY_RESUME" != "0" \
      && "$SURFACE_CANARY_RESUME" != "1" ]]; then
  echo "SURFACE_CANARY_RESUME must be 0 or 1" >&2
  exit 2
fi
GPU_INFO=""
for candidate in /proc/driver/nvidia/gpus/*/information; do
  if [[ -r "$candidate" ]] \
    && [[ "$(awk '/Device Minor:/ {print $3}' "$candidate")" == "$GPU" ]]; then
    GPU_INFO="$candidate"
    break
  fi
done
if [[ -z "$GPU_INFO" ]]; then
  echo "physical GPU1 has no NVIDIA driver record" >&2
  exit 2
fi
GPU_BUS_ID="$(awk '/Bus Location:/ {print $3}' "$GPU_INFO")"
GPU_CONFIG="/sys/bus/pci/devices/$GPU_BUS_ID/config"
GPU_CONFIG_PREFIX="$(od -An -tx1 -N16 "$GPU_CONFIG" 2>/dev/null | tr -d ' \n')"
if [[ -z "$GPU_CONFIG_PREFIX" || "$GPU_CONFIG_PREFIX" =~ ^f+$ ]]; then
  echo "physical GPU1 PCIe configuration space is not responding" >&2
  exit 2
fi
if ! timeout --kill-after=2s 10s nvidia-smi -i "$GPU" >/dev/null; then
  echo "physical GPU1 is not currently usable" >&2
  exit 2
fi
GPU_UUID="$(
  nvidia-smi -i "$GPU" --query-gpu=uuid \
    --format=csv,noheader,nounits | tr -d '[:space:]'
)"
if [[ -z "$GPU_UUID" ]]; then
  echo "physical GPU1 returned an empty UUID" >&2
  exit 2
fi
GPU_COMPUTE_OWNERS="$(
  nvidia-smi --query-compute-apps=gpu_uuid,pid \
    --format=csv,noheader,nounits \
    | awk -F', *' -v uuid="$GPU_UUID" '$1 == uuid {print $2}' \
    | paste -sd, -
)"
if [[ -n "$GPU_COMPUTE_OWNERS" ]]; then
  echo "physical GPU1 already has compute owner(s): $GPU_COMPUTE_OWNERS" >&2
  exit 2
fi
if [[ "$READOUT_SEEDS" != "0,1,2" ]]; then
  echo "the frozen query-free screen requires READOUT_SEEDS=0,1,2" >&2
  exit 2
fi
if [[ "$GPU_PEER_ACTIVITY_ACTION" != "terminate" ]]; then
  echo "this screen requires GPU_PEER_ACTIVITY_ACTION=terminate to release CUDA" >&2
  exit 2
fi
if [[ ! "$ADAPTOR_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ADAPTOR_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "$RADIO_THERMAL_PACING_SECONDS_PER_IMAGE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "RADIO_THERMAL_PACING_SECONDS_PER_IMAGE must be finite and non-negative" >&2
  exit 2
fi

for directory in \
  "$OUTPUT_ROOT" "$LOG_ROOT" "$CACHE_RESUME_ROOT" \
  "$ATTEMPT_RECEIPT_ROOT"; do
  if [[ -L "$directory" ]]; then
    echo "refusing symlinked SurfaceRegion run directory: $directory" >&2
    exit 2
  fi
done
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CACHE_RESUME_ROOT" \
  "$ATTEMPT_RECEIPT_ROOT"

bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" - \
  "$DATASET_ROOT" "$TRAIN_SPLIT" "$VALIDATION_SPLIT" \
  "$RADIO_REPO" "$RADIO_VERSION" "$RADIO_CHECKPOINT" \
  "$PFIR_DEV" "$PFIR_TEST" "$EXCLUDED_SCENES" \
  "$READOUT_SEEDS" "$OUTPUT_ROOT" "$MANIFEST" "$0" \
  "$THERMAL_GUARD" "$GPU_MAX_TEMP_C" "$GPU_START_MAX_TEMP_C" \
  "$GPU_MAX_POWER_LIMIT_W" "$GPU_POLL_SECONDS" \
  "$GPU_SOFT_PAUSE_TEMP_C" "$GPU_SOFT_RESUME_TEMP_C" \
  "$GPU_PEER_INDEX" "$GPU_PEER_PAUSE_TEMP_C" \
  "$GPU_PEER_RESUME_TEMP_C" "$GPU_PEER_QUIET_SECONDS" \
  "$GPU_PEER_MAX_POWER_W" "$GPU_PEER_MAX_MEMORY_MIB" \
  "$GPU_PEER_MAX_UTIL_PCT" \
  "$GPU_PEER_ACTIVITY_ACTION" "$GPU_PEER_INTERRUPT_EXIT_CODE" \
  "$ADAPTOR_BATCH_SIZE" \
  "$RADIO_THERMAL_PACING_SECONDS_PER_IMAGE" \
  "$GPU_UUID" "$GLOBAL_GPU1_LOCK" "$GPU1_SINGLETON_PROTOCOL" \
  "$CANARY_REPORT" \
  "$GPU_TELEMETRY_LOG" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from radio_gs.scripts.surface_region_run_guard import build_runtime_closure

(
    dataset_root,
    train_split,
    validation_split,
    radio_repo,
    radio_version,
    radio_checkpoint,
    pfir_dev,
    pfir_test,
    excluded_scenes,
    readout_seeds,
    output_root,
    manifest_path,
    runner_path,
    thermal_guard,
    gpu_max_temp_c,
    gpu_start_max_temp_c,
    gpu_max_power_limit_w,
    gpu_poll_seconds,
    gpu_soft_pause_temp_c,
    gpu_soft_resume_temp_c,
    gpu_peer_index,
    gpu_peer_pause_temp_c,
    gpu_peer_resume_temp_c,
    gpu_peer_quiet_seconds,
    gpu_peer_max_power_w,
    gpu_peer_max_memory_mib,
    gpu_peer_max_util_pct,
    gpu_peer_activity_action,
    gpu_peer_interrupt_exit_code,
    adaptor_batch_size,
    radio_thermal_pacing_seconds_per_image,
    gpu_uuid,
    global_gpu1_lock,
    gpu1_singleton_protocol,
    canary_report,
    gpu_telemetry_log,
) = sys.argv[1:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


repo_root = Path(runner_path).resolve().parents[2]
manifest = Path(manifest_path)
output = Path(output_root).resolve()
checkpoint_sha256 = sha256(Path(radio_checkpoint))
implementation = {
    relative: sha256(repo_root / relative)
    for relative in (
        "radio_gs/scripts/build_scannet_surface_region_cache.py",
        "radio_gs/scripts/surface_region_scene_resume.py",
        "radio_gs/scripts/train_surface_region_summary_readout.py",
        "radio_gs/interfaces/surface_region_contract.py",
        "radio_gs/interfaces/surface_region_summary.py",
        "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
        "radio_gs/scripts/surface_gpu1_lock_supervisor.py",
    )
}
runtime_closure = build_runtime_closure(
    repo_root=repo_root,
    radio_repo=radio_repo,
    radio_checkpoint=radio_checkpoint,
    checkpoint_sha256=checkpoint_sha256,
)
payload = {
    "schema_version": 1,
    "screen": "surface-region-fixed-teacher-replay-v2",
    "source_snapshot_root": str(repo_root),
    "source_snapshot_import_root": runtime_closure["runtime_fingerprint"][
        "repository_import_root"
    ],
    "source_snapshot_tree_sha256": runtime_closure[
        "repository_sources"
    ]["digest"],
    "dataset_root": str(Path(dataset_root).resolve()),
    "train_split": str(Path(train_split).resolve()),
    "train_split_sha256": sha256(Path(train_split)),
    "validation_split": str(Path(validation_split).resolve()),
    "validation_split_sha256": sha256(Path(validation_split)),
    "radio_repo": str(Path(radio_repo).resolve()),
    "radio_version": radio_version,
    "radio_checkpoint": str(Path(radio_checkpoint).resolve()),
    "radio_checkpoint_sha256": checkpoint_sha256,
    "exclusion_files": {
        str(Path(pfir_dev).resolve()): sha256(Path(pfir_dev)),
        str(Path(pfir_test).resolve()): sha256(Path(pfir_test)),
    },
    "excluded_scene_names": sorted(excluded_scenes.split(",")),
    "cache_contract": {
        "train_shards": 4,
        "validation_shards": 2,
        "frames_per_scene": 8,
        "regions_per_scene": 12,
        "region_radii_m": [0.25, 0.45, 0.70],
        "maximum_tokens": 256,
        "teacher_region_candidate_limit": 4096,
        "path_cost_mode": "appearance_boundary_geometric",
        "path_affinity_floor": 1e-4,
        "token_subsampling": "core_context_radial_stratified_v1",
        "core_token_fraction": 0.60,
        "teacher_views": 3,
        "adaptor_batch_size": int(adaptor_batch_size),
        "radio_thermal_pacing_seconds_per_image": float(
            radio_thermal_pacing_seconds_per_image
        ),
        "durable_scene_resume": {
            "artifact_type": "surface-region-scene-resume-contract-v1",
            "schema_version": 1,
            "root": str(output / "cache_scene_resume"),
            "partial_suffix": ".surface-scene.partial",
            "terminal_suffix": ".surface-scene.complete.json",
            "strict_complete_cli_and_input_binding": True,
            "external_sha_terminal": True,
            "weights_only_same_descriptor_load": True,
            "restore_python_rng_predecessor_successor": True,
            "partial_matches_final_pt_promotion_glob": False,
        },
        "seed": 0,
    },
    "candidates": {
        "control_c256_geometric": {
            "context_ratio": 1.20,
            "token_candidate_limit": 256,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "fresh_official_runtime",
        },
        "context_c1024_geometric": {
            "context_ratio": 1.20,
            "token_candidate_limit": 1024,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "exact_cache_replay",
        },
        "context_c1024_uniform": {
            "context_ratio": 1.20,
            "token_candidate_limit": 1024,
            "reliability": "uniform_valid",
            "teacher_source": "exact_cache_replay",
        },
        "core_c1024_geometric": {
            "context_ratio": 1.00,
            "token_candidate_limit": 1024,
            "reliability": "geometric_mean_observation_agreement",
            "teacher_source": "exact_cache_replay",
        },
    },
    "readout_contract": {
        "hidden_dim": 256,
        "epochs": 60,
        "patience": 10,
        "batch_size": 16,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "token_weight": 0.25,
        "relation_weight": 0.1,
        "reliability_attention_mode": "log_prior",
        "seeds": [int(value) for value in readout_seeds.split(",")],
    },
    "thermal_safety_contract": {
        "guard": str(Path(thermal_guard).resolve()),
        "guard_sha256": sha256(Path(thermal_guard)),
        "physical_gpu": 1,
        "maximum_temperature_c": int(gpu_max_temp_c),
        "maximum_start_temperature_c": int(gpu_start_max_temp_c),
        "maximum_power_limit_w": float(gpu_max_power_limit_w),
        "poll_seconds": int(gpu_poll_seconds),
        "soft_pause_temperature_c": int(gpu_soft_pause_temp_c),
        "soft_resume_temperature_c": int(gpu_soft_resume_temp_c),
        "peer_gpu": int(gpu_peer_index),
        "peer_pause_temperature_c": int(gpu_peer_pause_temp_c),
        "peer_resume_temperature_c": int(gpu_peer_resume_temp_c),
        "peer_quiet_seconds_before_launch": int(gpu_peer_quiet_seconds),
        "peer_max_power_w": float(gpu_peer_max_power_w),
        "peer_max_memory_mib": int(gpu_peer_max_memory_mib),
        "peer_max_utilization_pct": int(gpu_peer_max_util_pct),
        "peer_activity_action": gpu_peer_activity_action,
        "peer_activity_interrupt_exit_code": int(
            gpu_peer_interrupt_exit_code
        ),
        "peer_activity_retry_policy": (
            "retry_only_exit_87_after_cuda_release_with_fresh_"
            "quiet_owner_closure_checks_v1"
        ),
        "gpu_uuid": gpu_uuid,
        "global_gpu_lock": str(Path(global_gpu1_lock).resolve()),
        "global_gpu_lock_inherited_fd_verified": True,
        "global_gpu_kernel_singleton_protocol": gpu1_singleton_protocol,
        "global_gpu_kernel_singleton_inherited_fd_verified": True,
    },
    "canary_contract": {
        "stage": "cache_control_c256_geometric_train_0",
        "terminal": "caches/control_c256_geometric/train_shard0.pt",
        "report": str(Path(canary_report).resolve()),
        "maximum_temperature_c": 71,
        "reject_thermal_or_telemetry_abort": True,
        "reject_kernel_xid_or_pcie_fault": True,
        "require_peer_pause_resume_pairing": True,
        "default_fail_closed_after_pass": True,
        "resume_environment_variable": "SURFACE_CANARY_RESUME=1",
    },
    "attempt_receipt_contract": {
        "artifact_type": "surface-region-stage-attempt-v1",
        "schema_version": 1,
        "root": str(output / "stage_attempts"),
        "log_root": str(output / "logs"),
        "telemetry_path": str(Path(gpu_telemetry_log).resolve()),
        "immutable_no_clobber": True,
        "telemetry_interval_sha256_and_line_count_recorded": True,
        "telemetry_intervals_strictly_non_overlapping": True,
        "kernel_journal_file_record_and_fault_count_recorded": True,
        "command_and_manifest_sha_recorded": True,
        "automatic_retry_requires": {
            "exit_code": int(gpu_peer_interrupt_exit_code),
            "telemetry_event_prefix": (
                "peer_activity_interrupt_release_cuda_"
            ),
            "exact_event_count": 1,
            "independent_gpu_release_postflight": [
                "same_uuid",
                "pcie_responsive",
                "no_compute_owner",
            ],
            "kernel_journal_capture_status": 0,
            "kernel_xid_or_pcie_fault_count": 0,
        },
        "all_other_exit_codes_retry": False,
        "retry_limit": None,
        "fresh_preflight_each_attempt": [
            "runtime_closure",
            "gpu_uuid_compute_owner",
            "thermal_guard_quiet_and_start_temperature",
        ],
    },
    "selection_contract": {
        "minimum_mean_score_gain": 0.001,
        "minimum_seed_wins": 2,
        "maximum_component_drop": 0.002,
        "uses_benchmark_queries": False,
    },
    "runner_sha256": sha256(Path(runner_path)),
    "implementation_sources": implementation,
    "runtime_closure": runtime_closure,
}
if manifest.is_file():
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit(
            "SurfaceRegion OUTPUT_ROOT belongs to another immutable run"
        )
else:
    existing = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    ]
    if existing:
        raise SystemExit(
            "SurfaceRegion OUTPUT_ROOT contains artifacts without a manifest"
        )
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
PY

verify_run_closure() {
  local phase="$1"
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$CLOSURE_GUARD" verify-closure \
    --manifest "$MANIFEST" \
    --phase "$phase" >/dev/null
}

assert_gpu_uuid_unowned() {
  local current_uuid owners
  current_uuid="$(
    nvidia-smi -i "$GPU" --query-gpu=uuid \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  if [[ "$current_uuid" != "$GPU_UUID" ]]; then
    echo "physical GPU1 UUID changed during the SurfaceRegion run" >&2
    return 2
  fi
  owners="$(
    nvidia-smi --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits \
      | awk -F', *' -v uuid="$GPU_UUID" '$1 == uuid {print $2}' \
      | paste -sd, -
  )"
  if [[ -n "$owners" ]]; then
    echo "physical GPU1 acquired unexpected compute owner(s): $owners" >&2
    return 2
  fi
}

RUN_STAGE_DID_RUN=0
RUN_STAGE_TELEMETRY_START_LINE=0
RUN_STAGE_TELEMETRY_END_LINE=0
RUN_STAGE_START_EPOCH=0
RUN_STAGE_END_EPOCH=0

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
  local receipt="$1"
  local stage="$2"
  local attempt_index="$3"
  local command_status="$4"
  local result="$5"
  local attempt_log="$6"
  local telemetry_start="$7"
  local telemetry_end="$8"
  local start_epoch="$9"
  shift 9
  local end_epoch="$1"
  local terminal="$2"
  local sidecar="$3"
  local kernel_capture_status="$4"
  local kernel_log="$5"
  local postflight_capture_status="$6"
  local postflight_report="$7"
  shift 7
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" - \
    "$receipt" "$MANIFEST" "$stage" "$attempt_index" \
    "$command_status" "$result" "$attempt_log" \
    "$GPU_TELEMETRY_LOG" "$telemetry_start" "$telemetry_end" \
    "$start_epoch" "$end_epoch" "$terminal" "$sidecar" \
    "$kernel_capture_status" "$kernel_log" \
    "$postflight_capture_status" "$postflight_report" \
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
    receipt,
    manifest,
    stage,
    attempt_index,
    command_status,
    result,
    attempt_log,
    telemetry,
    telemetry_start,
    telemetry_end,
    start_epoch,
    end_epoch,
    terminal,
    sidecar,
    kernel_capture_status,
    kernel_log,
    postflight_capture_status,
    postflight_report,
    peer_action,
    peer_interrupt_exit_code,
    *command,
) = sys.argv[1:]


def optional_record(raw_path: str):
    path = Path(raw_path)
    if not os.path.lexists(path):
        return None
    return file_record(path)


kernel_encoded, _, _ = _stable_artifact_bytes(
    kernel_log,
    label="SurfaceRegion attempt kernel journal",
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
        telemetry,
        start_line=int(telemetry_start),
        end_line=int(telemetry_end),
    ),
    "kernel_journal": {
        "start_epoch": int(start_epoch),
        "end_epoch": int(end_epoch),
        "capture_status": int(kernel_capture_status),
        "fault_count": len(kernel_fault_lines(kernel_encoded)),
        "file": file_record(kernel_log),
    },
    "gpu_release_postflight": (
        None
        if not postflight_report
        else {
            "capture_status": int(postflight_capture_status),
            "report": optional_record(postflight_report),
        }
    ),
    "terminal": optional_record(terminal),
    "sidecar": optional_record(sidecar),
    "peer_activity_action": peer_action,
    "peer_activity_interrupt_exit_code": int(peer_interrupt_exit_code),
}
write_frozen_json(receipt, payload)
PY
}

verify_existing_attempt_receipt() {
  local receipt="$1"
  local stage="$2"
  local attempt_index="$3"
  local attempt_log="$4"
  shift 4
  local command=(
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
    "$CLOSURE_GUARD" verify-attempt
    --manifest "$MANIFEST"
    --receipt "$receipt"
    --stage "$stage"
    --index "$attempt_index"
    --log "$attempt_log"
    --allowed-result
    peer_activity_interrupted_cuda_released_retry_authorized
  )
  local argument
  for argument in "$@"; do
    command+=("--command-arg=$argument")
  done
  CUDA_VISIBLE_DEVICES="" "${command[@]}" >/dev/null
}

run_stage() {
  local stage="$1"
  local terminal="$2"
  shift 2
  local sidecar="${terminal}.json"
  verify_run_closure "pre_${stage}"
  RUN_STAGE_DID_RUN=0
  if [[ -L "$terminal" || -L "$sidecar" ]]; then
    echo "refusing symlinked SurfaceRegion terminal: $terminal" >&2
    exit 1
  fi
  if [[ -s "$terminal" && -s "$sidecar" ]]; then
    RUN_STAGE_TELEMETRY_START_LINE=0
    RUN_STAGE_TELEMETRY_END_LINE="$(telemetry_line_count)"
    RUN_STAGE_START_EPOCH="$(stat -c '%Y' "$MANIFEST")"
    RUN_STAGE_END_EPOCH="$(stat -c '%Y' "$sidecar")"
    return
  fi
  if [[ -e "$terminal" || -L "$terminal" \
        || -e "$sidecar" || -L "$sidecar" ]]; then
    echo "partial SurfaceRegion terminal requires inspection: $terminal" >&2
    exit 1
  fi
  RUN_STAGE_DID_RUN=1
  RUN_STAGE_TELEMETRY_START_LINE="$(telemetry_line_count)"
  RUN_STAGE_START_EPOCH="$(date +%s)"
  local attempt_dir="$ATTEMPT_RECEIPT_ROOT/$stage"
  if [[ -L "$attempt_dir" ]]; then
    echo "refusing symlinked stage attempt directory: $attempt_dir" >&2
    return 1
  fi
  mkdir -p "$attempt_dir"
  local attempt_index=1
  while true; do
    local attempt_tag receipt attempt_log kernel_log postflight_report
    local receipt_present log_present kernel_present postflight_present
    attempt_tag="$(printf '%06d' "$attempt_index")"
    receipt="$attempt_dir/attempt_${attempt_tag}.json"
    attempt_log="$LOG_ROOT/${stage}.attempt_${attempt_tag}.log"
    kernel_log="$LOG_ROOT/${stage}.attempt_${attempt_tag}.kernel.log"
    postflight_report="$LOG_ROOT/${stage}.attempt_${attempt_tag}.gpu_release_postflight.json"
    receipt_present=0
    log_present=0
    kernel_present=0
    postflight_present=0
    [[ -e "$receipt" || -L "$receipt" ]] && receipt_present=1
    [[ -e "$attempt_log" || -L "$attempt_log" ]] && log_present=1
    [[ -e "$kernel_log" || -L "$kernel_log" ]] && kernel_present=1
    [[ -e "$postflight_report" || -L "$postflight_report" ]] \
      && postflight_present=1
    if (( receipt_present && log_present )); then
      verify_existing_attempt_receipt \
        "$receipt" "$stage" "$attempt_index" "$attempt_log" "$@"
      attempt_index=$((attempt_index + 1))
      continue
    fi
    if (( receipt_present || log_present || kernel_present || postflight_present )); then
      echo "half-published SurfaceRegion attempt requires inspection: $stage/$attempt_tag" >&2
      return 1
    fi

    verify_run_closure "pre_${stage}_attempt_${attempt_tag}"
    assert_gpu_uuid_unowned
    local attempt_telemetry_start attempt_telemetry_end
    local attempt_start_epoch attempt_end_epoch command_status result
    local kernel_capture_status postflight_capture_status peer_event_count
    attempt_telemetry_start="$(telemetry_line_count)"
    attempt_start_epoch="$(date +%s)"
    command_status=0
    GPU="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
      GPU_TELEMETRY_LOG="$GPU_TELEMETRY_LOG" \
      GPU_MAX_TEMP_C="$GPU_MAX_TEMP_C" \
      GPU_START_MAX_TEMP_C="$GPU_START_MAX_TEMP_C" \
      GPU_MAX_POWER_LIMIT_W="$GPU_MAX_POWER_LIMIT_W" \
      GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
      GPU_SOFT_PAUSE_TEMP_C="$GPU_SOFT_PAUSE_TEMP_C" \
      GPU_SOFT_RESUME_TEMP_C="$GPU_SOFT_RESUME_TEMP_C" \
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
    attempt_end_epoch="$(date +%s)"
    attempt_telemetry_end="$(telemetry_line_count)"
    RUN_STAGE_END_EPOCH="$attempt_end_epoch"
    RUN_STAGE_TELEMETRY_END_LINE="$attempt_telemetry_end"
    kernel_capture_status=0
    CUDA_VISIBLE_DEVICES="" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$CLOSURE_GUARD" capture-kernel-journal \
      --start-epoch "$attempt_start_epoch" \
      --end-epoch "$attempt_end_epoch" \
      --output "$kernel_log" >/dev/null \
      || kernel_capture_status=$?
    postflight_capture_status=-1
    peer_event_count="$(attempt_peer_release_interrupt_count \
      "$attempt_telemetry_start" "$attempt_telemetry_end")"
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
    verify_run_closure "post_${stage}_attempt_${attempt_tag}"

    if (( kernel_capture_status != 0 )); then
      result="attempt_evidence_failed_no_retry"
    elif (( command_status == 0 )); then
      result="completed"
    elif (( command_status == GPU_PEER_INTERRUPT_EXIT_CODE )) \
        && (( peer_event_count == 1 )); then
      if (( postflight_capture_status == 0 )); then
        result="peer_activity_interrupted_cuda_released_retry_authorized"
      else
        result="peer_activity_interrupted_cuda_release_unverified_no_retry"
      fi
    else
      result="failed_no_retry"
    fi
    write_attempt_receipt \
      "$receipt" "$stage" "$attempt_index" "$command_status" \
      "$result" "$attempt_log" \
      "$attempt_telemetry_start" "$attempt_telemetry_end" \
      "$attempt_start_epoch" "$attempt_end_epoch" \
      "$terminal" "$sidecar" \
      "$kernel_capture_status" "$kernel_log" \
      "$postflight_capture_status" "$postflight_report" "$@"

    if [[ "$result" == "peer_activity_interrupted_cuda_released_retry_authorized" ]]; then
      if [[ -s "$terminal" && -s "$sidecar" ]]; then
        return 0
      fi
      if [[ -e "$terminal" || -L "$terminal" \
            || -e "$sidecar" || -L "$sidecar" ]]; then
        echo "peer interruption left a half-published final terminal: $stage" >&2
        return 1
      fi
      attempt_index=$((attempt_index + 1))
      continue
    fi
    if [[ "$result" == "peer_activity_interrupted_cuda_release_unverified_no_retry" \
          || "$result" == "attempt_evidence_failed_no_retry" ]]; then
      echo "SurfaceRegion attempt release/evidence audit failed closed: $stage" >&2
      return 86
    fi
    if (( command_status != 0 )); then
      echo "SurfaceRegion stage failed with status $command_status: $stage" >&2
      return "$command_status"
    fi
    if [[ ! -s "$terminal" || ! -s "$sidecar" ]]; then
      echo "SurfaceRegion stage did not produce an audited terminal: $stage" >&2
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

audit_control_shard0_canary() {
  local terminal="$1"
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$CLOSURE_GUARD" audit-canary \
    --manifest "$MANIFEST" \
    --telemetry "$GPU_TELEMETRY_LOG" \
    --terminal "$terminal" \
    --report "$CANARY_REPORT" \
    --start-line "$RUN_STAGE_TELEMETRY_START_LINE" \
    --end-line "$RUN_STAGE_TELEMETRY_END_LINE" \
    --start-epoch "$RUN_STAGE_START_EPOCH" \
    --end-epoch "$RUN_STAGE_END_EPOCH" >/dev/null
  if [[ "$SURFACE_CANARY_RESUME" != "1" ]]; then
    echo "SurfaceRegion full-shard canary passed and stopped fail-closed." >&2
    echo "Audit $CANARY_REPORT, then resume with SURFACE_CANARY_RESUME=1." >&2
    exit 3
  fi
}

build_candidate() {
  local candidate="$1"
  local context_ratio="$2"
  local candidate_limit="$3"
  local reliability="$4"
  local replay="$5"
  local role shard_count split shard
  for role in train validation; do
    if [[ "$role" == "train" ]]; then
      shard_count=4
      split="$TRAIN_SPLIT"
    else
      shard_count=2
      split="$VALIDATION_SPLIT"
    fi
    for ((shard=0; shard<shard_count; shard++)); do
      local output="$OUTPUT_ROOT/caches/$candidate/${role}_shard${shard}.pt"
      local resume_dir="$CACHE_RESUME_ROOT/$candidate/${role}_shard${shard}"
      local replay_args=()
      if [[ "$replay" == "yes" ]]; then
        replay_args=(
          --teacher-replay-cache
          "$OUTPUT_ROOT/caches/control_c256_geometric/${role}_shard${shard}.pt"
        )
      fi
      run_stage "cache_${candidate}_${role}_${shard}" "$output" \
        "${COMMON_CACHE_ARGS[@]}" \
        --split-file "$split" \
        --split-role "$role" \
        --shard-count "$shard_count" \
        --shard-index "$shard" \
        --max-scenes 100 \
        --context-ratio "$context_ratio" \
        --token-candidate-limit "$candidate_limit" \
        --region-reliability-mode "$reliability" \
        "${replay_args[@]}" \
        --resume-dir "$resume_dir" \
        --output "$output"
      if [[ "$candidate" == "control_c256_geometric" \
            && "$role" == "train" && "$shard" == "0" ]]; then
        audit_control_shard0_canary "$output"
      fi
    done
  done
}

build_candidate control_c256_geometric 1.20 256 \
  geometric_mean_observation_agreement no
build_candidate context_c1024_geometric 1.20 1024 \
  geometric_mean_observation_agreement yes
build_candidate context_c1024_uniform 1.20 1024 uniform_valid yes
build_candidate core_c1024_geometric 1.00 1024 \
  geometric_mean_observation_agreement yes

verify_run_closure "pre_cache_pairing"
bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" - \
  "$OUTPUT_ROOT" "$MANIFEST" "$PAIRING_REPORT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

from radio_gs.utils.immutable_artifacts import load_torch_mapping

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
output = Path(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
candidates = manifest["candidates"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_teacher_identity(record: dict) -> dict:
    return {
        key: record[key]
        for key in (
            "region_id",
            "scene",
            "seed",
            "physical_radius_m",
            "teacher_views",
            "teacher_medoid",
            "teacher_region_tokens",
            "teacher_support_sha256",
            "teacher_region_saturated",
            "teacher_target_sha256",
        )
    }


rows = []
for role, count in (("train", 4), ("validation", 2)):
    for shard in range(count):
        control_path = (
            root / "caches" / "control_c256_geometric"
            / f"{role}_shard{shard}.pt"
        )
        control, _, _ = load_torch_mapping(
            control_path,
            map_location="cpu",
            label="SurfaceRegion control cache",
        )
        control_meta = control["metadata"]
        if (
            control_meta["teacher_target_source"]
            != "fresh_official_runtime"
            or control_meta["failed_scenes"]
            or control_meta["teacher_regions_saturated"] != 0
        ):
            raise SystemExit(f"{control_path}: invalid fresh control")
        control_records = [
            record_teacher_identity(value)
            for value in control_meta["region_records"]
        ]
        for candidate, specification in candidates.items():
            path = root / "caches" / candidate / f"{role}_shard{shard}.pt"
            payload = (
                control
                if candidate == "control_c256_geometric"
                else load_torch_mapping(
                    path,
                    map_location="cpu",
                    label="SurfaceRegion treatment cache",
                )[0]
            )
            metadata = payload["metadata"]
            contract = metadata["region_contract"]
            candidate_limit = int(
                contract.get(
                    "token_candidate_limit",
                    contract["maximum_tokens"],
                )
            )
            if (
                float(contract["context_ratio"])
                != float(specification["context_ratio"])
                or candidate_limit
                != int(specification["token_candidate_limit"])
                or contract["reliability_semantics"]
                != specification["reliability"]
                or metadata["teacher_target_source"]
                != specification["teacher_source"]
                or metadata["teacher_target_protocol_sha256"]
                != control_meta["teacher_target_protocol_sha256"]
                or metadata["teacher_region_contract_sha256"]
                != control_meta["teacher_region_contract_sha256"]
                or metadata["scene_names"] != control_meta["scene_names"]
                or metadata["scene_region_counts"]
                != control_meta["scene_region_counts"]
                or metadata["failed_scenes"]
                or metadata["teacher_regions_saturated"] != 0
            ):
                raise SystemExit(
                    f"{path}: cache protocol differs from the screen manifest"
                )
            records = [
                record_teacher_identity(value)
                for value in metadata["region_records"]
            ]
            if records != control_records:
                raise SystemExit(
                    f"{path}: fixed teacher identities differ from control"
                )
            for key in (
                "official_summary_tokens",
                "official_crop_summaries",
                "teacher_mask",
            ):
                if not torch.equal(payload[key], control[key]):
                    raise SystemExit(
                        f"{path}: {key} is not an exact control replay"
                    )
            rows.append(
                {
                    "candidate": candidate,
                    "role": role,
                    "shard": shard,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "regions": len(records),
                    "teacher_target_protocol_sha256": metadata[
                        "teacher_target_protocol_sha256"
                    ],
                }
            )
            if payload is not control:
                del payload
        del control

report = {
    "schema_version": 1,
    "status": "exact_teacher_replay_verified",
    "run_manifest": str(manifest_path.resolve()),
    "run_manifest_sha256": sha256(manifest_path),
    "caches": rows,
    "benchmark_queries_opened": False,
    "benchmark_masks_opened": False,
}
temporary = output.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)
temporary.replace(output)
print(json.dumps(report, indent=2))
PY
verify_run_closure "post_cache_pairing"

IFS=',' read -r -a SEEDS <<<"$READOUT_SEEDS"
CANDIDATES=(
  control_c256_geometric
  context_c1024_geometric
  context_c1024_uniform
  core_c1024_geometric
)
for candidate in "${CANDIDATES[@]}"; do
  train_caches="$OUTPUT_ROOT/caches/$candidate/train_shard*.pt"
  validation_caches="$OUTPUT_ROOT/caches/$candidate/validation_shard*.pt"
  for seed in "${SEEDS[@]}"; do
    model="$OUTPUT_ROOT/readouts/${candidate}_seed${seed}.pt"
    run_stage "readout_${candidate}_seed${seed}" "$model" \
      radio_gs/scripts/train_surface_region_summary_readout.py \
      --train-caches "$train_caches" \
      --validation-caches "$validation_caches" \
      --output "$model" \
      --hidden-dim 256 \
      --epochs 60 \
      --patience 10 \
      --batch-size 16 \
      --learning-rate 2e-4 \
      --weight-decay 1e-4 \
      --token-weight 0.25 \
      --relation-weight 0.1 \
      --reliability-attention-mode log_prior \
      --seed "$seed" \
      --device cuda:0 \
      --radio-checkpoint "$RADIO_CHECKPOINT"
  done
done

verify_run_closure "pre_screen_report"
bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" - \
  "$OUTPUT_ROOT" "$MANIFEST" "$PAIRING_REPORT" "$SCREEN_REPORT" <<'PY'
import hashlib
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
pairing_path = Path(sys.argv[3])
output = Path(sys.argv[4])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
seeds = manifest["readout_contract"]["seeds"]
candidates = list(manifest["candidates"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


rows = {}
for candidate in candidates:
    reports = []
    for seed in seeds:
        model = root / "readouts" / f"{candidate}_seed{seed}.pt"
        report_path = model.with_suffix(model.suffix + ".json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append(
            {
                "seed": seed,
                "checkpoint": str(model.resolve()),
                "checkpoint_sha256": sha256(model),
                "best_epoch": report["best_epoch"],
                "best_selection_score": report["best_selection_score"],
                "selection_score_delta": report["selection_score_delta"],
                "validation": report["validation"],
            }
        )
    rows[candidate] = {
        "seeds": reports,
        "mean_selection_score": statistics.fmean(
            value["best_selection_score"] for value in reports
        ),
        "mean_validation": {
            key: statistics.fmean(
                value["validation"][key] for value in reports
            )
            for key in (
                "summary_token_cosine",
                "mean_descriptor_cosine",
                "all_view_descriptor_cosine",
            )
        },
    }

control_name = "control_c256_geometric"
control = rows[control_name]
minimum_gain = manifest["selection_contract"][
    "minimum_mean_score_gain"
]
minimum_wins = manifest["selection_contract"]["minimum_seed_wins"]
maximum_drop = manifest["selection_contract"][
    "maximum_component_drop"
]
eligible = []
for candidate, values in rows.items():
    values["mean_score_gain_over_control"] = (
        values["mean_selection_score"]
        - control["mean_selection_score"]
    )
    values["seed_wins_over_control"] = sum(
        current["best_selection_score"]
        > reference["best_selection_score"]
        for current, reference in zip(
            values["seeds"],
            control["seeds"],
        )
    )
    values["component_drops_from_control"] = {
        key: control["mean_validation"][key]
        - values["mean_validation"][key]
        for key in control["mean_validation"]
    }
    values["eligible_for_query_free_promotion"] = (
        candidate != control_name
        and values["mean_score_gain_over_control"] >= minimum_gain
        and values["seed_wins_over_control"] >= minimum_wins
        and max(values["component_drops_from_control"].values())
        <= maximum_drop
    )
    if values["eligible_for_query_free_promotion"]:
        eligible.append(candidate)

selected = (
    max(
        eligible,
        key=lambda name: (
            rows[name]["mean_selection_score"],
            rows[name]["seed_wins_over_control"],
            name,
        ),
    )
    if eligible
    else control_name
)
report = {
    "schema_version": 1,
    "selection_status": (
        "query_free_candidate_selected_benchmark_gate_still_closed"
        if selected != control_name
        else "query_free_control_retained"
    ),
    "selected_candidate": selected,
    "run_manifest": str(manifest_path.resolve()),
    "run_manifest_sha256": sha256(manifest_path),
    "cache_pairing_report": str(pairing_path.resolve()),
    "cache_pairing_report_sha256": sha256(pairing_path),
    "candidates": rows,
    "benchmark_queries_opened": False,
    "benchmark_masks_opened": False,
    "next_gate": (
        "freeze the selected query-free readout, then evaluate benchmarks "
        "without changing graph, unary, score, or connected-selection rules"
    ),
}
temporary = output.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)
temporary.replace(output)
print(json.dumps(report, indent=2))
PY

CUDA_VISIBLE_DEVICES="" \
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$CLOSURE_GUARD" verify-closure \
  --manifest "$MANIFEST" \
  --phase final_before_completion \
  --full-checkpoint \
  --attempt-root "$ATTEMPT_RECEIPT_ROOT" \
  --log-root "$LOG_ROOT" \
  --report "$CLOSURE_FINAL_REPORT" >/dev/null
date -Iseconds >"$OUTPUT_ROOT/screen.complete"
