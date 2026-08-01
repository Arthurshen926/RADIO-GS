#!/usr/bin/env bash

# Diagnostic NVOS queue for the hard-seed-anchored registered-region-v3 readout.
# The candidate changes only prompt compilation/readout over the already
# frozen canonical-mpr-v3 fields. It does not retrain geometry or capability
# features and never selects a stage or threshold from target masks.

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
cd "$REPO_ROOT"

MAIN_OUTPUT_ROOT="/root/RADIO-GS/output"
GLOBAL_GPU1_LOCK="$MAIN_OUTPUT_ROOT/.physical_gpu1.lock"
LOCK_SUPERVISOR="$REPO_ROOT/radio_gs/scripts/surface_gpu1_lock_supervisor.py"
GPU1_AUTHORITY="$REPO_ROOT/radio_gs/scripts/nvos_registered_region_v3_authority.py"
GPU1_SINGLETON_PROTOCOL="linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"
# This frozen host assignment is deliberately not environment-overridable.
EXPECTED_GPU1_UUID="GPU-0eac2c76-4004-49eb-bc0c-a9a30aec041a"
EXPECTED_GPU1_PROC_BUS_ID="0000:82:00.0"
EXPECTED_GPU1_NVIDIA_BUS_ID="00000000:82:00.0"
FROZEN_CUDA_DEVICE_ORDER="PCI_BUS_ID"
FROZEN_PYTHONDONTWRITEBYTECODE="1"
FROZEN_NUMBA_CACHE_DIR="/root/.cache/radio_gs/numba"
FROZEN_GPU_OWNER_PID_NAMESPACE_MODE="exclusive-singleton-after-clear-v1"

if [[ -n "${CUDA_DEVICE_ORDER+x}" \
      && "$CUDA_DEVICE_ORDER" != "$FROZEN_CUDA_DEVICE_ORDER" ]]; then
  echo "refusing non-frozen CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}" >&2
  exit 2
fi
if [[ -n "${PYTHONDONTWRITEBYTECODE+x}" \
      && "$PYTHONDONTWRITEBYTECODE" != "$FROZEN_PYTHONDONTWRITEBYTECODE" ]]; then
  echo "refusing non-frozen PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE}" >&2
  exit 2
fi
if [[ -n "${NUMBA_CACHE_DIR+x}" \
      && "$NUMBA_CACHE_DIR" != "$FROZEN_NUMBA_CACHE_DIR" ]]; then
  echo "refusing non-frozen NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}" >&2
  exit 2
fi
if [[ -n "${GPU_OWNER_PID_NAMESPACE_MODE+x}" \
      && "$GPU_OWNER_PID_NAMESPACE_MODE" \
      != "$FROZEN_GPU_OWNER_PID_NAMESPACE_MODE" ]]; then
  echo "refusing non-frozen GPU_OWNER_PID_NAMESPACE_MODE=${GPU_OWNER_PID_NAMESPACE_MODE}" >&2
  exit 2
fi
if [[ -n "${NVIDIA_VISIBLE_DEVICES+x}" \
      && "$NVIDIA_VISIBLE_DEVICES" != "$EXPECTED_GPU1_UUID" ]]; then
  echo "refusing non-frozen NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}" >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" \
      && "$CUDA_VISIBLE_DEVICES" != "$EXPECTED_GPU1_UUID" ]]; then
  echo "refusing numeric or foreign CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
export CUDA_DEVICE_ORDER="$FROZEN_CUDA_DEVICE_ORDER"
export NVIDIA_VISIBLE_DEVICES="$EXPECTED_GPU1_UUID"
export PYTHONDONTWRITEBYTECODE="$FROZEN_PYTHONDONTWRITEBYTECODE"
export NUMBA_CACHE_DIR="$FROZEN_NUMBA_CACHE_DIR"
export GPU_OWNER_PID_NAMESPACE_MODE="$FROZEN_GPU_OWNER_PID_NAMESPACE_MODE"
mkdir -p "$FROZEN_NUMBA_CACHE_DIR"
if [[ -L "$FROZEN_NUMBA_CACHE_DIR" \
      || "$(readlink -f -- "$FROZEN_NUMBA_CACHE_DIR")" \
      != "$FROZEN_NUMBA_CACHE_DIR" ]]; then
  echo "frozen NUMBA_CACHE_DIR is not a real external directory" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="" \
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$GPU1_AUTHORITY" verify-readonly-snapshot \
  --repo-root "$REPO_ROOT" >/dev/null

GPU="${GPU:-1}"
SOURCE_ROOT="${SOURCE_ROOT:-$MAIN_OUTPUT_ROOT/evaluation_closeout_20260716/canonical_mpr_v3_nvos8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN_OUTPUT_ROOT/optimization_20260731/nvos_registered_region_v3}"
QUEUE_PLAN="${QUEUE_PLAN:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/gaussfm_jobs/nvos_strict_unseen_v1/queue_plan.json}"
MANIFEST="${MANIFEST:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
PARENT_ASSET_MANIFEST="${PARENT_ASSET_MANIFEST:-$MAIN_OUTPUT_ROOT/optimization_20260731/nvos_registered_region_v2/run_manifest.json}"
CANDIDATE_CONTRACT="${CANDIDATE_CONTRACT:-$REPO_ROOT/paper/artifacts/nvos_registered_region_v3_candidate_20260731.yaml}"
SCENE_NAMES="${SCENE_NAMES:-fern flower fortress horns_center horns_left leaves orchids trex}"
CONTINUATION_SCREEN="$REPO_ROOT/radio_gs/scripts/screen_nvos_registered_region_v3_continuation.py"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"
GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-65}"
GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-3}"
GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-75}"
GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-70}"
GPU_PEER_INDEX="${GPU_PEER_INDEX:-}"
GPU_PEER_PAUSE_TEMP_C="${GPU_PEER_PAUSE_TEMP_C:-0}"
GPU_PEER_RESUME_TEMP_C="${GPU_PEER_RESUME_TEMP_C:-0}"
GPU_PEER_QUIET_SECONDS="${GPU_PEER_QUIET_SECONDS:-0}"
GPU_PEER_MAX_POWER_W="${GPU_PEER_MAX_POWER_W:-0}"
GPU_PEER_MAX_MEMORY_MIB="${GPU_PEER_MAX_MEMORY_MIB:-0}"
GPU_PEER_MAX_UTIL_PCT="${GPU_PEER_MAX_UTIL_PCT:-100}"

if [[ ! -f "$LOCK_SUPERVISOR" || -L "$LOCK_SUPERVISOR" \
      || ! -f "$GPU1_AUTHORITY" || -L "$GPU1_AUTHORITY" ]]; then
  echo "NVOS v3 source snapshot lacks a regular lock/closure authority" >&2
  exit 2
fi
# A snapshot has no independent GPU namespace.  The canonical supervisor owns
# both a no-follow main-tree flock and a pathname-independent kernel singleton
# for this runner's complete lifetime.  Environment variables alone never
# authorize entry into the inner runner.
if [[ -z "${RADIO_GS_GPU1_LOCK_FD:-}" \
      && -z "${RADIO_GS_GPU1_SINGLETON_FD:-}" ]]; then
  exec env CUDA_DEVICE_ORDER="$FROZEN_CUDA_DEVICE_ORDER" \
    PYTHONDONTWRITEBYTECODE="$FROZEN_PYTHONDONTWRITEBYTECODE" \
    NUMBA_CACHE_DIR="$FROZEN_NUMBA_CACHE_DIR" \
    GPU_OWNER_PID_NAMESPACE_MODE="$FROZEN_GPU_OWNER_PID_NAMESPACE_MODE" \
    NVIDIA_VISIBLE_DEVICES="$EXPECTED_GPU1_UUID" CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$LOCK_SUPERVISOR" run -- bash "$SCRIPT_PATH" "$@"
fi
if [[ -z "${RADIO_GS_GPU1_LOCK_FD:-}" \
      || -z "${RADIO_GS_GPU1_SINGLETON_FD:-}" ]]; then
  echo "physical GPU1 lock inheritance is incomplete" >&2
  exit 2
fi
if [[ "${RADIO_GS_GPU1_LOCK_PATH:-}" != "$GLOBAL_GPU1_LOCK" \
      || "${RADIO_GS_GPU1_SINGLETON_PROTOCOL:-}" \
      != "$GPU1_SINGLETON_PROTOCOL" ]]; then
  echo "physical GPU1 lock authority differs from the canonical contract" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="" \
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$LOCK_SUPERVISOR" verify-inherited \
  --fd "$RADIO_GS_GPU1_LOCK_FD" \
  --singleton-fd "$RADIO_GS_GPU1_SINGLETON_FD" >/dev/null

if [[ "$GPU" != "1" ]]; then
  echo "registered-region-v3 is assigned to physical GPU1; got GPU=$GPU" >&2
  exit 2
fi
if [[ ! -d "$MAIN_OUTPUT_ROOT" ]]; then
  echo "canonical main output root is unavailable: $MAIN_OUTPUT_ROOT" >&2
  exit 2
fi
MAIN_OUTPUT_REAL="$(readlink -f -- "$MAIN_OUTPUT_ROOT")"
if [[ -z "$MAIN_OUTPUT_REAL" || ! -d "$MAIN_OUTPUT_REAL" ]]; then
  echo "canonical main output target is unavailable: $MAIN_OUTPUT_ROOT" >&2
  exit 2
fi
OUTPUT_ROOT="$(realpath -ms -- "$OUTPUT_ROOT")"
case "$OUTPUT_ROOT" in
  "$MAIN_OUTPUT_ROOT"/*) ;;
  *)
    echo "NVOS v3 OUTPUT_ROOT must stay below $MAIN_OUTPUT_ROOT" >&2
    exit 2
    ;;
esac
OUTPUT_ROOT_REAL="$(readlink -m -- "$OUTPUT_ROOT")"
case "$OUTPUT_ROOT_REAL" in
  "$MAIN_OUTPUT_REAL"/*) ;;
  *)
    echo "NVOS v3 OUTPUT_ROOT resolves outside the canonical output target" >&2
    exit 2
    ;;
esac
RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
THREE_SCENE_SCREEN="$OUTPUT_ROOT/three_scene_screen.json"
PARTIAL_COMPLETION="$OUTPUT_ROOT/partial_completion.json"
LOG_ROOT="$OUTPUT_ROOT/logs"
LOCK_ROOT="$OUTPUT_ROOT/locks"
SCENE_RECEIPT_ROOT="$OUTPUT_ROOT/scene_receipts"
SCENE_ATTEMPT_ROOT="$OUTPUT_ROOT/scene_attempts"
for required in \
  "$SOURCE_ROOT" "$QUEUE_PLAN" "$MANIFEST" "$RADIO_CHECKPOINT" \
  "$PARENT_ASSET_MANIFEST" "$CANDIDATE_CONTRACT" \
  "$CONTINUATION_SCREEN" "$THERMAL_GUARD" "$GPU1_AUTHORITY" \
  "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"; do
  if [[ ! -e "$required" ]]; then
    echo "missing registered-region-v3 input: $required" >&2
    exit 2
  fi
done

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
GPU_PROC_UUID="$(awk '/GPU UUID:/ {print $3}' "$GPU_INFO")"
if [[ "$GPU_BUS_ID" != "$EXPECTED_GPU1_PROC_BUS_ID" \
      || "$GPU_PROC_UUID" != "$EXPECTED_GPU1_UUID" ]]; then
  echo "physical GPU1 proc UUID/PCI identity differs from the frozen host assignment" >&2
  exit 2
fi
GPU_CONFIG="/sys/bus/pci/devices/$GPU_BUS_ID/config"
GPU_CONFIG_PREFIX="$(od -An -tx1 -N16 "$GPU_CONFIG" 2>/dev/null | tr -d ' \n')"
if [[ ! "$GPU_CONFIG_PREFIX" =~ ^[0-9a-f]{32}$ \
      || "$GPU_CONFIG_PREFIX" =~ ^f+$ ]]; then
  echo "physical GPU1 PCIe configuration space is not responding" >&2
  exit 2
fi
if ! timeout --kill-after=2s 10s nvidia-smi -i "$GPU" >/dev/null; then
  echo "physical GPU1 is not usable by the current container" >&2
  exit 2
fi
GPU_UUID="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
GPU_NVIDIA_BUS_ID="$(nvidia-smi -i "$GPU" --query-gpu=pci.bus_id --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ "$GPU_UUID" != "$EXPECTED_GPU1_UUID" \
      || "$GPU_NVIDIA_BUS_ID" != "$EXPECTED_GPU1_NVIDIA_BUS_ID" ]]; then
  echo "nvidia-smi physical GPU1 UUID/PCI identity differs from the frozen host assignment" >&2
  exit 2
fi
GPU_COMPUTE_OWNERS="$(
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
    | awk -F', *' -v uuid="$GPU_UUID" '$1 == uuid {print $2}' \
    | paste -sd, -
)"
if [[ -n "$GPU_COMPUTE_OWNERS" ]]; then
  echo "physical GPU1 already has compute owner(s): $GPU_COMPUTE_OWNERS" >&2
  echo "wait for the current GPU1 stage to finish before launching NVOS v3" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="" \
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$GPU1_AUTHORITY" output-identity \
  --main-root "$MAIN_OUTPUT_ROOT" --output-root "$OUTPUT_ROOT" >/dev/null
mkdir -p \
  "$OUTPUT_ROOT" "$LOG_ROOT" "$LOCK_ROOT" \
  "$SCENE_RECEIPT_ROOT" "$SCENE_ATTEMPT_ROOT"
CUDA_VISIBLE_DEVICES="" \
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$GPU1_AUTHORITY" output-tree \
  --main-root "$MAIN_OUTPUT_ROOT" --output-root "$OUTPUT_ROOT" >/dev/null

exec {run_lock}>"$LOCK_ROOT/run.lock"
if ! flock -n "$run_lock"; then
  echo "another NVOS v3 runner owns $LOCK_ROOT/run.lock" >&2
  exit 2
fi

exec {manifest_lock}>"$LOCK_ROOT/run_manifest.lock"
flock "$manifest_lock"
bash radio_gs/scripts/run_repo_python.sh - \
  "$SOURCE_ROOT" "$QUEUE_PLAN" "$MANIFEST" "$RADIO_CHECKPOINT" \
  "$SCENE_NAMES" "$OUTPUT_ROOT" "$RUN_MANIFEST" "$0" \
  "$THERMAL_GUARD" "$GPU_MAX_TEMP_C" "$GPU_START_MAX_TEMP_C" \
  "$GPU_MAX_POWER_LIMIT_W" "$GPU_POLL_SECONDS" \
  "$GPU_SOFT_PAUSE_TEMP_C" "$GPU_SOFT_RESUME_TEMP_C" \
  "$GPU_PEER_INDEX" "$GPU_PEER_PAUSE_TEMP_C" \
  "$GPU_PEER_RESUME_TEMP_C" "$GPU_PEER_QUIET_SECONDS" \
  "$GPU_PEER_MAX_POWER_W" "$GPU_PEER_MAX_MEMORY_MIB" \
  "$GPU_PEER_MAX_UTIL_PCT" \
  "$PARENT_ASSET_MANIFEST" "$CANDIDATE_CONTRACT" \
  "$CONTINUATION_SCREEN" "$THREE_SCENE_SCREEN" \
  "$PARTIAL_COMPLETION" "$GPU1_AUTHORITY" \
  "$GPU_UUID" "$GPU_BUS_ID" "$GPU_NVIDIA_BUS_ID" \
  "$GPU_CONFIG_PREFIX" "$GLOBAL_GPU1_LOCK" \
  "$GPU1_SINGLETON_PROTOCOL" \
  "$FROZEN_GPU_OWNER_PID_NAMESPACE_MODE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

from radio_gs.scripts.nvos_registered_region_v3_authority import (
    build_runtime_closure,
    output_identity,
    verify_readonly_source_snapshot,
)
from radio_gs.scripts.screen_nvos_registered_region_v3_continuation import (
    SCREEN_CONTRACT,
    SCREEN_CONTRACT_SHA256,
)

(
    source_root,
    queue_plan,
    benchmark_manifest,
    radio_checkpoint,
    scene_names,
    output_root,
    run_manifest,
    runner,
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
    parent_asset_manifest,
    candidate_contract,
    continuation_screen,
    three_scene_screen,
    partial_completion,
    gpu1_authority,
    gpu_uuid,
    gpu_proc_bus_id,
    gpu_nvidia_bus_id,
    gpu_config_prefix,
    global_gpu1_lock,
    gpu1_singleton_protocol,
    gpu_owner_pid_namespace_mode,
) = sys.argv[1:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


repo = Path(runner).resolve().parents[2]
source_snapshot_authority = verify_readonly_source_snapshot(repo)
runtime_closure = build_runtime_closure(repo)
if (
    runtime_closure["source_snapshot_permissions"]
    != source_snapshot_authority["source_permissions"]
):
    raise SystemExit("source snapshot permissions changed during closure")
output_identity_record = output_identity(
    "/root/RADIO-GS/output", output_root
)
source = Path(source_root).resolve()
scenes = scene_names.split()
expected = [
    "fern",
    "flower",
    "fortress",
    "horns_center",
    "horns_left",
    "leaves",
    "orchids",
    "trex",
]
if scenes != expected:
    raise SystemExit(
        "registered-region-v3 requires the frozen ordered NVOS cohort: "
        + " ".join(expected)
    )
queue = json.loads(Path(queue_plan).read_text(encoding="utf-8"))
if (
    str(queue.get("benchmark")) != "nvos"
    or str(queue.get("protocol_hash")) != str(
        json.loads(Path(benchmark_manifest).read_text(encoding="utf-8")).get(
            "protocol_hash"
        )
    )
    or [str(row.get("scene_id")) for row in queue.get("scenes", [])]
    != expected
):
    raise SystemExit("queue/manifest do not match the frozen NVOS cohort")
parent_path = Path(parent_asset_manifest).resolve()
parent = json.loads(parent_path.read_text(encoding="utf-8"))
if (
    parent.get("candidate") != "registered-region-v2"
    or parent.get("scenes") != scenes
    or Path(str(parent.get("source_root", ""))).resolve() != source
    or Path(str(parent.get("queue_plan", ""))).resolve()
    != Path(queue_plan).resolve()
    or parent.get("queue_plan_sha256") != sha256(Path(queue_plan))
    or Path(str(parent.get("benchmark_manifest", ""))).resolve()
    != Path(benchmark_manifest).resolve()
    or parent.get("benchmark_manifest_sha256")
    != sha256(Path(benchmark_manifest))
    or Path(str(parent.get("radio_checkpoint", ""))).resolve()
    != Path(radio_checkpoint).resolve()
):
    raise SystemExit("parent asset manifest does not match the frozen NVOS inputs")
source_artifacts = json.loads(json.dumps(parent["source_artifacts"]))
queue_scene_inputs = json.loads(json.dumps(parent["queue_scene_inputs"]))
for scene in scenes:
    for name, record in source_artifacts[scene].items():
        path = Path(str(record["path"])).resolve()
        metadata = Path(str(record["metadata_path"])).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or not metadata.is_file()
            or sha256(metadata) != str(record["metadata_sha256"])
        ):
            raise SystemExit(f"{scene}: parent source record changed for {name}")
    for raw_path, record in queue_scene_inputs[scene].items():
        path = Path(str(raw_path)).resolve()
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise SystemExit(f"{scene}: parent renderer/view input changed: {path}")
parent_radio_sha256 = str(parent.get("radio_checkpoint_sha256", ""))
if len(parent_radio_sha256) != 64:
    raise SystemExit("parent asset manifest lacks the RADIO checkpoint digest")

candidate_payload = yaml.safe_load(Path(candidate_contract).read_text(encoding="utf-8"))
expected_thermal_plan = {
    "physical_gpu": 1,
    "maximum_temperature_c": int(gpu_max_temp_c),
    "maximum_start_temperature_c": int(gpu_start_max_temp_c),
    "soft_pause_temperature_c": int(gpu_soft_pause_temp_c),
    "soft_resume_temperature_c": int(gpu_soft_resume_temp_c),
    "peer_gpu": None if gpu_peer_index == "" else int(gpu_peer_index),
    "peer_pause_temperature_c": int(gpu_peer_pause_temp_c),
    "peer_resume_temperature_c": int(gpu_peer_resume_temp_c),
    "peer_quiet_seconds_before_launch": int(gpu_peer_quiet_seconds),
    "peer_maximum_power_w": float(gpu_peer_max_power_w),
    "peer_maximum_memory_mib": int(gpu_peer_max_memory_mib),
    "peer_maximum_utilization_percent": int(gpu_peer_max_util_pct),
    "poll_seconds": int(gpu_poll_seconds),
    "maximum_power_limit_w": float(gpu_max_power_limit_w),
}
if candidate_payload.get("thermal_plan") != expected_thermal_plan:
    raise SystemExit("candidate thermal plan differs from the queue safety contract")

if Path(gpu1_authority).resolve() != (
    repo / "radio_gs/scripts/nvos_registered_region_v3_authority.py"
).resolve():
    raise SystemExit("NVOS GPU1 authority escaped the source snapshot")
implementation = {
    relative: str(record["sha256"])
    for relative, record in runtime_closure["repository_sources"]["files"].items()
}
payload = {
    "schema_version": 2,
    "candidate": "registered-region-v3",
    "eligibility": "diagnostic_until_disjoint_registered_prompt_gate",
    "source_snapshot_root": runtime_closure["source_snapshot_root"],
    "source_snapshot_import_root": runtime_closure["repository_import_root"],
    "source_snapshot_tree_sha256": runtime_closure[
        "repository_sources"
    ]["digest"],
    "source_snapshot_permissions": source_snapshot_authority,
    "output_identity": output_identity_record,
    "scenes": scenes,
    "source_root": str(source),
    "source_artifacts": source_artifacts,
    "queue_scene_inputs": queue_scene_inputs,
    "queue_plan": str(Path(queue_plan).resolve()),
    "queue_plan_sha256": sha256(Path(queue_plan)),
    "benchmark_manifest": str(Path(benchmark_manifest).resolve()),
    "benchmark_manifest_sha256": sha256(Path(benchmark_manifest)),
    "radio_checkpoint": str(Path(radio_checkpoint).resolve()),
    "radio_checkpoint_sha256": parent_radio_sha256,
    "asset_manifest_parent": str(parent_path),
    "asset_manifest_parent_sha256": sha256(parent_path),
    "asset_reuse_contract": (
        "parent_sha256_plus_path_size_preflight_plus_scene_use_time_sha256"
    ),
    "continuation_screen": {
        "script": str(Path(continuation_screen).resolve()),
        "script_sha256": sha256(Path(continuation_screen)),
        "candidate_contract": str(Path(candidate_contract).resolve()),
        "candidate_contract_sha256": sha256(Path(candidate_contract)),
        "contract": SCREEN_CONTRACT,
        "contract_sha256": SCREEN_CONTRACT_SHA256,
        "output": str(Path(three_scene_screen).resolve()),
        "partial_completion_output": str(Path(partial_completion).resolve()),
    },
    "method_contract": {
        "support_mode": "canonical_support",
        "region_space": "sam3",
        "prompt_registration": {
            "mode": "raster_adjoint",
            "scale": 1.0,
            "alpha_threshold": 0.0,
            "depth_tolerance": 0.08,
            "relative_depth_tolerance": 0.02,
        },
        "seed_construction": "joint_signed",
        "seed_normalization": "none",
        "observation_fusion": "hard_seed_anchored_probability",
        "registered_seed_unary_weight": 0.0,
        "strong_unary": {
            "policy": "unit_confidence_on_shared_hard_seed_rows",
            "anchor_threshold_source": "solver.hard_seed_threshold",
            "anchor_threshold": 0.20,
            "formula": (
                "a=1[c>0 and abs(s)>=tau]; c_eff=a+(1-a)c; "
                "p=(1-c_eff)p_field+c_eff*q"
            ),
            "new_numeric_constant": False,
        },
        "observation_mass_source": (
            "raw_raster_adjoint_prompt_mass_times_"
            "labeled_footprint_coverage"
        ),
        "observation_confidence": "poisson_mass_coverage",
        "observation_mass_scale": 1.0,
        "observation_coverage_power": 1.0,
        "observation_constructed_before_capability_filter": True,
        "prompt_support_threshold": 0.0,
        "prototype_count": 4,
        "prototype_strategy": "spherical_mean_fps",
        "appearance_weight": 1.0,
        "boundary_weight": 0.35,
        "prototype_temperature": 0.07,
        "feature_calibration": "none",
        "background_centroids": 0,
        "score_calibration": "none",
        "negative_spatial_mode": "none",
        "diagnostic_selection_mode": "seeded_component",
        "selection_applied_to_main_output": False,
        "final_readout": "propagated",
        "graph": {
            "policy": "legacy",
            "component_policy": "same",
            "legacy_residual": 0.0,
            "channel_confidence_mode": "none",
        },
        "score_render": {
            "resolution": "prompt_native",
            "scale": 1.0,
            "valid_support_normalization": True,
            "valid_support_coverage_power": 1.0,
            "feature_contribution_gamma": 1.0,
            "score_chunk_size": 8192,
            "pixel_threshold": 0.5,
            "threshold_comparison": "greater_or_equal",
            "resize_to_ground_truth": "cv2.INTER_LINEAR",
        },
        "solver": {
            "type": "confidence_random_walker",
            "iterations": 12,
            "residual": 0.30,
            "unary_temperature": 0.10,
            "support_threshold": 0.50,
            "laplacian_weight": 1.0,
            "cg_iterations": 64,
            "cg_tolerance": 1e-5,
            "hard_seed_threshold": 0.20,
            "hard_seed_conflict_policy": "exclusive_relative",
            "hard_seed_conflict_margin": 0.0,
            "component_edge_threshold": 1e-5,
            "seeded_component_min_weight": 0.20,
        },
        "canonical_reliability_cache": "",
        "diagnostic_graph_affinity_override": "",
        "asset_hash_verification_required": True,
        "uses_target_calibration": False,
    },
    "runner": str(Path(runner).resolve()),
    "runner_sha256": sha256(Path(runner)),
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
        "peer_gpu": None if gpu_peer_index == "" else int(gpu_peer_index),
        "peer_pause_temperature_c": int(gpu_peer_pause_temp_c),
        "peer_resume_temperature_c": int(gpu_peer_resume_temp_c),
        "peer_quiet_seconds_before_launch": int(gpu_peer_quiet_seconds),
        "peer_maximum_power_w": float(gpu_peer_max_power_w),
        "peer_maximum_memory_mib": int(gpu_peer_max_memory_mib),
        "peer_maximum_utilization_percent": int(gpu_peer_max_util_pct),
        "gpu_uuid": gpu_uuid,
        "proc_pci_bus_id": gpu_proc_bus_id,
        "nvidia_smi_pci_bus_id": gpu_nvidia_bus_id,
        "pcie_config_prefix_16_bytes": gpu_config_prefix,
        "initial_compute_owners": [],
        "global_gpu_lock": global_gpu1_lock,
        "global_gpu_lock_resolved": str(Path(global_gpu1_lock).resolve()),
        "global_gpu_lock_inherited_fd_verified": True,
        "global_gpu_kernel_singleton_protocol": gpu1_singleton_protocol,
        "global_gpu_kernel_singleton_inherited_fd_verified": True,
        "scene_attempt_telemetry_root": str(
            Path(output_root).resolve() / "scene_attempts"
        ),
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": gpu_uuid,
        "nvidia_visible_devices": gpu_uuid,
        "cuda_child_attestation": (
            "torch_cuda0_empty_to_live_singleton_owner_uuid_pci_v2"
        ),
        "owner_audit": (
            "prelaunch_atomic_clear_plus_runtime_child_or_"
            "exclusive_invisible_host_pid_singleton_v2"
        ),
        "gpu_owner_pid_namespace_mode": gpu_owner_pid_namespace_mode,
    },
    "scene_receipt_contract": {
        "artifact_type": "nvos-v3-scene-receipt-v1",
        "receipt_root": str(Path(output_root).resolve() / "scene_receipts"),
        "attempt_root": str(Path(output_root).resolve() / "scene_attempts"),
        "skip_only_after_full_receipt_revalidation": True,
        "screen_requires_valid_receipts": ["fern", "flower", "fortress"],
        "aggregate_requires_all_scene_receipts": True,
        "postcheck": [
            "runtime_closure",
            "inherited_flock_and_kernel_singleton",
            "physical_gpu1_uuid_and_pci",
            "no_compute_owner",
            "result_sha256",
        ],
    },
    "implementation_sources": implementation,
    "runtime_closure": runtime_closure,
}
manifest = Path(run_manifest)
if manifest.is_symlink():
    raise SystemExit("refusing symlinked NVOS run manifest")
if manifest.is_file():
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    if previous != payload:
        raise SystemExit("OUTPUT_ROOT belongs to another immutable NVOS run")
else:
    if manifest.exists():
        raise SystemExit("NVOS run manifest path is not a regular file")
    allowed_pre_manifest_files = {
        (Path(output_root) / "locks" / "run.lock").resolve(),
        (Path(output_root) / "locks" / "run_manifest.lock").resolve(),
    }
    existing = [
        path
        for path in Path(output_root).rglob("*")
        if (
            path.is_file()
            and path.name != "run_manifest.json"
            and path.resolve() not in allowed_pre_manifest_files
        )
    ]
    if existing:
        raise SystemExit("OUTPUT_ROOT contains artifacts without a run manifest")
    temporary = manifest.with_suffix(".json.tmp")
    if os.path.lexists(temporary):
        raise SystemExit("NVOS run manifest temporary path already exists")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest)
PY
flock -u "$manifest_lock"
exec {manifest_lock}>&-

verify_gpu1_lock_authority() {
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$LOCK_SUPERVISOR" verify-inherited \
    --fd "$RADIO_GS_GPU1_LOCK_FD" \
    --singleton-fd "$RADIO_GS_GPU1_SINGLETON_FD" >/dev/null
}

verify_runtime_closure() {
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$GPU1_AUTHORITY" verify-readonly-snapshot \
    --repo-root "$REPO_ROOT" >/dev/null
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$GPU1_AUTHORITY" verify-closure \
    --repo-root "$REPO_ROOT" --manifest "$RUN_MANIFEST" >/dev/null
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$GPU1_AUTHORITY" verify-output-tree \
    --manifest "$RUN_MANIFEST" >/dev/null
}

assert_gpu1_identity_unowned() {
  local phase="$1"
  local current_bus current_proc_uuid current_prefix current_uuid
  local current_nvidia_bus current_owners
  current_bus="$(awk '/Bus Location:/ {print $3}' "$GPU_INFO")"
  current_proc_uuid="$(awk '/GPU UUID:/ {print $3}' "$GPU_INFO")"
  current_prefix="$(
    od -An -tx1 -N16 "$GPU_CONFIG" 2>/dev/null | tr -d ' \n'
  )"
  if [[ "$current_bus" != "$GPU_BUS_ID" \
        || "$current_proc_uuid" != "$GPU_UUID" \
        || "$current_prefix" != "$GPU_CONFIG_PREFIX" ]]; then
    echo "physical GPU1 proc/PCI identity changed at $phase" >&2
    return 2
  fi
  if ! timeout --kill-after=2s 10s nvidia-smi -i "$GPU" >/dev/null; then
    echo "physical GPU1 became unusable at $phase" >&2
    return 2
  fi
  current_uuid="$(
    nvidia-smi -i "$GPU" --query-gpu=uuid \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  current_nvidia_bus="$(
    nvidia-smi -i "$GPU" --query-gpu=pci.bus_id \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  if [[ "$current_uuid" != "$GPU_UUID" \
        || "$current_nvidia_bus" != "$GPU_NVIDIA_BUS_ID" ]]; then
    echo "physical GPU1 nvidia-smi identity changed at $phase" >&2
    return 2
  fi
  current_owners="$(
    nvidia-smi --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits \
      | awk -F', *' -v uuid="$GPU_UUID" '$1 == uuid {print $2}' \
      | paste -sd, -
  )"
  if [[ -n "$current_owners" ]]; then
    echo "physical GPU1 has unexpected compute owner(s) at $phase: $current_owners" >&2
    return 2
  fi
}

verify_gpu1_lock_authority
verify_runtime_closure

validate_scene_receipt() {
  local receipt_scene="$1"
  local receipt_result="$OUTPUT_ROOT/$receipt_scene/eval_full_mask_random_walker/${receipt_scene}_evaluation.json"
  local receipt_path="$SCENE_RECEIPT_ROOT/${receipt_scene}.receipt.json"
  CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$GPU1_AUTHORITY" validate-scene \
    --receipt "$receipt_path" --run-manifest "$RUN_MANIFEST" \
    --scene "$receipt_scene" --result "$receipt_result" >/dev/null
}

for scene in $SCENE_NAMES; do
  source="$SOURCE_ROOT/$scene"
  result_root="$OUTPUT_ROOT/$scene/eval_full_mask_random_walker"
  result="$result_root/${scene}_evaluation.json"
  receipt="$SCENE_RECEIPT_ROOT/${scene}.receipt.json"
  exec {scene_lock}>"$LOCK_ROOT/$scene.lock"
  flock "$scene_lock"
  result_present=0
  receipt_present=0
  [[ -e "$result" || -L "$result" ]] && result_present=1
  [[ -e "$receipt" || -L "$receipt" ]] && receipt_present=1
  if (( result_present || receipt_present )); then
    if (( ! result_present || ! receipt_present )) \
        || [[ ! -s "$result" || -L "$result" \
              || ! -s "$receipt" || -L "$receipt" ]]; then
      echo "$scene has incomplete result/receipt evidence; quarantine required" >&2
      exit 1
    fi
    if ! validate_scene_receipt "$scene"; then
      echo "$scene existing result failed exclusive-GPU receipt validation" >&2
      exit 1
    fi
  else
    verify_gpu1_lock_authority
    verify_runtime_closure
    assert_gpu1_identity_unowned "pre_${scene}"
    field="$source/canonical_d256_l128_capability_first.pth"
    capability="$source/official_dino_sam3_views.pt"
    graph="$source/shared_support_graph_k16.pt"
    field_sha="$(sha256sum "$field" | awk '{print $1}')"
    if [[ -d "$result_root" ]] \
        && find "$result_root" -mindepth 1 -type f -print -quit | grep -q .; then
      echo "$scene has unreceipted partial result artifacts; quarantine required" >&2
      exit 1
    fi
    mkdir -p "$result_root"
    attempt_index=1
    while [[ -e "$SCENE_ATTEMPT_ROOT/$scene/attempt_$(printf '%04d' "$attempt_index")" \
             || -L "$SCENE_ATTEMPT_ROOT/$scene/attempt_$(printf '%04d' "$attempt_index")" ]]; do
      attempt_index=$((attempt_index + 1))
    done
    attempt_root="$SCENE_ATTEMPT_ROOT/$scene/attempt_$(printf '%04d' "$attempt_index")"
    mkdir -p "$attempt_root"
    verify_runtime_closure
    telemetry="$attempt_root/telemetry.csv"
    owner_audit="$attempt_root/owner_audit.csv"
    attestation="$attempt_root/cuda_attestation.json"
    command_record="$attempt_root/command.json"
    postcheck="$attempt_root/postcheck.json"
    attempt_log="$attempt_root/evaluator.log"
    evaluator_command=(
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
      "$REPO_ROOT/radio_gs/scripts/eval_nvos_gaussian_first.py"
      --manifest "$MANIFEST"
      --queue-root "$(dirname "$QUEUE_PLAN")"
      --scene-id "$scene"
      --output-dir "$result_root"
      --run-manifest "$RUN_MANIFEST"
      --candidate-id registered-region-v3
      --device cuda:0
      --gpu-attestation-output "$attestation"
      --expected-gpu-uuid "$GPU_UUID"
      --expected-gpu-bus-id "$GPU_NVIDIA_BUS_ID"
      --radio-checkpoint "$RADIO_CHECKPOINT"
      --region-space sam3
      --support-mode canonical_support
      --canonical-capability-cache "$capability"
      --canonical-support-graph "$graph"
      --canonical-field-sha256 "$field_sha"
      --prompt-registration-mode raster_adjoint
      --prompt-registration-scale 1.0
      --alpha-threshold 0.0
      --support-threshold 0.0
      --prototype-count 4
      --prototype-strategy spherical_mean_fps
      --registered-seed-construction joint_signed
      --registered-observation-fusion hard_seed_anchored_probability
      --registered-observation-confidence poisson_mass_coverage
      --registered-observation-mass-scale 1.0
      --registered-observation-coverage-power 1.0
      --registered-seed-unary-weight 0.0
      --registered-selection-mode seeded_component
      --registered-readout-stage propagated
      --score-render-resolution prompt_native
      --score-render-scale 1.0
      --valid-support-normalization
      --valid-support-coverage-power 1.0
      --feature-contribution-gamma 1.0
      --graph-policy legacy
      --component-graph-policy same
      --channel-confidence-mode none
      --negative-spatial-mode none
      --appearance-weight 1.0
      --boundary-weight 0.35
      --prototype-temperature 0.07
      --feature-calibration none
      --background-centroids 0
      --score-calibration none
      --score-chunk-size 8192
      --solver-type confidence_random_walker
      --laplacian-weight 1.0
      --cg-iterations 64
      --cg-tolerance 1e-5
      --hard-seed-threshold 0.20
      --hard-seed-conflict-policy exclusive_relative
      --hard-seed-conflict-margin 0.0
      --component-edge-threshold 1e-5
      --seeded-component-min-weight 0.20
      --solver-iterations 12
      --solver-residual 0.30
      --solver-unary-temperature 0.10
      --solver-support-threshold 0.50
      --require-asset-hashes
    )
    CUDA_VISIBLE_DEVICES="" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$GPU1_AUTHORITY" prepare-scene \
      --output "$command_record" --run-manifest "$RUN_MANIFEST" \
      --scene "$scene" --result "$result" \
      --telemetry "$telemetry" --owner-audit "$owner_audit" \
      --attestation "$attestation" --postcheck "$postcheck" \
      --receipt "$receipt" --evaluator-log "$attempt_log" \
      --guard "$THERMAL_GUARD" \
      --gpu-uuid "$GPU_UUID" --gpu-bus-id "$GPU_NVIDIA_BUS_ID" \
      -- "${evaluator_command[@]}" >/dev/null
    verify_runtime_closure
    command_status=0
    GPU="$GPU" CUDA_DEVICE_ORDER="$FROZEN_CUDA_DEVICE_ORDER" \
      NVIDIA_VISIBLE_DEVICES="$GPU_UUID" CUDA_VISIBLE_DEVICES="$GPU_UUID" \
      GPU_TELEMETRY_LOG="$telemetry" \
      GPU_OWNER_AUDIT_LOG="$owner_audit" \
      GPU_OWNER_PID_NAMESPACE_MODE="$FROZEN_GPU_OWNER_PID_NAMESPACE_MODE" \
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
      bash "$THERMAL_GUARD" -- \
      "${evaluator_command[@]}" \
      >"$attempt_log" 2>&1 || command_status=$?
    verify_gpu1_lock_authority
    verify_runtime_closure
    if (( command_status != 0 )); then
      echo "registered-region-v3 scene failed: $scene (status=$command_status)" >&2
      exit "$command_status"
    fi
    assert_gpu1_identity_unowned "post_${scene}"
    CUDA_VISIBLE_DEVICES="" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$GPU1_AUTHORITY" postcheck-scene \
      --output "$postcheck" --run-manifest "$RUN_MANIFEST" \
      --scene "$scene" --result "$result" \
      --gpu-uuid "$GPU_UUID" --gpu-bus-id "$GPU_NVIDIA_BUS_ID" \
      --lock-fd "$RADIO_GS_GPU1_LOCK_FD" \
      --singleton-fd "$RADIO_GS_GPU1_SINGLETON_FD" >/dev/null
    CUDA_VISIBLE_DEVICES="" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$GPU1_AUTHORITY" finalize-scene \
      --output "$receipt" --command-record "$command_record" \
      --postcheck "$postcheck" >/dev/null
    validate_scene_receipt "$scene"
    verify_gpu1_lock_authority
    verify_runtime_closure
  fi
  flock -u "$scene_lock"
  exec {scene_lock}>&-
  unset scene_lock

  if [[ "$scene" == "fortress" ]]; then
    verify_gpu1_lock_authority
    verify_runtime_closure
    for screened_scene in fern flower fortress; do
      validate_scene_receipt "$screened_scene"
    done
    exec {screen_lock}>"$LOCK_ROOT/three_scene_screen.lock"
    flock "$screen_lock"
    bash radio_gs/scripts/run_repo_python.sh \
      "$CONTINUATION_SCREEN" \
      --result-root "$OUTPUT_ROOT" \
      --run-manifest "$RUN_MANIFEST" \
      --candidate-contract "$CANDIDATE_CONTRACT" \
      --output "$THREE_SCENE_SCREEN" \
      --partial-completion-output "$PARTIAL_COMPLETION" \
      >"$OUTPUT_ROOT/three_scene_screen.log" 2>&1
    verify_gpu1_lock_authority
    verify_runtime_closure
    for screened_scene in fern flower fortress; do
      validate_scene_receipt "$screened_scene"
    done
    flock -u "$screen_lock"
    exec {screen_lock}>&-
    screen_decision="$(
      bash radio_gs/scripts/run_repo_python.sh - "$THREE_SCENE_SCREEN" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = payload.get("decision")
if decision not in {"continue_full_eight", "reject_stop_after_three"}:
    raise SystemExit("invalid registered-region-v3 continuation decision")
print(decision)
PY
    )"
    verify_gpu1_lock_authority
    verify_runtime_closure
    if [[ "$screen_decision" == "reject_stop_after_three" ]]; then
      echo "registered-region-v3 rejected by frozen three-scene screen; stopping normally"
      exit 0
    fi
  fi
done

verify_gpu1_lock_authority
verify_runtime_closure
for completed_scene in $SCENE_NAMES; do
  validate_scene_receipt "$completed_scene"
done
exec {aggregate_lock}>"$LOCK_ROOT/aggregate.lock"
flock "$aggregate_lock"
bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/aggregate_registered_prompt_closeout.py \
  --queue-plan "$QUEUE_PLAN" \
  --result-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.json" \
  --run-manifest "$RUN_MANIFEST" \
  --expected-candidate registered-region-v3 \
  --require-method-config \
  >"$OUTPUT_ROOT/aggregate.log" 2>&1
verify_gpu1_lock_authority
verify_runtime_closure
for completed_scene in $SCENE_NAMES; do
  validate_scene_receipt "$completed_scene"
done
flock -u "$aggregate_lock"
exec {aggregate_lock}>&-
