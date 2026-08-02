#!/usr/bin/env bash

# Memory-bounded fixed-full-eight queue for nvos-forward-beta-coverage-v1.
#
# The method/protocol receipt is generated before snapshot staging.  This
# runner only validates the embedded receipt, binds GPU0/GPU1 from live
# inventory, preserves the fixed 4+4 GPU assignment, but admits exactly one
# resident scene evaluator at a time.  The alternating order is fixed before
# target results exist and aggregation starts only after all eight
# scene-authority receipts exist.  No metric controls execution or stopping.

set -euo pipefail

EXECUTING_SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_PATH="${RUNNER_AUTHORITY_PATH:-$EXECUTING_SCRIPT_PATH}"
REPO_ROOT="$(cd "$(dirname "$EXECUTING_SCRIPT_PATH")/../.." && pwd -P)"
cd "$REPO_ROOT"

MAIN_OUTPUT_ROOT="/root/RADIO-GS/output"
FORWARD_BETA_VARIANT="${FORWARD_BETA_VARIANT:-v1}"
SNAPSHOT_STAGER="${SNAPSHOT_STAGER:-$REPO_ROOT/radio_gs/scripts/stage_nvos_forward_beta_coverage_v1_snapshot.py}"
SCENE_AUTHORITY="${SCENE_AUTHORITY:-$REPO_ROOT/radio_gs/scripts/nvos_forward_beta_scene_authority.py}"
AGGREGATOR="${AGGREGATOR:-$REPO_ROOT/radio_gs/scripts/aggregate_nvos_forward_beta_full8_nonexact.py}"
CANDIDATE_ID="${CANDIDATE_ID:-nvos-forward-beta-coverage-v1}"
FORWARD_MODE="${FORWARD_MODE:-beta_coverage_v1}"
STAGER_MODULE="${STAGER_MODULE:-radio_gs.scripts.stage_nvos_forward_beta_coverage_v1_snapshot}"
STAGING_MANIFEST_RELATIVE="${STAGING_MANIFEST_RELATIVE:-paper/artifacts/nvos_forward_beta_coverage_v1_snapshot_staging.json}"
CANDIDATE_CONTRACT="${CANDIDATE_CONTRACT:-$REPO_ROOT/paper/artifacts/nvos_forward_beta_coverage_v1_candidate_20260802.yaml}"
PROTOCOL_AUTHORITY_RECEIPT="${PROTOCOL_AUTHORITY_RECEIPT:-$REPO_ROOT/paper/artifacts/nvos_forward_beta_coverage_v1_protocol_authority.json}"
RELIABILITY_CACHE_MANIFEST="${RELIABILITY_CACHE_MANIFEST:-}"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
RUN_REPO_PYTHON="$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
FROZEN_CUDA_DEVICE_ORDER="PCI_BUS_ID"
FROZEN_PYTHONDONTWRITEBYTECODE="1"
FROZEN_NUMBA_CACHE_DIR="/root/.cache/radio_gs/numba"
FROZEN_OWNER_MODE="exclusive-singleton-after-clear-v1"

SOURCE_ROOT="${SOURCE_ROOT:-$MAIN_OUTPUT_ROOT/evaluation_closeout_20260716/canonical_mpr_v3_nvos8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MAIN_OUTPUT_ROOT/optimization_20260802/nvos_forward_beta_coverage_v1}"
QUEUE_PLAN="${QUEUE_PLAN:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/gaussfm_jobs/nvos_strict_unseen_v1/queue_plan.json}"
MANIFEST="${MANIFEST:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
PARENT_ASSET_MANIFEST="${PARENT_ASSET_MANIFEST:-$MAIN_OUTPUT_ROOT/optimization_20260731/nvos_registered_region_v2/run_manifest.json}"

# GPU ownership and the serial admission order are fixed before any target
# result exists.  Keeping these arrays explicit makes assignment drift
# fail-closed in both the run-manifest and CPU contract tests.
GPU0_SCENES=(fern flower fortress horns_center)
GPU1_SCENES=(horns_left leaves orchids trex)
ALL_SCENES=(fern flower fortress horns_center horns_left leaves orchids trex)
SERIAL_SCENE_GPU_PLAN=(
  "0:fern" "1:horns_left"
  "0:flower" "1:leaves"
  "0:fortress" "1:orchids"
  "0:horns_center" "1:trex"
)

# Low-frequency protection validated for the host's capped 3090s.
GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-86}"
GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-80}"
GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-82}"
GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-78}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-20}"
GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"
GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES="${GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES:-3}"

for assignment in \
  "CUDA_DEVICE_ORDER:$FROZEN_CUDA_DEVICE_ORDER" \
  "PYTHONDONTWRITEBYTECODE:$FROZEN_PYTHONDONTWRITEBYTECODE" \
  "NUMBA_CACHE_DIR:$FROZEN_NUMBA_CACHE_DIR" \
  "GPU_OWNER_PID_NAMESPACE_MODE:$FROZEN_OWNER_MODE"; do
  name="${assignment%%:*}"
  expected="${assignment#*:}"
  if [[ -n "${!name+x}" && "${!name}" != "$expected" ]]; then
    echo "refusing non-frozen $name=${!name}" >&2
    exit 2
  fi
done
export CUDA_DEVICE_ORDER="$FROZEN_CUDA_DEVICE_ORDER"
export PYTHONDONTWRITEBYTECODE="$FROZEN_PYTHONDONTWRITEBYTECODE"
export NUMBA_CACHE_DIR="$FROZEN_NUMBA_CACHE_DIR"
export GPU_OWNER_PID_NAMESPACE_MODE="$FROZEN_OWNER_MODE"
mkdir -p "$FROZEN_NUMBA_CACHE_DIR"
if [[ -L "$FROZEN_NUMBA_CACHE_DIR" \
      || "$(readlink -f -- "$FROZEN_NUMBA_CACHE_DIR")" \
      != "$FROZEN_NUMBA_CACHE_DIR" ]]; then
  echo "frozen NUMBA_CACHE_DIR is not a real external directory" >&2
  exit 2
fi

for required_source in \
  "$SNAPSHOT_STAGER" "$SCENE_AUTHORITY" "$AGGREGATOR" \
  "$THERMAL_GUARD" "$RUN_REPO_PYTHON" \
  "$CANDIDATE_CONTRACT" "$PROTOCOL_AUTHORITY_RECEIPT" \
  "$REPO_ROOT/$STAGING_MANIFEST_RELATIVE"; do
  if [[ ! -f "$required_source" || -L "$required_source" ]]; then
    echo "forward-Beta snapshot input is missing or unsafe: $required_source" >&2
    exit 2
  fi
done
if [[ "$FORWARD_BETA_VARIANT" == "v2" ]]; then
  if [[ -z "$RELIABILITY_CACHE_MANIFEST" \
        || ! -f "$RELIABILITY_CACHE_MANIFEST" \
        || -L "$RELIABILITY_CACHE_MANIFEST" ]]; then
    echo "v2 reliability cache manifest is missing or unsafe" >&2
    exit 2
  fi
fi
for required_input in \
  "$SOURCE_ROOT" "$QUEUE_PLAN" "$MANIFEST" "$RADIO_CHECKPOINT" \
  "$PARENT_ASSET_MANIFEST"; do
  if [[ ! -e "$required_input" ]]; then
    echo "missing forward-Beta input: $required_input" >&2
    exit 2
  fi
done

CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
  bash "$RUN_REPO_PYTHON" "$SNAPSHOT_STAGER" validate-snapshot \
  --snapshot-root "$REPO_ROOT" >/dev/null

if [[ ! -d "$MAIN_OUTPUT_ROOT" ]]; then
  echo "canonical output root is unavailable: $MAIN_OUTPUT_ROOT" >&2
  exit 2
fi
MAIN_OUTPUT_REAL="$(readlink -f -- "$MAIN_OUTPUT_ROOT")"
OUTPUT_ROOT="$(realpath -ms -- "$OUTPUT_ROOT")"
OUTPUT_ROOT_REAL="$(readlink -m -- "$OUTPUT_ROOT")"
case "$OUTPUT_ROOT" in "$MAIN_OUTPUT_ROOT"/*) ;; *)
  echo "OUTPUT_ROOT must stay below $MAIN_OUTPUT_ROOT" >&2; exit 2 ;;
esac
case "$OUTPUT_ROOT_REAL" in "$MAIN_OUTPUT_REAL"/*) ;; *)
  echo "OUTPUT_ROOT resolves outside the canonical output target" >&2; exit 2 ;;
esac
# The canonical /root/RADIO-GS/output entry is a legitimate mount symlink.
# Keep both containment checks above lexical, then use only the resolved target
# below so immutable resume reads never traverse that symlink.
OUTPUT_ROOT="$OUTPUT_ROOT_REAL"

RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
LOCK_ROOT="$OUTPUT_ROOT/locks"
LOG_ROOT="$OUTPUT_ROOT/logs"
SCENE_RECEIPT_ROOT="$OUTPUT_ROOT/scene_receipts"
SCENE_ATTEMPT_ROOT="$OUTPUT_ROOT/scene_attempts"
mkdir -p "$OUTPUT_ROOT" "$LOCK_ROOT" "$LOG_ROOT" \
  "$SCENE_RECEIPT_ROOT" "$SCENE_ATTEMPT_ROOT"

exec {run_lock}>"$LOCK_ROOT/run.lock"
if ! flock -n "$run_lock"; then
  echo "another forward-Beta runner owns $LOCK_ROOT/run.lock" >&2
  exit 2
fi

exec {manifest_lock}>"$LOCK_ROOT/run_manifest.lock"
flock "$manifest_lock"
CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
  bash "$RUN_REPO_PYTHON" - \
  "$REPO_ROOT" "$SOURCE_ROOT" "$QUEUE_PLAN" "$MANIFEST" \
  "$RADIO_CHECKPOINT" "$PARENT_ASSET_MANIFEST" "$OUTPUT_ROOT" \
  "$RUN_MANIFEST" "$SCRIPT_PATH" "$THERMAL_GUARD" "$SCENE_AUTHORITY" \
  "$STAGER_MODULE" "$STAGING_MANIFEST_RELATIVE" \
  "$RELIABILITY_CACHE_MANIFEST" \
  "$GPU_MAX_TEMP_C" "$GPU_START_MAX_TEMP_C" \
  "$GPU_SOFT_PAUSE_TEMP_C" "$GPU_SOFT_RESUME_TEMP_C" \
  "$GPU_POLL_SECONDS" "$GPU_MAX_POWER_LIMIT_W" \
  "$GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES" <<'PY'
import importlib
import inspect
import sys
from pathlib import Path

(
    repo, source, queue, benchmark, checkpoint, parent, output_root,
    run_manifest, runner, guard, scene_authority,
    stager_module, staging_manifest_relative, reliability_cache_manifest,
    max_temp, start_temp, soft_pause, soft_resume, poll_seconds, max_power,
    max_consecutive_telemetry_failures,
) = sys.argv[1:]
stager = importlib.import_module(stager_module)
build_run_manifest_payload = stager.build_run_manifest_payload
validate_snapshot = stager.validate_snapshot
write_run_manifest = stager.write_run_manifest
repo_path = Path(repo).resolve()
staging = validate_snapshot(repo_path)
files = staging["snapshot_files"]
runtime_sources = {
    "selection": files["selection"],
    "files": files["files"],
    "digest": files["digest"],
}
runtime_closure = {
    "repository_import_root": str(repo_path),
    "repository_sources": runtime_sources,
    "digest": files["digest"],
}
snapshot_authority = {
    "status": "readonly_non_live_source_snapshot_verified",
    "staging_manifest": str(
        repo_path / staging_manifest_relative
    ),
    "snapshot_files_digest": files["digest"],
}
thermal = {
    "policy": "dual_gpu_independent_low_frequency_single_resident_v1",
    "physical_gpus": [0, 1],
    "maximum_concurrent_scene_evaluators": 1,
    "host_memory_policy": "fixed_mapping_single_scene_resident_v1",
    "maximum_temperature_c": int(max_temp),
    "maximum_start_temperature_c": int(start_temp),
    "soft_pause_temperature_c": int(soft_pause),
    "soft_resume_temperature_c": int(soft_resume),
    "poll_seconds": int(poll_seconds),
    "maximum_power_limit_w": float(max_power),
    "maximum_consecutive_telemetry_failures": int(
        max_consecutive_telemetry_failures
    ),
    "peer_coupling": False,
}
identity = {
    "status": "canonical_output_subtree_lexically_bound",
    "main_output_root": "/root/RADIO-GS/output",
    "output_root": str(Path(output_root).resolve()),
}
manifest_kwargs = dict(
    snapshot_root=repo_path,
    source_root=source,
    queue_plan=queue,
    benchmark_manifest=benchmark,
    radio_checkpoint=checkpoint,
    parent_asset_manifest=parent,
    output_root=output_root,
    runner=runner,
    thermal_guard=guard,
    gpu_authority=scene_authority,
    runtime_closure=runtime_closure,
    source_snapshot_authority=snapshot_authority,
    thermal_safety_contract=thermal,
    output_identity=identity,
)
if "reliability_cache_manifest" in inspect.signature(
    build_run_manifest_payload
).parameters:
    manifest_kwargs["reliability_cache_manifest"] = reliability_cache_manifest
payload = build_run_manifest_payload(**manifest_kwargs)
write_run_manifest(run_manifest, payload)
PY
flock -u "$manifest_lock"
exec {manifest_lock}>&-

validate_manifest_scene() {
  local scene="$1"
  CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
    bash "$RUN_REPO_PYTHON" "$SCENE_AUTHORITY" validate-manifest \
    --run-manifest "$RUN_MANIFEST" --scene "$scene" >/dev/null
}

validate_scene_receipt() {
  local scene="$1"
  local result="$OUTPUT_ROOT/$scene/eval_full_mask_random_walker/${scene}_evaluation.json"
  local receipt="$SCENE_RECEIPT_ROOT/$scene/scene_receipt.json"
  CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
    bash "$RUN_REPO_PYTHON" "$SCENE_AUTHORITY" validate-scene \
    --receipt "$receipt" --run-manifest "$RUN_MANIFEST" \
    --scene "$scene" --result "$result" >/dev/null
}

gpu_inventory_identity() {
  local physical_index="$1"
  timeout --kill-after=2s 10s nvidia-smi -i "$physical_index" \
    --query-gpu=index,uuid,pci.bus_id --format=csv,noheader,nounits
}

GPU_UUIDS=("" "")
GPU_BUS_IDS=("" "")

bind_reserved_gpu_identity() {
  local physical_index="$1"
  local inventory observed_index gpu_uuid gpu_bus_id owners
  inventory="$(gpu_inventory_identity "$physical_index")"
  IFS=',' read -r observed_index gpu_uuid gpu_bus_id <<<"$inventory"
  observed_index="$(tr -d '[:space:]' <<<"$observed_index")"
  gpu_uuid="$(tr -d '[:space:]' <<<"$gpu_uuid")"
  gpu_bus_id="$(tr -d '[:space:]' <<<"$gpu_bus_id")"
  if [[ "$observed_index" != "$physical_index" \
        || ! "$gpu_uuid" =~ ^GPU-[0-9A-Fa-f-]{32,}$ \
        || ! "$gpu_bus_id" =~ ^([0-9A-Fa-f]{8}:)?[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]]; then
    echo "physical GPU$physical_index returned invalid live inventory: $inventory" >&2
    return 2
  fi
  owners="$(
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
      | awk -F', *' -v uuid="$gpu_uuid" '$1 == uuid {print $2}' | paste -sd, -
  )"
  if [[ -n "$owners" ]]; then
    echo "physical GPU$physical_index already has compute owner(s): $owners" >&2
    return 2
  fi
  GPU_UUIDS[$physical_index]="$gpu_uuid"
  GPU_BUS_IDS[$physical_index]="$gpu_bus_id"
}

# Reserve both assigned cards for the complete serial cohort.  This prevents a
# second cooperative runner from filling host memory on the currently idle
# card while the admitted scene is resident on the other card.
exec {gpu0_lock}>"$MAIN_OUTPUT_ROOT/.physical_gpu0.lock"
flock "$gpu0_lock"
exec {gpu1_lock}>"$MAIN_OUTPUT_ROOT/.physical_gpu1.lock"
flock "$gpu1_lock"
bind_reserved_gpu_identity 0
bind_reserved_gpu_identity 1

run_gpu_scene() {
  local physical_index="$1"
  local scene="$2"
  if [[ "$physical_index" != "0" && "$physical_index" != "1" ]]; then
    echo "serial plan contains invalid physical GPU index: $physical_index" >&2
    return 2
  fi
  local gpu_uuid="${GPU_UUIDS[$physical_index]}"
  local gpu_bus_id="${GPU_BUS_IDS[$physical_index]}"

  validate_manifest_scene "$scene"
  local source="$SOURCE_ROOT/$scene"
  local result_root="$OUTPUT_ROOT/$scene/eval_full_mask_random_walker"
  local result="$result_root/${scene}_evaluation.json"
  local receipt_dir="$SCENE_RECEIPT_ROOT/$scene"
  local receipt="$receipt_dir/scene_receipt.json"
  exec {scene_lock}>"$LOCK_ROOT/$scene.lock"
  flock "$scene_lock"
  if [[ -e "$result" || -e "$receipt" || -L "$result" || -L "$receipt" ]]; then
    if [[ ! -s "$result" || -L "$result" \
          || ! -s "$receipt" || -L "$receipt" ]]; then
      echo "$scene has incomplete result/receipt evidence; quarantine required" >&2
      return 1
    fi
    validate_scene_receipt "$scene"
    flock -u "$scene_lock"
    exec {scene_lock}>&-
    unset scene_lock
    return 0
  fi

  local field="$source/canonical_d256_l128_capability_first.pth"
  local capability="$source/official_dino_sam3_views.pt"
  local graph="$source/shared_support_graph_k16.pt"
  local reliability=""
  if [[ "$FORWARD_BETA_VARIANT" == "v2" ]]; then
    reliability="$(
      CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
        bash "$RUN_REPO_PYTHON" - "$RUN_MANIFEST" "$scene" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
record = manifest["source_artifacts"][sys.argv[2]][
    "canonical_primitive_reliability_v1.pt"
]
print(record["path"])
PY
    )"
  fi
  local field_sha
  field_sha="$(sha256sum "$field" | awk '{print $1}')"
  mkdir -p "$result_root" "$receipt_dir" "$SCENE_ATTEMPT_ROOT/$scene"
  local attempt_index=1
  while [[ -e "$SCENE_ATTEMPT_ROOT/$scene/attempt_$(printf '%04d' "$attempt_index")" ]]; do
    attempt_index=$((attempt_index + 1))
  done
  local attempt_root="$SCENE_ATTEMPT_ROOT/$scene/attempt_$(printf '%04d' "$attempt_index")"
  mkdir -p "$attempt_root"
  local telemetry="$attempt_root/telemetry.csv"
  local owner_audit="$attempt_root/owner_audit.csv"
  local attestation="$attempt_root/cuda_attestation.json"
  local command_record="$attempt_root/command.json"
  local postcheck="$attempt_root/postcheck.json"
  local attempt_log="$attempt_root/evaluator.log"
  local evaluator_command=(
      bash "$RUN_REPO_PYTHON"
      "$REPO_ROOT/radio_gs/scripts/eval_nvos_gaussian_first.py"
      --manifest "$MANIFEST"
      --queue-root "$(dirname "$QUEUE_PLAN")"
      --scene-id "$scene"
      --output-dir "$result_root"
      --run-manifest "$RUN_MANIFEST"
      --device cuda:0
      --gpu-attestation-output "$attestation"
      --expected-gpu-physical-index "$physical_index"
      --expected-gpu-uuid "$gpu_uuid"
      --expected-gpu-bus-id "$gpu_bus_id"
      --radio-checkpoint "$RADIO_CHECKPOINT"
      --region-space sam3
      --support-mode canonical_support
      --canonical-capability-cache "$capability"
      --canonical-support-graph "$graph"
      --canonical-field-sha256 "$field_sha"
      --prompt-registration-mode raster_adjoint
      --prompt-registration-scale 1.0
      --alpha-threshold 0.0
      --depth-tolerance 0.08
      --relative-depth-tolerance 0.02
      --support-threshold 0.0
      --prototype-count 4
      --prototype-strategy spherical_mean_fps
      --registered-seed-construction joint_signed
      --registered-observation-fusion probability_mixture
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
      --graph-legacy-residual 0.0
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
  if [[ "$FORWARD_BETA_VARIANT" == "v1" ]]; then
    evaluator_command+=(
      --candidate-id nvos-forward-beta-coverage-v1
      --registered-forward-unary beta_coverage_v1
    )
  else
    evaluator_command+=(
      --candidate-id "$CANDIDATE_ID"
      --registered-forward-unary "$FORWARD_MODE"
      --canonical-reliability-cache "$reliability"
    )
  fi
  CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
    bash "$RUN_REPO_PYTHON" "$SCENE_AUTHORITY" prepare-scene \
      --output "$command_record" --run-manifest "$RUN_MANIFEST" \
      --scene "$scene" --result "$result" --telemetry "$telemetry" \
      --owner-audit "$owner_audit" --attestation "$attestation" \
      --postcheck "$postcheck" --receipt "$receipt" \
      --evaluator-log "$attempt_log" --guard "$THERMAL_GUARD" \
      --physical-index "$physical_index" --gpu-uuid "$gpu_uuid" \
    --gpu-bus-id "$gpu_bus_id" -- "${evaluator_command[@]}" >/dev/null

  local command_status=0
  GPU="$physical_index" CUDA_DEVICE_ORDER="$FROZEN_CUDA_DEVICE_ORDER" \
      NVIDIA_VISIBLE_DEVICES="$gpu_uuid" CUDA_VISIBLE_DEVICES="$gpu_uuid" \
      GPU_TELEMETRY_LOG="$telemetry" GPU_OWNER_AUDIT_LOG="$owner_audit" \
      GPU_OWNER_PID_NAMESPACE_MODE="$FROZEN_OWNER_MODE" \
      GPU_MAX_TEMP_C="$GPU_MAX_TEMP_C" \
      GPU_START_MAX_TEMP_C="$GPU_START_MAX_TEMP_C" \
      GPU_SOFT_PAUSE_TEMP_C="$GPU_SOFT_PAUSE_TEMP_C" \
      GPU_SOFT_RESUME_TEMP_C="$GPU_SOFT_RESUME_TEMP_C" \
      GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
      GPU_MAX_POWER_LIMIT_W="$GPU_MAX_POWER_LIMIT_W" \
      GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES="$GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES" \
      GPU_PEER_INDEX="" GPU_PEER_PAUSE_TEMP_C=0 GPU_PEER_RESUME_TEMP_C=0 \
      GPU_PEER_QUIET_SECONDS=0 GPU_PEER_MAX_POWER_W=0 \
      GPU_PEER_MAX_MEMORY_MIB=0 GPU_PEER_MAX_UTIL_PCT=100 \
    bash "$THERMAL_GUARD" -- "${evaluator_command[@]}" \
    >"$attempt_log" 2>&1 || command_status=$?
  if (( command_status != 0 )); then
    echo "forward-Beta GPU$physical_index scene failed: $scene (status=$command_status)" >&2
    return "$command_status"
  fi

  CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
    bash "$RUN_REPO_PYTHON" "$SCENE_AUTHORITY" postcheck-scene \
      --output "$postcheck" --command-record "$command_record" >/dev/null
    CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
      bash "$RUN_REPO_PYTHON" "$SCENE_AUTHORITY" finalize-scene \
      --output "$receipt" --command-record "$command_record" \
    --postcheck "$postcheck" >/dev/null
  validate_scene_receipt "$scene"
  flock -u "$scene_lock"
  exec {scene_lock}>&-
  unset scene_lock
}

for scene_gpu in "${SERIAL_SCENE_GPU_PLAN[@]}"; do
  physical_index="${scene_gpu%%:*}"
  scene="${scene_gpu#*:}"
  run_gpu_scene "$physical_index" "$scene"
done

CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
  bash "$RUN_REPO_PYTHON" "$SNAPSHOT_STAGER" validate-snapshot \
  --snapshot-root "$REPO_ROOT" >/dev/null
for scene in "${ALL_SCENES[@]}"; do
  validate_scene_receipt "$scene"
done
exec {aggregate_lock}>"$LOCK_ROOT/aggregate.lock"
flock "$aggregate_lock"
CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES="" \
  bash "$RUN_REPO_PYTHON" "$AGGREGATOR" \
  --run-manifest "$RUN_MANIFEST" --result-root "$OUTPUT_ROOT" \
  --receipt-root "$SCENE_RECEIPT_ROOT" \
  --output "$OUTPUT_ROOT/summary.json" \
  >"$OUTPUT_ROOT/aggregate.log" 2>&1
for scene in "${ALL_SCENES[@]}"; do
  validate_scene_receipt "$scene"
done
flock -u "$aggregate_lock"
exec {aggregate_lock}>&-
flock -u "$gpu1_lock"
exec {gpu1_lock}>&-
flock -u "$gpu0_lock"
exec {gpu0_lock}>&-
