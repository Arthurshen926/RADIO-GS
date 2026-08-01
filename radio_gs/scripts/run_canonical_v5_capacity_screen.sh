#!/usr/bin/env bash

# Query-free canonical-field capacity screen.  All three candidates share the
# same normalized mean-resultant MPR caches.  Width (W), spatial content (S),
# and nonlinear depth (D) are added one at a time over the reliability control.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-1}"
GLOBAL_GPU1_LOCK="$REPO_ROOT/output/.physical_gpu1.lock"
if [[ "${CANONICAL_V5_PHYSICAL_GPU_LOCK_HELD:-0}" != "1" ]]; then
  mkdir -p "$REPO_ROOT/output"
  exec python3 - "$GLOBAL_GPU1_LOCK" "$0" "$@" <<'PY'
import errno
import fcntl
import os
import stat
import sys

lock_path, script, *arguments = sys.argv[1:]
if not hasattr(os, "O_NOFOLLOW"):
    raise SystemExit("physical GPU1 lock requires O_NOFOLLOW support")
try:
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
except OSError as exc:
    if exc.errno == errno.ELOOP:
        raise SystemExit(f"refusing symlinked physical GPU1 lock: {lock_path}")
    raise
info = os.fstat(descriptor)
path_info = os.stat(lock_path, follow_symlinks=False)
if not stat.S_ISREG(info.st_mode) or (
    info.st_dev != path_info.st_dev or info.st_ino != path_info.st_ino
):
    raise SystemExit(f"physical GPU1 lock is not one stable regular file: {lock_path}")
try:
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("another RADIO-GS task owns physical GPU1")
os.set_inheritable(descriptor, True)
environment = dict(os.environ)
environment["CANONICAL_V5_PHYSICAL_GPU_LOCK_HELD"] = "1"
environment["CANONICAL_V5_PHYSICAL_GPU_LOCK_FD"] = str(descriptor)
os.execve("/bin/bash", ["bash", script, *arguments], environment)
PY
fi
python3 - "$GLOBAL_GPU1_LOCK" \
  "${CANONICAL_V5_PHYSICAL_GPU_LOCK_FD:-}" <<'PY'
import os
import stat
import sys

path, raw_descriptor = sys.argv[1:]
try:
    descriptor = int(raw_descriptor)
    info = os.fstat(descriptor)
    path_info = os.stat(path, follow_symlinks=False)
except (OSError, TypeError, ValueError) as exc:
    raise SystemExit("physical GPU1 lock inheritance is invalid") from exc
if not stat.S_ISREG(info.st_mode) or (
    info.st_dev != path_info.st_dev or info.st_ino != path_info.st_ino
):
    raise SystemExit("physical GPU1 lock identity changed during exec")
PY

CONFIG="${CONFIG:?set CONFIG to the frozen render config}"
GEOMETRY_CHECKPOINT="${GEOMETRY_CHECKPOINT:?set GEOMETRY_CHECKPOINT}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:?set RADIO_CHECKPOINT}"
EXCLUDE_FRAME_IDS="${EXCLUDE_FRAME_IDS:?set held-out frame IDs or file}"
FIDELITY_FRAME_IDS="${FIDELITY_FRAME_IDS:-$EXCLUDE_FRAME_IDS}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set a new query-free v5 output root}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-0}"
CAPABILITY_MAP_SOURCE="${CAPABILITY_MAP_SOURCE:-official_extracted}"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
GPU_TELEMETRY_LOG="${GPU_TELEMETRY_LOG:-$OUTPUT_ROOT/gpu${GPU}_telemetry.csv}"
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
RADIO_THERMAL_PACING_SECONDS_PER_IMAGE="${RADIO_THERMAL_PACING_SECONDS_PER_IMAGE:-8.0}"
V5_CONTINUOUS_STAGE_POLICY="${V5_CONTINUOUS_STAGE_POLICY:-uncharacterized_continuous_hard_abort_only}"
V5_CONTINUOUS_CANARY_RECORD="${V5_CONTINUOUS_CANARY_RECORD:-}"
V5_CONTINUOUS_CANARY_RECORD_SHA256="${V5_CONTINUOUS_CANARY_RECORD_SHA256:-}"

if [[ "$GPU" != "1" ]]; then
  echo "canonical-v5 is assigned to physical GPU1; got GPU=$GPU" >&2
  exit 2
fi

if [[ "$CAPABILITY_MAP_SOURCE" != "official_extracted" ]]; then
  echo "the v5 promotion screen requires CAPABILITY_MAP_SOURCE=official_extracted" >&2
  exit 2
fi
if [[ "$V5_CONTINUOUS_STAGE_POLICY" != "uncharacterized_continuous_hard_abort_only" ]]; then
  echo "unsupported v5 continuous-stage policy" >&2
  exit 2
fi
if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$RADIO_THERMAL_PACING_SECONDS_PER_IMAGE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "RADIO_THERMAL_PACING_SECONDS_PER_IMAGE must be finite and non-negative" >&2
  exit 2
fi
for required in \
  "$CONFIG" "$GEOMETRY_CHECKPOINT" "$RADIO_CHECKPOINT" "$THERMAL_GUARD"; do
  if [[ ! -s "$required" ]]; then
    echo "missing v5 screen input: $required" >&2
    exit 2
  fi
done
MPR_ROOT="$OUTPUT_ROOT/mpr"
FIELD_ROOT="$OUTPUT_ROOT/fields"
AUDIT_ROOT="$OUTPUT_ROOT/fidelity"
LOG_ROOT="$OUTPUT_ROOT/logs"
RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
RESPONSIBILITY="$MPR_ROOT/shared_responsibility.pt"
RAW_MPR="$MPR_ROOT/raw_radio_mean_resultant.pt"
DINO_MPR="$MPR_ROOT/dino_v3_mean_resultant.pt"
SAM3_MPR="$MPR_ROOT/sam3_mean_resultant.pt"
mkdir -p "$OUTPUT_ROOT"
RUN_LOCK="$OUTPUT_ROOT/.canonical_v5.lock"
exec {RUN_LOCK_FD}>"$RUN_LOCK"
if ! flock -n "$RUN_LOCK_FD"; then
  echo "another canonical-v5 runner owns OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$MPR_ROOT" "$FIELD_ROOT" "$AUDIT_ROOT" "$LOG_ROOT"

mapfile -t FEATURE_INPUTS < <(
  bash radio_gs/scripts/run_repo_python.sh - "$CONFIG" <<'PY'
import sys
from pathlib import Path

from radio_gs.config import load_config

config = load_config(sys.argv[1])
print(config.scene)
print(Path(config.scene_root).resolve() / "color")
print(Path(config.feature_dir).resolve())
print(config.radio_repo or "/root/RADIO")
print(config.radio_version)
PY
)
if [[ "${#FEATURE_INPUTS[@]}" -ne 5 ]]; then
  echo "could not resolve verified feature-extraction inputs" >&2
  exit 2
fi
FEATURE_SCENE="${FEATURE_INPUTS[0]}"
FEATURE_IMAGE_DIR="${FEATURE_INPUTS[1]}"
VERIFIED_FEATURE_DIR="${FEATURE_INPUTS[2]}"
FEATURE_RADIO_REPO="${FEATURE_INPUTS[3]}"
FEATURE_RADIO_VERSION="${FEATURE_INPUTS[4]}"

GPU_INFO=""
for candidate in /proc/driver/nvidia/gpus/*/information; do
  if [[ -r "$candidate" ]] \
    && [[ "$(awk '/Device Minor:/ {print $3}' "$candidate")" == "$GPU" ]]; then
    GPU_INFO="$candidate"
    break
  fi
done
if [[ -z "$GPU_INFO" ]]; then
  echo "physical GPU $GPU has no NVIDIA driver record" >&2
  exit 2
fi
GPU_BUS_ID="$(awk '/Bus Location:/ {print $3}' "$GPU_INFO")"
GPU_CONFIG="/sys/bus/pci/devices/$GPU_BUS_ID/config"
GPU_CONFIG_PREFIX="$(od -An -tx1 -N16 "$GPU_CONFIG" 2>/dev/null | tr -d ' \n')"
if [[ -z "$GPU_CONFIG_PREFIX" || "$GPU_CONFIG_PREFIX" =~ ^f+$ ]]; then
  echo "physical GPU $GPU PCIe configuration space is not responding" >&2
  exit 2
fi
if ! timeout --kill-after=2s 10s nvidia-smi -i "$GPU" >/dev/null; then
  echo "physical GPU $GPU is not available to the current container" >&2
  exit 2
fi
GPU_UUID="$(
  timeout --kill-after=2s 10s nvidia-smi -i "$GPU" \
    --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]'
)"
if [[ ! "$GPU_UUID" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  echo "physical GPU $GPU returned an invalid UUID" >&2
  exit 2
fi

assert_gpu1_unowned() {
  local owners
  owners="$(
    timeout --kill-after=2s 10s nvidia-smi \
      --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits \
      | awk -F', *' -v uuid="$GPU_UUID" '$1 == uuid {print $2}' \
      | paste -sd, -
  )" || {
    echo "could not query compute owners for physical GPU1 UUID $GPU_UUID" >&2
    return 2
  }
  if [[ -n "$owners" ]]; then
    echo "physical GPU1 UUID $GPU_UUID already has compute owner(s): $owners" >&2
    return 2
  fi
}

assert_gpu1_unowned

if [[ -z "$V5_CONTINUOUS_CANARY_RECORD" \
      || ! "$V5_CONTINUOUS_CANARY_RECORD_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "canonical-v5 continuous MPR/field/audit launch remains closed" >&2
  echo "provide an externally trusted V5_CONTINUOUS_CANARY_RECORD and SHA256" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="" bash radio_gs/scripts/run_repo_python.sh - \
  "$V5_CONTINUOUS_CANARY_RECORD" \
  "$V5_CONTINUOUS_CANARY_RECORD_SHA256" \
  "$GPU_UUID" "$V5_CONTINUOUS_STAGE_POLICY" "$THERMAL_GUARD" \
  "$GPU_MAX_TEMP_C" "$GPU_MAX_POWER_LIMIT_W" <<'PY'
import sys

from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file

(
    path,
    expected_sha256,
    gpu_uuid,
    policy,
    thermal_guard,
    maximum_temperature,
    maximum_power_limit,
) = sys.argv[1:]
record, observed_sha256, _ = load_json_object(
    path,
    expected_sha256=expected_sha256,
    label="canonical-v5 continuous-stage canary authority",
)
expected_keys = {
    "schema_version",
    "contract",
    "status",
    "physical_gpu_uuid",
    "stage_policy",
    "thermal_guard_sha256",
    "maximum_observed_temperature_c",
    "observed_power_limit_w",
    "hard_abort_only",
    "completed_at",
}
if set(record) != expected_keys:
    raise SystemExit("continuous-stage canary authority schema differs")
if (
    record["schema_version"] != 1
    or record["contract"] != "canonical-v5-continuous-gpu-canary-v1"
    or record["status"] != "passed"
    or record["physical_gpu_uuid"] != gpu_uuid
    or record["stage_policy"] != policy
    or record["thermal_guard_sha256"] != sha256_file(thermal_guard)
    or record["hard_abort_only"] is not True
    or not str(record["completed_at"])
    or float(record["maximum_observed_temperature_c"])
    >= float(maximum_temperature)
    or float(record["observed_power_limit_w"]) > float(maximum_power_limit)
):
    raise SystemExit("continuous-stage canary authority does not approve this run")
if observed_sha256 != expected_sha256:
    raise SystemExit("continuous-stage canary authority digest differs")
PY

run_guarded_command() {
  assert_gpu1_unowned
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
    bash "$THERMAL_GUARD" -- "$@"
}

validate_feature_output() {
  local feature_root="$1"
  CUDA_VISIBLE_DEVICES="" bash radio_gs/scripts/run_repo_python.sh - \
    "$feature_root" <<'PY'
import sys
from pathlib import Path

from radio_gs.scripts.extract_radio_features import (
    _validate_final_output_bundle,
)

root = Path(sys.argv[1]).resolve()
_validate_final_output_bundle(root)
PY
}

FEATURE_STAGING="${VERIFIED_FEATURE_DIR}.incomplete"
FEATURE_STAGING_LOCK="${VERIFIED_FEATURE_DIR}.incomplete.lock"
if [[ ! -s "$VERIFIED_FEATURE_DIR/frame_manifest.json" ]]; then
  mkdir -p "$(dirname "$VERIFIED_FEATURE_DIR")"
  exec {FEATURE_STAGING_LOCK_FD}>"$FEATURE_STAGING_LOCK"
  if ! flock -n "$FEATURE_STAGING_LOCK_FD"; then
    echo "another v5 extraction owns the verified feature staging path" >&2
    exit 2
  fi
  # Recheck after taking the lock in case another invocation just promoted it.
  if [[ ! -s "$VERIFIED_FEATURE_DIR/frame_manifest.json" ]]; then
    if [[ ! -d "$FEATURE_IMAGE_DIR" ]]; then
      echo "verified feature source images are missing: $FEATURE_IMAGE_DIR" >&2
      exit 2
    fi
    if [[ -d "$VERIFIED_FEATURE_DIR" ]] \
      && [[ -n "$(find "$VERIFIED_FEATURE_DIR" -mindepth 1 -print -quit)" ]]; then
      echo "verified feature directory is partial and will not be overwritten" >&2
      exit 2
    fi
    if [[ -d "$VERIFIED_FEATURE_DIR" ]]; then
      rmdir "$VERIFIED_FEATURE_DIR"
    fi
    if [[ -e "$FEATURE_STAGING" && ! -d "$FEATURE_STAGING" ]]; then
      echo "verified feature staging path is not a directory: $FEATURE_STAGING" >&2
      exit 2
    fi
    run_guarded_command bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/extract_radio_features.py \
      --scene "$FEATURE_SCENE" \
      --image_dir "$FEATURE_IMAGE_DIR" \
      --output_dir "$FEATURE_STAGING" \
      --radio_repo "$FEATURE_RADIO_REPO" \
      --radio_version "$FEATURE_RADIO_VERSION" \
      --radio_checkpoint "$RADIO_CHECKPOINT" \
      --batch_size 1 \
      --frame_stride 1 \
      --frame-id-mode auto \
      --extract_adaptors \
      --adaptor_names dino_v3_7b,sam3 \
      --resolution_scale 1.0 \
      --device cuda:0 \
      --amp \
      --skip_pca_stats \
      --resume-partial \
      --radio-thermal-pacing-seconds-per-image \
      "$RADIO_THERMAL_PACING_SECONDS_PER_IMAGE" \
      >>"$LOG_ROOT/verified_feature_extraction.log" 2>&1
    if [[ ! -s "$FEATURE_STAGING/frame_manifest.json" ]]; then
      echo "verified feature extraction did not produce its manifest" >&2
      exit 1
    fi
    if ! validate_feature_output "$FEATURE_STAGING"; then
      echo "verified feature staging failed final output-bundle validation" >&2
      echo "move it to a separate quarantine path, then rerun: $FEATURE_STAGING" >&2
      exit 2
    fi
    if [[ -e "$VERIFIED_FEATURE_DIR" ]]; then
      echo "verified feature target appeared while staging was locked" >&2
      exit 2
    fi
    mv "$FEATURE_STAGING" "$VERIFIED_FEATURE_DIR"
  fi
  flock -u "$FEATURE_STAGING_LOCK_FD"
fi
if ! validate_feature_output "$VERIFIED_FEATURE_DIR"; then
  echo "verified feature target is stale or partial; refusing overwrite" >&2
  echo "move it to a separate quarantine path, then rerun: $VERIFIED_FEATURE_DIR" >&2
  exit 2
fi
mapfile -t FEATURE_BUNDLE_AUTHORITY < <(
  CUDA_VISIBLE_DEVICES="" bash radio_gs/scripts/run_repo_python.sh - \
    "$VERIFIED_FEATURE_DIR" <<'PY'
import sys

from radio_gs.scripts.extract_radio_features import _validate_final_output_bundle

result = _validate_final_output_bundle(sys.argv[1])
print(result["output_bundle_sha256"])
print(result["manifest_sha256"])
PY
)
if [[ "${#FEATURE_BUNDLE_AUTHORITY[@]}" -ne 2 \
      || ! "${FEATURE_BUNDLE_AUTHORITY[0]}" =~ ^[0-9a-f]{64}$ \
      || ! "${FEATURE_BUNDLE_AUTHORITY[1]}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "verified feature bundle did not yield stable SHA-256 authority" >&2
  exit 2
fi
FEATURE_OUTPUT_BUNDLE_SHA256="${FEATURE_BUNDLE_AUTHORITY[0]}"
FEATURE_MANIFEST_SHA256="${FEATURE_BUNDLE_AUTHORITY[1]}"
mapfile -t INPUT_SHA256_VALUES < <(
  CUDA_VISIBLE_DEVICES="" bash radio_gs/scripts/run_repo_python.sh - \
    "$CONFIG" "$GEOMETRY_CHECKPOINT" "$RADIO_CHECKPOINT" <<'PY'
import sys
from radio_gs.utils.immutable_artifacts import sha256_file
for path in sys.argv[1:]:
    print(sha256_file(path))
PY
)
if [[ "${#INPUT_SHA256_VALUES[@]}" -ne 3 \
      || ! "${INPUT_SHA256_VALUES[0]}" =~ ^[0-9a-f]{64}$ \
      || ! "${INPUT_SHA256_VALUES[1]}" =~ ^[0-9a-f]{64}$ \
      || ! "${INPUT_SHA256_VALUES[2]}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "v5 inputs did not yield stable SHA-256 values" >&2
  exit 2
fi
CONFIG_SHA256="${INPUT_SHA256_VALUES[0]}"
GEOMETRY_CHECKPOINT_SHA256="${INPUT_SHA256_VALUES[1]}"
RADIO_CHECKPOINT_SHA256="${INPUT_SHA256_VALUES[2]}"

bash radio_gs/scripts/run_repo_python.sh - \
  "$CONFIG" "$GEOMETRY_CHECKPOINT" "$RADIO_CHECKPOINT" \
  "$EXCLUDE_FRAME_IDS" "$FIDELITY_FRAME_IDS" "$EPOCHS" "$SEED" \
  "$CAPABILITY_MAP_SOURCE" "$OUTPUT_ROOT" "$RUN_MANIFEST" "$0" \
  "$THERMAL_GUARD" "$GPU" "$GPU_MAX_TEMP_C" \
  "$GPU_START_MAX_TEMP_C" "$GPU_MAX_POWER_LIMIT_W" \
  "$GPU_POLL_SECONDS" "$GPU_SOFT_PAUSE_TEMP_C" \
  "$GPU_SOFT_RESUME_TEMP_C" "$GPU_PEER_INDEX" \
  "$GPU_PEER_PAUSE_TEMP_C" "$GPU_PEER_RESUME_TEMP_C" \
  "$GPU_PEER_QUIET_SECONDS" "$GPU_PEER_MAX_POWER_W" \
  "$GPU_PEER_MAX_MEMORY_MIB" "$GPU_PEER_MAX_UTIL_PCT" \
  "$RADIO_THERMAL_PACING_SECONDS_PER_IMAGE" "$GPU_UUID" \
  "$V5_CONTINUOUS_STAGE_POLICY" "$V5_CONTINUOUS_CANARY_RECORD" \
  "$V5_CONTINUOUS_CANARY_RECORD_SHA256" "$CONFIG_SHA256" \
  "$GEOMETRY_CHECKPOINT_SHA256" "$RADIO_CHECKPOINT_SHA256" \
  "$FEATURE_OUTPUT_BUNDLE_SHA256" "$FEATURE_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from radio_gs.config import config_to_dict, load_config
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _parse_frame_ids,
    _resolve_extracted_capability_source,
)
from radio_gs.scripts.extract_radio_features import (
    _canonical_json_sha256,
    _python_source_tree_fingerprint,
    _validate_final_output_bundle,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    write_frozen_json,
)

(
    config_arg,
    geometry_arg,
    radio_arg,
    exclude_arg,
    fidelity_arg,
    epochs_arg,
    seed_arg,
    capability_map_source,
    output_arg,
    manifest_arg,
    runner_arg,
    thermal_guard_arg,
    gpu_arg,
    gpu_max_temp_arg,
    gpu_start_max_temp_arg,
    gpu_max_power_limit_arg,
    gpu_poll_seconds_arg,
    gpu_soft_pause_temp_arg,
    gpu_soft_resume_temp_arg,
    gpu_peer_index_arg,
    gpu_peer_pause_temp_arg,
    gpu_peer_resume_temp_arg,
    gpu_peer_quiet_seconds_arg,
    gpu_peer_max_power_arg,
    gpu_peer_max_memory_arg,
    gpu_peer_max_util_arg,
    radio_thermal_pacing_arg,
    gpu_uuid_arg,
    continuous_stage_policy_arg,
    continuous_canary_arg,
    continuous_canary_sha256_arg,
    expected_config_sha256_arg,
    expected_geometry_sha256_arg,
    expected_radio_sha256_arg,
    expected_feature_bundle_sha256_arg,
    expected_feature_manifest_sha256_arg,
) = sys.argv[1:]


def sha256(path: Path) -> str:
    return sha256_file(path)


config_path = Path(config_arg).resolve()
geometry_path = Path(geometry_arg).resolve()
radio_path = Path(radio_arg).resolve()
runner_path = Path(runner_arg).resolve()
repo_root = runner_path.parents[2]
output_root = Path(output_arg).resolve()
manifest_path = Path(manifest_arg)
config = load_config(str(config_path))
resolved_config = config_to_dict(config)
resolved_config_sha256 = hashlib.sha256(
    json.dumps(
        resolved_config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
radio_sha256 = sha256(radio_path)
if (
    sha256(config_path) != expected_config_sha256_arg
    or sha256(geometry_path) != expected_geometry_sha256_arg
    or radio_sha256 != expected_radio_sha256_arg
):
    raise SystemExit("v5 immutable input changed before manifest creation")
implementation_relpaths = (
    "radio_gs/scripts/extract_radio_features.py",
    "radio_gs/scripts/build_gaussian_multiview_teacher_cache.py",
    "radio_gs/scripts/train_canonical_radio_field.py",
    "radio_gs/scripts/audit_canonical_capability_fidelity.py",
    "radio_gs/scripts/eval_lerf_grounding.py",
    "radio_gs/evaluation/capability_fidelity.py",
    "radio_gs/field/canonical_gaussian_field.py",
    "radio_gs/field/primitive_fusion.py",
    "radio_gs/field/spatial_hash.py",
    "radio_gs/training/canonical_field_losses.py",
    "radio_gs/training/tensor_cache_io.py",
    "radio_gs/rendering/coefficient_renderer.py",
    "radio_gs/field/checkpoint.py",
    "radio_gs/interfaces/frozen_radio_views.py",
    "radio_gs/models/radio_adaptors.py",
    "radio_gs/utils/checkpoint_io.py",
    "radio_gs/utils/immutable_artifacts.py",
    "radio_gs/scripts/run_repo_python.sh",
    "radio_gs/scripts/run_with_gpu_thermal_guard.sh",
)
implementation_sources = {
    relative: sha256(repo_root / relative)
    for relative in implementation_relpaths
}
implementation_source_tree = _python_source_tree_fingerprint(
    repo_root / "radio_gs"
)
feature_dir = Path(str(config.feature_dir))
feature_manifest_path = feature_dir / "frame_manifest.json"
try:
    (
        feature_manifest_payload,
        feature_manifest_sha256,
        _feature_manifest_source,
    ) = load_json_object(
        feature_manifest_path,
        label="verified RADIO feature manifest",
    )
except (OSError, ValueError) as exc:
    raise SystemExit(
        f"verified feature manifest is unreadable: {feature_manifest_path}"
    ) from exc
feature_bundle_validation = _validate_final_output_bundle(
    feature_dir,
    feature_manifest_payload,
    expected_manifest_sha256=feature_manifest_sha256,
    expected_output_bundle_sha256=expected_feature_bundle_sha256_arg,
)
if feature_manifest_sha256 != expected_feature_manifest_sha256_arg:
    raise SystemExit("verified feature manifest changed before run anchoring")
feature_radio = feature_manifest_payload.get("radio")
feature_execution = feature_manifest_payload.get("execution")
if not isinstance(feature_radio, dict) or not isinstance(feature_execution, dict):
    raise SystemExit("verified feature source/runtime provenance is incomplete")
if (
    feature_radio.get("checkpoint_sha256") != radio_sha256
    or feature_radio.get("checkpoint_provenance")
    != "explicit_file_sha256"
    or feature_radio.get("checkpoint_load_contract")
    != "external_sha256_same_fd_restricted_pickle_hub_injection_v1"
):
    raise SystemExit("verified feature checkpoint loading was not restricted")
recorded_radio_tree = feature_radio.get("python_source_tree")
if not isinstance(recorded_radio_tree, dict):
    raise SystemExit("verified feature manifest has no RADIO source-tree digest")
current_radio_tree = _python_source_tree_fingerprint(feature_radio["repo"])
if current_radio_tree != recorded_radio_tree:
    raise SystemExit("local RADIO Python source tree differs from extraction")
runtime_fingerprint = feature_execution.get("runtime_fingerprint")
if (
    not isinstance(runtime_fingerprint, dict)
    or runtime_fingerprint.get("contract")
        != "radio-extraction-runtime-fingerprint-v2"
    or runtime_fingerprint.get("fingerprint_sha256")
    != _canonical_json_sha256(
        {
            key: value
            for key, value in runtime_fingerprint.items()
            if key != "fingerprint_sha256"
        }
    )
):
    raise SystemExit("verified feature runtime fingerprint is invalid")
official_sources = {}
for name in ("dino_v3", "sam3"):
    source = _resolve_extracted_capability_source(
        feature_dir,
        name,
        expected_radio_checkpoint_sha256=radio_sha256,
        expected_scene=str(config.scene),
        expected_image_dir=(
            Path(str(config.scene_root)).resolve() / "color"
        ),
        expected_output_bundle_sha256=feature_bundle_validation[
            "output_bundle_sha256"
        ],
    )
    source_record = {
        key: source[key]
        for key in (
            "adaptor_name",
            "native_grid",
            "frame_manifest",
            "frame_manifest_sha256",
            "radio_version",
            "radio_checkpoint",
            "radio_checkpoint_sha256",
            "radio_checkpoint_provenance",
            "radio_checkpoint_load_contract",
            "scene",
            "image_dir",
            "frame_indices_sha256",
            "output_bundle_sha256",
            "execution",
        )
    }
    extraction_execution = feature_manifest_payload.get("execution")
    if not isinstance(extraction_execution, dict):
        raise SystemExit(
            "official extracted capability source lacks its execution contract"
        )
    if (
        extraction_execution.get("resume_partial") is not True
        or extraction_execution.get("atomic_tensor_commit")
        != "same_directory_temp_then_os_replace_v1"
        or extraction_execution.get("committed_frame_validation")
        != "same_fd_sha256_weights_only_dtype_shape_finite_v2"
        or extraction_execution.get("invalid_or_missing_frame_policy")
        != "recompute_entire_frame_v1"
        or extraction_execution.get("pacing_order")
        != "frame_commit_then_cuda_synchronize_then_sleep_v1"
        or float(
            extraction_execution.get(
                "radio_thermal_pacing_seconds_per_image",
                -1.0,
            )
        )
        != float(radio_thermal_pacing_arg)
    ):
        raise SystemExit(
            "official extracted capability source violates the v5 extraction "
            "safety contract"
        )
    source_record["output_bundle_sha256"] = feature_bundle_validation[
        "output_bundle_sha256"
    ]
    source_record["resume_contract_sha256"] = feature_bundle_validation[
        "resume_contract_sha256"
    ]
    source_record["radio_source_tree_sha256"] = recorded_radio_tree[
        "tree_sha256"
    ]
    source_record["runtime_fingerprint_sha256"] = runtime_fingerprint[
        "fingerprint_sha256"
    ]
    source_record["feature_extraction_execution"] = extraction_execution
    official_sources[name] = source_record
payload = {
    "schema_version": 1,
    "screen": "canonical-v5-query-free-capacity",
    "config": str(config_path),
    "config_sha256": expected_config_sha256_arg,
    "resolved_config_sha256": resolved_config_sha256,
    "geometry_checkpoint": str(geometry_path),
    "geometry_checkpoint_sha256": expected_geometry_sha256_arg,
    "radio_checkpoint": str(radio_path),
    "radio_checkpoint_sha256": expected_radio_sha256_arg,
    "exclude_frame_ids": sorted(_parse_frame_ids(exclude_arg)),
    "fidelity_frame_ids": sorted(_parse_frame_ids(fidelity_arg)),
    "capability_map_source": capability_map_source,
    "official_capability_sources": official_sources,
    "epochs": int(epochs_arg),
    "seed": int(seed_arg),
    "runner": str(runner_path),
    "runner_sha256": sha256(runner_path),
    "feature_extraction_safety_contract": {
        "resume_partial": True,
        "staging": "deterministic_sibling_incomplete_v1",
        "staging_trust_boundary": (
            "mutable_until_output_bundle_is_anchored_by_run_manifest"
        ),
        "radio_python_source_tree": "ordered-relative-python-source-tree-sha256-v1",
        "runtime_fingerprint": "radio-extraction-runtime-fingerprint-v2",
        "radio_checkpoint_load": (
            "external_sha256_same_fd_restricted_pickle_hub_injection_v1"
        ),
        "atomic_tensor_commit": "same_directory_temp_then_os_replace_v1",
        "committed_frame_validation": (
            "same_fd_sha256_weights_only_dtype_shape_finite_v2"
        ),
        "final_output_bundle": "radio-feature-output-bundle-v1",
        "final_output_bundle_sha256": feature_bundle_validation[
            "output_bundle_sha256"
        ],
        "invalid_or_missing_frame_policy": "recompute_entire_frame_v1",
        "radio_thermal_pacing_seconds_per_image": float(
            radio_thermal_pacing_arg
        ),
        "pacing_order": "frame_commit_then_cuda_synchronize_then_sleep_v1",
    },
    "thermal_safety_contract": {
        "guard": str(Path(thermal_guard_arg).resolve()),
        "guard_sha256": sha256(Path(thermal_guard_arg)),
        "physical_gpu": int(gpu_arg),
        "physical_gpu_uuid": gpu_uuid_arg,
        "maximum_temperature_c": int(gpu_max_temp_arg),
        "maximum_start_temperature_c": int(gpu_start_max_temp_arg),
        "maximum_power_limit_w": float(gpu_max_power_limit_arg),
        "poll_seconds": int(gpu_poll_seconds_arg),
        "soft_pause_temperature_c": int(gpu_soft_pause_temp_arg),
        "soft_resume_temperature_c": int(gpu_soft_resume_temp_arg),
        "peer_gpu": int(gpu_peer_index_arg),
        "peer_pause_temperature_c": int(gpu_peer_pause_temp_arg),
        "peer_resume_temperature_c": int(gpu_peer_resume_temp_arg),
        "peer_quiet_seconds_before_launch": int(gpu_peer_quiet_seconds_arg),
        "peer_maximum_power_w": float(gpu_peer_max_power_arg),
        "peer_maximum_memory_mib": int(gpu_peer_max_memory_arg),
        "peer_maximum_utilization_percent": int(gpu_peer_max_util_arg),
    },
    "continuous_stage_safety_contract": {
        "mpr": continuous_stage_policy_arg,
        "field": continuous_stage_policy_arg,
        "audit": continuous_stage_policy_arg,
        "formal_launch_gate": "externally_trusted_short_canary_record_v1",
        "canary_record": str(Path(continuous_canary_arg).resolve()),
        "canary_record_sha256": continuous_canary_sha256_arg,
    },
    "implementation_sources": implementation_sources,
    "implementation_source_tree": implementation_source_tree,
    "fixed_training_contract": {
        "epochs": int(epochs_arg),
        "seed": int(seed_arg),
        "observation_contract": "canonical-mpr-v1",
        "raster_reliability_mode": "mean_resultant",
        "coefficient_dim": 256,
        "local_dim": 128,
        "fusion_reliability": True,
        "hash": {
            "levels": 8,
            "features_per_level": 2,
            "log2_size": 15,
            "base_resolution": 8,
            "max_resolution": 512,
            "hidden_dim": 64,
        },
        "loss_weights": {"mpr": 1.0, "dino": 0.2, "sam3": 0.2},
        "regularization_weights": {
            "coefficient": 1e-5,
            "basis_orthogonality": 1e-3,
            "weight_decay": 1e-5,
        },
        "pca_samples": 50000,
        "standardize": True,
        "freeze_basis": False,
        "batch_size": 4096,
        "eval_batch_size": 16384,
        "learning_rate": 0.002,
        "validation_fraction": 0.05,
        "target_cosine": 0.985,
        "candidates": {
            "v5_r_reliability": [0, 192, 0],
            "v5_w_width": [0, 512, 0],
            "v5_s_spatial": [64, 512, 0],
            "v5_d_deep": [64, 512, 2],
        },
    },
    "fixed_audit_contract": {
        "alpha_threshold": 0.02,
        "support_eps": 1e-6,
        "boundary_quantile": 0.2,
        "residual_mode": "none",
    },
    "fixed_selection_contract": {
        "baseline": "v5_r_reliability",
        "max_mean_dense_drop": 0.005,
        "max_p05_dense_drop": 0.01,
        "max_unsupported_fraction": 0.005,
        "min_relation_gain": 0.005,
        "objective": (
            "maximize_mean_official_dino_sam_affinity_pearson_and_"
            "boundary_margin_retention_under_dense_and_support_guards"
        ),
    },
    "benchmark_queries_opened": False,
    "benchmark_masks_opened": False,
}
if manifest_path.exists():
    previous, _digest, _source = load_json_object(
        manifest_path,
        label="canonical-v5 run manifest",
    )
    if previous != payload:
        raise SystemExit(
            "v5 OUTPUT_ROOT belongs to a different immutable run manifest"
        )
else:
    existing_terminals = [
        path
        for subdir in ("mpr", "fields", "fidelity")
        for path in (output_root / subdir).glob("*")
        if path.is_file()
    ]
    existing_terminals.extend(
        path
        for path in (
            output_root / "capacity_screen.json",
            output_root / "capacity_screen.complete",
            output_root / "capacity_screen.complete.json",
        )
        if path.is_file()
    )
    if existing_terminals:
        raise SystemExit(
            "v5 OUTPUT_ROOT contains stage artifacts but no run manifest"
        )
    write_frozen_json(manifest_path, payload)
PY

stale_stage_error() {
  local stage="$1"
  shift
  echo "v5 stage is partial or fails independent validation: $stage" >&2
  echo "refusing to overwrite or delete existing artifacts" >&2
  echo "move these paths to a separate quarantine directory, then rerun:" >&2
  printf '  %s\n' "$@" >&2
  exit 2
}

validate_torch_stage() {
  local stage="$1"
  local terminal="$2"
  CUDA_VISIBLE_DEVICES="" bash radio_gs/scripts/run_repo_python.sh - \
    "$stage" "$terminal" "$RUN_MANIFEST" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.scripts.build_gaussian_multiview_teacher_cache import (
    _load_responsibility_cache,
)
from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file

stage = sys.argv[1]
terminal = Path(sys.argv[2]).resolve()
report_path = Path(str(terminal) + ".json")
run_manifest_path = Path(sys.argv[3]).resolve()


def sha256(path: Path) -> str:
    return sha256_file(path)


try:
    report, _report_sha256, _ = load_json_object(
        report_path,
        label="v5 stage report",
    )
    manifest, _manifest_sha256, _ = load_json_object(
        run_manifest_path,
        label="v5 run manifest",
    )
except (OSError, ValueError) as exc:
    raise SystemExit(f"stage JSON provenance is unreadable: {exc}") from exc
if Path(str(report.get("output", ""))).resolve() != terminal:
    raise SystemExit("stage report points to another terminal")

if stage.startswith("mpr_"):
    expected_space = {
        "mpr_raw_mean_resultant": "radio",
        "mpr_dino_v3_mean_resultant": "dino_v3",
        "mpr_sam3_mean_resultant": "sam3",
    }.get(stage)
    if expected_space is None:
        raise SystemExit("MPR stage name is not frozen")
    try:
        payload, _terminal_sha256, _ = load_mpr_cache(
            terminal,
            expected_feature_space=expected_space,
            require_reliability=True,
            require_formal_safety=True,
        )
    except Exception as exc:
        raise SystemExit(f"stage terminal cannot be safely reopened: {exc}") from exc
    metadata = payload["metadata"]
    if (
        report.get("metadata")
        != json.loads(json.dumps(metadata, sort_keys=True))
    ):
        raise SystemExit("MPR report metadata differs from its terminal")
    responsibility = terminal.parent / "shared_responsibility.pt"
    try:
        _assignments, responsibility_sha256 = _load_responsibility_cache(
            responsibility,
            expected_contract=dict(
                metadata.get("registration_responsibility_contract", {})
            ),
            num_gaussians=int(payload["xyz"].shape[0]),
            expected_sha256=str(
                metadata.get("registration_responsibility_cache_sha256", "")
            ),
        )
    except Exception as exc:
        raise SystemExit(
            f"responsibility cache cannot be safely reopened: {exc}"
        ) from exc
    if (
        metadata.get("feature_space") != expected_space
        or metadata.get("raster_reliability_mode") != "mean_resultant"
        or metadata.get("normalize_each_view") is not True
        or metadata.get("observation_lifting_contract", {}).get("name")
        != "canonical-mpr-v1"
        or Path(str(metadata.get("config", ""))).resolve()
        != Path(manifest["config"]).resolve()
        or Path(str(metadata.get("checkpoint", ""))).resolve()
        != Path(manifest["geometry_checkpoint"]).resolve()
        or sorted(metadata.get("excluded_frame_ids", []))
        != manifest["exclude_frame_ids"]
        or metadata.get("registration_responsibility_cache_sha256")
        != responsibility_sha256
        or metadata.get("shared_registration_responsibility") is not True
        or metadata.get("feature_output_bundle_sha256")
        != manifest["feature_extraction_safety_contract"][
            "final_output_bundle_sha256"
        ]
    ):
        raise SystemExit("MPR terminal differs from the immutable run contract")
    if expected_space != "radio":
        source = manifest["official_capability_sources"][expected_space]
        if (
            metadata.get("capability_map_source") != "official_extracted"
            or metadata.get("capability_adaptor_execution")
            != "official_c_radio_runtime_adaptor_output"
            or metadata.get("official_adaptor_checkpoint_sha256")
            != manifest["radio_checkpoint_sha256"]
            or metadata.get("official_adaptor_checkpoint_provenance")
            != "explicit_file_sha256"
            or metadata.get("capability_native_map_manifest_sha256")
            != source["frame_manifest_sha256"]
            or metadata.get("capability_native_map_frame_indices_sha256")
            != source["frame_indices_sha256"]
            or metadata.get("capability_native_map_output_bundle_sha256")
            != source["output_bundle_sha256"]
            or metadata.get(
                "capability_native_map_radio_checkpoint_load_contract"
            )
            != source["radio_checkpoint_load_contract"]
        ):
            raise SystemExit("capability MPR source provenance differs")
elif stage.startswith("field_"):
    try:
        _field, payload = load_canonical_field_checkpoint(
            terminal,
            map_location="cpu",
        )
    except Exception as exc:
        raise SystemExit(f"stage terminal cannot be safely reopened: {exc}") from exc
    name = stage.removeprefix("field_")
    candidate = manifest["fixed_training_contract"]["candidates"].get(name)
    if not isinstance(candidate, list) or len(candidate) != 3:
        raise SystemExit("field stage is not a frozen v5 candidate")
    architecture = payload.get("architecture")
    training = payload.get("training_config")
    if not isinstance(architecture, dict) or not isinstance(training, dict):
        raise SystemExit("field checkpoint lacks architecture/training provenance")
    coarse, hidden, blocks = map(int, candidate)
    if (
        architecture.get("coarse_dim") != coarse
        or architecture.get("hidden_dim") != hidden
        or architecture.get("fusion_residual_blocks") != blocks
        or architecture.get("fusion_reliability") is not True
        or architecture.get("use_fusion") is not True
        or payload.get("benchmark_masks_opened") is not False
        or payload.get("text_queries_opened") is not False
        or len(payload.get("history", [])) != int(manifest["epochs"])
        or training.get("seed") != int(manifest["seed"])
        or training.get("epochs") != int(manifest["epochs"])
        or training.get("min_epochs") != int(manifest["epochs"])
        or training.get("spatial_coarse_dim") != coarse
        or training.get("hidden_dim") != hidden
        or training.get("fusion_residual_blocks") != blocks
        or payload.get("feature_output_bundle_sha256")
        != manifest["feature_extraction_safety_contract"][
            "final_output_bundle_sha256"
        ]
    ):
        raise SystemExit("field checkpoint differs from its frozen v5 rung")
    training_sha = hashlib.sha256(
        json.dumps(training, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    mpr_path = Path(str(training.get("mpr_cache", ""))).resolve()
    if (
        payload.get("training_config_sha256") != training_sha
        or not mpr_path.is_file()
        or payload.get("mpr_cache_sha256") != sha256(mpr_path)
        or report.get("training_config_sha256") != training_sha
        or report.get("mpr_cache_sha256") != payload.get("mpr_cache_sha256")
        or report.get("final_metrics") != payload.get("final_metrics")
        or report.get("final_capability_metrics")
        != payload.get("final_capability_metrics")
    ):
        raise SystemExit("field report/checkpoint binding differs")
else:
    raise SystemExit(f"unsupported v5 torch stage: {stage}")
PY
}

run_gpu_stage() {
  local stage="$1"
  local terminal="$2"
  shift 2
  if [[ -e "$terminal" || -e "$terminal.json" ]]; then
    if [[ ! -s "$terminal" || ! -s "$terminal.json" ]]; then
      stale_stage_error "$stage" "$terminal" "$terminal.json"
    fi
    if ! validate_torch_stage "$stage" "$terminal"; then
      stale_stage_error "$stage" "$terminal" "$terminal.json"
    fi
    return 0
  fi
  run_guarded_command bash radio_gs/scripts/run_repo_python.sh "$@" \
    >"$LOG_ROOT/${stage}.log" 2>&1
  if [[ ! -s "$terminal" || ! -s "$terminal.json" ]]; then
    echo "v5 stage did not produce its audited terminal: $stage" >&2
    exit 1
  fi
  if ! validate_torch_stage "$stage" "$terminal"; then
    stale_stage_error "$stage" "$terminal" "$terminal.json"
  fi
}

validate_audit_stage() {
  local name="$1"
  local audit_path="$2"
  CUDA_VISIBLE_DEVICES="" bash radio_gs/scripts/run_repo_python.sh - \
    "$name" "$audit_path" "$FIELD_ROOT/${name}.pth" "$RUN_MANIFEST" <<'PY'
import hashlib
import math
import sys
from pathlib import Path

from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file

name = sys.argv[1]
audit_path = Path(sys.argv[2]).resolve()
field_path = Path(sys.argv[3]).resolve()
manifest_path = Path(sys.argv[4]).resolve()


def sha256(path: Path) -> str:
    return sha256_file(path)


try:
    audit, _audit_sha256, _ = load_json_object(
        audit_path,
        label="canonical-v5 fidelity audit",
    )
    manifest, _manifest_sha256, _ = load_json_object(
        manifest_path,
        label="canonical-v5 run manifest",
    )
except (OSError, ValueError) as exc:
    raise SystemExit(f"fidelity audit provenance is unreadable: {exc}") from exc
if set(audit) != {
    "schema_version",
    "audit",
    "protocol",
    "artifacts",
    "aggregate",
    "per_frame",
} or audit.get("schema_version") != 1 or audit.get("audit") != "canonical_capability_fidelity_v1":
    raise SystemExit("fidelity audit top-level schema differs")
if name not in manifest["fixed_training_contract"]["candidates"]:
    raise SystemExit("fidelity audit is not for a frozen v5 candidate")
protocol = audit.get("protocol")
artifacts = audit.get("artifacts")
aggregate = audit.get("aggregate")
contract = manifest["fixed_audit_contract"]
if not isinstance(protocol, dict) or not isinstance(artifacts, dict):
    raise SystemExit("fidelity audit lacks protocol/artifact provenance")
if not isinstance(aggregate, dict) or not aggregate:
    raise SystemExit("fidelity audit has no aggregate metrics")
if set(aggregate) != {
    "raw_radio",
    "official_dino_v3",
    "official_sam3",
    "support_fraction_on_visible",
    "supported_visible_pixels",
    "total_visible_pixels",
}:
    raise SystemExit("fidelity audit aggregate schema differs")


def finite_tree(value, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SystemExit(f"{label} contains a non-finite value")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{label}.{key}")
        return
    raise SystemExit(f"{label} contains an unsupported JSON value")


finite_tree(audit, "audit")
per_frame = audit.get("per_frame")
if (
    not isinstance(per_frame, list)
    or [row.get("frame_id") for row in per_frame if isinstance(row, dict)]
    != protocol.get("frame_ids")
):
    raise SystemExit("fidelity audit per-frame coverage differs")
support_fraction = aggregate["support_fraction_on_visible"]
supported_pixels = aggregate["supported_visible_pixels"]
visible_pixels = aggregate["total_visible_pixels"]
if (
    not 0.0 <= float(support_fraction) <= 1.0
    or not isinstance(supported_pixels, int)
    or not isinstance(visible_pixels, int)
    or supported_pixels < 0
    or visible_pixels < supported_pixels
):
    raise SystemExit("fidelity audit support summary is invalid")
if (
    protocol.get("held_out_from_mpr") is not True
    or protocol.get("frame_ids") != manifest["fidelity_frame_ids"]
    or protocol.get("benchmark_masks_opened") is not False
    or protocol.get("text_queries_opened") is not False
    or protocol.get("capability_map_source") != "official_extracted"
    or protocol.get("alpha_threshold") != contract["alpha_threshold"]
    or protocol.get("support_eps") != contract["support_eps"]
    or protocol.get("boundary_quantile") != contract["boundary_quantile"]
    or protocol.get("residual_mode") != contract["residual_mode"]
):
    raise SystemExit("fidelity audit protocol differs from the immutable run")
if (
    not field_path.is_file()
    or Path(str(artifacts.get("field_checkpoint", ""))).resolve() != field_path
    or artifacts.get("field_checkpoint_sha256") != sha256(field_path)
    or artifacts.get("config_sha256") != manifest["config_sha256"]
    or artifacts.get("resolved_config_sha256")
    != manifest["resolved_config_sha256"]
    or artifacts.get("geometry_checkpoint_sha256")
    != manifest["geometry_checkpoint_sha256"]
    or artifacts.get("radio_checkpoint_sha256")
    != manifest["radio_checkpoint_sha256"]
    or artifacts.get("view_residual_checkpoint") != ""
    or artifacts.get("boundary_residual_checkpoint") != ""
):
    raise SystemExit("fidelity audit artifact binding differs")
try:
    _field, field_payload = load_canonical_field_checkpoint(
        field_path,
        map_location="cpu",
        expected_sha256=str(artifacts["field_checkpoint_sha256"]),
    )
except Exception as exc:
    raise SystemExit(f"fidelity field checkpoint cannot be safely reopened: {exc}") from exc
if (
    field_payload.get("feature_output_bundle_sha256")
    != manifest["feature_extraction_safety_contract"][
        "final_output_bundle_sha256"
    ]
):
    raise SystemExit("fidelity field belongs to another feature output bundle")
for source_name in ("dino_v3", "sam3"):
    actual = artifacts.get("official_capability_sources", {}).get(source_name, {})
    expected = manifest["official_capability_sources"][source_name]
    for key in (
        "frame_manifest_sha256",
        "radio_checkpoint_sha256",
        "radio_checkpoint_load_contract",
        "scene",
        "image_dir",
        "frame_indices_sha256",
        "output_bundle_sha256",
    ):
        if actual.get(key) != expected.get(key):
            raise SystemExit(
                f"fidelity audit {source_name} source binding differs: {key}"
            )
PY
}

COMMON_MPR_ARGS=(
  radio_gs/scripts/build_gaussian_multiview_teacher_cache.py
  --config "$CONFIG"
  --checkpoint "$GEOMETRY_CHECKPOINT"
  --device cuda:0
  --observation-contract canonical-mpr-v1
  --expected-geometry-checkpoint-sha256 "$GEOMETRY_CHECKPOINT_SHA256"
  --exclude-frame-ids "$EXCLUDE_FRAME_IDS"
  --normalize-each-view
  --raster-reliability-mode mean_resultant
  --expected-feature-scene "$FEATURE_SCENE"
  --expected-feature-image-dir "$FEATURE_IMAGE_DIR"
  --expected-feature-output-bundle-sha256 "$FEATURE_OUTPUT_BUNDLE_SHA256"
)

if [[ -e "$RAW_MPR" || -e "$RAW_MPR.json" || -e "$RESPONSIBILITY" ]]; then
  if [[ ! -s "$RAW_MPR" || ! -s "$RAW_MPR.json" || ! -s "$RESPONSIBILITY" ]]; then
    stale_stage_error "mpr_raw_mean_resultant" \
      "$RAW_MPR" "$RAW_MPR.json" "$RESPONSIBILITY"
  fi
  if ! validate_torch_stage "mpr_raw_mean_resultant" "$RAW_MPR"; then
    stale_stage_error "mpr_raw_mean_resultant" \
      "$RAW_MPR" "$RAW_MPR.json" "$RESPONSIBILITY"
  fi
else
  run_guarded_command bash radio_gs/scripts/run_repo_python.sh \
    "${COMMON_MPR_ARGS[@]}" \
    --feature-space radio \
    --save-responsibility-cache "$RESPONSIBILITY" \
    --output "$RAW_MPR" \
    >"$LOG_ROOT/mpr_raw_mean_resultant.log" 2>&1
  if [[ ! -s "$RAW_MPR" || ! -s "$RAW_MPR.json" || ! -s "$RESPONSIBILITY" ]] \
    || ! validate_torch_stage "mpr_raw_mean_resultant" "$RAW_MPR"; then
    stale_stage_error "mpr_raw_mean_resultant" \
      "$RAW_MPR" "$RAW_MPR.json" "$RESPONSIBILITY"
  fi
fi
RESPONSIBILITY_SHA256="$(
  CUDA_VISIBLE_DEVICES="" bash radio_gs/scripts/run_repo_python.sh - \
    "$RESPONSIBILITY" <<'PY'
import sys
from radio_gs.utils.immutable_artifacts import sha256_file
print(sha256_file(sys.argv[1]))
PY
)"
if [[ ! "$RESPONSIBILITY_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "shared responsibility cache did not yield a stable SHA-256" >&2
  exit 2
fi
for capability in dino_v3 sam3; do
  if [[ "$capability" == "dino_v3" ]]; then
    target="$DINO_MPR"
  else
    target="$SAM3_MPR"
  fi
  run_gpu_stage "mpr_${capability}_mean_resultant" "$target" \
    "${COMMON_MPR_ARGS[@]}" \
    --feature-space "$capability" \
    --radio-checkpoint "$RADIO_CHECKPOINT" \
    --capability-map-source "$CAPABILITY_MAP_SOURCE" \
    --responsibility-cache "$RESPONSIBILITY" \
    --expected-responsibility-cache-sha256 "$RESPONSIBILITY_SHA256" \
    --output "$target"
done

bash radio_gs/scripts/run_repo_python.sh - \
  "$RAW_MPR" "$DINO_MPR" "$SAM3_MPR" "$RESPONSIBILITY" \
  "$RUN_MANIFEST" <<'PY'
import sys
from pathlib import Path

import torch

from radio_gs.training.tensor_cache_io import load_mpr_cache
from radio_gs.utils.immutable_artifacts import load_json_object, sha256_file

(
    raw_path,
    dino_path,
    sam_path,
    responsibility_path,
    run_manifest_path,
) = map(Path, sys.argv[1:])
run_manifest, _manifest_sha256, _ = load_json_object(
    run_manifest_path,
    label="canonical-v5 run manifest",
)
raw, _raw_sha256, _ = load_mpr_cache(
    raw_path,
    expected_feature_space="radio",
    require_reliability=True,
    require_formal_safety=True,
)
xyz = torch.as_tensor(raw["xyz"]).clone()
valid = torch.as_tensor(raw["valid"]).bool().clone()
counts = torch.as_tensor(raw["view_counts"]).long().clone()

responsibility_sha = sha256_file(responsibility_path)

reference_metadata = raw.get("metadata", {})
if (
    reference_metadata.get("feature_space") != "radio"
    or Path(reference_metadata.get("config", "")).resolve()
    != Path(run_manifest["config"]).resolve()
    or Path(reference_metadata.get("checkpoint", "")).resolve()
    != Path(run_manifest["geometry_checkpoint"]).resolve()
    or sorted(reference_metadata.get("excluded_frame_ids", []))
    != run_manifest["exclude_frame_ids"]
    or reference_metadata.get("registration_responsibility_cache_sha256")
    != responsibility_sha
    or reference_metadata.get("shared_registration_responsibility") is not True
    or reference_metadata.get("feature_output_bundle_sha256")
    != run_manifest["feature_extraction_safety_contract"][
        "final_output_bundle_sha256"
    ]
):
    raise SystemExit("raw MPR does not match the immutable v5 run manifest")
reference_frames = reference_metadata.get("selected_frame_indices")
reference_contract = reference_metadata.get(
    "registration_responsibility_contract"
)
for path, expected_space in (
    (raw_path, "radio"),
    (dino_path, "dino_v3"),
    (sam_path, "sam3"),
):
    payload = raw if path == raw_path else load_mpr_cache(
        path,
        expected_feature_space=expected_space,
        require_reliability=True,
        require_formal_safety=True,
    )[0]
    metadata = payload.get("metadata", {})
    if metadata.get("raster_reliability_mode") != "mean_resultant":
        raise SystemExit(f"{path}: not a mean-resultant MPR")
    if metadata.get("normalize_each_view") is not True:
        raise SystemExit(f"{path}: observations were not normalized")
    if metadata.get("observation_lifting_contract", {}).get("name") != "canonical-mpr-v1":
        raise SystemExit(f"{path}: wrong observation contract")
    if not torch.equal(torch.as_tensor(payload["xyz"]), xyz):
        raise SystemExit(f"{path}: Gaussian rows differ")
    if not torch.equal(torch.as_tensor(payload["valid"]).bool(), valid):
        raise SystemExit(f"{path}: valid support differs")
    if not torch.equal(torch.as_tensor(payload["view_counts"]).long(), counts):
        raise SystemExit(f"{path}: view counts differ")
    if (
        metadata.get("selected_frame_indices") != reference_frames
        or metadata.get("registration_responsibility_contract")
        != reference_contract
        or metadata.get("registration_responsibility_cache_sha256")
        != responsibility_sha
        or metadata.get("shared_registration_responsibility") is not True
        or metadata.get("feature_output_bundle_sha256")
        != run_manifest["feature_extraction_safety_contract"][
            "final_output_bundle_sha256"
        ]
    ):
        raise SystemExit(f"{path}: shared responsibility contract differs")
    if path != raw_path:
        name = "dino_v3" if path == dino_path else "sam3"
        expected_source = run_manifest["official_capability_sources"][name]
        if (
            metadata.get("feature_space") != name
            or metadata.get("capability_map_source")
            != "official_extracted"
            or metadata.get("capability_adaptor_execution")
            != "official_c_radio_runtime_adaptor_output"
            or metadata.get("official_adaptor_checkpoint_provenance")
            != "explicit_file_sha256"
            or metadata.get("official_adaptor_checkpoint_sha256")
            != run_manifest["radio_checkpoint_sha256"]
            or metadata.get("capability_native_map_manifest_sha256")
            != expected_source["frame_manifest_sha256"]
            or metadata.get("capability_native_map_grid")
            != expected_source["native_grid"]
            or metadata.get("capability_native_map_scene")
            != expected_source["scene"]
            or metadata.get("capability_native_map_image_dir")
            != expected_source["image_dir"]
            or metadata.get(
                "capability_native_map_frame_indices_sha256"
            )
            != expected_source["frame_indices_sha256"]
            or metadata.get("capability_native_map_output_bundle_sha256")
            != expected_source["output_bundle_sha256"]
        ):
            raise SystemExit(
                f"{path}: native official capability provenance differs"
            )
        if metadata.get("registration_responsibility_cache_sha256") != responsibility_sha:
            raise SystemExit(
                f"{path}: capability MPR did not reuse raw responsibility"
            )
        del payload
PY

mapfile -t MPR_SHA256_VALUES < <(
  CUDA_VISIBLE_DEVICES="" bash radio_gs/scripts/run_repo_python.sh - \
    "$RAW_MPR" "$DINO_MPR" "$SAM3_MPR" <<'PY'
import sys
from radio_gs.utils.immutable_artifacts import sha256_file
for path in sys.argv[1:]:
    print(sha256_file(path))
PY
)
if [[ "${#MPR_SHA256_VALUES[@]}" -ne 3 \
      || ! "${MPR_SHA256_VALUES[0]}" =~ ^[0-9a-f]{64}$ \
      || ! "${MPR_SHA256_VALUES[1]}" =~ ^[0-9a-f]{64}$ \
      || ! "${MPR_SHA256_VALUES[2]}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "validated MPR terminals did not yield stable SHA-256 values" >&2
  exit 2
fi
RAW_MPR_SHA256="${MPR_SHA256_VALUES[0]}"
DINO_MPR_SHA256="${MPR_SHA256_VALUES[1]}"
SAM3_MPR_SHA256="${MPR_SHA256_VALUES[2]}"

train_candidate() {
  local name="$1"
  local coarse="$2"
  local hidden="$3"
  local residual_blocks="$4"
  local output="$FIELD_ROOT/${name}.pth"
  run_gpu_stage "field_${name}" "$output" \
    radio_gs/scripts/train_canonical_radio_field.py \
    --mpr-cache "$RAW_MPR" \
    --expected-mpr-cache-sha256 "$RAW_MPR_SHA256" \
    --observation-contract canonical-mpr-v1 \
    --radio-checkpoint "$RADIO_CHECKPOINT" \
    --expected-radio-checkpoint-sha256 "$RADIO_CHECKPOINT_SHA256" \
    --expected-feature-output-bundle-sha256 "$FEATURE_OUTPUT_BUNDLE_SHA256" \
    --output "$output" \
    --device cuda:0 \
    --coefficient-dim 256 \
    --local-dim 128 \
    --primitive-fusion \
    --fusion-reliability \
    --spatial-coarse-dim "$coarse" \
    --hidden-dim "$hidden" \
    --fusion-residual-blocks "$residual_blocks" \
    --hash-levels 8 \
    --hash-features-per-level 2 \
    --hash-log2-size 15 \
    --hash-base-resolution 8 \
    --hash-max-resolution 512 \
    --hash-hidden-dim 64 \
    --official-capability-loss \
    --dino-mpr-cache "$DINO_MPR" \
    --expected-dino-v3-mpr-cache-sha256 "$DINO_MPR_SHA256" \
    --sam3-mpr-cache "$SAM3_MPR" \
    --expected-sam3-mpr-cache-sha256 "$SAM3_MPR_SHA256" \
    --mpr-weight 1.0 \
    --dino-weight 0.20 \
    --sam3-weight 0.20 \
    --coefficient-weight 1e-5 \
    --basis-orthogonality-weight 1e-3 \
    --epochs "$EPOCHS" \
    --min-epochs "$EPOCHS" \
    --batch-size 4096 \
    --eval-batch-size 16384 \
    --learning-rate 2e-3 \
    --weight-decay 1e-5 \
    --validation-fraction 0.05 \
    --target-cosine 0.985 \
    --pca-samples 50000 \
    --radio-version "$FEATURE_RADIO_VERSION" \
    --seed "$SEED"
}

train_candidate v5_r_reliability 0 192 0
train_candidate v5_w_width 0 512 0
train_candidate v5_s_spatial 64 512 0
train_candidate v5_d_deep 64 512 2

for name in v5_r_reliability v5_w_width v5_s_spatial v5_d_deep; do
  audit_output="$AUDIT_ROOT/${name}.json"
  field_checkpoint="$FIELD_ROOT/${name}.pth"
  field_checkpoint_sha256="$(
    CUDA_VISIBLE_DEVICES="" bash radio_gs/scripts/run_repo_python.sh - \
      "$field_checkpoint" <<'PY'
import sys
from radio_gs.utils.immutable_artifacts import sha256_file
print(sha256_file(sys.argv[1]))
PY
  )"
  if [[ ! "$field_checkpoint_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "field checkpoint did not yield a stable SHA-256: $field_checkpoint" >&2
    exit 2
  fi
  if [[ -e "$audit_output" ]]; then
    if [[ ! -s "$audit_output" ]] \
      || ! validate_audit_stage "$name" "$audit_output"; then
      stale_stage_error "fidelity_${name}" "$audit_output"
    fi
  else
    run_guarded_command bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/audit_canonical_capability_fidelity.py \
      --config "$CONFIG" \
      --expected-config-sha256 "$CONFIG_SHA256" \
      --geometry-checkpoint "$GEOMETRY_CHECKPOINT" \
      --expected-geometry-checkpoint-sha256 "$GEOMETRY_CHECKPOINT_SHA256" \
      --field-checkpoint "$field_checkpoint" \
      --expected-field-checkpoint-sha256 "$field_checkpoint_sha256" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --expected-radio-checkpoint-sha256 "$RADIO_CHECKPOINT_SHA256" \
      --expected-feature-output-bundle-sha256 \
      "$FEATURE_OUTPUT_BUNDLE_SHA256" \
      --capability-map-source "$CAPABILITY_MAP_SOURCE" \
      --frame-ids "$FIDELITY_FRAME_IDS" \
      --alpha-threshold 0.02 \
      --support-eps 1e-6 \
      --boundary-quantile 0.2 \
      --output "$audit_output" \
      --device cuda:0 \
      >"$LOG_ROOT/fidelity_${name}.log" 2>&1
    if [[ ! -s "$audit_output" ]] \
      || ! validate_audit_stage "$name" "$audit_output"; then
      stale_stage_error "fidelity_${name}" "$audit_output"
    fi
  fi
done

bash radio_gs/scripts/run_repo_python.sh - \
  "$FIELD_ROOT" "$AUDIT_ROOT" "$EPOCHS" "$SEED" \
  "$CAPABILITY_MAP_SOURCE" "$OUTPUT_ROOT/capacity_screen.json" \
  "$RAW_MPR" "$DINO_MPR" "$SAM3_MPR" "$RUN_MANIFEST" \
  "$OUTPUT_ROOT/capacity_screen.complete.json" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

from radio_gs.evaluation.capability_fidelity import select_query_free_compositor
from radio_gs.field import load_canonical_field_checkpoint
from radio_gs.scripts.extract_radio_features import (
    _python_source_tree_fingerprint,
    _validate_final_output_bundle,
)
from radio_gs.utils.immutable_artifacts import (
    load_json_object,
    sha256_file,
    write_frozen_json,
)

field_root = Path(sys.argv[1])
audit_root = Path(sys.argv[2])
epochs = int(sys.argv[3])
seed = int(sys.argv[4])
capability_map_source = sys.argv[5]
output = Path(sys.argv[6])
raw_mpr = Path(sys.argv[7])
dino_mpr = Path(sys.argv[8])
sam3_mpr = Path(sys.argv[9])
run_manifest_path = Path(sys.argv[10])
completion_output = Path(sys.argv[11])
run_manifest, run_manifest_sha256, _ = load_json_object(
    run_manifest_path,
    label="canonical-v5 run manifest",
)
expected = {
    "v5_r_reliability": (0, 192, 0),
    "v5_w_width": (0, 512, 0),
    "v5_s_spatial": (64, 512, 0),
    "v5_d_deep": (64, 512, 2),
}
rows = []
fidelity_variants = {}


def sha256(path: Path) -> str:
    return sha256_file(path)


runner_path = Path(run_manifest["runner"]).resolve()
repo_root = runner_path.parents[2]
if (
    sha256(runner_path) != run_manifest["runner_sha256"]
    or _python_source_tree_fingerprint(repo_root / "radio_gs")
    != run_manifest["implementation_source_tree"]
    or any(
        sha256(repo_root / relative) != expected_sha256
        for relative, expected_sha256 in run_manifest[
            "implementation_sources"
        ].items()
    )
    or sha256(Path(run_manifest["thermal_safety_contract"]["guard"]))
    != run_manifest["thermal_safety_contract"]["guard_sha256"]
    or sha256(
        Path(
            run_manifest["continuous_stage_safety_contract"][
                "canary_record"
            ]
        )
    )
    != run_manifest["continuous_stage_safety_contract"][
        "canary_record_sha256"
    ]
):
    raise SystemExit("v5 implementation source tree changed during the run")

feature_source = run_manifest["official_capability_sources"]["dino_v3"]
feature_manifest_path = Path(feature_source["frame_manifest"])
feature_manifest, _feature_manifest_sha256, _ = load_json_object(
    feature_manifest_path,
    expected_sha256=feature_source["frame_manifest_sha256"],
    label="final RADIO feature manifest",
)
_validate_final_output_bundle(
    feature_manifest_path.parent,
    feature_manifest,
    expected_manifest_sha256=feature_source["frame_manifest_sha256"],
    expected_output_bundle_sha256=feature_source["output_bundle_sha256"],
)
if (
    _python_source_tree_fingerprint(feature_manifest["radio"]["repo"])
    != feature_manifest["radio"]["python_source_tree"]
):
    raise SystemExit("RADIO source tree changed before v5 completion")


raw_mpr_sha256 = sha256(raw_mpr)
expected_num_gaussians = None
for name, (coarse, hidden, blocks) in expected.items():
    checkpoint = field_root / f"{name}.pth"
    checkpoint_sha256 = sha256(checkpoint)
    _field, payload = load_canonical_field_checkpoint(
        checkpoint,
        map_location="cpu",
        expected_sha256=checkpoint_sha256,
    )
    architecture = payload["architecture"]
    spatial_hash = (
        {
            "output_dim": coarse,
            "num_levels": 8,
            "features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 8,
            "max_resolution": 512,
            "hidden_dim": 64,
        }
        if coarse
        else None
    )
    expected_architecture = {
        "feature_dim": 1280,
        "coefficient_dim": 256,
        "local_dim": 128,
        "coarse_dim": coarse,
        "spatial_hash": spatial_hash,
        "position_storage": (
            "normalized_fp16" if coarse else "none"
        ),
        "fusion_reliability": True,
        "hidden_dim": hidden,
        "fusion_residual_blocks": blocks,
        "use_fusion": True,
        "trainable_basis": True,
        "trainable_statistics": False,
    }
    if any(
        architecture.get(key) != value
        for key, value in expected_architecture.items()
    ):
        raise SystemExit(
            f"{checkpoint}: complete architecture is not the frozen v5 rung"
        )
    num_gaussians = int(architecture.get("num_gaussians", 0))
    if num_gaussians <= 0:
        raise SystemExit(f"{checkpoint}: invalid Gaussian count")
    if expected_num_gaussians is None:
        expected_num_gaussians = num_gaussians
    elif num_gaussians != expected_num_gaussians:
        raise SystemExit(f"{checkpoint}: Gaussian count differs across rungs")
    if payload.get("benchmark_masks_opened") or payload.get("text_queries_opened"):
        raise SystemExit(f"{checkpoint}: query-dependent training provenance")
    if (
        Path(payload.get("mpr_cache", "")).resolve() != raw_mpr.resolve()
        or payload.get("mpr_cache_sha256") != raw_mpr_sha256
        or len(payload.get("history", [])) != epochs
        or payload.get("feature_output_bundle_sha256")
        != run_manifest["feature_extraction_safety_contract"][
            "final_output_bundle_sha256"
        ]
    ):
        raise SystemExit(f"{checkpoint}: training input or epoch contract differs")
    training = payload.get("training_config", {})
    expected_training = {
        "observation_contract": "canonical-mpr-v1",
        "coefficient_dim": 256,
        "local_dim": 128,
        "spatial_coarse_dim": coarse,
        "hash_levels": 8,
        "hash_features_per_level": 2,
        "hash_log2_size": 15,
        "hash_base_resolution": 8,
        "hash_max_resolution": 512,
        "hash_hidden_dim": 64,
        "fusion_reliability": True,
        "hidden_dim": hidden,
        "fusion_residual_blocks": blocks,
        "primitive_fusion": True,
        "pca_samples": 50000,
        "no_standardize": False,
        "freeze_basis": False,
        "official_capability_loss": True,
        "mpr_weight": 1.0,
        "dino_weight": 0.2,
        "sam3_weight": 0.2,
        "coefficient_weight": 1e-5,
        "basis_orthogonality_weight": 1e-3,
        "epochs": epochs,
        "min_epochs": epochs,
        "batch_size": 4096,
        "eval_batch_size": 16384,
        "learning_rate": 0.002,
        "weight_decay": 1e-5,
        "validation_fraction": 0.05,
        "target_cosine": 0.985,
        "seed": seed,
    }
    if any(
        training.get(key) != value
        for key, value in expected_training.items()
    ):
        raise SystemExit(
            f"{checkpoint}: actual training parameters differ from v5"
        )
    if (
        Path(training.get("mpr_cache", "")).resolve()
        != raw_mpr.resolve()
        or Path(training.get("dino_mpr_cache", "")).resolve()
        != dino_mpr.resolve()
        or Path(training.get("sam3_mpr_cache", "")).resolve()
        != sam3_mpr.resolve()
        or Path(training.get("radio_checkpoint", "")).resolve()
        != Path(run_manifest["radio_checkpoint"]).resolve()
        or training.get("radio_version")
        != run_manifest["official_capability_sources"]["dino_v3"][
            "radio_version"
        ]
        or payload.get("training_config_sha256")
        != hashlib.sha256(
            json.dumps(
                training,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    ):
        raise SystemExit(
            f"{checkpoint}: training parameter provenance is inconsistent"
        )
    loss_config = payload.get("loss_config", {})
    expected_losses = (
        ("mpr_weight", 1.0),
        ("dino_weight", 0.2),
        ("sam3_weight", 0.2),
        ("relation_weight", 0.0),
    )
    if any(
        loss_config.get(key) is None
        or not math.isclose(
            float(loss_config[key]),
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for key, expected_value in expected_losses
    ):
        raise SystemExit(f"{checkpoint}: frozen v5 loss contract differs")
    target_provenance = payload.get("capability_mpr_targets", {})
    for target_name, target_path in (
        ("dino_v3", dino_mpr),
        ("sam3", sam3_mpr),
    ):
        provenance = target_provenance.get(target_name, {})
        if (
            Path(provenance.get("path", "")).resolve()
            != target_path.resolve()
            or provenance.get("sha256") != sha256(target_path)
            or provenance.get("capability_map_source")
            != "official_extracted"
            or provenance.get("official_adaptor_checkpoint_sha256")
            != run_manifest["radio_checkpoint_sha256"]
            or provenance.get("official_adaptor_checkpoint_provenance")
            != "explicit_file_sha256"
            or provenance.get("feature_output_bundle_sha256")
            != run_manifest["feature_extraction_safety_contract"][
                "final_output_bundle_sha256"
            ]
            or provenance.get("capability_native_map_output_bundle_sha256")
            != run_manifest["official_capability_sources"][target_name][
                "output_bundle_sha256"
            ]
            or provenance.get(
                "capability_native_map_radio_checkpoint_load_contract"
            )
            != run_manifest["official_capability_sources"][target_name][
                "radio_checkpoint_load_contract"
            ]
        ):
            raise SystemExit(
                f"{checkpoint}: {target_name} target provenance differs"
            )
    audit_path = audit_root / f"{name}.json"
    audit, audit_sha256, _ = load_json_object(
        audit_path,
        label=f"canonical-v5 {name} fidelity audit",
    )
    protocol = audit.get("protocol", {})
    artifacts = audit.get("artifacts", {})
    audit_contract = run_manifest["fixed_audit_contract"]
    if (
        protocol.get("held_out_from_mpr") is not True
        or protocol.get("frame_ids")
        != run_manifest["fidelity_frame_ids"]
        or protocol.get("benchmark_masks_opened") is not False
        or protocol.get("text_queries_opened") is not False
        or protocol.get("capability_map_source") != capability_map_source
        or protocol.get("alpha_threshold")
        != audit_contract["alpha_threshold"]
        or protocol.get("support_eps") != audit_contract["support_eps"]
        or protocol.get("boundary_quantile")
        != audit_contract["boundary_quantile"]
        or protocol.get("residual_mode")
        != audit_contract["residual_mode"]
    ):
        raise SystemExit(f"{audit_path}: fidelity split/provenance is not query-free")
    if (
        Path(artifacts.get("field_checkpoint", "")).resolve()
        != checkpoint.resolve()
        or artifacts.get("field_checkpoint_sha256") != sha256(checkpoint)
        or artifacts.get("config_sha256")
        != run_manifest["config_sha256"]
        or artifacts.get("resolved_config_sha256")
        != run_manifest["resolved_config_sha256"]
        or artifacts.get("geometry_checkpoint_sha256")
        != run_manifest["geometry_checkpoint_sha256"]
        or artifacts.get("radio_checkpoint_sha256")
        != run_manifest["radio_checkpoint_sha256"]
        or artifacts.get("feature_output_bundle_sha256")
        != run_manifest["feature_extraction_safety_contract"][
            "final_output_bundle_sha256"
        ]
        or artifacts.get("view_residual_checkpoint") != ""
        or artifacts.get("boundary_residual_checkpoint") != ""
    ):
        raise SystemExit(f"{audit_path}: fidelity report belongs to another field")
    for source_name in ("dino_v3", "sam3"):
        actual_source = artifacts.get(
            "official_capability_sources", {}
        ).get(source_name, {})
        expected_source = run_manifest[
            "official_capability_sources"
        ][source_name]
        if (
            actual_source.get("frame_manifest_sha256")
            != expected_source["frame_manifest_sha256"]
            or actual_source.get("radio_checkpoint_sha256")
            != run_manifest["radio_checkpoint_sha256"]
            or actual_source.get("radio_checkpoint_provenance")
            != "explicit_file_sha256"
            or actual_source.get("radio_checkpoint_load_contract")
            != (
                "external_sha256_same_fd_restricted_pickle_"
                "hub_injection_v1"
            )
            or actual_source.get("scene")
            != expected_source["scene"]
            or actual_source.get("image_dir")
            != expected_source["image_dir"]
            or actual_source.get("frame_indices_sha256")
            != expected_source["frame_indices_sha256"]
            or actual_source.get("output_bundle_sha256")
            != expected_source["output_bundle_sha256"]
        ):
            raise SystemExit(
                f"{audit_path}: {source_name} source provenance differs"
            )
    fidelity_variants[name] = audit["aggregate"]
    rows.append(
        {
            "name": name,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "audit": str(audit_path.resolve()),
            "audit_sha256": audit_sha256,
            "architecture": architecture,
            "final_metrics": payload["final_metrics"],
            "final_capability_metrics": payload["final_capability_metrics"],
            "held_out_fidelity": audit["aggregate"],
        }
    )
selection_contract = run_manifest.get("fixed_selection_contract")
expected_selection_contract = {
    "baseline": "v5_r_reliability",
    "max_mean_dense_drop": 0.005,
    "max_p05_dense_drop": 0.01,
    "max_unsupported_fraction": 0.005,
    "min_relation_gain": 0.005,
    "objective": (
        "maximize_mean_official_dino_sam_affinity_pearson_and_"
        "boundary_margin_retention_under_dense_and_support_guards"
    ),
}
if selection_contract != expected_selection_contract:
    raise SystemExit("run manifest has a different v5 selection contract")
selection = select_query_free_compositor(
    fidelity_variants,
    baseline=selection_contract["baseline"],
    max_mean_dense_drop=selection_contract["max_mean_dense_drop"],
    max_p05_dense_drop=selection_contract["max_p05_dense_drop"],
    max_unsupported_fraction=selection_contract["max_unsupported_fraction"],
    min_relation_gain=selection_contract["min_relation_gain"],
)
report = {
    "schema_version": 1,
    "selection_status": (
        "single_scene_query_free_provisional_cross_scene_confirmation_required"
        if selection["selected_variant"] is not None
        else selection["selection_status"]
    ),
    "epochs": epochs,
    "seed": seed,
    "run_manifest": str(run_manifest_path.resolve()),
    "run_manifest_sha256": run_manifest_sha256,
    "candidates": rows,
    "fixed_selection_contract": selection_contract,
    "query_free_selection": selection,
    "next_gate": (
        "repeat the identical screen on a second development scene; promote "
        "only a candidate selected consistently without benchmark queries"
    ),
    "benchmark_queries_opened": False,
    "benchmark_masks_opened": False,
}
write_frozen_json(output, report)
report_sha256 = sha256(output)
completion = {
    "schema_version": 1,
    "contract": "canonical-v5-capacity-screen-completion-v1",
    "screen": "canonical-v5-query-free-capacity",
    "scene": run_manifest["official_capability_sources"]["dino_v3"][
        "scene"
    ],
    "report": {"path": str(output.resolve()), "sha256": report_sha256},
    "run_manifest": {
        "path": str(run_manifest_path.resolve()),
        "sha256": run_manifest_sha256,
    },
    "candidates": {
        row["name"]: {
            "checkpoint": {
                "path": row["checkpoint"],
                "sha256": row["checkpoint_sha256"],
            },
            "audit": {
                "path": row["audit"],
                "sha256": row["audit_sha256"],
            },
        }
        for row in rows
    },
    "feature_output_bundle_sha256": run_manifest[
        "feature_extraction_safety_contract"
    ]["final_output_bundle_sha256"],
    "implementation_source_tree_sha256": run_manifest[
        "implementation_source_tree"
    ]["tree_sha256"],
    "benchmark_queries_opened": False,
    "benchmark_masks_opened": False,
}
write_frozen_json(completion_output, completion)
print(json.dumps(report, indent=2))
PY
