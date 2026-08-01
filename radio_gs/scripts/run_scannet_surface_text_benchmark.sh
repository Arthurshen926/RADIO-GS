#!/usr/bin/env bash

# Formal promoted Surface text benchmark: an immutable 3-seed x 3-scene grid.
# Authority is accepted by a ScanNet-free process before any benchmark input is
# opened.  The only GPU phase is durable/resumable and guarded on physical GPU1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SURFACE_MANIFEST="${SURFACE_MANIFEST:?set frozen Surface promotion manifest}"
SURFACE_COMPLETION="${SURFACE_COMPLETION:?set frozen Surface completion}"
TEXT_AUDIT_MANIFEST="${TEXT_AUDIT_MANIFEST:?set accepted text-response audit manifest}"
TEXT_AUDIT_COMPLETION="${TEXT_AUDIT_COMPLETION:?set accepted text-response audit completion}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set a dedicated benchmark output root}"

GPU="${GPU:-1}"
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
SEMANTIC_RADIO_BATCH_SIZE="${SEMANTIC_RADIO_BATCH_SIZE:-1024}"
SEMANTIC_BATCH_SIZE="${SEMANTIC_BATCH_SIZE:-64}"
SEMANTIC_PACING_SECONDS="${SEMANTIC_PACING_SECONDS:-4.0}"

RUN_REPO_PYTHON="$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
AUTHORITY_GATE="$REPO_ROOT/radio_gs/scripts/finalize_scannet_surface_text_authority_gate.py"
FINALIZER="$REPO_ROOT/radio_gs/scripts/finalize_scannet_surface_text_benchmark.py"
GUARD_RECEIPT="$REPO_ROOT/radio_gs/scripts/finalize_gpu_guard_receipt.py"
SEMANTIC_BUILDER="$REPO_ROOT/radio_gs/scripts/build_surface_region_semantic_cache.py"
EVALUATOR="$REPO_ROOT/radio_gs/scripts/eval_scannet_canonical_text_query.py"
PREPARED_ROOT="/mnt/pool/sqy/3d_understanding/scannet_og"
TEXT_CACHE_BASE="$REPO_ROOT/checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt"

if [[ "$GPU" != "1" ]]; then
  echo "formal ScanNet text benchmark is fixed to physical GPU1" >&2
  exit 2
fi
if [[ ! "$SEMANTIC_RADIO_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$SEMANTIC_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$SEMANTIC_PACING_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || ! awk -v value="$SEMANTIC_PACING_SECONDS" 'BEGIN { exit !(value > 0) }'; then
  echo "semantic batch sizes and pacing must be positive" >&2
  exit 2
fi
if (( GPU_SOFT_PAUSE_TEMP_C > 0 )); then
  if (( GPU_SOFT_RESUME_TEMP_C >= GPU_SOFT_PAUSE_TEMP_C \
        || GPU_SOFT_PAUSE_TEMP_C >= GPU_MAX_TEMP_C )); then
    echo "optional soft pause thresholds are invalid" >&2
    exit 2
  fi
elif (( GPU_SOFT_RESUME_TEMP_C != 0 )); then
  echo "soft resume requires a positive soft pause threshold" >&2
  exit 2
fi
for authority_input in \
  "$SURFACE_MANIFEST" "$SURFACE_COMPLETION" \
  "$TEXT_AUDIT_MANIFEST" "$TEXT_AUDIT_COMPLETION" \
  "$AUTHORITY_GATE" "$RUN_REPO_PYTHON"; do
  if [[ ! -s "$authority_input" || -L "$authority_input" ]]; then
    echo "missing or symlinked authority input: $authority_input" >&2
    exit 2
  fi
done

OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"
if [[ -L "$OUTPUT_ROOT" ]]; then
  echo "refuse symlink benchmark output root: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

open_lock_file() {
  local lock_path="$1"
  local descriptor_name="$2"
  if [[ -L "$lock_path" ]]; then
    echo "refuse symlink lock: $lock_path" >&2
    return 2
  fi
  if [[ ! -e "$lock_path" ]]; then
    (set -o noclobber; : >"$lock_path") 2>/dev/null || true
  fi
  if [[ -L "$lock_path" || ! -f "$lock_path" ]]; then
    echo "lock is not a regular non-symlink file: $lock_path" >&2
    return 2
  fi
  local opened_fd
  exec {opened_fd}<>"$lock_path"
  if [[ "$(stat -Lc '%d:%i' "/proc/$$/fd/$opened_fd")" \
        != "$(stat -c '%d:%i' "$lock_path")" ]]; then
    echo "lock path identity changed: $lock_path" >&2
    return 2
  fi
  printf -v "$descriptor_name" '%s' "$opened_fd"
}

# The output-root lock is deliberately acquired before the authority gate.
RUNNER_LOCK_PATH="$OUTPUT_ROOT/.runner.lock"
open_lock_file "$RUNNER_LOCK_PATH" RUNNER_LOCK_FD
if ! flock -n "$RUNNER_LOCK_FD"; then
  echo "another process owns this formal benchmark output root" >&2
  exit 2
fi

PHYSICAL_GPU_LOCK="$REPO_ROOT/output/.physical_gpu1.lock"
PHYSICAL_GPU_LOCK_FD=""
cleanup_locks() {
  if [[ -n "$PHYSICAL_GPU_LOCK_FD" ]]; then
    flock -u "$PHYSICAL_GPU_LOCK_FD" 2>/dev/null || true
  fi
  flock -u "$RUNNER_LOCK_FD" 2>/dev/null || true
}
trap cleanup_locks EXIT INT TERM

AUTHORITY_RECEIPT_PATH="$OUTPUT_ROOT/authority.receipt.json"
GATE_RESULT="$({
  CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" "$AUTHORITY_GATE" \
    --surface-manifest "$SURFACE_MANIFEST" \
    --surface-completion "$SURFACE_COMPLETION" \
    --audit-manifest "$TEXT_AUDIT_MANIFEST" \
    --audit-completion "$TEXT_AUDIT_COMPLETION" \
    --output "$AUTHORITY_RECEIPT_PATH"
} | tail -n 1)"
AUTHORITY_RECEIPT_SHA256="$(
  printf '%s' "$GATE_RESULT" | bash "$RUN_REPO_PYTHON" -c \
    'import json,sys; print(json.load(sys.stdin)["receipt_sha256"])'
)"

# This is the one repository-global physical GPU1 mutex.  It is intentionally
# located at the exact canonical path below and rejects a final symlink.
open_lock_file "$PHYSICAL_GPU_LOCK" PHYSICAL_GPU_LOCK_FD
if ! flock -n "$PHYSICAL_GPU_LOCK_FD"; then
  echo "physical GPU1 is owned by another RADIO-GS workflow" >&2
  exit 2
fi

IFS=',' read -r GPU_UUID GPU_BUS_ID <<<"$(
  nvidia-smi -i 1 --query-gpu=uuid,pci.bus_id --format=csv,noheader,nounits
)"
GPU_UUID="$(printf '%s' "$GPU_UUID" | xargs)"
GPU_BUS_ID="$(printf '%s' "$GPU_BUS_ID" | xargs)"
if [[ "$GPU_UUID" != GPU-* || -z "$GPU_BUS_ID" ]]; then
  echo "cannot bind physical GPU1 UUID/bus identity" >&2
  exit 2
fi

reject_existing_gpu1_compute_owner() {
  local owners
  owners="$(
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' -v expected="$GPU_UUID" '
          { gpu=$1; gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu) }
          gpu == expected { print $0 }
        '
  )"
  if [[ -n "$owners" ]]; then
    echo "physical GPU1 already has a compute owner:" >&2
    printf '%s\n' "$owners" >&2
    return 2
  fi
}
reject_existing_gpu1_compute_owner

for implementation in \
  "$THERMAL_GUARD" "$FINALIZER" "$GUARD_RECEIPT" \
  "$SEMANTIC_BUILDER" "$EVALUATOR"; do
  if [[ ! -s "$implementation" || -L "$implementation" ]]; then
    echo "missing or symlinked formal implementation: $implementation" >&2
    exit 2
  fi
done

RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" "$FINALIZER" preflight \
  --authority-receipt "$AUTHORITY_RECEIPT_PATH" \
  --authority-receipt-sha256 "$AUTHORITY_RECEIPT_SHA256" \
  --output-root "$OUTPUT_ROOT" \
  --run-manifest "$RUN_MANIFEST" \
  --semantic-radio-batch-size "$SEMANTIC_RADIO_BATCH_SIZE" \
  --semantic-batch-size "$SEMANTIC_BATCH_SIZE" \
  --semantic-pacing-seconds "$SEMANTIC_PACING_SECONDS" \
  --thermal-guard "$THERMAL_GUARD" \
  --gpu "$GPU" \
  --gpu-uuid "$GPU_UUID" \
  --gpu-bus-id "$GPU_BUS_ID" \
  --gpu-max-temp-c "$GPU_MAX_TEMP_C" \
  --gpu-start-max-temp-c "$GPU_START_MAX_TEMP_C" \
  --gpu-max-power-limit-w "$GPU_MAX_POWER_LIMIT_W" \
  --gpu-poll-seconds "$GPU_POLL_SECONDS" \
  --gpu-soft-pause-temp-c "$GPU_SOFT_PAUSE_TEMP_C" \
  --gpu-soft-resume-temp-c "$GPU_SOFT_RESUME_TEMP_C" \
  --gpu-peer-index "$GPU_PEER_INDEX" \
  --gpu-peer-pause-temp-c "$GPU_PEER_PAUSE_TEMP_C" \
  --gpu-peer-resume-temp-c "$GPU_PEER_RESUME_TEMP_C" \
  --gpu-peer-quiet-seconds "$GPU_PEER_QUIET_SECONDS" \
  --gpu-peer-max-power-w "$GPU_PEER_MAX_POWER_W" \
  --gpu-peer-max-memory-mib "$GPU_PEER_MAX_MEMORY_MIB" \
  --gpu-peer-max-util-pct "$GPU_PEER_MAX_UTIL_PCT"

mkdir -p "$OUTPUT_ROOT/logs"

quarantine_required() {
  local stage_root="$1"
  local role="$2"
  local quarantine_path="$stage_root/quarantine/${role}.manual-$(date -u +%Y%m%dT%H%M%SZ)"
  echo "stale/corrupt $role; quarantine path: $quarantine_path; nothing was deleted" >&2
  exit 1
}

run_guarded() {
  local telemetry="$1"
  shift
  # CUDA receives the UUID, while the guard keeps the physical nvidia-smi
  # index.  This prevents a different enumeration order from mapping cuda:0
  # to anything other than the GPU1 device frozen in the run manifest.
  GPU="$GPU" CUDA_VISIBLE_DEVICES="$GPU_UUID" \
    GPU_TELEMETRY_LOG="$telemetry" \
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

SCENES=(scene0062_00 scene0140_00 scene0200_00)
SEEDS=(0 1 2)
for seed in "${SEEDS[@]}"; do
  for scene in "${SCENES[@]}"; do
    STAGE_JSON="$(
      CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" "$FINALIZER" stage-info \
        --run-manifest "$RUN_MANIFEST" --seed "$seed" --scene "$scene"
    )"
    mapfile -t INFO < <(
      printf '%s' "$STAGE_JSON" | bash "$RUN_REPO_PYTHON" -c '
import json,sys
v=json.load(sys.stdin); s=v["stage"]; i=v["scene_inputs"]; r=v["readout"]; q=v["radio_checkpoint"]
for x in (s["semantic_cache"],s["semantic_terminal"],s["semantic_resume_dir"],s["semantic_guard_receipt"],s["evaluation_report"],s["evaluation_terminal"],i["field"]["path"],i["field"]["sha256"],i["mpr"]["sha256"],i["graph"]["path"],i["graph"]["sha256"],i["label"]["path"],r["checkpoint"]["path"],r["checkpoint"]["sha256"],q["path"],q["sha256"]): print(x)
'
    )
    if [[ "${#INFO[@]}" != "16" ]]; then
      echo "stage-info did not return the frozen registry" >&2
      exit 2
    fi
    SEMANTIC="${INFO[0]}"
    SEMANTIC_TERMINAL="${INFO[1]}"
    RESUME_DIR="${INFO[2]}"
    SEMANTIC_GUARD_RECEIPT="${INFO[3]}"
    REPORT="${INFO[4]}"
    REPORT_TERMINAL="${INFO[5]}"
    FIELD="${INFO[6]}"
    FIELD_SHA256="${INFO[7]}"
    MPR_SHA256="${INFO[8]}"
    GRAPH="${INFO[9]}"
    GRAPH_SHA256="${INFO[10]}"
    LABEL="${INFO[11]}"
    READOUT="${INFO[12]}"
    READOUT_SHA256="${INFO[13]}"
    RADIO_CHECKPOINT="${INFO[14]}"
    RADIO_CHECKPOINT_SHA256="${INFO[15]}"
    STAGE_ROOT="$(dirname "$SEMANTIC")"
    mkdir -p "$STAGE_ROOT"

    if [[ -s "$SEMANTIC" && -s "$SEMANTIC_GUARD_RECEIPT" ]]; then
      if ! CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" "$FINALIZER" \
        finalize-semantic --run-manifest "$RUN_MANIFEST" \
        --seed "$seed" --scene "$scene" --semantic-cache "$SEMANTIC" \
        --guard-receipt "$SEMANTIC_GUARD_RECEIPT" \
        --terminal "$SEMANTIC_TERMINAL"; then
        quarantine_required "$STAGE_ROOT" "semantic"
      fi
    elif [[ -e "$SEMANTIC" || -e "$SEMANTIC_GUARD_RECEIPT" \
            || -e "$SEMANTIC_TERMINAL" ]]; then
      quarantine_required "$STAGE_ROOT" "semantic-incomplete-terminal"
    else
      reject_existing_gpu1_compute_owner
      ATTEMPTS_ROOT="$STAGE_ROOT/semantic.guard.attempts"
      mkdir -p "$ATTEMPTS_ROOT"
      attempt_index=1
      while [[ -e "$ATTEMPTS_ROOT/attempt_$(printf '%04d' "$attempt_index")" ]]; do
        attempt_index=$((attempt_index + 1))
      done
      ATTEMPT_ROOT="$ATTEMPTS_ROOT/attempt_$(printf '%04d' "$attempt_index")"
      mkdir "$ATTEMPT_ROOT"
      COMMAND_RECORD="$ATTEMPT_ROOT/command.json"
      TELEMETRY="$ATTEMPT_ROOT/telemetry.csv"
      SEMANTIC_LOG="$ATTEMPT_ROOT/semantic.log"
      BUILDER_COMMAND=(
        bash "$RUN_REPO_PYTHON" "$SEMANTIC_BUILDER"
        --field-checkpoint "$FIELD"
        --field-checkpoint-sha256 "$FIELD_SHA256"
        --support-graph "$GRAPH"
        --support-graph-sha256 "$GRAPH_SHA256"
        --readout-checkpoint "$READOUT"
        --readout-checkpoint-sha256 "$READOUT_SHA256"
        --mpr-cache-sha256 "$MPR_SHA256"
        --output "$SEMANTIC"
        --query-output "$STAGE_ROOT/semantic_query.pt"
        --resume-dir "$RESUME_DIR"
        --radio-batch-size "$SEMANTIC_RADIO_BATCH_SIZE"
        --semantic-batch-size "$SEMANTIC_BATCH_SIZE"
        --thermal-pacing-seconds-per-batch "$SEMANTIC_PACING_SECONDS"
        --radio-checkpoint "$RADIO_CHECKPOINT"
        --radio-checkpoint-sha256 "$RADIO_CHECKPOINT_SHA256"
        --device cuda:0
      )
      CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" "$GUARD_RECEIPT" \
        prepare-command --output "$COMMAND_RECORD" \
        --run-manifest "$RUN_MANIFEST" --seed "$seed" --scene "$scene" \
        --gpu "$GPU" --gpu-uuid "$GPU_UUID" --gpu-bus-id "$GPU_BUS_ID" \
        -- "${BUILDER_COMMAND[@]}"
      set +e
      run_guarded "$TELEMETRY" "${BUILDER_COMMAND[@]}" \
        >"$SEMANTIC_LOG" 2>&1
      guarded_status=$?
      set -e
      if (( guarded_status != 0 )); then
        echo "guarded semantic stage exited $guarded_status; durable resume retained at $RESUME_DIR; attempt evidence retained at $ATTEMPT_ROOT" >&2
        exit "$guarded_status"
      fi
      CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" "$GUARD_RECEIPT" finalize \
        --output "$SEMANTIC_GUARD_RECEIPT" \
        --command-record "$COMMAND_RECORD" --telemetry "$TELEMETRY" \
        --guard "$THERMAL_GUARD" --stage-output "$SEMANTIC" --exit-status 0
      CUDA_VISIBLE_DEVICES="" bash "$RUN_REPO_PYTHON" "$FINALIZER" \
        finalize-semantic --run-manifest "$RUN_MANIFEST" \
        --seed "$seed" --scene "$scene" --semantic-cache "$SEMANTIC" \
        --guard-receipt "$SEMANTIC_GUARD_RECEIPT" \
        --terminal "$SEMANTIC_TERMINAL"
    fi

    if [[ -s "$REPORT" ]]; then
      if ! CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 bash "$RUN_REPO_PYTHON" "$FINALIZER" \
        finalize-eval --run-manifest "$RUN_MANIFEST" --seed "$seed" \
        --scene "$scene" --report "$REPORT" --terminal "$REPORT_TERMINAL"; then
        quarantine_required "$STAGE_ROOT" "evaluation"
      fi
    elif [[ -e "$REPORT" || -e "$REPORT_TERMINAL" ]]; then
      quarantine_required "$STAGE_ROOT" "evaluation-incomplete-terminal"
    else
      EVALUATION_LOG="$OUTPUT_ROOT/logs/seed${seed}_${scene}_evaluation.log"
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 bash "$RUN_REPO_PYTHON" "$EVALUATOR" \
        --scene "$scene" --prepared-root "$PREPARED_ROOT" --label-ply "$LABEL" \
        --semantic-cache "$SEMANTIC" --output "$REPORT" \
        --class-splits 19,15,10 --projection-k 8 --distance-epsilon 0.0001 \
        --chunk-size 2048 --knn-workers 1 --torch-threads 1 \
        --prompt-templates '{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}' \
        --text-embedding-cache "$TEXT_CACHE_BASE" --scale-aggregation max \
        --scale-specificity-margin 0.0 --device cpu \
        >"$EVALUATION_LOG" 2>&1
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 bash "$RUN_REPO_PYTHON" "$FINALIZER" \
        finalize-eval --run-manifest "$RUN_MANIFEST" --seed "$seed" \
        --scene "$scene" --report "$REPORT" --terminal "$REPORT_TERMINAL"
    fi
  done
done

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 bash "$RUN_REPO_PYTHON" "$FINALIZER" \
  finalize-benchmark --run-manifest "$RUN_MANIFEST" \
  --aggregate-output "$OUTPUT_ROOT/aggregate.json" \
  --completion "$OUTPUT_ROOT/benchmark.complete.json"
