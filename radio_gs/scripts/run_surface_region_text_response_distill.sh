#!/usr/bin/env bash

# Authority-qualified, three-seed, target-blind Surface text-response treatment.
# CPU calibration and audits never expose CUDA.  Every GPU1 seed is separately
# locked, guarded, receipted, Xid/PCIe-audited, and resumed only from an exact
# immutable terminal.

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
cd "$REPO_ROOT"

LOCK_ROOT="/root/RADIO-GS/output"
GLOBAL_GPU1_LOCK="$LOCK_ROOT/.physical_gpu1.lock"
GPU1_SINGLETON_PROTOCOL="linux-abstract-af-unix-stream-v1:radio-gs-physical-gpu1-v1"

GPU="${GPU:-1}"
CANDIDATE="${CANDIDATE:?set CANDIDATE to the frozen SurfaceRegion cache candidate}"
CONTEXT_POOLING_MODE="joint_attention_v1"
SURFACE_ROOT="${SURFACE_ROOT:?set SURFACE_ROOT to the frozen attention post-cache Surface authority root}"
FIT_TEXT_BANK="${FIT_TEXT_BANK:?set FIT_TEXT_BANK to the frozen fit-split .pt artifact}"
FIT_TEXT_BANK_MANIFEST="${FIT_TEXT_BANK_MANIFEST:?set FIT_TEXT_BANK_MANIFEST to its sidecar JSON}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new lexical child of /root/RADIO-GS/output}"
TRAIN_CACHES="${TRAIN_CACHES:?set TRAIN_CACHES to the frozen four-shard training cache glob}"
VALIDATION_CACHES="${VALIDATION_CACHES:?set VALIDATION_CACHES to the frozen two-shard validation cache glob}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-$OUTPUT_ROOT/calibrations}"
GRADIENT_DIAGNOSTIC="${GRADIENT_DIAGNOSTIC:-/root/RADIO-GS/output/optimization_20260801/warmstart_gradient_diagnostic_gpu1_seed0_v3/result.json}"
GRADIENT_DIAGNOSTIC_SHA256="${GRADIENT_DIAGNOSTIC_SHA256:-bff5f97c949559a0c0a7d60b7509be2caf3c65bf27e61a129cb867bd5d1cc4cd}"
RUN_MANIFEST="$OUTPUT_ROOT/run_manifest.json"
COMPLETION="$OUTPUT_ROOT/text_response_distill.complete"
INITIAL_GPU_PREFLIGHT="$OUTPUT_ROOT/receipts/gpu1.initial.json"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
AUTHORITY="$REPO_ROOT/radio_gs/scripts/surface_text_response_distill_authority.py"
RECEIPT_FINALIZER="$REPO_ROOT/radio_gs/scripts/finalize_gpu_guard_receipt.py"

# GPU1-only policy validated by the Surface p4 canary and post-cache readouts.
# No peer board is queried or used as a launch/runtime condition.
GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-78}"
GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-65}"
GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-3}"
GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-75}"
GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-70}"
GPU_PEER_INDEX=""
GPU_PEER_PAUSE_TEMP_C=0
GPU_PEER_RESUME_TEMP_C=0
GPU_PEER_QUIET_SECONDS="${GPU_PEER_QUIET_SECONDS:-0}"
GPU_PEER_MAX_POWER_W=0
GPU_PEER_MAX_MEMORY_MIB=0
GPU_PEER_MAX_UTIL_PCT=100
GPU_PEER_ACTIVITY_ACTION="terminate"
GPU_OWNER_PID_NAMESPACE_MODE="exclusive-singleton-after-clear-v1"

case "$CANDIDATE" in
  control_c256_geometric|context_c1024_geometric|context_c1024_uniform|core_c1024_geometric)
    ;;
  *)
    echo "unknown frozen SurfaceRegion candidate: $CANDIDATE" >&2
    exit 2
    ;;
esac
if [[ "$GPU" != "1" ]]; then
  echo "text-response recovery is assigned to physical GPU1; got GPU=$GPU" >&2
  exit 2
fi
if [[ "$OUTPUT_ROOT" != /* || "$LOCK_ROOT" != /* ]]; then
  echo "authority output and canonical lock roots must be absolute" >&2
  exit 2
fi

# The Python supervisor opens both locks with O_NOFOLLOW and LOCK_NB and passes
# their live file descriptors into this child.  No environment-only bypass is
# accepted.  A source snapshot needs no .git directory and still shares the
# literal main-repository /root/RADIO-GS/output/.physical_gpu1.lock.
if [[ -z "${TEXT_RESPONSE_DISTILL_GLOBAL_LOCK_FD:-}" \
      && -z "${TEXT_RESPONSE_DISTILL_RUN_LOCK_FD:-}" \
      && -z "${RADIO_GS_GPU1_SINGLETON_FD:-}" ]]; then
  exec env CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" "$AUTHORITY" run-locked \
      --repo-root "$REPO_ROOT" \
      --lock-root "$LOCK_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      -- "$SCRIPT_PATH" "$@"
fi
if [[ -z "${TEXT_RESPONSE_DISTILL_GLOBAL_LOCK_FD:-}" \
      || -z "${TEXT_RESPONSE_DISTILL_RUN_LOCK_FD:-}" \
      || -z "${RADIO_GS_GPU1_SINGLETON_FD:-}" ]]; then
  echo "incomplete inherited authority lock descriptor contract" >&2
  exit 2
fi
if [[ "${RADIO_GS_GPU1_SINGLETON_PROTOCOL:-}" \
      != "$GPU1_SINGLETON_PROTOCOL" ]]; then
  echo "physical GPU1 kernel singleton protocol was not inherited" >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES="" bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$AUTHORITY" verify-lock-fds \
    --lock-root "$LOCK_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --global-fd "$TEXT_RESPONSE_DISTILL_GLOBAL_LOCK_FD" \
    --run-fd "$TEXT_RESPONSE_DISTILL_RUN_LOCK_FD" \
    --singleton-fd "$RADIO_GS_GPU1_SINGLETON_FD"

for required in \
  "$FIT_TEXT_BANK" "$FIT_TEXT_BANK_MANIFEST" "$RADIO_CHECKPOINT" \
  "$GRADIENT_DIAGNOSTIC" \
  "$THERMAL_GUARD" "$AUTHORITY" "$RECEIPT_FINALIZER" \
  "$SURFACE_ROOT/run_manifest.json" "$SURFACE_ROOT/cache_pairing.json" \
  "$SURFACE_ROOT/screen.complete"; do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "missing or symlinked text-response authority input: $required" >&2
    exit 2
  fi
done
if [[ ! "$GRADIENT_DIAGNOSTIC_SHA256" =~ ^[0-9a-f]{64}$ \
      || "$(sha256sum "$GRADIENT_DIAGNOSTIC" | awk '{print $1}')" \
      != "$GRADIENT_DIAGNOSTIC_SHA256" ]]; then
  echo "formal gradient design diagnostic SHA-256 differs" >&2
  exit 2
fi
if [[ ! -f "$SURFACE_ROOT/attention_pooling_screen.json" \
      || -L "$SURFACE_ROOT/attention_pooling_screen.json" \
      || ! -f "$SURFACE_ROOT/runtime_closure_final.json" \
      || -L "$SURFACE_ROOT/runtime_closure_final.json" \
      || -e "$SURFACE_ROOT/query_free_promotion_bundle.json" ]]; then
  echo "warm-start distillation requires only the attention-postcache Surface authority" >&2
  exit 2
fi
for seed in 0 1 2; do
  control="$SURFACE_ROOT/readouts/${CONTEXT_POOLING_MODE}_seed${seed}.pt"
  if [[ ! -f "$control" || -L "$control" \
        || ! -f "${control}.json" || -L "${control}.json" ]]; then
    echo "missing or symlinked seed-$seed frozen Surface control: $control" >&2
    exit 2
  fi
done
if ! compgen -G "$TRAIN_CACHES" >/dev/null; then
  echo "training cache glob is empty: $TRAIN_CACHES" >&2
  exit 2
fi
if ! compgen -G "$VALIDATION_CACHES" >/dev/null; then
  echo "validation cache glob is empty: $VALIDATION_CACHES" >&2
  exit 2
fi

mkdir -p \
  "$OUTPUT_ROOT/readouts" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/audits" \
  "$OUTPUT_ROOT/receipts" "$OUTPUT_ROOT/telemetry" "$CALIBRATION_ROOT"

authority() {
  CUDA_VISIBLE_DEVICES="" bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$AUTHORITY" "$@"
}

bind_surface_control() {
  local seed="$1"
  local lexical="$SURFACE_ROOT/readouts/${CONTEXT_POOLING_MODE}_seed${seed}.pt"
  SURFACE_CONTROL_CHECKPOINT="$(readlink -f "$lexical")"
  if [[ -z "$SURFACE_CONTROL_CHECKPOINT" \
        || ! -f "$SURFACE_CONTROL_CHECKPOINT" \
        || -L "$SURFACE_CONTROL_CHECKPOINT" ]]; then
    echo "seed-$seed Surface control path cannot be frozen" >&2
    return 1
  fi
  SURFACE_CONTROL_CHECKPOINT_SHA256="$(sha256sum "$SURFACE_CONTROL_CHECKPOINT" | awk '{print $1}')"
  if [[ ! "$SURFACE_CONTROL_CHECKPOINT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "seed-$seed Surface control SHA-256 is invalid" >&2
    return 1
  fi
}

seed_calibration_manifest() {
  printf '%s/seed%s.json' "$CALIBRATION_ROOT" "$1"
}

canonical_seed_calibration_manifest() {
  local seed="$1"
  local canonical
  canonical="$(readlink -f "$(seed_calibration_manifest "$seed")")"
  if [[ -z "$canonical" || ! -f "$canonical" || -L "$canonical" ]]; then
    echo "seed-$seed calibration manifest cannot be canonically frozen" >&2
    return 1
  fi
  printf '%s' "$canonical"
}

seed_calibration_audit() {
  printf '%s/audits/calibration_seed%s.json' "$OUTPUT_ROOT" "$1"
}

freeze_command_output() {
  local output="$1"
  shift
  local partial="${output}.partial"
  if [[ -e "$partial" || -L "$partial" ]]; then
    echo "partial authority output requires quarantine: $partial" >&2
    return 1
  fi
  if [[ -e "$output" || -L "$output" ]]; then
    local temporary
    temporary="$(mktemp)"
    if ! "$@" >"$temporary"; then
      rm -f "$temporary"
      return 1
    fi
    if ! cmp -s "$temporary" "$output"; then
      rm -f "$temporary"
      echo "existing immutable command output differs: $output" >&2
      return 1
    fi
    rm -f "$temporary"
    return 0
  fi
  if ! "$@" >"$partial"; then
    echo "command failed; partial output retained for inspection: $partial" >&2
    return 1
  fi
  if ! mv -n "$partial" "$output" || [[ ! -s "$output" ]]; then
    echo "failed to publish immutable command output: $output" >&2
    return 1
  fi
}

for seed in 0 1 2; do
  bind_surface_control "$seed"
  calibration_manifest="$(seed_calibration_manifest "$seed")"
  calibration_log="$OUTPUT_ROOT/logs/calibrate_response_lambda_seed${seed}.log"
  calibration_audit="$(seed_calibration_audit "$seed")"
  common_calibration_args=(
    --train-caches "$TRAIN_CACHES"
    --validation-caches "$VALIDATION_CACHES"
    --fit-text-bank "$FIT_TEXT_BANK"
    --fit-text-bank-manifest "$FIT_TEXT_BANK_MANIFEST"
    --surface-control-checkpoint "$SURFACE_CONTROL_CHECKPOINT"
    --surface-control-checkpoint-sha256 "$SURFACE_CONTROL_CHECKPOINT_SHA256"
    --gradient-diagnostic "$GRADIENT_DIAGNOSTIC"
    --gradient-diagnostic-sha256 "$GRADIENT_DIAGNOSTIC_SHA256"
    --hidden-dim 256
    --reliability-attention-mode log_prior
    --context-pooling-mode "$CONTEXT_POOLING_MODE"
    --radio-checkpoint "$RADIO_CHECKPOINT"
    --token-weight 0.25
    --relation-weight 0.1
    --seed "$seed"
    --device cpu
  )
  if [[ ! -e "$calibration_manifest" ]]; then
    if [[ -e "$calibration_log" || -L "$calibration_log" ]]; then
      echo "seed-$seed calibration log exists without terminal manifest" >&2
      exit 1
    fi
    freeze_command_output "$calibration_log" \
      env CUDA_VISIBLE_DEVICES="" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        radio_gs/scripts/train_surface_region_text_response_distill.py calibrate \
        "${common_calibration_args[@]}" \
        --output "$calibration_manifest"
  elif [[ ! -s "$calibration_manifest" || ! -s "$calibration_log" ]]; then
    echo "seed-$seed partial calibration state requires quarantine" >&2
    exit 1
  fi
  freeze_command_output "$calibration_audit" \
    env CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      radio_gs/scripts/train_surface_region_text_response_distill.py audit-calibration \
      "${common_calibration_args[@]}" \
      --calibration-manifest "$calibration_manifest"
done

GPU_UUID=""
GPU_BUS_ID=""
GPU_PROC_BUS_ID=""
GPU_PCI_PREFIX=""

observe_gpu1() {
  local gpu_info=""
  local candidate_info
  for candidate_info in /proc/driver/nvidia/gpus/*/information; do
    if [[ -r "$candidate_info" ]] \
      && [[ "$(awk '/Device Minor:/ {print $3}' "$candidate_info")" == "$GPU" ]]; then
      gpu_info="$candidate_info"
      break
    fi
  done
  if [[ -z "$gpu_info" ]]; then
    echo "physical GPU1 has no NVIDIA driver record" >&2
    return 2
  fi
  GPU_PROC_BUS_ID="$(awk '/Bus Location:/ {print $3}' "$gpu_info")"
  GPU_PCI_PREFIX="$(
    od -An -tx1 -N16 "/sys/bus/pci/devices/$GPU_PROC_BUS_ID/config" 2>/dev/null \
      | tr -d ' \n'
  )"
  local identity
  if ! identity="$(
    timeout --kill-after=2s 10s nvidia-smi -i "$GPU" \
      --query-gpu=uuid,pci.bus_id --format=csv,noheader,nounits
  )"; then
    echo "physical GPU1 is not currently queryable" >&2
    return 2
  fi
  IFS=',' read -r GPU_UUID GPU_BUS_ID <<<"$identity"
  GPU_UUID="${GPU_UUID//[[:space:]]/}"
  GPU_BUS_ID="${GPU_BUS_ID//[[:space:]]/}"
  local owners
  if ! owners="$(
    timeout --kill-after=2s 10s nvidia-smi -i "$GPU" \
      --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits
  )"; then
    echo "cannot audit physical GPU1 compute owners" >&2
    return 2
  fi
  GPU_COMPUTE_OWNERS=()
  while IFS=',' read -r owner_uuid owner_pid; do
    owner_uuid="${owner_uuid//[[:space:]]/}"
    owner_pid="${owner_pid//[[:space:]]/}"
    if [[ "$owner_uuid" == "$GPU_UUID" && -n "$owner_pid" ]]; then
      GPU_COMPUTE_OWNERS+=("$owner_pid")
    fi
  done <<<"$owners"
}

record_gpu_check() {
  local phase="$1"
  local output="$2"
  local manifest="${3:-}"
  observe_gpu1
  local observed_epoch
  observed_epoch="$(date +%s)"
  local arguments=(
    record-gpu-check
    --output "$output"
    --phase "$phase"
    --gpu-uuid "$GPU_UUID"
    --gpu-bus-id "$GPU_BUS_ID"
    --proc-bus-id "$GPU_PROC_BUS_ID"
    --pci-prefix "$GPU_PCI_PREFIX"
    --observed-epoch "$observed_epoch"
  )
  local owner
  for owner in "${GPU_COMPUTE_OWNERS[@]}"; do
    arguments+=(--compute-owner "$owner")
  done
  if [[ -n "$manifest" ]]; then
    arguments+=(--run-manifest "$manifest")
  fi
  authority "${arguments[@]}"
}

if [[ ! -e "$RUN_MANIFEST" ]]; then
  if [[ -e "$INITIAL_GPU_PREFLIGHT" || -L "$INITIAL_GPU_PREFLIGHT" ]]; then
    echo "initial GPU preflight exists without its run manifest" >&2
    exit 1
  fi
  record_gpu_check "initial_manifest_binding" "$INITIAL_GPU_PREFLIGHT"
elif [[ ! -s "$RUN_MANIFEST" || ! -s "$INITIAL_GPU_PREFLIGHT" ]]; then
  echo "partial run manifest/GPU identity state requires quarantine" >&2
  exit 1
fi

MANIFEST_ARGUMENTS=(
  --repo-root "$REPO_ROOT"
  --lock-root "$LOCK_ROOT"
  --candidate "$CANDIDATE"
  --surface-root "$SURFACE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --train-caches "$TRAIN_CACHES"
  --validation-caches "$VALIDATION_CACHES"
  --fit-text-bank "$FIT_TEXT_BANK"
  --fit-text-bank-manifest "$FIT_TEXT_BANK_MANIFEST"
  --radio-checkpoint "$RADIO_CHECKPOINT"
  --gradient-diagnostic "$GRADIENT_DIAGNOSTIC"
  --gradient-diagnostic-sha256 "$GRADIENT_DIAGNOSTIC_SHA256"
  --initial-gpu-preflight "$INITIAL_GPU_PREFLIGHT"
  --thermal-guard "$THERMAL_GUARD"
  --run-manifest "$RUN_MANIFEST"
  --gpu-max-temp-c "$GPU_MAX_TEMP_C"
  --gpu-start-max-temp-c "$GPU_START_MAX_TEMP_C"
  --gpu-max-power-limit-w "$GPU_MAX_POWER_LIMIT_W"
  --gpu-poll-seconds "$GPU_POLL_SECONDS"
  --gpu-soft-pause-temp-c "$GPU_SOFT_PAUSE_TEMP_C"
  --gpu-soft-resume-temp-c "$GPU_SOFT_RESUME_TEMP_C"
  --gpu-peer-pause-temp-c "$GPU_PEER_PAUSE_TEMP_C"
  --gpu-peer-resume-temp-c "$GPU_PEER_RESUME_TEMP_C"
  --gpu-peer-quiet-seconds "$GPU_PEER_QUIET_SECONDS"
  --gpu-peer-max-power-w "$GPU_PEER_MAX_POWER_W"
  --gpu-peer-max-memory-mib "$GPU_PEER_MAX_MEMORY_MIB"
  --gpu-peer-max-util-pct "$GPU_PEER_MAX_UTIL_PCT"
  --gpu-peer-activity-action "$GPU_PEER_ACTIVITY_ACTION"
  --gpu-owner-pid-namespace-mode "$GPU_OWNER_PID_NAMESPACE_MODE"
)
for seed in 0 1 2; do
  MANIFEST_ARGUMENTS+=(
    --calibration-manifest "$seed=$(canonical_seed_calibration_manifest "$seed")"
    --calibration-audit "$seed=$(seed_calibration_audit "$seed")"
  )
done
if [[ -n "$GPU_PEER_INDEX" ]]; then
  MANIFEST_ARGUMENTS+=(--gpu-peer-index "$GPU_PEER_INDEX")
fi

if [[ -s "$RUN_MANIFEST" ]]; then
  authority verify-manifest "${MANIFEST_ARGUMENTS[@]}"
else
  authority create-manifest "${MANIFEST_ARGUMENTS[@]}"
fi

audit_seed() {
  local seed="$1"
  bind_surface_control "$seed"
  local calibration_manifest
  calibration_manifest="$(canonical_seed_calibration_manifest "$seed")"
  local checkpoint="$OUTPUT_ROOT/readouts/${CANDIDATE}_text_response_seed${seed}.pt"
  local audit="$OUTPUT_ROOT/audits/audit_seed${seed}.json"
  freeze_command_output "$audit" \
    env CUDA_VISIBLE_DEVICES="" \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      radio_gs/scripts/train_surface_region_text_response_distill.py audit-checkpoint \
      --train-caches "$TRAIN_CACHES" \
      --validation-caches "$VALIDATION_CACHES" \
      --fit-text-bank "$FIT_TEXT_BANK" \
      --fit-text-bank-manifest "$FIT_TEXT_BANK_MANIFEST" \
      --calibration-manifest "$calibration_manifest" \
      --run-manifest "$RUN_MANIFEST" \
      --surface-control-checkpoint "$SURFACE_CONTROL_CHECKPOINT" \
      --surface-control-checkpoint-sha256 "$SURFACE_CONTROL_CHECKPOINT_SHA256" \
      --output "$checkpoint" \
      --hidden-dim 256 \
      --epochs 60 \
      --patience 10 \
      --batch-size 16 \
      --learning-rate 2e-4 \
      --weight-decay 1e-4 \
      --token-weight 0.25 \
      --relation-weight 0.1 \
      --reliability-attention-mode log_prior \
      --context-pooling-mode "$CONTEXT_POOLING_MODE" \
      --seed "$seed" \
      --device cuda:0 \
      --radio-checkpoint "$RADIO_CHECKPOINT"
}

PENDING_SEEDS=()
for seed in 0 1 2; do
  state="$(authority classify-seed --run-manifest "$RUN_MANIFEST" --seed "$seed")"
  if [[ "$state" == '{"state": "complete"}' ]]; then
    authority verify-seed --run-manifest "$RUN_MANIFEST" --seed "$seed"
  elif [[ "$state" == '{"state": "pending"}' ]]; then
    PENDING_SEEDS+=("$seed")
  else
    echo "invalid seed authority state: $state" >&2
    exit 1
  fi
done

run_seed() {
  local seed="$1"
  bind_surface_control "$seed"
  local calibration_manifest
  calibration_manifest="$(canonical_seed_calibration_manifest "$seed")"
  local checkpoint="$OUTPUT_ROOT/readouts/${CANDIDATE}_text_response_seed${seed}.pt"
  local training_log="$OUTPUT_ROOT/logs/train_seed${seed}.log"
  local command_record="$OUTPUT_ROOT/receipts/seed${seed}.command.json"
  local telemetry="$OUTPUT_ROOT/telemetry/seed${seed}.csv"
  local receipt="$OUTPUT_ROOT/receipts/seed${seed}.guard.json"
  local kernel_log="$OUTPUT_ROOT/receipts/seed${seed}.kernel.log"
  local gpu_pre="$OUTPUT_ROOT/receipts/seed${seed}.gpu_pre.json"
  local gpu_post="$OUTPUT_ROOT/receipts/seed${seed}.gpu_post.json"
  local terminal="$OUTPUT_ROOT/receipts/seed${seed}.complete.json"

  authority verify-manifest "${MANIFEST_ARGUMENTS[@]}"
  record_gpu_check "pre_seed${seed}" "$gpu_pre" "$RUN_MANIFEST"

  local train_command=(
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
    radio_gs/scripts/train_surface_region_text_response_distill.py train
    --train-caches "$TRAIN_CACHES"
    --validation-caches "$VALIDATION_CACHES"
    --fit-text-bank "$FIT_TEXT_BANK"
    --fit-text-bank-manifest "$FIT_TEXT_BANK_MANIFEST"
    --calibration-manifest "$calibration_manifest"
    --run-manifest "$RUN_MANIFEST"
    --surface-control-checkpoint "$SURFACE_CONTROL_CHECKPOINT"
    --surface-control-checkpoint-sha256 "$SURFACE_CONTROL_CHECKPOINT_SHA256"
    --output "$checkpoint"
    --hidden-dim 256
    --epochs 60
    --patience 10
    --batch-size 16
    --learning-rate 2e-4
    --weight-decay 1e-4
    --token-weight 0.25
    --relation-weight 0.1
    --reliability-attention-mode log_prior
    --context-pooling-mode "$CONTEXT_POOLING_MODE"
    --seed "$seed"
    --device cuda:0
    --radio-checkpoint "$RADIO_CHECKPOINT"
  )
  local command_prepared_epoch
  command_prepared_epoch="$(date +%s)"
  CUDA_VISIBLE_DEVICES="" bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$RECEIPT_FINALIZER" prepare-command \
      --output "$command_record" \
      --run-manifest "$RUN_MANIFEST" \
      --seed "$seed" \
      --scene "$CANDIDATE" \
      --gpu 1 \
      --gpu-uuid "$GPU_UUID" \
      --gpu-bus-id "$GPU_BUS_ID" \
      --prepared-epoch "$command_prepared_epoch" \
      -- "${train_command[@]}"

  local journal_start journal_end guard_status journal_status filter_status
  journal_start="$(date +%s)"
  guard_status=0
  GPU="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
    GPU_TELEMETRY_LOG="$telemetry" \
    GPU_MAX_TEMP_C="$GPU_MAX_TEMP_C" \
    GPU_START_MAX_TEMP_C="$GPU_START_MAX_TEMP_C" \
    GPU_MAX_POWER_LIMIT_W="$GPU_MAX_POWER_LIMIT_W" \
    GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
    GPU_SOFT_PAUSE_TEMP_C="$GPU_SOFT_PAUSE_TEMP_C" \
    GPU_SOFT_RESUME_TEMP_C="$GPU_SOFT_RESUME_TEMP_C" \
    GPU_OWNER_PID_NAMESPACE_MODE="$GPU_OWNER_PID_NAMESPACE_MODE" \
    GPU_PEER_INDEX="$GPU_PEER_INDEX" \
    GPU_PEER_PAUSE_TEMP_C="$GPU_PEER_PAUSE_TEMP_C" \
    GPU_PEER_RESUME_TEMP_C="$GPU_PEER_RESUME_TEMP_C" \
    GPU_PEER_QUIET_SECONDS="$GPU_PEER_QUIET_SECONDS" \
    GPU_PEER_MAX_POWER_W="$GPU_PEER_MAX_POWER_W" \
    GPU_PEER_MAX_MEMORY_MIB="$GPU_PEER_MAX_MEMORY_MIB" \
    GPU_PEER_MAX_UTIL_PCT="$GPU_PEER_MAX_UTIL_PCT" \
    GPU_PEER_ACTIVITY_ACTION="$GPU_PEER_ACTIVITY_ACTION" \
    bash "$THERMAL_GUARD" -- "${train_command[@]}" \
    >"$training_log" 2>&1 || guard_status=$?
  journal_end="$(date +%s)"
  if (( guard_status != 0 )); then
    echo "seed-$seed guarded training failed with status $guard_status; partial artifacts require quarantine" >&2
    return "$guard_status"
  fi
  if [[ ! -s "$checkpoint" || ! -s "${checkpoint}.json" || ! -s "$telemetry" ]]; then
    echo "seed-$seed guarded training produced incomplete artifacts" >&2
    return 1
  fi

  local kernel_partial="${kernel_log}.partial"
  if [[ -e "$kernel_partial" || -e "$kernel_log" || -L "$kernel_partial" || -L "$kernel_log" ]]; then
    echo "seed-$seed kernel journal output already exists" >&2
    return 1
  fi
  printf 'surface_text_response_seed=%s\tstart_epoch=%s\tend_epoch=%s\n' \
    "$seed" "$journal_start" "$journal_end" >"$kernel_partial"
  set +e
  journalctl -k --since "@$journal_start" --until "@$journal_end" \
      --no-pager -o short-iso \
    | rg -i "${GPU_UUID}|${GPU_PROC_BUS_ID}|${GPU_PROC_BUS_ID#*:}|${GPU_BUS_ID}" \
      >>"$kernel_partial"
  local pipeline_status=("${PIPESTATUS[@]}")
  journal_status="${pipeline_status[0]}"
  filter_status="${pipeline_status[1]}"
  set -e
  if (( journal_status != 0 || (filter_status != 0 && filter_status != 1) )); then
    echo "cannot audit kernel Xid/PCIe interval for seed-$seed" >&2
    return 1
  fi
  mv -n "$kernel_partial" "$kernel_log"

  record_gpu_check "post_seed${seed}" "$gpu_post" "$RUN_MANIFEST"
  audit_seed "$seed"
  CUDA_VISIBLE_DEVICES="" bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$RECEIPT_FINALIZER" finalize \
      --output "$receipt" \
      --command-record "$command_record" \
      --telemetry "$telemetry" \
      --guard "$THERMAL_GUARD" \
      --stage-output "$checkpoint" \
      --exit-status 0

  authority verify-manifest "${MANIFEST_ARGUMENTS[@]}"
  authority finalize-seed \
    --run-manifest "$RUN_MANIFEST" --seed "$seed" --terminal "$terminal" \
    --journal-start-epoch "$journal_start" --journal-end-epoch "$journal_end"
  authority verify-seed --run-manifest "$RUN_MANIFEST" --seed "$seed"
}

for seed in "${PENDING_SEEDS[@]}"; do
  run_seed "$seed"
done

authority verify-manifest "${MANIFEST_ARGUMENTS[@]}"
authority finalize-run --run-manifest "$RUN_MANIFEST" --output "$COMPLETION"
