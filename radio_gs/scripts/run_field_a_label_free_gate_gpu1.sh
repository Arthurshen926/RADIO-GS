#!/usr/bin/env bash

# Recompute the preregistered, label-free Field-A gate on physical GPU1.
# This script never invokes a downstream benchmark evaluator.

set -euo pipefail

REPO_ROOT=/root/RADIO-GS
RUN_PYTHON="$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
EVALUATOR="$REPO_ROOT/radio_gs/scripts/eval_field_a_label_free_gate.py"
GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
OUT_ROOT=/mnt/pool/sqy/results/RADIO-GS/output
RUN_ROOT="$OUT_ROOT/optimization_20260803/field_a"
RECEIPT="$RUN_ROOT/figurines_label_free_gate_checkpoint_receipt.json"
STDOUT_LOG="$RUN_ROOT/logs/figurines_field_a_gate_gpu1.stdout.log"
TELEMETRY="$RUN_ROOT/logs/figurines_field_a_gate_gpu1.telemetry.csv"
OWNER_AUDIT="$RUN_ROOT/logs/figurines_field_a_gate_gpu1.owner_audit.csv"
TRAIN_TELEMETRY="$RUN_ROOT/logs/figurines_field_a_train_gpu1_attempt3.telemetry.csv"
TRAIN_OWNER_AUDIT="$RUN_ROOT/logs/figurines_field_a_train_gpu1_attempt3.owner_audit.csv"

REGISTRATION="$REPO_ROOT/paper/artifacts/canonical_field_a_exact_capability_registration_20260803.json"
REGISTRATION_SHA=c90ee536a4c38a0c6e7254cca607da79f24f719a965bb12775866aa67868571f
PRIMARY="$OUT_ROOT/canonical_mpr/figurines_raw_radio_top1_120_plus_adjoint16_support_verified_pose.pt"
PRIMARY_SHA=df01507d65b6a6e6ad75e001fd926b30e18482dd64cb065f3c58710c17969f81
OBSERVATION="$OUT_ROOT/canonical_mpr/figurines_raw_radio_adjoint_train16_verified_pose.pt"
OBSERVATION_SHA=9d9c06223e638b978434fd6fe8057dc0c5fc00e989e3892ff2cfff262795119f
DINO="$OUT_ROOT/canonical_mpr/figurines_dino_v3_adjoint_train16_mean_resultant_field_a.pt"
DINO_SHA=9c08de38a240623ee4588f3baba8c1450cf96b2abc2aeb4c6f73acf52352815e
SAM="$OUT_ROOT/canonical_mpr/figurines_sam3_adjoint_train16_mean_resultant_field_a.pt"
SAM_SHA=016d2274bb6a031cc2919299772132b129cf724e962ed3853191692b010716d6
RADIO=/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
RADIO_SHA=bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9
FEATURE_BUNDLE_SHA=2fc5c37e1790afba62ccb18175c1a853d624b5bd9f039a8701fbdb7d783b08e3
INITIAL="$OUT_ROOT/canonical_fields/figurines_compact_d256_l128_primary_frozen_adjoint16_fallback_caploss_seed0.pth"
INITIAL_SHA=328ba9f9f19f69f02a118462cbb427fac7670cbc83e4d4eade7e66902943aa66
FINAL="$OUT_ROOT/canonical_fields/figurines_compact_d256_l128_field_a_exact_capability_seed0.pth"
FINAL_SHA=9753eeb9bba7062b26f2443ee61be8bf2be4b4eedb3516a21984f62188a27067

mkdir -p "$RUN_ROOT/logs"
for path in "$RECEIPT" "$STDOUT_LOG" "$TELEMETRY" "$OWNER_AUDIT"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite the registered Field-A gate artifact: $path" >&2
    exit 2
  fi
done

GPU=1 \
GPU_TELEMETRY_LOG="$TELEMETRY" \
GPU_OWNER_AUDIT_LOG="$OWNER_AUDIT" \
GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
GPU_POLL_SECONDS=20 \
GPU_START_MAX_TEMP_C=78 \
GPU_SOFT_PAUSE_TEMP_C=81 \
GPU_SOFT_RESUME_TEMP_C=76 \
GPU_MAX_TEMP_C=84 \
GPU_MAX_POWER_LIMIT_W=300.5 \
CUDA_VISIBLE_DEVICES=1 \
  bash "$GUARD" -- \
    bash "$RUN_PYTHON" "$EVALUATOR" \
      --experiment-registration "$REGISTRATION" \
      --expected-experiment-registration-sha256 "$REGISTRATION_SHA" \
      --mpr-cache "$PRIMARY" \
      --expected-mpr-cache-sha256 "$PRIMARY_SHA" \
      --capability-observation-reference-mpr-cache "$OBSERVATION" \
      --expected-capability-observation-reference-mpr-cache-sha256 "$OBSERVATION_SHA" \
      --dino-mpr-cache "$DINO" \
      --expected-dino-v3-mpr-cache-sha256 "$DINO_SHA" \
      --sam3-mpr-cache "$SAM" \
      --expected-sam3-mpr-cache-sha256 "$SAM_SHA" \
      --radio-checkpoint "$RADIO" \
      --expected-radio-checkpoint-sha256 "$RADIO_SHA" \
      --expected-feature-output-bundle-sha256 "$FEATURE_BUNDLE_SHA" \
      --initial-field-checkpoint "$INITIAL" \
      --expected-initial-field-checkpoint-sha256 "$INITIAL_SHA" \
      --final-field-checkpoint "$FINAL" \
      --expected-final-field-checkpoint-sha256 "$FINAL_SHA" \
      --training-telemetry "$TRAIN_TELEMETRY" \
      --training-owner-audit "$TRAIN_OWNER_AUDIT" \
      --output "$RECEIPT" \
      --device cuda:0 \
      --batch-size 16384 \
      >"$STDOUT_LOG" 2>&1

