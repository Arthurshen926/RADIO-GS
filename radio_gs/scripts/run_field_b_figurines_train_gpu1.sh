#!/usr/bin/env bash

# The single preregistered figurines Field-B run. This script trains the
# query-independent field only and never invokes a downstream evaluator.

set -euo pipefail

REPO_ROOT=/root/RADIO-GS
RUN_PYTHON="$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
TRAINER="$REPO_ROOT/radio_gs/scripts/train_canonical_radio_field.py"
GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
OUT_ROOT=/mnt/pool/sqy/results/RADIO-GS/output
RUN_ROOT="$OUT_ROOT/optimization_20260803/field_b"
OUTPUT="$OUT_ROOT/canonical_fields/figurines_compact_d256_l128_field_b_boundary_relation_seed0.pth"
STDOUT_LOG="$RUN_ROOT/logs/figurines_field_b_train_gpu1_attempt1.stdout.log"
TELEMETRY="$RUN_ROOT/logs/figurines_field_b_train_gpu1_attempt1.telemetry.csv"
OWNER_AUDIT="$RUN_ROOT/logs/figurines_field_b_train_gpu1_attempt1.owner_audit.csv"

REGISTRATION="$REPO_ROOT/paper/artifacts/canonical_field_b_boundary_relation_registration_20260803.json"
REGISTRATION_SHA=bbff75dbc7583025d3648009896632241eb04d94ae2ab5f5434ebfa5b0961986
PRIMARY="$OUT_ROOT/canonical_mpr/figurines_raw_radio_top1_120_plus_adjoint16_support_verified_pose.pt"
PRIMARY_SHA=df01507d65b6a6e6ad75e001fd926b30e18482dd64cb065f3c58710c17969f81
OBSERVATION="$OUT_ROOT/canonical_mpr/figurines_raw_radio_adjoint_train16_verified_pose.pt"
OBSERVATION_SHA=9d9c06223e638b978434fd6fe8057dc0c5fc00e989e3892ff2cfff262795119f
DINO="$OUT_ROOT/canonical_mpr/figurines_dino_v3_adjoint_train16_mean_resultant_field_a.pt"
DINO_SHA=9c08de38a240623ee4588f3baba8c1450cf96b2abc2aeb4c6f73acf52352815e
SAM="$OUT_ROOT/canonical_mpr/figurines_sam3_adjoint_train16_mean_resultant_field_a.pt"
SAM_SHA=016d2274bb6a031cc2919299772132b129cf724e962ed3853191692b010716d6
TRIPLETS="$RUN_ROOT/figurines_exact_capability_boundary_triplets_k16_v1.pt"
TRIPLETS_SHA=7fec5dbe2b042d41b73cdbdad719bfbf12f9ebeed511abbd80f18bdd378b7100
INITIAL="$OUT_ROOT/canonical_fields/figurines_compact_d256_l128_field_a_exact_capability_seed0.pth"
INITIAL_SHA=9753eeb9bba7062b26f2443ee61be8bf2be4b4eedb3516a21984f62188a27067
RADIO=/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
RADIO_SHA=bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9
FEATURE_BUNDLE_SHA=2fc5c37e1790afba62ccb18175c1a853d624b5bd9f039a8701fbdb7d783b08e3

mkdir -p "$RUN_ROOT/logs"
for path in "$OUTPUT" "$OUTPUT.json" "$STDOUT_LOG" "$TELEMETRY" "$OWNER_AUDIT"; do
  if [[ -e "$path" ]]; then
    echo "refusing to overwrite the registered Field-B artifact: $path" >&2
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
    bash "$RUN_PYTHON" "$TRAINER" \
      --mpr-cache "$PRIMARY" \
      --expected-mpr-cache-sha256 "$PRIMARY_SHA" \
      --observation-contract compatible-legacy \
      --radio-checkpoint "$RADIO" \
      --expected-radio-checkpoint-sha256 "$RADIO_SHA" \
      --expected-feature-output-bundle-sha256 "$FEATURE_BUNDLE_SHA" \
      --output "$OUTPUT" \
      --initial-field-checkpoint "$INITIAL" \
      --expected-initial-field-checkpoint-sha256 "$INITIAL_SHA" \
      --radio-version c-radio_v4-h \
      --device cuda:0 \
      --coefficient-dim 256 \
      --local-dim 128 \
      --spatial-coarse-dim 0 \
      --primitive-fusion \
      --fusion-reliability \
      --hidden-dim 192 \
      --fusion-residual-blocks 0 \
      --official-capability-loss \
      --capability-target-contract field_a_exact_adjoint \
      --capability-observation-reference-mpr-cache "$OBSERVATION" \
      --expected-capability-observation-reference-mpr-cache-sha256 "$OBSERVATION_SHA" \
      --dino-mpr-cache "$DINO" \
      --expected-dino-v3-mpr-cache-sha256 "$DINO_SHA" \
      --sam3-mpr-cache "$SAM" \
      --expected-sam3-mpr-cache-sha256 "$SAM_SHA" \
      --relation-objective field_b_boundary_ranking_v1 \
      --field-b-experiment-registration "$REGISTRATION" \
      --expected-field-b-experiment-registration-sha256 "$REGISTRATION_SHA" \
      --relation-triplet-cache "$TRIPLETS" \
      --expected-relation-triplet-cache-sha256 "$TRIPLETS_SHA" \
      --mpr-weight 1.0 \
      --dino-weight 0.2 \
      --sam3-weight 0.2 \
      --relation-weight 0.05 \
      --coefficient-weight 0.00001 \
      --basis-orthogonality-weight 0.001 \
      --epochs 20 \
      --min-epochs 20 \
      --batch-size 4096 \
      --eval-batch-size 16384 \
      --learning-rate 0.002 \
      --weight-decay 0.00001 \
      --validation-fraction 0.05 \
      --target-cosine 0.985 \
      --seed 0 \
      >"$STDOUT_LOG" 2>&1
