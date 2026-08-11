#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/RADIO-GS
PYTHON=/root/miniconda3/envs/cybersim_agent/bin/python
RUN_ROOT="$ROOT/local_ssd_results/source_sam_single_radio_lerf_capability_relative_v3/figurines"
OUTPUT="$RUN_ROOT/canonical_radio_source_sam_capability_relative_e5_seed0.pth"
LOG="$RUN_ROOT/canonical_radio_source_sam_capability_relative_e5_seed0.train.log"
TELEMETRY="$RUN_ROOT/canonical_radio_source_sam_capability_relative_e5_seed0.telemetry.csv"
MANIFEST="$RUN_ROOT/source_only_sam_capability_relative_manifest.v3.1.json"
MANIFEST_SHA=a2ea46527974e202f218788b31c5fd693e3eac245c07912a6d25540f151082d0

if [[ -e "$OUTPUT" || -e "$OUTPUT.json" || -e "$LOG" || -e "$TELEMETRY" ]]; then
  echo "refusing to clobber v3 output/log/telemetry" >&2
  exit 2
fi

read -r PRE_TEMP PRE_MEM PRE_UTIL < <(
  nvidia-smi -i 0 --query-gpu=temperature.gpu,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' '
)
if (( PRE_TEMP >= 82 || PRE_MEM > 256 || PRE_UTIL > 5 )); then
  echo "GPU0 preflight failed: temp=$PRE_TEMP mem=$PRE_MEM util=$PRE_UTIL" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ROOT/local_ssd_results/nvidia_driver_535_runtime:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"

"$PYTHON" "$ROOT/radio_gs/scripts/train_canonical_radio_field.py" \
  --mpr-cache /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/factorized_raw_radio_exact_marginal.pt \
  --expected-mpr-cache-sha256 4bad5345f6721f7fb2fab5a234a93ae80c0e5ce39217d1bd841e29559fabbf4b \
  --observation-contract canonical-factorized-radio-v1 \
  --radio-checkpoint /root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar \
  --expected-radio-checkpoint-sha256 bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9 \
  --expected-feature-output-bundle-sha256 2fc5c37e1790afba62ccb18175c1a853d624b5bd9f039a8701fbdb7d783b08e3 \
  --output "$OUTPUT" \
  --initial-field-checkpoint /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/fields/trainable_basis_d512_l512_exact_marginal.pth \
  --expected-initial-field-checkpoint-sha256 c6c7d8b378c830a1effa3e86b4f4251f00d73a597ab76812add15745ddd352f5 \
  --source-only-sam-capability-relative-structure-manifest "$MANIFEST" \
  --expected-source-only-sam-capability-relative-structure-manifest-sha256 "$MANIFEST_SHA" \
  --official-capability-loss \
  --capability-target-contract matched_exact_marginal \
  --dino-mpr-cache /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/dino_v3_matched_exact_marginal.pt \
  --expected-dino-v3-mpr-cache-sha256 dd2c3623a206e0bdf04615c8086cbda29556e170af019f5b298cc5a5346b8067 \
  --sam3-mpr-cache /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/sam3_matched_exact_marginal.pt \
  --expected-sam3-mpr-cache-sha256 8a6a5d289e04f4b3e4587eca7085a67f617ef17d01ff55934ca73952c66fdb33 \
  --factorized-capability-reference-mpr-cache /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/raw_radio_matched_exact_marginal.pt \
  --expected-factorized-capability-reference-mpr-cache-sha256 9c82f3d33101cb1e088af3cdf021df89130166a4b49e4005378f5d672108fb7c \
  --factorized-capability-cohort-authority "$ROOT/paper/artifacts/canonical_factorized_radio_v1_figurines_exact_marginal_v3_capability_cohort_authority_20260805.json" \
  --expected-factorized-capability-cohort-authority-sha256 8f490353a890101c00a9047c401d5cd6464dfd2c4bf6656696e5e26ad339f11a \
  --no-fusion-reliability \
  --epochs 5 \
  --min-epochs 5 \
  --batch-size 1024 \
  --eval-batch-size 8192 \
  --learning-rate 0.0002 \
  --weight-decay 0.00001 \
  --validation-fraction 0.05 \
  --mpr-weight 1.0 \
  --dino-weight 0.2 \
  --sam3-weight 0.2 \
  --seed 0 \
  --device cuda:0 >"$LOG" 2>&1 &
TRAIN_PID=$!

echo "timestamp,gpu,temp_c,power_w,power_limit_w,memory_mib,util_percent,state" >"$TELEMETRY"
PAUSED=0
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  LINE=$(nvidia-smi -i 0 --query-gpu=timestamp,index,temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  TEMP=$(printf '%s' "$LINE" | awk -F, '{print $3}')
  if (( TEMP >= 85 )); then
    printf '%s,hard_stop\n' "$LINE" >>"$TELEMETRY"
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" || true
    exit 85
  fi
  if (( TEMP >= 82 && PAUSED == 0 )); then
    kill -STOP "$TRAIN_PID"
    PAUSED=1
  elif (( TEMP <= 78 && PAUSED == 1 )); then
    kill -CONT "$TRAIN_PID"
    PAUSED=0
  fi
  if (( PAUSED == 1 )); then STATE=paused; else STATE=running; fi
  printf '%s,%s\n' "$LINE" "$STATE" >>"$TELEMETRY"
  sleep 60
done
wait "$TRAIN_PID"
