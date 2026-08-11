#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/RADIO-GS
PYTHON=/root/miniconda3/envs/cybersim_agent/bin/python
RUN_ROOT="$ROOT/local_ssd_results/source_sam_single_radio_lerf_capability_relative_v3/figurines/rgb_free_benchmark_sentinel"
FIELD="$ROOT/local_ssd_results/source_sam_single_radio_lerf_capability_relative_v3/figurines/canonical_radio_source_sam_capability_relative_e5_seed0.pth"
FIELD_SHA=d38256ed91c4f373759355395bd8c8f6ddcd4b4f59018f9d27e53525a35f31b9
MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/factorized_raw_radio_exact_marginal.pt
MPR_SHA=4bad5345f6721f7fb2fab5a234a93ae80c0e5ce39217d1bd841e29559fabbf4b
RADIO=/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
RADIO_SHA=bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9
READOUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260717/surface_region_contract_v2/readout_v2_clean_h256.pth
READOUT_SHA=5b2d123a7827d9ab79aa4aa5a70077f00a656beebcf4c95ea5a3c9efdbe13ccb
GEOMETRY=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
GEOMETRY_SHA=6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2
POS_TEXT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260715/text/siglip2_lerf_figurines_officialcanonical_prompt5.pt
POS_TEXT_SHA=08fa6f870c824fde212b302d6c88cf74487f1122055fac709708399fb480578b
NEG_TEXT="$ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt"
NEG_TEXT_SHA=18d2aac56b50a9670ffe04b397d23a4652dd44fe8f18ed7a309a82b6c1102b67
PREREG="$ROOT/paper/artifacts/source_sam_single_radio_lerf_figurines_capability_relative_v3_rgb_free_benchmark_preregistration_20260811.json"
PREREG_SHA=979ea3ace8168618ef2c40ecd68fdc3531736a1932e89354db142384801f95c2
SEAL="$ROOT/paper/artifacts/source_sam_single_radio_lerf_figurines_capability_relative_v3_query_score_batch_seal_20260811.json"
LOG="$RUN_ROOT/materialization.log"
TELEMETRY="$RUN_ROOT/gpu0_telemetry.csv"

if [[ -e "$RUN_ROOT" || -e "$SEAL" ]]; then
  echo "refusing to clobber v3 derivative root or seal" >&2
  exit 2
fi
if [[ "$(sha256sum "$PREREG" | awk '{print $1}')" != "$PREREG_SHA" ]]; then
  echo "preregistration SHA mismatch" >&2
  exit 3
fi
if [[ "$(sha256sum "$FIELD" | awk '{print $1}')" != "$FIELD_SHA" ]]; then
  echo "field SHA mismatch" >&2
  exit 4
fi

read -r PRE_TEMP PRE_MEM PRE_UTIL < <(
  nvidia-smi -i 0 --query-gpu=temperature.gpu,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' '
)
if (( PRE_TEMP >= 82 || PRE_MEM > 256 || PRE_UTIL > 5 )); then
  echo "GPU0 preflight failed: temp=$PRE_TEMP mem=$PRE_MEM util=$PRE_UTIL" >&2
  exit 5
fi

mkdir -p "$RUN_ROOT"
touch "$LOG"
echo "timestamp,gpu,temp_c,power_w,power_limit_w,memory_mib,util_percent,stage,state" >"$TELEMETRY"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$ROOT/local_ssd_results/nvidia_driver_535_runtime:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"

run_guarded() {
  local stage=$1
  shift
  printf '[stage=%s] start %s\n' "$stage" "$(date --iso-8601=seconds)" >>"$LOG"
  "$@" >>"$LOG" 2>&1 &
  local child_pid=$!
  local paused=0
  while kill -0 "$child_pid" 2>/dev/null; do
    local line temp
    line=$(nvidia-smi -i 0 --query-gpu=timestamp,index,temperature.gpu,power.draw,power.limit,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    temp=$(printf '%s' "$line" | awk -F, '{print $3}')
    if (( temp >= 85 )); then
      printf '%s,%s,hard_stop\n' "$line" "$stage" >>"$TELEMETRY"
      kill -TERM "$child_pid" 2>/dev/null || true
      wait "$child_pid" || true
      exit 85
    fi
    if (( temp >= 82 && paused == 0 )); then
      kill -STOP "$child_pid"
      paused=1
    elif (( temp <= 78 && paused == 1 )); then
      kill -CONT "$child_pid"
      paused=0
    fi
    if (( paused == 1 )); then
      printf '%s,%s,paused\n' "$line" "$stage" >>"$TELEMETRY"
    else
      printf '%s,%s,running\n' "$line" "$stage" >>"$TELEMETRY"
    fi
    sleep 60
  done
  wait "$child_pid"
  printf '[stage=%s] complete %s\n' "$stage" "$(date --iso-8601=seconds)" >>"$LOG"
}

STATE="$RUN_ROOT/factorized_primitive_state_v3.pt"
CAPABILITY="$RUN_ROOT/official_dino_sam3_views_v3.pt"
GRAPH="$RUN_ROOT/support_graph_v3_v2_isomorphic.pt"
DESCRIPTOR="$RUN_ROOT/descriptor_v3_v2_isomorphic.pt"
QUERY_DESCRIPTOR="$RUN_ROOT/descriptor_v3_v2_isomorphic_query.pt"

run_guarded factorized_state "$PYTHON" "$ROOT/radio_gs/scripts/build_factorized_primitive_state.py" \
  --field-checkpoint "$FIELD" \
  --expected-field-checkpoint-sha256 "$FIELD_SHA" \
  --factorized-radio-cache "$MPR" \
  --expected-factorized-radio-cache-sha256 "$MPR_SHA" \
  --output "$STATE" \
  --chunk-size 4096

run_guarded capability "$PYTHON" "$ROOT/radio_gs/scripts/build_canonical_capability_views.py" \
  --field-checkpoint "$FIELD" \
  --field-checkpoint-schema factorized-v2 \
  --expected-field-checkpoint-sha256 "$FIELD_SHA" \
  --output "$CAPABILITY" \
  --mpr-cache "$MPR" \
  --expected-mpr-cache-sha256 "$MPR_SHA" \
  --observation-contract canonical \
  --radio-checkpoint "$RADIO" \
  --expected-radio-checkpoint-sha256 "$RADIO_SHA" \
  --batch-size 2048 \
  --device cuda:0

CAPABILITY_SHA=$(sha256sum "$CAPABILITY" | awk '{print $1}')
run_guarded support_graph "$PYTHON" "$ROOT/radio_gs/scripts/build_canonical_support_graph.py" \
  --capability-cache "$CAPABILITY" \
  --expected-capability-cache-sha256 "$CAPABILITY_SHA" \
  --output "$GRAPH" \
  --neighbors 16 \
  --spatial-scale 2.0 \
  --appearance-temperature 0.1 \
  --boundary-temperature 0.1 \
  --normal-temperature 0.2 \
  --surface-tangent-temperature 0.2 \
  --surface-relation none \
  --surface-normal-neighbors 24 \
  --surface-normal-batch-size 8192 \
  --surface-normal-min-planarity 0.0 \
  --surface-topology-min-affinity 0.0 \
  --affinity-dim 128 \
  --hash-batch-size 8192 \
  --capability-affinity-mode signed_hash \
  --affinity-device cpu \
  --affinity-chunk-size 65536 \
  --topology-mode symmetric_union

STATE_SHA=$(sha256sum "$STATE" | awk '{print $1}')
GRAPH_SHA=$(sha256sum "$GRAPH" | awk '{print $1}')
run_guarded surface_region "$PYTHON" "$ROOT/radio_gs/scripts/build_surface_region_semantic_cache.py" \
  --field-checkpoint "$FIELD" \
  --field-checkpoint-schema factorized-v2 \
  --factorized-primitive-state "$STATE" \
  --factorized-primitive-state-sha256 "$STATE_SHA" \
  --field-checkpoint-sha256 "$FIELD_SHA" \
  --support-graph "$GRAPH" \
  --support-graph-sha256 "$GRAPH_SHA" \
  --readout-checkpoint "$READOUT" \
  --readout-checkpoint-sha256 "$READOUT_SHA" \
  --mpr-cache-sha256 "$MPR_SHA" \
  --output "$DESCRIPTOR" \
  --query-output "$QUERY_DESCRIPTOR" \
  --region-radii 0.25,0.45,0.7 \
  --graph-neighbors 16 \
  --radio-batch-size 4096 \
  --semantic-batch-size 512 \
  --resume-dir "$RUN_ROOT/resume_descriptor" \
  --radio-feature-normalization legacy_raw \
  --thermal-pacing-seconds-per-batch 0.05 \
  --device cuda:0 \
  --radio-checkpoint "$RADIO" \
  --radio-checkpoint-sha256 "$RADIO_SHA"

DESCRIPTOR_SHA=$(sha256sum "$DESCRIPTOR" | awk '{print $1}')
run_guarded positive_fp32 "$PYTHON" "$ROOT/radio_gs/scripts/materialize_lerf_multiscale_query_score_cache_fp32.py" \
  --descriptor-cache "$DESCRIPTOR" \
  --descriptor-cache-sha256 "$DESCRIPTOR_SHA" \
  --text-query-cache "$POS_TEXT" \
  --text-query-cache-sha256 "$POS_TEXT_SHA" \
  --field-checkpoint "$FIELD" \
  --field-checkpoint-sha256 "$FIELD_SHA" \
  --readout-checkpoint "$READOUT" \
  --readout-checkpoint-sha256 "$READOUT_SHA" \
  --renderer-geometry-checkpoint "$GEOMETRY" \
  --renderer-geometry-checkpoint-sha256 "$GEOMETRY_SHA" \
  --output "$RUN_ROOT/positive_fp32.pt" \
  --chunk-size 4096

run_guarded negative_fp32 "$PYTHON" "$ROOT/radio_gs/scripts/materialize_lerf_multiscale_query_score_cache_fp32.py" \
  --descriptor-cache "$DESCRIPTOR" \
  --descriptor-cache-sha256 "$DESCRIPTOR_SHA" \
  --text-query-cache "$NEG_TEXT" \
  --text-query-cache-sha256 "$NEG_TEXT_SHA" \
  --field-checkpoint "$FIELD" \
  --field-checkpoint-sha256 "$FIELD_SHA" \
  --readout-checkpoint "$READOUT" \
  --readout-checkpoint-sha256 "$READOUT_SHA" \
  --renderer-geometry-checkpoint "$GEOMETRY" \
  --renderer-geometry-checkpoint-sha256 "$GEOMETRY_SHA" \
  --output "$RUN_ROOT/negative_fp32.pt" \
  --chunk-size 4096 \
  --allow-missing-text-canonicalization-metadata

"$PYTHON" "$ROOT/radio_gs/scripts/seal_lerf_rgb_free_query_score_batch.py" \
  --root "$RUN_ROOT" \
  --field-checkpoint "$FIELD" \
  --expected-field-checkpoint-sha256 "$FIELD_SHA" \
  --preregistration "$PREREG" \
  --expected-preregistration-sha256 "$PREREG_SHA" \
  --output "$SEAL" >>"$LOG" 2>&1

printf '[all] sealed %s\n' "$(date --iso-8601=seconds)" >>"$LOG"
