#!/usr/bin/env bash

# Post-freeze combination: rebuild the canonical capability graph from the
# first frozen Field-A checkpoint, then apply the already frozen LERF text
# query-conditioned support operator without changing any parameter.

set -euo pipefail

REPO_ROOT=/root/RADIO-GS
RUN_PYTHON="$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
OUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260803/lerf_field_a_query_conditioned_support_v1
FIELD=/mnt/pool/sqy/results/RADIO-GS/output/canonical_fields/figurines_compact_d256_l128_field_a_exact_capability_seed0.pth
FIELD_SHA=9753eeb9bba7062b26f2443ee61be8bf2be4b4eedb3516a21984f62188a27067
MPR=/mnt/pool/sqy/results/RADIO-GS/output/canonical_mpr/figurines_raw_radio_top1_120_plus_adjoint16_support_verified_pose.pt
MPR_SHA=df01507d65b6a6e6ad75e001fd926b30e18482dd64cb065f3c58710c17969f81
RADIO=/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
SCORE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260803/lerf_field_a_figurines_v1/query_scores/figurines_field_a.pt
RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
CONFIG="$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"
CAPABILITY="$OUT/figurines_field_a_capability.pt"
GRAPH="$OUT/figurines_field_a_support_graph_k16.pt"
RESULT="$OUT/figurines/figurines/lerf_direct_3d_selection_results.json"

guard() {
  local tag="$1"
  shift
  GPU=1 \
  GPU_TELEMETRY_LOG="$OUT/logs/${tag}.telemetry.csv" \
  GPU_OWNER_AUDIT_LOG="$OUT/logs/${tag}.owner_audit.csv" \
  GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
  GPU_POLL_SECONDS=20 \
  GPU_START_MAX_TEMP_C=78 \
  GPU_SOFT_PAUSE_TEMP_C=81 \
  GPU_SOFT_RESUME_TEMP_C=76 \
  GPU_MAX_TEMP_C=84 \
  GPU_MAX_POWER_LIMIT_W=300.5 \
  CUDA_VISIBLE_DEVICES=1 \
    bash "$GUARD" -- "$@"
}

mkdir -p "$OUT/logs"
for path in "$CAPABILITY" "$CAPABILITY.json" "$GRAPH" "$GRAPH.json" \
  "$OUT/figurines/method_receipt.prelabel.json" "$RESULT"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing to overwrite Field-A support artifact: $path" >&2
    exit 2
  fi
done

guard capability \
  bash "$RUN_PYTHON" \
    "$REPO_ROOT/radio_gs/scripts/build_canonical_capability_views.py" \
    --field-checkpoint "$FIELD" \
    --expected-field-checkpoint-sha256 "$FIELD_SHA" \
    --mpr-cache "$MPR" \
    --expected-mpr-cache-sha256 "$MPR_SHA" \
    --observation-contract compatible-legacy \
    --radio-checkpoint "$RADIO" \
    --output "$CAPABILITY" \
    --batch-size 2048 \
    --device cuda:0 \
    >"$OUT/logs/capability.stdout.log" 2>&1

# Preserve the exact historic support_graph_v3 configuration. Graph topology
# and signed feature hashing are CPU operations in that frozen construction.
bash "$RUN_PYTHON" \
  "$REPO_ROOT/radio_gs/scripts/build_canonical_support_graph.py" \
  --capability-cache "$CAPABILITY" \
  --output "$GRAPH" \
  --neighbors 16 \
  --spatial-scale 2.0 \
  --appearance-temperature 0.1 \
  --boundary-temperature 0.1 \
  --normal-temperature 0.2 \
  --surface-relation none \
  --covisibility-weight 0.0 \
  --affinity-dim 128 \
  --hash-batch-size 8192 \
  --capability-affinity-mode signed_hash \
  --affinity-device cpu \
  --affinity-chunk-size 65536 \
  --topology-mode symmetric_union \
  >"$OUT/logs/graph.stdout.log" 2>&1

guard query_support \
  bash "$RUN_PYTHON" \
    "$REPO_ROOT/radio_gs/scripts/eval_lerf_query_conditioned_support.py" \
    --config "$CONFIG" \
    --checkpoint "$RENDERER" \
    --scene figurines \
    --ours_multiscale_query_score_cache "$SCORE" \
    --support_graph "$GRAPH" \
    --output_dir "$OUT/figurines" \
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
    --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
    --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
    --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
    --gpu 0 \
    >"$OUT/logs/query_support.stdout.log" 2>&1
