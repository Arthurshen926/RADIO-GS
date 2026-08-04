#!/usr/bin/env bash

# Frozen figurines sentinel for Field-C.  Every downstream component is held
# identical to the accepted Field-A comparison; only the field checkpoint is
# replaced after its label-free gate passes.

set -euo pipefail

REPO_ROOT=/root/RADIO-GS
RUN_PYTHON="$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
OUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260804/lerf_field_c_figurines_v2
FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260804/field_c/figurines_compact_d256_l128_field_c_exact_center_uncertainty_seed0.pth
FIELD_SHA=7b8c1f3feddeb6f18af756fa5cf00816af3042e574355d610264245a6cdd52c6
MPR_SHA=83c7d7db89fdea3f864200c145693c5e401d7d2506d28f70e1db44dff6e7bf28
GRAPH=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260716/global_3d_readout/figurines_support_graph_v3.pt
GRAPH_SHA=abcdd466fbbd726f277b59b137a59ac93b0a2c270a7557fc9916a478a66a1451
READOUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260717/surface_region_contract_v2/readout_v2_clean_h256.pth
READOUT_SHA=5b2d123a7827d9ab79aa4aa5a70077f00a656beebcf4c95ea5a3c9efdbe13ccb
TEXT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260715/text/siglip2_lerf_figurines_officialcanonical_prompt5.pt
TEXT_SHA=08fa6f870c824fde212b302d6c88cf74487f1122055fac709708399fb480578b
RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
RENDERER_SHA=6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2
RADIO=/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
RADIO_SHA=bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9
CONFIG="$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"
DESCRIPTOR="$OUT/descriptors/figurines_field_c.pt"
SCORE_CACHE="$OUT/query_scores/figurines_field_c.pt"
FIXED_RESULT="$OUT/fixed/figurines/figurines/lerf_direct_3d_selection_results.json"
OTSU_RESULT="$OUT/otsu3/figurines/figurines/lerf_direct_3d_selection_results.json"
SCORE_RESULT="$OUT/score_quality/figurines/figurines/lerf_direct_3d_selection_results.json"

guard() {
  local tag="$1"
  shift
  GPU=1 \
  GPU_TELEMETRY_LOG="$OUT/logs/${tag}.telemetry.csv" \
  GPU_OWNER_AUDIT_LOG="$OUT/logs/${tag}.owner_audit.csv" \
  GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
  GPU_POLL_SECONDS=120 \
  GPU_START_MAX_TEMP_C=83 \
  GPU_MAX_TEMP_C=87 \
  GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS=2 \
  GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=2 \
  GPU_SOFT_PAUSE_TEMP_C=0 \
  GPU_SOFT_RESUME_TEMP_C=0 \
  GPU_MAX_POWER_LIMIT_W=300.5 \
  CUDA_VISIBLE_DEVICES=1 \
    bash "$GUARD" -- "$@"
}

mkdir -p "$OUT/descriptors" "$OUT/query_scores" "$OUT/logs"
for path in "$DESCRIPTOR" "$DESCRIPTOR.json" "$DESCRIPTOR.provenance.json" \
  "$OUT/descriptors/figurines_field_c_query.pt" "$SCORE_CACHE" \
  "$SCORE_CACHE.json" "$FIXED_RESULT" "$OTSU_RESULT" "$SCORE_RESULT"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing to overwrite Field-C LERF artifact: $path" >&2
    exit 2
  fi
done

guard descriptor \
  bash "$RUN_PYTHON" \
    "$REPO_ROOT/radio_gs/scripts/build_surface_region_semantic_cache.py" \
    --field-checkpoint "$FIELD" \
    --field-checkpoint-sha256 "$FIELD_SHA" \
    --canonical-radio-source field_decode \
    --support-graph "$GRAPH" \
    --support-graph-sha256 "$GRAPH_SHA" \
    --readout-checkpoint "$READOUT" \
    --readout-checkpoint-sha256 "$READOUT_SHA" \
    --mpr-cache-sha256 "$MPR_SHA" \
    --radio-checkpoint "$RADIO" \
    --radio-checkpoint-sha256 "$RADIO_SHA" \
    --output "$DESCRIPTOR" \
    --query-output "$OUT/descriptors/figurines_field_c_query.pt" \
    --resume-dir "$OUT/resume_figurines" \
    --radio-batch-size 4096 \
    --semantic-batch-size 256 \
    --thermal-pacing-seconds-per-batch 0.05 \
    --device cuda:0 \
    >"$OUT/logs/descriptor.stdout.log" 2>&1

DESCRIPTOR_SHA="$(sha256sum "$DESCRIPTOR" | awk '{print $1}')"
bash "$RUN_PYTHON" \
  "$REPO_ROOT/radio_gs/scripts/materialize_lerf_multiscale_query_score_cache.py" \
  --descriptor-cache "$DESCRIPTOR" \
  --descriptor-cache-sha256 "$DESCRIPTOR_SHA" \
  --text-query-cache "$TEXT" \
  --text-query-cache-sha256 "$TEXT_SHA" \
  --field-checkpoint "$FIELD" \
  --field-checkpoint-sha256 "$FIELD_SHA" \
  --readout-checkpoint "$READOUT" \
  --readout-checkpoint-sha256 "$READOUT_SHA" \
  --renderer-geometry-checkpoint "$RENDERER" \
  --renderer-geometry-checkpoint-sha256 "$RENDERER_SHA" \
  --output "$SCORE_CACHE" \
  --chunk-size 4096 \
  >"$OUT/logs/query_scores.stdout.log" 2>&1

guard fixed \
  bash "$RUN_PYTHON" \
    "$REPO_ROOT/radio_gs/scripts/eval_lerf_direct_3d_selection.py" \
    --config "$CONFIG" \
    --checkpoint "$RENDERER" \
    --scene figurines \
    --protocol_preset vala_repo_3d \
    --score_threshold 0.6 \
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
    --output_dir "$OUT/fixed/figurines" \
    --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
    --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
    --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
    --ours_multiscale_query_score_cache "$SCORE_CACHE" \
    --gpu 0 \
    >"$OUT/logs/fixed.stdout.log" 2>&1

guard otsu3 \
  bash "$RUN_PYTHON" \
    "$REPO_ROOT/radio_gs/scripts/eval_lerf_adaptive_support_diagnostic.py" \
    --config "$CONFIG" \
    --checkpoint "$RENDERER" \
    --scene figurines \
    --ours_multiscale_query_score_cache "$SCORE_CACHE" \
    --output_dir "$OUT/otsu3/figurines" \
    --calibration_mode recursive_upper_otsu3 \
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
    --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
    --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
    --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
    --gpu 0 \
    >"$OUT/logs/otsu3.stdout.log" 2>&1

guard score_quality \
  bash "$RUN_PYTHON" \
    "$REPO_ROOT/radio_gs/scripts/eval_lerf_score_quality_diagnostic.py" \
    --config "$CONFIG" \
    --checkpoint "$RENDERER" \
    --scene figurines \
    --ours_multiscale_query_score_cache "$SCORE_CACHE" \
    --frozen_formal_result "$FIXED_RESULT" \
    --target_blind_adaptive_result "$OTSU_RESULT" \
    --output_dir "$OUT/score_quality/figurines" \
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
    --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
    --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
    --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
    --gpu 0 \
    >"$OUT/logs/score_quality.stdout.log" 2>&1
