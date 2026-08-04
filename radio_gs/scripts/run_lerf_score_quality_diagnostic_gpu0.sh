#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/root/RADIO-GS
OUT="$REPO_ROOT/output/optimization_20260803/lerf_text_audit/score_quality_v1"
FORMAL_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260802/lerf_direct3d_ours_native_multiscale_v2
ADAPTIVE_ROOT="$REPO_ROOT/output/optimization_20260803/lerf_text_audit/adaptive_support/recursive_upper_otsu3"

run_scene() {
  local scene="$1"
  local config="$2"
  local checkpoint="$3"
  local score_cache="$4"
  local scene_out="$OUT/$scene"
  mkdir -p "$scene_out" "$OUT/logs"
  GPU=0 \
  GPU_TELEMETRY_LOG="$OUT/logs/${scene}.telemetry.csv" \
  GPU_OWNER_AUDIT_LOG="$OUT/logs/${scene}.owner_audit.csv" \
  GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
  GPU_POLL_SECONDS=20 \
  GPU_START_MAX_TEMP_C=78 \
  GPU_SOFT_PAUSE_TEMP_C=81 \
  GPU_SOFT_RESUME_TEMP_C=76 \
  GPU_MAX_TEMP_C=84 \
  CUDA_VISIBLE_DEVICES=0 \
    bash "$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh" -- \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$REPO_ROOT/radio_gs/scripts/eval_lerf_score_quality_diagnostic.py" \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --scene "$scene" \
        --ours_multiscale_query_score_cache "$score_cache" \
        --frozen_formal_result "$FORMAL_ROOT/$scene/$scene/lerf_direct_3d_selection_results.json" \
        --target_blind_adaptive_result "$ADAPTIVE_ROOT/$scene/$scene/lerf_direct_3d_selection_results.json" \
        --output_dir "$scene_out" \
        --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
        --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
        --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
        --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
        --gpu 0
}

for scene in figurines waldo_kitchen ramen teatime; do
  case "$scene" in
    figurines)
      run_scene "$scene" \
        "$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml" \
        /mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth \
        /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260802/lerf_native_multiscale_query_scores_v2/figurines.pt
      ;;
    waldo_kitchen)
      run_scene "$scene" \
        "$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_waldo_kitchen_radio_verified_pose.yaml" \
        /mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth \
        /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260802/lerf_native_multiscale_query_scores_v2/waldo_kitchen.pt
      ;;
    ramen)
      run_scene "$scene" \
        "$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_ramen_radio_verified_pose.yaml" \
        /mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth \
        /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260802/lerf_native_multiscale_query_scores_v2/ramen.pt
      ;;
    teatime)
      run_scene "$scene" \
        "$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_teatime_radio_verified_pose.yaml" \
        /mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth \
        /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260802/lerf_native_multiscale_query_scores_v2/teatime.pt
      ;;
  esac
done
