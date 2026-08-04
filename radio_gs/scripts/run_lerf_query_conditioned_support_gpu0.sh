#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/root/RADIO-GS
OUT="$REPO_ROOT/output/optimization_20260803/lerf_text_audit/query_conditioned_support_v1"
SCENES="${SCENES:-figurines}"

run_scene() {
  local scene="$1"
  local config checkpoint score_cache graph scene_out result
  case "$scene" in
    figurines)
      config="$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"
      checkpoint=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
      ;;
    waldo_kitchen)
      config="$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_waldo_kitchen_radio_verified_pose.yaml"
      checkpoint=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth
      ;;
    ramen)
      config="$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_ramen_radio_verified_pose.yaml"
      checkpoint=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth
      ;;
    teatime)
      config="$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_teatime_radio_verified_pose.yaml"
      checkpoint=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth
      ;;
    *)
      echo "unsupported LERF scene: $scene" >&2
      return 2
      ;;
  esac
  score_cache=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260802/lerf_native_multiscale_query_scores_v2/${scene}.pt
  graph="$REPO_ROOT/output/optimization_20260716/global_3d_readout/${scene}_support_graph_v3.pt"
  scene_out="$OUT/$scene"
  result="$scene_out/$scene/lerf_direct_3d_selection_results.json"
  if [[ -f "$result" ]]; then
    echo "skip completed $scene: $result"
    return 0
  fi
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
        "$REPO_ROOT/radio_gs/scripts/eval_lerf_query_conditioned_support.py" \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --scene "$scene" \
        --ours_multiscale_query_score_cache "$score_cache" \
        --support_graph "$graph" \
        --output_dir "$scene_out" \
        --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
        --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
        --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
        --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
        --gpu 0
}

for scene in $SCENES; do
  run_scene "$scene"
done
