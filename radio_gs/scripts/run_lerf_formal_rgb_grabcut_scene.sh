#!/usr/bin/env bash
set -euo pipefail

SCENE=${1:?usage: run_lerf_formal_rgb_grabcut_scene.sh SCENE GPU}
GPU=${2:?usage: run_lerf_formal_rgb_grabcut_scene.sh SCENE GPU}
ROOT=/root/RADIO-GS
OUT_ROOT=${OUT_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260825/lerf_formal_improvement_v1}
SCORE_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf_sam_siglip_object_posterior_source32_v6_relevancy_identity_extent/scores

case "$SCENE" in
  figurines)
    CONFIG=lerf_figurines_radio_verified_pose.yaml
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth ;;
  ramen)
    CONFIG=lerf_ramen_radio_verified_pose.yaml
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth ;;
  teatime)
    CONFIG=lerf_teatime_radio_verified_pose.yaml
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth ;;
  waldo_kitchen)
    CONFIG=lerf_waldo_kitchen_radio_verified_pose.yaml
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth ;;
  *) echo "unsupported LERF scene: $SCENE" >&2; exit 2 ;;
esac

CUDA_VISIBLE_DEVICES=$GPU bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$ROOT/radio_gs/scripts/eval_lerf_grounding.py" \
  --config "$ROOT/radio_gs/configs/generated/frozen_eval_20260802/$CONFIG" \
  --checkpoint "$CHECKPOINT" --rendered_only --render_readout primitive_posterior \
  --primitive_score_cache "$SCORE_ROOT/$SCENE.pt" \
  --primitive_query_cache "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/$SCENE/primitive_query_method_v1.pth" \
  --primitive_valid_normalization --scene "$SCENE" \
  --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
  --output_dir "$OUT_ROOT/${SCENE}_rgb_grabcut" \
  --text_embedding_cache "$ROOT/checkpoints/siglip2_lerf_all_exact_official.pt" \
  --prompt_templates '{query}' --iou_threshold 0.6 --threshold_mode fixed \
  --scoring relevancy --relevancy_temp 0.1 \
  --canonical_embedding_cache "$ROOT/checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt" \
  --heatmap_upsample 4 --eval_at_image_resolution \
  --localization_mode bbox_smoothed_peak --localization_smoothing_kernel 30 \
  --mask_refinement rgb_grabcut --mask_refinement_iters 1 \
  --mask_refinement_dilate 5 --mask_refinement_erode 2 --gpu 0
