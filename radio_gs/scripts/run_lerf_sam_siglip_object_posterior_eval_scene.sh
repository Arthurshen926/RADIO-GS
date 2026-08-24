#!/usr/bin/env bash
set -euo pipefail

SCENE=${1:?usage: run_lerf_sam_siglip_object_posterior_eval_scene.sh SCENE}
ROOT=/root/RADIO-GS
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf_sam_siglip_object_posterior_source32_v1}
SCORE_ROOT=${SCORE_ROOT:-$RUN_ROOT}
SCORE_DIR=${SCORE_DIR:-$SCORE_ROOT/scores}
PYTHON_RUNNER=${PYTHON_RUNNER:-$ROOT/radio_gs/scripts/run_repo_python.sh}
SCORE_THRESHOLD=${SCORE_THRESHOLD:-0.6}

case "$SCENE" in
  figurines)
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_figurines_radio_verified_pose.yaml
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
    ;;
  ramen)
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_ramen_radio_verified_pose.yaml
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth
    ;;
  teatime)
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_teatime_radio_verified_pose.yaml
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth
    ;;
  waldo_kitchen)
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_waldo_kitchen_radio_verified_pose.yaml
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth
    ;;
  *)
    echo "unsupported LERF scene: $SCENE" >&2
    exit 2
    ;;
esac

SCORES=$SCORE_DIR/$SCENE.pt
LERF3D_ROOT=$RUN_ROOT/lerf3d/$SCENE
LERF3D_RESULT=$LERF3D_ROOT/$SCENE/lerf_direct_3d_selection_results.json
LERF2D_ROOT=$RUN_ROOT/lerf2d/${SCENE}_eval
LERF2D_RESULT=$LERF2D_ROOT/lerf_ovs_results.json
mkdir -p "$RUN_ROOT/logs" "$LERF3D_ROOT" "$LERF2D_ROOT"
test -f "$SCORES"

if [[ ! -f "$LERF3D_RESULT" ]]; then
  bash "$PYTHON_RUNNER" radio_gs/scripts/eval_lerf_direct_3d_selection.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --scene "$SCENE" \
    --protocol_preset none \
    --external_query_score_cache "$SCORES" \
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
    --output_dir "$LERF3D_ROOT" \
    --summary_head_weights checkpoints/siglip2_summary_head.pth \
    --text_embedding_cache checkpoints/siglip2_lerf_all_exact_official.pt \
    --canonical_embedding_cache checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt \
    --prompt_templates '{query}' \
    --selection_mode score_threshold \
    --score_threshold "$SCORE_THRESHOLD" \
    --score_source direct \
    --scoring relevancy \
    --softmax_temperature 10 \
    --score_postprocess none \
    --projection_mode selected_only_alpha \
    --silhouette_threshold 0.0392156862745098 \
    --alpha_binarization png_uint8_gt10 \
    --mask_refinement peak_component_retention_guard \
    --component_guard_min_largest_fraction 0.65 \
    --min_select 0 \
    --gpu 0 \
    >"$RUN_ROOT/logs/${SCENE}_lerf3d.log" 2>&1
fi

if [[ ! -f "$LERF2D_RESULT" ]]; then
  bash "$PYTHON_RUNNER" radio_gs/scripts/eval_lerf_grounding.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --rendered_only \
    --render_readout primitive_posterior \
    --primitive_score_cache "$SCORES" \
    --primitive_query_cache "/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/$SCENE/primitive_query_method_v1.pth" \
    --primitive_valid_normalization \
    --scene "$SCENE" \
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
    --output_dir "$LERF2D_ROOT" \
    --text_embedding_cache checkpoints/siglip2_lerf_all_exact_official.pt \
    --prompt_templates '{query}' \
    --iou_threshold "$SCORE_THRESHOLD" \
    --threshold_mode fixed \
    --scoring relevancy \
    --relevancy_temp 0.1 \
    --canonical_embedding_cache checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt \
    --heatmap_upsample 4 \
    --eval_at_image_resolution \
    --localization_mode bbox_smoothed_peak \
    --localization_smoothing_kernel 30 \
    --mask_refinement none \
    --gpu 0 \
    >"$RUN_ROOT/logs/${SCENE}_lerf2d.log" 2>&1
fi

sha256sum "$SCORES" "$LERF3D_RESULT" "$LERF2D_RESULT"
