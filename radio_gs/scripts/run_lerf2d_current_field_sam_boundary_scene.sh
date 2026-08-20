#!/usr/bin/env bash

set -euo pipefail

ROOT=/root/RADIO-GS
SCENE=${SCENE:?set SCENE}
PHYSICAL_GPU=${PHYSICAL_GPU:?set PHYSICAL_GPU}
OUT_ROOT=${OUT_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf2d_primitive_peak_feature_sam_boundary_v1}
RENDER_READOUT=${RENDER_READOUT:-primitive_score}
PRIMITIVE_SCORE_CACHE=${PRIMITIVE_SCORE_CACHE:-}
IOU_THRESHOLD=${IOU_THRESHOLD:-0.5}
THRESHOLD_MODE=${THRESHOLD_MODE:-absolute}
PRIMITIVE_VALID_NORMALIZATION=${PRIMITIVE_VALID_NORMALIZATION:-0}
METHOD_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1
GEOMETRY_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs

case "$SCENE" in
  figurines)
    CONFIG="$ROOT/radio_gs/configs/generated/canonical_render/lerf_figurines_source_only_siglip2_spatial.yaml"
    CHECKPOINT="$GEOMETRY_ROOT/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth"
    PRIMITIVE="$METHOD_ROOT/figurines/primitive_query_method_v1.pth"
    PRIMITIVE_SHA=acc0b8b4cbf429d92e2f9df05865898066349fb79bcbe0bd3933ae1e504f1e18
    FEATURE_DIR="$METHOD_ROOT/figurines/rendered_benchmark_method_v1_lineage/figurines"
    HEAD="$ROOT/output/radio_gs/prompt_sam3_mask_head_20260523/figurines_trainviews_lerf2dcoarse_e60_cache/prompt_conditioned_sam3_mask_head.pth"
    ;;
  ramen)
    CONFIG="$ROOT/radio_gs/configs/generated/canonical_render/lerf_ramen_source_only_siglip2_spatial.yaml"
    CHECKPOINT="$GEOMETRY_ROOT/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth"
    PRIMITIVE="$METHOD_ROOT/ramen/primitive_query_method_v1.pth"
    PRIMITIVE_SHA=893fda2a90142f71ee8175e666f12353a93e08a8125d8d5bdaf26d3a95dc54b5
    FEATURE_DIR="$METHOD_ROOT/ramen/rendered_benchmark_method_v1/ramen"
    HEAD="$ROOT/output/radio_gs/prompt_sam3_mask_head_20260523/ramen_trainviews_lerf2dcoarse_e60_cache/prompt_conditioned_sam3_mask_head.pth"
    ;;
  teatime)
    CONFIG="$ROOT/radio_gs/configs/generated/canonical_render/lerf_teatime_source_only_siglip2_spatial.yaml"
    CHECKPOINT="$GEOMETRY_ROOT/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth"
    PRIMITIVE="$METHOD_ROOT/teatime/primitive_query_method_v1.pth"
    PRIMITIVE_SHA=3938c13cd5f2c78cc2522aeff26cb0f77ba08cbeb519288b4b564dffd629b96b
    FEATURE_DIR="$METHOD_ROOT/teatime/rendered_benchmark_method_v1/teatime"
    HEAD="$ROOT/output/radio_gs/prompt_sam3_mask_head_20260523/teatime_trainviews_lerf2dcoarse_e60_cache/prompt_conditioned_sam3_mask_head.pth"
    ;;
  waldo_kitchen)
    CONFIG="$ROOT/radio_gs/configs/generated/canonical_render/lerf_waldo_kitchen_source_only_siglip2_spatial.yaml"
    CHECKPOINT="$GEOMETRY_ROOT/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth"
    PRIMITIVE="$METHOD_ROOT/waldo_kitchen/primitive_query_method_v1.pth"
    PRIMITIVE_SHA=01ffe08e54466dc0da720bcc2e25029ae2b085e24e78f8ac5ad9ced28085159f
    FEATURE_DIR="$METHOD_ROOT/waldo_kitchen/rendered_benchmark_method_v1/waldo_kitchen"
    HEAD="$ROOT/output/radio_gs/prompt_sam3_mask_head_20260523/waldo_kitchen_trainviews_lerf2dcoarse_e60_cache/prompt_conditioned_sam3_mask_head.pth"
    ;;
  *)
    echo "unsupported scene: $SCENE" >&2
    exit 2
    ;;
esac

OUTPUT="$OUT_ROOT/$SCENE"
LOG="$OUTPUT/run.log"
RESULT="$OUTPUT/lerf_ovs_results.json"
if [[ -e "$RESULT" || -e "$LOG" ]]; then
  echo "refusing to overwrite an existing result or log for $SCENE" >&2
  exit 3
fi
for path in "$CONFIG" "$CHECKPOINT" "$PRIMITIVE" "$HEAD"; do
  [[ -r "$path" ]] || { echo "required input is absent: $path" >&2; exit 4; }
done
[[ -d "$FEATURE_DIR/backbone" ]] || { echo "current-field boundary features are absent: $FEATURE_DIR" >&2; exit 4; }
[[ "$(sha256sum "$PRIMITIVE" | cut -d' ' -f1)" == "$PRIMITIVE_SHA" ]] || {
  echo "primitive query cache SHA-256 differs" >&2
  exit 4
}

mkdir -p "$OUTPUT"
cd "$ROOT"
READOUT_ARGS=(--render_readout "$RENDER_READOUT")
if [[ "$RENDER_READOUT" == primitive_posterior ]]; then
  [[ -r "$PRIMITIVE_SCORE_CACHE" ]] || {
    echo "primitive posterior cache is absent: $PRIMITIVE_SCORE_CACHE" >&2
    exit 4
  }
  READOUT_ARGS+=(--primitive_score_cache "$PRIMITIVE_SCORE_CACHE")
  if [[ "$PRIMITIVE_VALID_NORMALIZATION" == 1 ]]; then
    READOUT_ARGS+=(--primitive_valid_normalization)
  fi
fi
CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
  bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.scripts.eval_lerf_grounding \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --rendered_only \
  "${READOUT_ARGS[@]}" \
  --primitive_query_cache "$PRIMITIVE" \
  --primitive_confidence none \
  --primitive_fallback_blend direct \
  --scene "$SCENE" \
  --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
  --output_dir "$OUTPUT" \
  --summary_head_weights checkpoints/siglip2_summary_head.pth \
  --use_summary_head \
  --text_embedding_cache checkpoints/siglip2_lerf_all_exact_official.pt \
  --canonical_embedding_cache checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt \
  --text_encoder siglip2 \
  --prompt_templates '{query}' \
  --protocol_preset none \
  --scoring relevancy \
  --relevancy_temp 0.1 \
  --threshold_mode "$THRESHOLD_MODE" \
  --iou_threshold "$IOU_THRESHOLD" \
  --heatmap_upsample 1 \
  --eval_at_image_resolution \
  --localization_mode bbox_smoothed_peak \
  --localization_smoothing_kernel 30 \
  --mask_refinement sam3_prompt_mask_head \
  --sam3_prompt_mask_head_checkpoint "$HEAD" \
  --sam3_prompt_mask_head_feature_dir "$FEATURE_DIR" \
  --sam3_prompt_mask_head_logit_threshold 0.0 \
  --sam3_prompt_mask_head_min_initial_iou 0.5 \
  --sam3_prompt_mask_head_max_initial_area_fraction 1.0 \
  --sam3_prompt_mask_head_min_refined_area_ratio 0.7 \
  --sam3_prompt_mask_head_max_refined_area_ratio 1.3 \
  --sam3_prompt_mask_head_support_dilate 12 \
  --sam3_prompt_mask_head_coarse_dilate 1 \
  --sam3_prompt_mask_head_coarse_threshold 0.5 \
  --sam3_prompt_mask_head_min_heatmap_mean_ratio 0.85 \
  --sam3_prompt_mask_head_min_heatmap_mass_ratio 0.25 \
  --sam3_prompt_mask_head_require_peak_in_refined \
  --sam3_prompt_mask_head_initial_refinement peak_component \
  --sam3_prompt_mask_head_apply_to rendered \
  --gpu 0 \
  >"$LOG" 2>&1
sha256sum "$RESULT"
