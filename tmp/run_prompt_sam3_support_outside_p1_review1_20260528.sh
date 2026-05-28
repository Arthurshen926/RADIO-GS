#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/RADIO-GS"
cd "${REPO_ROOT}"

GPU="${1:-5}"
VARIANT="${2:-trainviews_directcoarse_supportoutside_w025_d8_e60_20260528}"
shift 2 || true
SCENES=("$@")
if [[ ${#SCENES[@]} -eq 0 ]]; then
  SCENES=(ramen figurines)
fi

cfg_for_scene() {
  case "$1" in
    ramen)
      echo "tmp/lerf_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528/lerf_ramen_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528.yaml"
      ;;
    figurines)
      echo "tmp/lerf_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528/lerf_figurines_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528.yaml"
      ;;
    teatime)
      if [[ -f "tmp/lerf_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528/lerf_teatime_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528.yaml" ]]; then
        echo "tmp/lerf_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528/lerf_teatime_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528.yaml"
      else
        echo "tmp/lerf_v14_fdh_ws240_240ep/lerf_teatime_v14_fdh_ws240_240ep.yaml"
      fi
      ;;
    waldo_kitchen)
      if [[ -f "tmp/lerf_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528/lerf_waldo_kitchen_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528.yaml" ]]; then
        echo "tmp/lerf_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528/lerf_waldo_kitchen_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528.yaml"
      else
        echo "tmp/lerf_v14_fdh_ws240_240ep/lerf_waldo_kitchen_v14_fdh_ws240_240ep.yaml"
      fi
      ;;
    *)
      echo "unknown scene: $1" >&2
      return 2
      ;;
  esac
}

ckpt_for_scene() {
  case "$1" in
    ramen)
      echo "output/radio_gs/lerf_ramen_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528/checkpoints/best.pth"
      ;;
    figurines)
      echo "output/radio_gs/lerf_figurines_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528/checkpoints/best.pth"
      ;;
    teatime)
      if [[ -f "output/radio_gs/lerf_teatime_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528/checkpoints/best.pth" ]]; then
        echo "output/radio_gs/lerf_teatime_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528/checkpoints/best.pth"
      else
        echo "output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep/checkpoints/best.pth"
      fi
      ;;
    waldo_kitchen)
      if [[ -f "output/radio_gs/lerf_waldo_kitchen_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528/checkpoints/best.pth" ]]; then
        echo "output/radio_gs/lerf_waldo_kitchen_adjoint_vpr_nocache_querydistill_rendercons_p0_ft20_mf16_20260528/checkpoints/best.pth"
      else
        echo "output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth"
      fi
      ;;
    *)
      echo "unknown scene: $1" >&2
      return 2
      ;;
  esac
}

for SCENE in "${SCENES[@]}"; do
  CFG="$(cfg_for_scene "${SCENE}")"
  CKPT="$(ckpt_for_scene "${SCENE}")"
  HEAD_DIR="output/radio_gs/prompt_sam3_mask_head_20260528/${SCENE}_${VARIANT}"
  COARSE_DIR="output/radio_gs/prompt_sam3_trainview_coarse_20260523/${SCENE}"
  SAM3_CACHE="output/radio_gs/foundation_cache_sam3_modelscope_mapped_trainviews/${SCENE}"
  EVAL_DIR="output/radio_gs/lerf_${SCENE}_direct_prompt_sam3_${VARIANT}_eval"

  if [[ ! -f "${CFG}" ]]; then
    echo "missing config for ${SCENE}: ${CFG}" >&2
    exit 3
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "missing checkpoint for ${SCENE}: ${CKPT}" >&2
    exit 3
  fi

  echo "[p1-prompt-sam3-supportoutside] train gpu=${GPU} scene=${SCENE} variant=${VARIANT}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/train_prompt_conditioned_sam3_mask_head.py \
    --scene "${SCENE}" \
    --sam3_cache_root "${SAM3_CACHE}" \
    --text_embedding_cache checkpoints/siglip2_lerf_text_embeddings.pt \
    --output_dir "${HEAD_DIR}" \
    --source rendered \
    --config "${CFG}" \
    --checkpoint "${CKPT}" \
    --coarse_mask_dir "${COARSE_DIR}" \
    --device cuda \
    --train_size 240 320 \
    --epochs 60 \
    --lr 2e-4 \
    --dice_weight 0.5 \
    --boundary_weight 0.25 \
    --support_outside_weight 0.25 \
    --support_outside_dilate 8 \
    --coarse_dilate 3 \
    --coarse_threshold 0.5 \
    --target_activation binary \
    --target_threshold 0.5 \
    --cache_source_features \
    --feature_cache_dtype float16

  echo "[p1-prompt-sam3-supportoutside] eval gpu=${GPU} scene=${SCENE}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_lerf_direct_3d_selection.py \
    --config "${CFG}" \
    --checkpoint "${CKPT}" \
    --scene "${SCENE}" \
    --output_dir "${EVAL_DIR}" \
    --score_source direct \
    --scoring softmax_scene \
    --softmax_temperature 50 \
    --selection_mode score_threshold \
    --threshold_sweep 0.15,0.25,0.35,0.45,0.55 \
    --selection_min_ratio 0.005 \
    --use_point_summary_adapter \
    --strict_direct_head_consistency \
    --point_summary_adapter_blend_alpha 1.0 \
    --point_summary_adapter_valid_mask_mode teacher_cache \
    --direct_primitive_confidence_mode teacher_cache_valid \
    --direct_primitive_confidence_blend 1.0 \
    --direct_primitive_opacity_threshold 0.02 \
    --silhouette_threshold 0.6 \
    --mask_refinement sam3_prompt_mask_head \
    --sam3_prompt_mask_head_checkpoint "${HEAD_DIR}/prompt_conditioned_sam3_mask_head.pth" \
    --sam3_prompt_mask_head_text_embedding_cache checkpoints/siglip2_lerf_text_embeddings.pt \
    --sam3_prompt_mask_head_logit_threshold -1.0 \
    --sam3_prompt_mask_head_min_initial_iou 0.05 \
    --sam3_prompt_mask_head_max_initial_area_fraction 0.85 \
    --sam3_prompt_mask_head_min_refined_area_ratio 0.25 \
    --sam3_prompt_mask_head_max_refined_area_ratio 1.8 \
    --sam3_prompt_mask_head_support_dilate 12 \
    --sam3_prompt_mask_head_coarse_dilate 1 \
    --sam3_prompt_mask_head_initial_refinement peak_component \
    --sam3_prompt_mask_head_require_peak_in_refined \
    --sam3_prompt_mask_head_min_heatmap_mean_ratio 0.85 \
    --sam3_prompt_mask_head_min_heatmap_mass_ratio 0.25 \
    --sam3_refinement_geometry_gate \
    --sam3_refinement_gate_min_area_ratio 0.25 \
    --sam3_refinement_gate_max_area_ratio 1.8 \
    --sam3_refinement_gate_min_boundary_gain 0.0
done
