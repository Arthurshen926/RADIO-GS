#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/RADIO-GS"
cd "${REPO_ROOT}"

GPU="${1:-4}"
SCENE="${2:-figurines}"
CFG="${3:?config path required}"
CHECKPOINT="${4:?checkpoint path required}"
OUT_ROOT="${5:-output/radio_gs/lerf_${SCENE}_direct_prompt_sam3_heatguard_review1_20260527}"
HEAD_VARIANT="${6:-trainviews_directcoarse_e60_cache}"

head_dir="output/radio_gs/prompt_sam3_mask_head_20260523/${SCENE}_${HEAD_VARIANT}"
head_ckpt="${head_dir}/prompt_conditioned_sam3_mask_head.pth"
if [[ ! -f "${head_ckpt}" ]]; then
  echo "missing prompt-conditioned SAM3 mask-head checkpoint: ${head_ckpt}" >&2
  exit 2
fi

echo "[direct-prompt-sam3-head-eval] gpu=${GPU} scene=${SCENE} cfg=${CFG} checkpoint=${CHECKPOINT}"
CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/eval_lerf_direct_3d_selection.py \
  --config "${CFG}" \
  --checkpoint "${CHECKPOINT}" \
  --scene "${SCENE}" \
  --output_dir "${OUT_ROOT}" \
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
  --sam3_prompt_mask_head_checkpoint "${head_ckpt}" \
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
