#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash tmp/run_direct3d_lowmem_global_selector_sweep_20260516.sh <gpu> [scene ...]" >&2
  exit 1
fi

GPU="$1"
shift 1
ROOT="/root/RADIO-GS"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="output/radio_gs/lerf_direct3d_lowmem_global_selector_${STAMP}"
LOG_DIR="output/radio_gs/logs/direct3d_lowmem_global_selector_${STAMP}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

THRESHOLDS="0.05,0.07,0.09,0.11,0.13,0.15,0.18,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55"
PROMPTENS_TEMPLATES='{query}|a photo of a {query}|a close-up photo of the {query}|a 3d scan of a {query}|an object called {query}'
DEFAULT_ALL_TEMPLATES='{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}'

scene_config() {
  case "$1" in
    figurines) echo "radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml" ;;
    ramen) echo "radio_gs/configs/lerf_hybrid_v14_ramen_fdh_ws240_240ep.yaml" ;;
    teatime) echo "radio_gs/configs/lerf_hybrid_v14_teatime_fdh_ws240_240ep.yaml" ;;
    waldo_kitchen) echo "radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep.yaml" ;;
    *) echo "unknown scene: $1" >&2; return 1 ;;
  esac
}

scene_checkpoint() {
  case "$1" in
    figurines) echo "output/radio_gs/vpr_field_consistency_20260515/figurines/checkpoints/best.pth" ;;
    ramen) echo "output/radio_gs/vpr_field_consistency_weighted_log_20260515/ramen/checkpoints/best.pth" ;;
    teatime) echo "output/radio_gs/vpr_field_journal_rank_clipped_20260515/teatime/checkpoints/best.pth" ;;
    waldo_kitchen) echo "output/radio_gs/vpr_field_consistency_weighted_log_20260515/waldo_kitchen/checkpoints/best.pth" ;;
    *) echo "unknown scene: $1" >&2; return 1 ;;
  esac
}

scene_text_cache() {
  case "$1" in
    figurines) echo "checkpoints/siglip2_lerf_text_embeddings_promptens_20260515_figurines.pt" ;;
    ramen) echo "checkpoints/siglip2_lerf_text_embeddings_query_20260515_ramen.pt" ;;
    teatime) echo "checkpoints/siglip2_lerf_text_embeddings_promptens_default_all_20260515.pt" ;;
    waldo_kitchen) echo "checkpoints/siglip2_lerf_text_embeddings_promptens_20260515_waldo_kitchen.pt" ;;
    *) echo "unknown scene: $1" >&2; return 1 ;;
  esac
}

scene_score_cache() {
  case "$1" in
    figurines) echo "output/radio_gs/vpr_field_consistency_promptens_score_cache_20260515/figurines_promptens_scene_a05.pt" ;;
    ramen) echo "output/radio_gs/vpr_field_weighted_score_cache_20260516/ramen_query_a05_sil055.pt" ;;
    teatime) echo "output/radio_gs/vpr_field_journal_rank_clipped_promptens_score_cache_20260515/teatime_promptens_a05.pt" ;;
    waldo_kitchen) echo "output/radio_gs/vpr_field_consistency_weighted_promptens_score_cache_20260515/waldo_kitchen_promptens_scene_a05.pt" ;;
    *) echo "unknown scene: $1" >&2; return 1 ;;
  esac
}

scene_prompt_templates() {
  case "$1" in
    ramen) echo "{query}" ;;
    figurines) echo "$PROMPTENS_TEMPLATES" ;;
    teatime|waldo_kitchen) echo "$DEFAULT_ALL_TEMPLATES" ;;
    *) echo "unknown scene: $1" >&2; return 1 ;;
  esac
}

scene_min_ratio() {
  case "$1" in
    ramen) echo "0.005" ;;
    *) echo "0.003" ;;
  esac
}

scene_max_ratio() {
  case "$1" in
    ramen) echo "0.018" ;;
    waldo_kitchen) echo "0.07" ;;
    *) echo "0.06" ;;
  esac
}

scene_silhouette() {
  case "$1" in
    ramen) echo "0.6" ;;
    *) echo "0.55" ;;
  esac
}

run_scene() {
  local scene="$1"
  local log_path="$LOG_DIR/${scene}_gpu${GPU}.log"
  echo "[$(date '+%F %T')] start lowmem scene=${scene} gpu=${GPU}" | tee -a "$LOG_DIR/queue.log"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.scripts.eval_lerf_direct_3d_selection \
    --config "$(scene_config "$scene")" \
    --checkpoint "$(scene_checkpoint "$scene")" \
    --scene "$scene" \
    --output_dir "$OUT_ROOT" \
    --text_embedding_cache "$(scene_text_cache "$scene")" \
    --score_cache "$(scene_score_cache "$scene")" \
    --prompt_templates "$(scene_prompt_templates "$scene")" \
    --score_source direct \
    --use_point_summary_adapter \
    --point_summary_adapter_blend_alpha 0.5 \
    --scoring softmax_scene \
    --softmax_temperature 50 \
    --selection_mode score_threshold \
    --score_threshold 0.25 \
    --threshold_sweep "$THRESHOLDS" \
    --selection_min_ratio "$(scene_min_ratio "$scene")" \
    --selection_max_ratio "$(scene_max_ratio "$scene")" \
    --silhouette_threshold "$(scene_silhouette "$scene")" \
    --mask_refinement largest_component_rgb_grabcut \
    --chunk_size 2048 \
    --gpu 0 2>&1 | tee "$log_path"
}

if [[ $# -gt 0 ]]; then
  SCENES=("$@")
else
  SCENES=(figurines ramen teatime waldo_kitchen)
fi

for scene in "${SCENES[@]}"; do
  run_scene "$scene"
done

echo "[$(date '+%F %T')] complete lowmem gpu=${GPU} out=${OUT_ROOT}" | tee -a "$LOG_DIR/queue.log"
