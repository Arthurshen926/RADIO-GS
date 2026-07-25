#!/usr/bin/env bash

# Evaluate the ScanNet-frozen specificity-preserving scale rule on all four
# LERF scenes.  The primitive unaries are compiled before this queue starts;
# this script only performs the unchanged frozen rendering/evaluator pass.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SOURCE_ROOT="${SOURCE_ROOT:-output/optimization_20260724/text_specificity_margin002}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/optimization_20260724/text_specificity_margin002}"
AFTER_MARKER="${AFTER_MARKER:-}"

wait_for_gpu() {
  local available=0
  while (( available < 2 )); do
    local values used util
    values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU")"
    used="${values%%,*}"; util="${values##*,}"
    used="${used// /}"; util="${util// /}"
    if (( used < 1200 && util < 10 )); then
      available=$((available + 1))
    else
      available=0
    fi
    if (( available < 2 )); then sleep 20; fi
  done
}

if [[ -n "$AFTER_MARKER" ]]; then
  while [[ ! -s "$AFTER_MARKER" ]]; do sleep 30; done
fi

declare -A CONFIGS=(
  [figurines]="radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"
  [ramen]="radio_gs/configs/generated/query_consistency/lerf_ramen_radio_verified_pose.yaml"
  [teatime]="radio_gs/configs/generated/query_consistency/lerf_teatime_radio_verified_pose.yaml"
  [waldo_kitchen]="radio_gs/configs/generated/query_consistency/lerf_waldo_kitchen_radio_verified_pose.yaml"
)
declare -A CHECKPOINTS=(
  [figurines]="output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth"
  [ramen]="output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth"
  [teatime]="output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep/checkpoints/best.pth"
  [waldo_kitchen]="output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth"
)

mkdir -p "$OUTPUT_ROOT/logs"
for scene in figurines ramen teatime waldo_kitchen; do
  unary="$SOURCE_ROOT/${scene}_unary_specificity002.pt"
  result="$OUTPUT_ROOT/${scene}_eval_specificity002/lerf_ovs_results.json"
  if [[ -s "$result" ]]; then
    continue
  fi
  for required in "$unary" "${CONFIGS[$scene]}" "${CHECKPOINTS[$scene]}"; do
    if [[ ! -s "$required" ]]; then
      echo "missing LERF specificity input: $required" >&2
      exit 2
    fi
  done
  wait_for_gpu
  CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_lerf_grounding.py \
    --config "${CONFIGS[$scene]}" \
    --checkpoint "${CHECKPOINTS[$scene]}" \
    --rendered_only \
    --render_readout primitive_unary \
    --primitive_score_cache "$unary" \
    --scene "$scene" \
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
    --output_dir "$OUTPUT_ROOT/${scene}_eval_specificity002" \
    --text_embedding_cache checkpoints/siglip2_lerf_text_embeddings_query_all_20260515.pt \
    --prompt_templates '{query}' \
    --iou_threshold 0.6 \
    --threshold_mode fixed \
    --scoring cosine \
    --heatmap_upsample 4 \
    --localization_mode polygon_argmax \
    --mask_refinement none \
    --gpu 0 \
    >"$OUTPUT_ROOT/logs/${scene}_specificity002.log" 2>&1
done

date -Iseconds >"$OUTPUT_ROOT/lerf_specificity002.complete"
