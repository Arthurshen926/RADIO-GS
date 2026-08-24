#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/RADIO-GS}
ROOT=/mnt/pool/sqy/results/RADIO-GS/output
RUN=$ROOT/optimization_20260824/query_native_gaussian_memory/shared_coeff_canon8_factorized_queryholdout4_seed24_v3

sha() { sha256sum "$1" | awk '{print $1}'; }

evaluate_scene() {
  local scene=$1 gpu=$2
  local core=$ROOT/optimization_20260815/core_method_v1/$scene
  local cache=$RUN/$scene/source_distilled_scores.pt
  local output=$RUN/eval/$scene
  local result=$output/scannet_vala_gaussian_protocol_results.json
  local config=$REPO/radio_gs/configs/generated/frozen_eval_20260802/scannet_${scene%_00}_canonical_mpr_v3_paper8.yaml
  local label=/mnt/pool/sqy/3d_understanding/scannet_og/$scene/${scene}_vh_clean_2.labels.ply
  local checkpoint
  case "$scene" in
    scene0070_00) checkpoint=$ROOT/radio_gs/scannet_og_scene0070_00_v14_b1/checkpoints/best.pth ;;
    scene0400_00) checkpoint=$ROOT/radio_gs/scannet_og_scene0400_00_v67_dino_cv001_b2_s32768_ft20/checkpoints/best.pth ;;
    *) checkpoint=$ROOT/radio_gs/scannet_og_${scene}_v14/checkpoints/best.pth ;;
  esac
  if [[ -s "$result" ]]; then
    echo "[$scene] complete; skipping immutable result"
    return 0
  fi
  [[ -s "$cache" ]] || { echo "[$scene] missing passed shared cache" >&2; return 1; }
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" bash "$REPO/radio_gs/scripts/run_repo_python.sh" \
    "$REPO/radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py" \
    --scene_list "$scene" --prepared_root /mnt/pool/sqy/3d_understanding/scannet_og \
    --config "$config" --checkpoint "$checkpoint" --label_ply "$label" \
    --output_dir "$output" --class_splits 19,15,10 \
    --feature_chunk_size 8192 --pseudo_chunk_size 512 \
    --pseudo_gt_cache_dir "$core/scannet_vala_method_v1/pseudo_gt" \
    --radius_factor 5.0 --candidate_k 1000 --fallback_k 1 \
    --row_opacity_threshold 0.1 --compact_feature_key features \
    --prompt_templates '{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}' \
    --text_embedding_cache "$REPO/checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt" \
    --text_encoder siglip2 --class_aliases none \
    --projection_weights "$REPO/checkpoints/siglip2_feat_projection.pth" \
    --summary_head_weights "$REPO/checkpoints/siglip2_summary_head.pth" \
    --external_query_score_cache "$cache" \
    --expected_external_query_score_cache_sha256 "$(sha "$cache")" --device cuda \
    >"$RUN/eval_${scene}.log" 2>&1
}

(evaluate_scene scene0000_00 0; evaluate_scene scene0400_00 0) & p0=$!
(evaluate_scene scene0062_00 1; evaluate_scene scene0590_00 1) & p1=$!
evaluate_scene scene0070_00 2 & p2=$!
evaluate_scene scene0097_00 3 & p3=$!
evaluate_scene scene0140_00 4 & p4=$!
evaluate_scene scene0347_00 5 & p5=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do
  wait "$pid" || status=1
done
exit "$status"
