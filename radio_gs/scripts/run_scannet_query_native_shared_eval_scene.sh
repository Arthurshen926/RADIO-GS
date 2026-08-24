#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SCENE GPU SHARED_RUN" >&2
  exit 2
fi
SCENE=$1
GPU=$2
SHARED_RUN=$3
REPO=${REPO:-/root/RADIO-GS}
ROOT=/mnt/pool/sqy/results/RADIO-GS/output
CORE=$ROOT/optimization_20260815/core_method_v1/$SCENE
CACHE=$SHARED_RUN/$SCENE/source_distilled_scores.pt
RUN=$SHARED_RUN/$SCENE/eval
RESULT=$RUN/scannet_vala_gaussian_protocol_results.json
case "$SCENE" in
  scene0070_00) CHECKPOINT=$ROOT/radio_gs/scannet_og_scene0070_00_v14_b1/checkpoints/best.pth ;;
  scene0400_00) CHECKPOINT=$ROOT/radio_gs/scannet_og_scene0400_00_v67_dino_cv001_b2_s32768_ft20/checkpoints/best.pth ;;
  *) CHECKPOINT=$ROOT/radio_gs/scannet_og_${SCENE}_v14/checkpoints/best.pth ;;
esac
CONFIG=$REPO/radio_gs/configs/generated/frozen_eval_20260802/scannet_${SCENE%_00}_canonical_mpr_v3_paper8.yaml
LABEL=/mnt/pool/sqy/3d_understanding/scannet_og/$SCENE/${SCENE}_vh_clean_2.labels.ply
for value in "$CACHE" "$CHECKPOINT" "$CONFIG" "$LABEL"; do
  [[ -s "$value" ]] || { echo "incomplete input: $value" >&2; exit 1; }
done
if [[ ! -s "$RESULT" ]]; then
  mkdir -p "$RUN"
  CUDA_VISIBLE_DEVICES=$GPU bash "$REPO/radio_gs/scripts/run_repo_python.sh" \
    "$REPO/radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py" \
    --scene_list "$SCENE" --prepared_root /mnt/pool/sqy/3d_understanding/scannet_og \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --label_ply "$LABEL" \
    --output_dir "$RUN" --class_splits 19,15,10 \
    --feature_chunk_size 8192 --pseudo_chunk_size 512 \
    --pseudo_gt_cache_dir "$CORE/scannet_vala_method_v1/pseudo_gt" \
    --radius_factor 5.0 --candidate_k 1000 --fallback_k 1 \
    --row_opacity_threshold 0.1 --compact_feature_key features \
    --prompt_templates '{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}' \
    --text_embedding_cache "$REPO/checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt" \
    --text_encoder siglip2 --class_aliases none \
    --projection_weights "$REPO/checkpoints/siglip2_feat_projection.pth" \
    --summary_head_weights "$REPO/checkpoints/siglip2_summary_head.pth" \
    --external_query_score_cache "$CACHE" \
    --expected_external_query_score_cache_sha256 "$(sha256sum "$CACHE" | awk '{print $1}')" --device cuda
fi
chmod -R o+rwX "$SHARED_RUN/$SCENE"
