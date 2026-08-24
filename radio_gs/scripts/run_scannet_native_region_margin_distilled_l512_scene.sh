#!/usr/bin/env bash
set -euo pipefail

# Query-response/top-two-margin distillation of the fixed native region teacher
# into frozen L512.  The protocol class vocabulary is opened explicitly, while
# benchmark RGB, labels and masks remain unavailable during fitting.

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SCENE GPU" >&2
  exit 2
fi

SCENE=$1
GPU=$2
ROOT=${ROOT:-/root/RADIO-GS}
CORE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/$SCENE
BASE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260823/scannet_semantic_ladder/restored_direct_capability/$SCENE/primitive_query_restored_direct_capability.pt
FIELD=$CORE/generic_text_response_w005_s0_64.pth
TEACHER_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/$SCENE/native_region_distilled_l512_v1
TARGET=$TEACHER_ROOT/native_region_residual_target.pt
VARIANT=${VARIANT:-native_region_margin_distilled_l512_v3}
RUN=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/$SCENE/$VARIANT
QUERY=$RUN/primitive_query_region_margin_distilled.pt
RESULT=$RUN/eval/scannet_vala_gaussian_protocol_results.json
TEXT=${RESPONSE_TEXT:-$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_ens5_split19.pt}

case "$SCENE" in
  scene0070_00)
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/scannet_og_scene0070_00_v14_b1/checkpoints/best.pth
    ;;
  scene0400_00)
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/scannet_og_scene0400_00_v67_dino_cv001_b2_s32768_ft20/checkpoints/best.pth
    ;;
  *)
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/scannet_og_${SCENE}_v14/checkpoints/best.pth
    ;;
esac
CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/scannet_${SCENE%_00}_canonical_mpr_v3_paper8.yaml
LABEL=/mnt/pool/sqy/3d_understanding/scannet_og/$SCENE/${SCENE}_vh_clean_2.labels.ply

for required in "$TARGET" "$BASE" "$FIELD" "$TEXT" "$CHECKPOINT" "$CONFIG" "$LABEL"; do
  if [[ ! -s "$required" ]]; then
    echo "required asset is incomplete: $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN"
exec 9>"$RUN/.materialize.lock"
flock 9

TARGET_SHA=$(sha256sum "$TARGET" | awk '{print $1}')
BASE_SHA=$(sha256sum "$BASE" | awk '{print $1}')
FIELD_SHA=$(sha256sum "$FIELD" | awk '{print $1}')
TEXT_SHA=$(sha256sum "$TEXT" | awk '{print $1}')

if [[ ! -s "$QUERY" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.train_frozen_latent_direct_capability_decoder \
      --scene "$SCENE" --field "$FIELD" --expected-field-sha256 "$FIELD_SHA" \
      --direct-capability-target "$TARGET" \
      --expected-direct-capability-target-sha256 "$TARGET_SHA" \
      --baseline-query-cache "$BASE" \
      --expected-baseline-query-cache-sha256 "$BASE_SHA" \
      --response-embedding-cache "$TEXT" \
      --expected-response-embedding-cache-sha256 "$TEXT_SHA" \
      --response-weight 1 --margin-weight 1 --response-logit-scale 100 \
      --output-model "$RUN/decoder.pt" --output-query-cache "$QUERY" \
      --device cuda:0 --hidden-dim 512 --steps 400 --batch-size 4096 \
      --validation-interval 25 --maximum-p05-drop 0.002 \
      --teacher-order native_sam_extent_then_native_siglip_region_then_query_margin
fi

if [[ ! -s "$RESULT" ]]; then
  QUERY_SHA=$(sha256sum "$QUERY" | awk '{print $1}')
  CUDA_VISIBLE_DEVICES=$GPU \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$ROOT/radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py" \
      --scene_list "$SCENE" \
      --prepared_root /mnt/pool/sqy/3d_understanding/scannet_og \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --label_ply "$LABEL" \
      --output_dir "$RUN/eval" --class_splits 19,15,10 \
      --feature_chunk_size 8192 --pseudo_chunk_size 512 \
      --pseudo_gt_cache_dir "$CORE/scannet_vala_method_v1/pseudo_gt" \
      --radius_factor 5.0 --candidate_k 1000 --fallback_k 1 \
      --row_opacity_threshold 0.1 --compact_feature_key features \
      --prompt_templates '{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}' \
      --text_embedding_cache "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt" \
      --text_encoder siglip2 --class_aliases none \
      --projection_weights "$ROOT/checkpoints/siglip2_feat_projection.pth" \
      --summary_head_weights "$ROOT/checkpoints/siglip2_summary_head.pth" \
      --external_query_feature_cache "$QUERY" \
      --expected_external_query_feature_cache_sha256 "$QUERY_SHA" \
      --device cuda
fi

chmod -R o+rwX "$RUN"
