#!/usr/bin/env bash
set -euo pipefail

# Evaluate the fixed native SAM3+SigLIP2 region residual on top of the already
# promoted deployable direct-capability decoder, rather than against the older
# primitive baseline.  Alpha=0.25 is frozen from the four-scene source-method
# confirmation before the remaining paper8 scenes are opened.

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SCENE" >&2
  exit 2
fi

SCENE=$1
ROOT=${ROOT:-/root/RADIO-GS}
GPU=${GPU:-2}
CORE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/$SCENE
NATIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/$SCENE/native_sam_siglip
RESTORED=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260823/scannet_semantic_ladder/restored_direct_capability/$SCENE/primitive_query_restored_direct_capability.pt
RUN_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/$SCENE/native_sam_siglip_restored_baseline
MEMBERSHIP=$NATIVE/native_sam3_multiscale_memberships.pt
TEACHER=$NATIVE/native_siglip2_sam_crop_teacher.pt
SCORE=$RUN_ROOT/restored_score_cache/development/${SCENE}_scores.npz
RESULT=$RUN_ROOT/native_sam_siglip_region_vote_on_restored.json

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

if [[ ! -s "$MEMBERSHIP" || ! -s "$TEACHER" ]]; then
  echo "native source teacher is incomplete for $SCENE" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
# Multiple GPU queues may discover the same newly completed native teacher at
# nearly the same time.  Serialize this scene's cache/result materialization so
# only one writer can enter; the follower rechecks the immutable outputs below
# and becomes a no-op.
exec 9>"$RUN_ROOT/.materialize.lock"
flock 9

if [[ ! -s "$SCORE" ]]; then
  RESTORED_SHA=$(sha256sum "$RESTORED" | awk '{print $1}')
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$ROOT/radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py" \
      --scene_list "$SCENE" \
      --prepared_root /mnt/pool/sqy/3d_understanding/scannet_og \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --label_ply "$LABEL" \
      --output_dir "$RUN_ROOT/restored_score_cache" \
      --class_splits 19,15,10 --feature_chunk_size 8192 --pseudo_chunk_size 512 \
      --pseudo_gt_cache_dir "$CORE/scannet_vala_method_v1/pseudo_gt" \
      --radius_factor 5.0 --candidate_k 1000 --fallback_k 1 \
      --row_opacity_threshold 0.1 --compact_feature_key features \
      --prompt_templates '{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}' \
      --text_embedding_cache "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt" \
      --text_encoder siglip2 --class_aliases none \
      --projection_weights "$ROOT/checkpoints/siglip2_feat_projection.pth" \
      --summary_head_weights "$ROOT/checkpoints/siglip2_summary_head.pth" \
      --external_query_feature_cache "$RESTORED" \
      --expected_external_query_feature_cache_sha256 "$RESTORED_SHA" \
      --save_development_score_cache --score_cache_only \
      --allow_topology_free_score_cache --device cuda \
      >"$RUN_ROOT/restored_score_cache.log" 2>&1
fi

if [[ ! -s "$RESULT" ]]; then
  bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
    -m radio_gs.scripts.evaluate_scannet_native_sam_siglip_region_vote \
    --scene "$SCENE" --membership "$MEMBERSHIP" --proposal-teacher "$TEACHER" \
    --score-cache "$SCORE" \
    --text-cache-19 "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_exact_split19.pt" \
    --text-cache-15 "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_exact_split15.pt" \
    --text-cache-10 "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_exact_split10.pt" \
    --minimum-views 2 --minimum-view-agreement 0.5 --blend-weights 0.25 \
    --output "$RESULT" >"$RUN_ROOT/native_region_on_restored.log" 2>&1
fi

chmod -R o+rwX "$RUN_ROOT"
