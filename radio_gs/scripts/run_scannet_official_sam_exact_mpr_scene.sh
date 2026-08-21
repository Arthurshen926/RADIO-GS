#!/usr/bin/env bash
set -euo pipefail

SCENE=${1:?usage: run_scannet_official_sam_exact_mpr_scene.sh SCENE GPU}
PHYSICAL_GPU=${2:?usage: run_scannet_official_sam_exact_mpr_scene.sh SCENE GPU}
ROOT=/root/RADIO-GS
CORE_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/scannet_official_sam_instance_v1}
EVAL_ROOT=${EVAL_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/scannet_official_sam_exact_mpr_full8_v1}
SAM_CHECKPOINT=$ROOT/checkpoints/sam3_modelscope/sam3.pt
SAM_SHA=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
SAM_SEED_MARGIN=${SAM_SEED_MARGIN:-0.04}
SAM_UPDATE_MARGIN=${SAM_UPDATE_MARGIN:-0.01}
SAM_SEMANTIC_TOLERANCE=${SAM_SEMANTIC_TOLERANCE:-0.001}
SAM_CONSENSUS=${SAM_CONSENSUS:-0.80}
SAM_MIN_PROPOSALS=${SAM_MIN_PROPOSALS:-4}
SAM_MIN_VIEWS=${SAM_MIN_VIEWS:-3}
SAM_ITERATIONS=${SAM_ITERATIONS:-1}
EXTRA_EVAL_ARGS=()
if [[ "${SAVE_DEVELOPMENT_SCORE_CACHE:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--save_development_score_cache)
fi
if [[ "${SCORE_CACHE_ONLY:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--score_cache_only)
fi

case "$SCENE" in
  scene0000_00) SHORT=0000; QUERY_SHA=bd8d0f9e448ab2954edef02db2a6240eea85c98f984d6388139b201c3ba99c52; CHECKPOINT_TAG=scene0000_00_v14 ;;
  scene0062_00) SHORT=0062; QUERY_SHA=9226255874f0a1a82ef69909c01533dbe89d1af9ed8ef336efb12219310f1abd; CHECKPOINT_TAG=scene0062_00_v14 ;;
  scene0070_00) SHORT=0070; QUERY_SHA=ec0c0942804723f89e91a48a0d7d48665d0c9d40bd7cf397aca3c82b1938b177; CHECKPOINT_TAG=scene0070_00_v14_b1 ;;
  scene0097_00) SHORT=0097; QUERY_SHA=4a34500f1429dd747ec48b9b32af9bf593e38d395f3c34f0bf768a7e632dec19; CHECKPOINT_TAG=scene0097_00_v14 ;;
  scene0140_00) SHORT=0140; QUERY_SHA=06bd53b8f0cbe49e2e2f2681df5074801ecb22adc8cc63c4dc73a237d6af267f; CHECKPOINT_TAG=scene0140_00_v14 ;;
  scene0347_00) SHORT=0347; QUERY_SHA=3dbb367f4c213c7fc6c446b10e4c7b343621368218221d565a50c27f3949df37; CHECKPOINT_TAG=scene0347_00_v14 ;;
  scene0400_00) SHORT=0400; QUERY_SHA=e168d80c787a3627abed0d7cc9a30764caafe41e02d1a3ab786fe9cf1a82c106; CHECKPOINT_TAG=scene0400_00_v67_dino_cv001_b2_s32768_ft20 ;;
  scene0590_00) SHORT=0590; QUERY_SHA=e8b1fde7a1e4e19e967785bfff4677bee61552c1dddaa23e494a40a1805a21e1; CHECKPOINT_TAG=scene0590_00_v14 ;;
  *) echo "unsupported ScanNet paper8 scene: $SCENE" >&2; exit 2 ;;
esac

CORE=$CORE_ROOT/$SCENE
SCENE_ROOT=$RUN_ROOT/$SCENE
MASK_PARENT=$SCENE_ROOT/rebuilt_official_sam3_masks
MASK_ROOT=$MASK_PARENT/shard0
MEMBERSHIP=$SCENE_ROOT/official_sam3_exact_mpr_memberships.pt
AUTH=$CORE/exact_marginal_responsibility_heldout4.json
FEATURE_MANIFEST=$CORE/source_only_siglip2_features/frame_manifest.json
PRIMITIVE=$CORE/primitive_query_method_v1.pth
IMAGE_ROOT=/mnt/pool/sqy/3d_understanding/scannet_og/$SCENE/color
mkdir -p "$SCENE_ROOT/logs" "$MASK_PARENT/logs" "$EVAL_ROOT/$SCENE"

if [[ ! -f "$MEMBERSHIP" ]]; then
  if [[ ! -f "$MASK_ROOT/rebuild_receipt.json" ]]; then
    AUTH_SHA=$(sha256sum "$AUTH" | awk '{print $1}')
    FEATURE_SHA=$(sha256sum "$FEATURE_MANIFEST" | awk '{print $1}')
    CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
      bash "$ROOT/radio_gs/scripts/run_official_sam3_python.sh" \
      -m radio_gs.scripts.rebuild_scannet_source_sam_hierarchy \
      --exact-mpr-authority "$AUTH" \
      --expected-exact-mpr-sha256 "$AUTH_SHA" \
      --current-feature-manifest "$FEATURE_MANIFEST" \
      --expected-feature-manifest-sha256 "$FEATURE_SHA" \
      --image-root "$IMAGE_ROOT" \
      --output-root "$MASK_PARENT" \
      --shard-index 0 \
      --shard-count 1 \
      --device cuda:0 \
      --checkpoint-path "$SAM_CHECKPOINT" \
      --expected-checkpoint-sha256 "$SAM_SHA" \
      >"$SCENE_ROOT/logs/official_sam3_masks.log" 2>&1
  fi
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
    -m radio_gs.scripts.build_scannet_official_sam3_exact_mpr_memberships \
    --scene "$SCENE" \
    --responsibility-authority "$AUTH" \
    --primitive-cache "$PRIMITIVE" \
    --mask-root "$MASK_ROOT" \
    --expected-checkpoint-sha256 "$SAM_SHA" \
    --expected-grid-size 12 \
    --min-membership 0.5 \
    --device cuda:0 \
    --output "$MEMBERSHIP" \
    >"$SCENE_ROOT/logs/exact_mpr_memberships.log" 2>&1
fi

# Build proposal evidence ahead of a frozen confirmation without opening the
# benchmark labels.  This keeps expensive official-SAM work parallel while a
# globally fixed categorical readout is selected on a disjoint development
# cohort.
if [[ "${STOP_BEFORE_EVAL:-0}" == "1" ]]; then
  sha256sum "$MEMBERSHIP"
  exit 0
fi

RESULT=$EVAL_ROOT/$SCENE/scannet_vala_gaussian_protocol_results.json
if [[ ! -f "$RESULT" ]]; then
  MEMBERSHIP_SHA=$(sha256sum "$MEMBERSHIP" | awk '{print $1}')
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
    "$ROOT/radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py" \
    --scene_list "$SCENE" \
    --prepared_root /mnt/pool/sqy/3d_understanding/scannet_og \
    --config "$ROOT/radio_gs/configs/generated/frozen_eval_20260802/scannet_scene${SHORT}_canonical_mpr_v3_paper8.yaml" \
    --checkpoint "/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/scannet_og_${CHECKPOINT_TAG}/checkpoints/best.pth" \
    --label_ply "/mnt/pool/sqy/3d_understanding/scannet_og/$SCENE/${SCENE}_vh_clean_2.labels.ply" \
    --output_dir "$EVAL_ROOT/$SCENE" \
    --class_splits 19,15,10 \
    --feature_chunk_size 8192 \
    --pseudo_chunk_size 512 \
    --pseudo_gt_cache_dir "$CORE/scannet_vala_method_v1/pseudo_gt" \
    --radius_factor 5.0 \
    --candidate_k 1000 \
    --fallback_k 1 \
    --row_opacity_threshold 0.1 \
    --compact_feature_key features \
    --prompt_templates '{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}' \
    --text_embedding_cache "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt" \
    --text_encoder siglip2 \
    --class_aliases none \
    --projection_weights "$ROOT/checkpoints/siglip2_feat_projection.pth" \
    --summary_head_weights "$ROOT/checkpoints/siglip2_summary_head.pth" \
    --external_query_feature_cache "$PRIMITIVE" \
    --expected_external_query_feature_cache_sha256 "$QUERY_SHA" \
    --sam_proposal_membership_cache "$MEMBERSHIP" \
    --expected_sam_proposal_membership_cache_sha256 "$MEMBERSHIP_SHA" \
    --sam_instance_topology \
    --sam_instance_seed_margin "$SAM_SEED_MARGIN" \
    --sam_instance_update_margin "$SAM_UPDATE_MARGIN" \
    --sam_instance_semantic_tolerance "$SAM_SEMANTIC_TOLERANCE" \
    --sam_instance_consensus "$SAM_CONSENSUS" \
    --sam_instance_minimum_supporting_proposals "$SAM_MIN_PROPOSALS" \
    --sam_instance_minimum_supporting_views "$SAM_MIN_VIEWS" \
    --sam_instance_iterations "$SAM_ITERATIONS" \
    "${EXTRA_EVAL_ARGS[@]}" \
    --device cuda \
    >"$SCENE_ROOT/logs/frozen_eval.log" 2>&1
fi

sha256sum "$MEMBERSHIP" "$RESULT"
