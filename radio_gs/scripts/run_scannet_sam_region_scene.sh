#!/usr/bin/env bash

set -euo pipefail

SCENE=${SCENE:?set SCENE}
PHYSICAL_GPU=${PHYSICAL_GPU:?set PHYSICAL_GPU}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT}
SAM_REGION_ALPHA=${SAM_REGION_ALPHA:-0.25}
SAM_REGION_MARGIN_THRESHOLD=${SAM_REGION_MARGIN_THRESHOLD:-0.03}

case "$SCENE" in
  scene0000_00) SHORT=0000; QUERY_SHA=bd8d0f9e448ab2954edef02db2a6240eea85c98f984d6388139b201c3ba99c52; SAM_SHA=19308ebf4f7ce995464424030c5f33443d8133fc15a7f400d5dd5f8624ec6b29; CHECKPOINT_TAG=scene0000_00_v14 ;;
  scene0062_00) SHORT=0062; QUERY_SHA=9226255874f0a1a82ef69909c01533dbe89d1af9ed8ef336efb12219310f1abd; SAM_SHA=532eeb4ba4969ef8a36d18a4486bf17c26fdf0abcc64f3d1a66f134ae4d8e0d3; CHECKPOINT_TAG=scene0062_00_v14 ;;
  scene0070_00) SHORT=0070; QUERY_SHA=ec0c0942804723f89e91a48a0d7d48665d0c9d40bd7cf397aca3c82b1938b177; SAM_SHA=a4476d21ca3e2842dc2e95737df32b2ef6dd90f3bc36223e4de41177502f185c; CHECKPOINT_TAG=scene0070_00_v14_b1 ;;
  scene0097_00) SHORT=0097; QUERY_SHA=4a34500f1429dd747ec48b9b32af9bf593e38d395f3c34f0bf768a7e632dec19; SAM_SHA=31a7748e3bd34cd9e2e16ddb6bca74bb7758d6dd83c0bb5cec3129dfb1bd6bb3; CHECKPOINT_TAG=scene0097_00_v14 ;;
  scene0140_00) SHORT=0140; QUERY_SHA=06bd53b8f0cbe49e2e2f2681df5074801ecb22adc8cc63c4dc73a237d6af267f; SAM_SHA=01ecc982b46f1971bd1ef20bd3d5714b3a5415739c8329ec2bb2a37279253aaa; CHECKPOINT_TAG=scene0140_00_v14 ;;
  scene0347_00) SHORT=0347; QUERY_SHA=3dbb367f4c213c7fc6c446b10e4c7b343621368218221d565a50c27f3949df37; SAM_SHA=a7765246152aaa7ea0c915f6c6391e651f3da6fe6b7275f0f8e31bf43001ef26; CHECKPOINT_TAG=scene0347_00_v14 ;;
  scene0400_00) SHORT=0400; QUERY_SHA=e168d80c787a3627abed0d7cc9a30764caafe41e02d1a3ab786fe9cf1a82c106; SAM_SHA=25bea01e62aeace042d723278e35fdc9a20e8aaf6221aa7c526f654254acd1ff; CHECKPOINT_TAG=scene0400_00_v67_dino_cv001_b2_s32768_ft20 ;;
  scene0590_00) SHORT=0590; QUERY_SHA=e8b1fde7a1e4e19e967785bfff4677bee61552c1dddaa23e494a40a1805a21e1; SAM_SHA=08f161021e04f46ccc0b508425068f37f8ca1d867919cb6ebda330fb5ef13547; CHECKPOINT_TAG=scene0590_00_v14 ;;
  *) echo "unsupported scene: $SCENE" >&2; exit 2 ;;
esac

CORE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/$SCENE
RESULT=$OUTPUT_ROOT/$SCENE/scannet_vala_gaussian_protocol_results.json
if [[ -e "$RESULT" ]]; then
  echo "existing result must be audited explicitly: $RESULT" >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
  bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py \
  --scene_list "$SCENE" \
  --prepared_root /mnt/pool/sqy/3d_understanding/scannet_og \
  --config "radio_gs/configs/generated/frozen_eval_20260802/scannet_scene${SHORT}_canonical_mpr_v3_paper8.yaml" \
  --checkpoint "/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/scannet_og_${CHECKPOINT_TAG}/checkpoints/best.pth" \
  --label_ply "/mnt/pool/sqy/3d_understanding/scannet_og/$SCENE/${SCENE}_vh_clean_2.labels.ply" \
  --output_dir "$OUTPUT_ROOT/$SCENE" \
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
  --text_embedding_cache checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt \
  --text_encoder siglip2 \
  --class_aliases none \
  --projection_weights checkpoints/siglip2_feat_projection.pth \
  --summary_head_weights checkpoints/siglip2_summary_head.pth \
  --external_query_feature_cache "$CORE/primitive_query_method_v1.pth" \
  --expected_external_query_feature_cache_sha256 "$QUERY_SHA" \
  --sam_region_feature_cache "$CORE/sam3_matched_exact_marginal_heldout4.pt" \
  --expected_sam_region_feature_cache_sha256 "$SAM_SHA" \
  --sam_region_k 8 \
  --sam_region_radius 0.10 \
  --sam_region_similarity_threshold 0.50 \
  --sam_region_alpha "$SAM_REGION_ALPHA" \
  --sam_region_margin_threshold "$SAM_REGION_MARGIN_THRESHOLD" \
  --device cuda
