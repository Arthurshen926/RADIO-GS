#!/usr/bin/env bash

set -euo pipefail

PHYSICAL_GPU=${PHYSICAL_GPU:?set PHYSICAL_GPU}
SAM_REGION_ALPHA=${SAM_REGION_ALPHA:?set SAM_REGION_ALPHA}
SAM_REGION_MARGIN_THRESHOLD=${SAM_REGION_MARGIN_THRESHOLD:?set SAM_REGION_MARGIN_THRESHOLD}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR}

CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
  bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py \
  --scene_list scene0000_00 \
  --prepared_root /mnt/pool/sqy/3d_understanding/scannet_og \
  --config radio_gs/configs/generated/frozen_eval_20260802/scannet_scene0000_canonical_mpr_v3_paper8.yaml \
  --checkpoint /mnt/pool/sqy/results/RADIO-GS/output/radio_gs/scannet_og_scene0000_00_v14/checkpoints/best.pth \
  --label_ply /mnt/pool/sqy/3d_understanding/scannet_og/scene0000_00/scene0000_00_vh_clean_2.labels.ply \
  --output_dir "$OUTPUT_DIR" \
  --class_splits 19,15,10 \
  --feature_chunk_size 8192 \
  --pseudo_chunk_size 512 \
  --pseudo_gt_cache_dir /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/scene0000_00/scannet_vala_method_v1/pseudo_gt \
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
  --external_query_feature_cache /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/scene0000_00/primitive_query_method_v1.pth \
  --expected_external_query_feature_cache_sha256 bd8d0f9e448ab2954edef02db2a6240eea85c98f984d6388139b201c3ba99c52 \
  --sam_region_feature_cache /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/scene0000_00/sam3_matched_exact_marginal_heldout4.pt \
  --expected_sam_region_feature_cache_sha256 19308ebf4f7ce995464424030c5f33443d8133fc15a7f400d5dd5f8624ec6b29 \
  --sam_region_k 8 \
  --sam_region_radius 0.10 \
  --sam_region_similarity_threshold 0.50 \
  --sam_region_alpha "$SAM_REGION_ALPHA" \
  --sam_region_margin_threshold "$SAM_REGION_MARGIN_THRESHOLD" \
  --device cuda
