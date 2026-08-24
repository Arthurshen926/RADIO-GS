#!/usr/bin/env bash
set -euo pipefail

# Source-only native SAM3 + native SigLIP2 region residual for one ScanNet
# development scene.  All thresholds and fusion weights are scene-invariant.

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SCENE" >&2
  exit 2
fi

SCENE=$1
ROOT=${ROOT:-/root/RADIO-GS}
GPU=${GPU:-0}
MAXIMUM_IMAGES=${MAXIMUM_IMAGES:-16}
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/$SCENE/native_sam_siglip}
CORE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/$SCENE
MPR=$CORE/exact_marginal_responsibility_heldout4.json
PRIMITIVE=$CORE/primitive_query_method_v1.pth
FRAME_MANIFEST=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260802/scannet_canonical_mpr_v3_paper8/feature_bundles/$SCENE/frame_manifest.json
if [[ ! -s "$FRAME_MANIFEST" ]]; then
  # Some paper8 scenes retain the hash-bound legacy frame manifest even when
  # their old dense feature tensors were pruned.  Native extraction needs only
  # the registered RGB identities and hashes, not those legacy tensors.
  FRAME_MANIFEST=/mnt/pool/sqy/results/RADIO-GS/output/radio_features_scannet_og/$SCENE/frame_manifest.json
fi
SCORE_CACHE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/scannet_official_sam_dev4_score_cache_v1/$SCENE/development/${SCENE}_scores.npz
if [[ ! -s "$SCORE_CACHE" ]]; then
  SCORE_CACHE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260821/scannet_object_tracks_paper8_score_cache_v1/$SCENE/development/${SCENE}_scores.npz
fi
AUTHORITY=$RUN_ROOT/source_rgb_exact_mpr_uniform${MAXIMUM_IMAGES}.json
MASK_ROOT=$RUN_ROOT/native_sam3_multiscale_uniform${MAXIMUM_IMAGES}
MANIFEST_NAME=manifest_grid8_crop2.json
MEMBERSHIP=$RUN_ROOT/native_sam3_multiscale_memberships.pt
TEACHER=$RUN_ROOT/native_siglip2_sam_crop_teacher.pt
RESULT=$RUN_ROOT/native_sam_siglip_region_vote.json

mkdir -p "$RUN_ROOT"
if [[ ! -s "$AUTHORITY" ]]; then
  bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
    -m radio_gs.scripts.materialize_exact_mpr_source_rgb_authority \
    --scene "$SCENE" \
    --exact-mpr-authority "$MPR" \
    --frame-manifest "$FRAME_MANIFEST" \
    --image-dir-override /mnt/pool/sqy/3d_understanding/scannet_og/$SCENE/color \
    --maximum-images "$MAXIMUM_IMAGES" \
    --output "$AUTHORITY" >"$RUN_ROOT/source_authority.log" 2>&1
fi
chmod -R o+rwX "$RUN_ROOT"

if [[ ! -s "$MASK_ROOT/$MANIFEST_NAME" ]]; then
  AUTHORITY_SHA=$(sha256sum "$AUTHORITY" | awk '{print $1}')
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_official_sam3_python.sh" \
      "$ROOT/radio_gs/scripts/build_sam3_multiscale_hierarchy_cache.py" \
      --source-authority "$AUTHORITY" \
      --source-authority-sha256 "$AUTHORITY_SHA" \
      --maximum-images "$MAXIMUM_IMAGES" \
      --output-root "$MASK_ROOT" \
      --manifest-name "$MANIFEST_NAME" \
      --checkpoint-path "$ROOT/checkpoints/sam3_modelscope/sam3.pt" \
      --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e \
      --device cuda:0 --dtype bfloat16 --resolution 1008 \
      --crop-layers 2 --crop-overlap-ratio 0.25 \
      --points-per-side 8 --point-grid-downscale-factor 2 \
      --minimum-quality 0.70 --minimum-stability 0.0 \
      --minimum-crop-area-fraction 0.001 \
      --minimum-full-image-area-fraction 0.0001 \
      --maximum-full-image-area-fraction 0.90 \
      --crop-edge-tolerance-pixels 2 --dedup-iou 0.85 \
      --dedup-near-equal-area-ratio 0.90 --maximum-masks 256 \
      --containment-threshold 0.90 --minimum-parent-area-ratio 1.05 \
      --skip-existing >"$RUN_ROOT/native_sam3.log" 2>&1
fi
chmod -R o+rwX "$RUN_ROOT"

if [[ ! -s "$MEMBERSHIP" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.build_lerf_multiscale_sam3_exact_mpr_memberships \
      --scene "$SCENE" --responsibility-authority "$MPR" \
      --primitive-cache "$PRIMITIVE" --source-authority "$AUTHORITY" \
      --mask-root "$MASK_ROOT" --manifest-name "$MANIFEST_NAME" \
      --min-membership 0.5 --device cuda:0 --output "$MEMBERSHIP" \
      >"$RUN_ROOT/membership.log" 2>&1
fi

if [[ ! -s "$TEACHER" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher \
      --scene "$SCENE" --mask-root "$MASK_ROOT" --manifest-name "$MANIFEST_NAME" \
      --encoder-backend native_siglip2_vision \
      --native-siglip2-model /root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots/a713301b217d38485fb2204c808367d10bc3cc40 \
      --context-expansion 1.5 --crop-resolution 384 --batch-size 1 \
      --device cuda:0 --output "$TEACHER" >"$RUN_ROOT/native_siglip.log" 2>&1
fi
chmod -R o+rwX "$RUN_ROOT"

if [[ ! -s "$RESULT" ]]; then
  bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
    -m radio_gs.scripts.evaluate_scannet_native_sam_siglip_region_vote \
    --scene "$SCENE" --membership "$MEMBERSHIP" --proposal-teacher "$TEACHER" \
    --score-cache "$SCORE_CACHE" \
    --text-cache-19 "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_exact_split19.pt" \
    --text-cache-15 "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_exact_split15.pt" \
    --text-cache-10 "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_exact_split10.pt" \
    --minimum-views 2 --minimum-view-agreement 0.5 \
    --blend-weights 0.25,0.5,1.0 --output "$RESULT" \
    >"$RUN_ROOT/native_sam_siglip_region_vote.log" 2>&1
fi
chmod -R o+rwX "$RUN_ROOT"
