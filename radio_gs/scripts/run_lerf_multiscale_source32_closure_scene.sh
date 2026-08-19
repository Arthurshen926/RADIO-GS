#!/usr/bin/env bash
set -euo pipefail

SCENE=${1:?usage: run_lerf_multiscale_source32_closure_scene.sh SCENE [DEVICE]}
DEVICE=${2:-cpu}
ROOT=/root/RADIO-GS
OUT_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf_multiscale_sam3_source32
PYTHON=${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}

case "$SCENE" in
  figurines)
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/exact_marginal_responsibility_authority.json
    ;;
  ramen)
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/ramen/fix4c_exact_marginal_v1/exact_marginal_responsibility_authority.json
    ;;
  teatime)
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/teatime/exact_marginal_target_v1/exact_marginal_responsibility_authority.json
    ;;
  waldo_kitchen)
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/exact_marginal_responsibility_authority.json
    ;;
  *)
    echo "unsupported LERF scene: $SCENE" >&2
    exit 2
    ;;
esac

SCENE_ROOT="$OUT_ROOT/$SCENE"
PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/$SCENE/primitive_query_method_v1.pth
SOURCE_AUTHORITY="$ROOT/paper/artifacts/lerf_${SCENE}_source32_exact_mpr_rgb_authority_20260817.json"
FEATURE_MANIFEST=/mnt/pool/sqy/results/RADIO-GS/output/canonical_teacher_features_v2/${SCENE}_source_only_siglip2/frame_manifest.json
MEMBERSHIP="$SCENE_ROOT/gaussian_multiscale_memberships.pt"
SPATIAL_TEACHER="$SCENE_ROOT/mask_aligned_siglip2_spatial_teacher.pt"

test -f "$SCENE_ROOT/manifest.json"
test -f "$PRIMITIVE"
test -f "$SOURCE_AUTHORITY"
test -f "$FEATURE_MANIFEST"

if [[ ! -f "$MEMBERSHIP" ]]; then
  "$PYTHON" -m radio_gs.scripts.build_lerf_multiscale_sam3_exact_mpr_memberships \
    --scene "$SCENE" \
    --responsibility-authority "$MPR" \
    --primitive-cache "$PRIMITIVE" \
    --source-authority "$SOURCE_AUTHORITY" \
    --mask-root "$SCENE_ROOT" \
    --min-membership 0.5 \
    --device "$DEVICE" \
    --output "$MEMBERSHIP"
fi

if [[ ! -f "$SPATIAL_TEACHER" ]]; then
  "$PYTHON" -m radio_gs.scripts.build_sam_mask_aligned_siglip2_spatial_teacher \
    --scene "$SCENE" \
    --mask-root "$SCENE_ROOT" \
    --feature-manifest "$FEATURE_MANIFEST" \
    --output "$SPATIAL_TEACHER"
fi
