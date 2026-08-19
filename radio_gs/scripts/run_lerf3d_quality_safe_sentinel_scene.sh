#!/usr/bin/env bash

set -euo pipefail

ROOT=/root/RADIO-GS
SCENE=${SCENE:?set SCENE to figurines or ramen}
PHYSICAL_GPU=${PHYSICAL_GPU:?set PHYSICAL_GPU}
MEMBERSHIP_ROOT=${MEMBERSHIP_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf_sam3_exact_mpr_memberships_v2_probability_correct}
OUT_ROOT=${OUT_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf3d_object_topology_quality_safe_sentinel_v1}

case "$SCENE" in
  figurines)
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/figurines/primitive_query_method_v1.pth
    PRIMITIVE_SHA=acc0b8b4cbf429d92e2f9df05865898066349fb79bcbe0bd3933ae1e504f1e18
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_figurines_radio_verified_pose.yaml
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
    QUERIES='bag,green apple,green toy chair,jake,miffy,old camera,pikachu,pink ice cream,pirate hat,porcelain hand,pumpkin,red apple,red toy chair,rubber duck with buoy,rubber duck with hat,rubics cube,spatula,tesla door handle,toy cat statue,toy elephant,waldo'
    ;;
  ramen)
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/ramen/primitive_query_method_v1.pth
    PRIMITIVE_SHA=893fda2a90142f71ee8175e666f12353a93e08a8125d8d5bdaf26d3a95dc54b5
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_ramen_radio_verified_pose.yaml
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth
    QUERIES='bowl,chopsticks,corn,egg,glass of water,hand,kamaboko,napkin,nori,onion segments,plate,sake cup,spoon,wavy noodles'
    ;;
  *) echo "unsupported sentinel scene: $SCENE" >&2; exit 2 ;;
esac

[[ "$(sha256sum "$PRIMITIVE" | cut -d' ' -f1)" == "$PRIMITIVE_SHA" ]] || {
  echo "primitive cache SHA-256 differs" >&2
  exit 3
}
MEMBERSHIP=$MEMBERSHIP_ROOT/$SCENE.pt
[[ -f "$MEMBERSHIP" ]] || { echo "corrected v2 membership cache is absent" >&2; exit 4; }

SCORES=$OUT_ROOT/scores/$SCENE.pt
EVAL_ROOT=$OUT_ROOT/eval/$SCENE
RESULT=$EVAL_ROOT/$SCENE/lerf_direct_3d_selection_results.json
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/scores" "$EVAL_ROOT"
cd "$ROOT"

# Frozen before either sentinel result: proposal confidence may rank candidates
# and weight accepted noisy-or observations, but never changes the pure
# probability membership threshold.  Sparse absence preserves the text prior.
if [[ ! -f "$SCORES" ]]; then
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.scripts.build_lerf_identity_seeded_object_topology_scores \
    --scene "$SCENE" \
    --primitive-query-cache "$PRIMITIVE" \
    --membership-cache "$MEMBERSHIP" \
    --text-embedding-cache checkpoints/siglip2_lerf_all_exact_official.pt \
    --canonical-embedding-cache checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt \
    --query-names "$QUERIES" \
    --posterior-mode object_topology \
    --seed-support-ratio 0.8 \
    --identity-core-ratio 0.8 \
    --minimum-object-views 2 \
    --minimum-row-views 1 \
    --extent-membership-floor 0.5 \
    --sibling-exclusion-strength 0 \
    --unknown-policy preserve_text_prior \
    --membership-calibration pure_probability \
    --use-proposal-quality \
    --device cuda:0 \
    --output "$SCORES" \
    >"$OUT_ROOT/logs/${SCENE}_scores.log" 2>&1
fi

if [[ ! -f "$RESULT" ]]; then
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_lerf_direct_3d_selection.py \
    --config "$CONFIG" \
    --checkpoint "$RENDERER" \
    --scene "$SCENE" \
    --protocol_preset none \
    --external_query_score_cache "$SCORES" \
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
    --output_dir "$EVAL_ROOT" \
    --summary_head_weights checkpoints/siglip2_summary_head.pth \
    --text_embedding_cache checkpoints/siglip2_lerf_all_exact_official.pt \
    --canonical_embedding_cache checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt \
    --prompt_templates '{query}' \
    --selection_mode score_threshold \
    --score_threshold 0.6 \
    --score_source direct \
    --scoring relevancy \
    --softmax_temperature 10 \
    --score_postprocess none \
    --projection_mode selected_only_alpha \
    --silhouette_threshold 0.0392156862745098 \
    --alpha_binarization png_uint8_gt10 \
    --mask_refinement peak_component_retention_guard \
    --component_guard_min_largest_fraction 0.65 \
    --min_select 0 \
    --gpu 0 \
    >"$OUT_ROOT/logs/${SCENE}_eval.log" 2>&1
fi

sha256sum "$MEMBERSHIP" "$SCORES" "$RESULT"
