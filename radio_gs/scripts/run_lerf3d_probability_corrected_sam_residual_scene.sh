#!/usr/bin/env bash

set -euo pipefail

ROOT=/root/RADIO-GS
SCENE=${SCENE:?set SCENE}
PHYSICAL_GPU=${PHYSICAL_GPU:?set PHYSICAL_GPU}
MEMBERSHIP_ROOT=${MEMBERSHIP_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf_sam3_exact_mpr_memberships_v2_probability_correct}
OUT_ROOT=${OUT_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf3d_seeded_residual_probability_correct_v2_alpha025}
SAM_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/foundation_cache_sam3_modelscope_mapped_trainviews

case "$SCENE" in
  figurines)
    RESPONSIBILITY=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/exact_marginal_responsibility_authority.json
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/figurines/primitive_query_method_v1.pth
    PRIMITIVE_SHA=acc0b8b4cbf429d92e2f9df05865898066349fb79bcbe0bd3933ae1e504f1e18
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_figurines_radio_verified_pose.yaml
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
    QUERIES='bag,green apple,green toy chair,jake,miffy,old camera,pikachu,pink ice cream,pirate hat,porcelain hand,pumpkin,red apple,red toy chair,rubber duck with buoy,rubber duck with hat,rubics cube,spatula,tesla door handle,toy cat statue,toy elephant,waldo'
    ;;
  ramen)
    RESPONSIBILITY=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/ramen/fix4c_exact_marginal_v1/exact_marginal_responsibility_authority.json
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/ramen/primitive_query_method_v1.pth
    PRIMITIVE_SHA=893fda2a90142f71ee8175e666f12353a93e08a8125d8d5bdaf26d3a95dc54b5
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_ramen_radio_verified_pose.yaml
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth
    QUERIES='bowl,chopsticks,corn,egg,glass of water,hand,kamaboko,napkin,nori,onion segments,plate,sake cup,spoon,wavy noodles'
    ;;
  teatime)
    RESPONSIBILITY=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/teatime/exact_marginal_target_v1/exact_marginal_responsibility_authority.json
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/teatime/primitive_query_method_v1.pth
    PRIMITIVE_SHA=3938c13cd5f2c78cc2522aeff26cb0f77ba08cbeb519288b4b564dffd629b96b
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_teatime_radio_verified_pose.yaml
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth
    QUERIES='apple,bag of cookies,bear nose,coffee,coffee mug,dall-e brand,hooves,paper napkin,plate,sheep,stuffed bear,tea in a glass,three cookies,yellow pouf'
    ;;
  waldo_kitchen)
    RESPONSIBILITY=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/exact_marginal_responsibility_authority.json
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/primitive_query_method_v1.pth
    PRIMITIVE_SHA=01ffe08e54466dc0da720bcc2e25029ae2b085e24e78f8ac5ad9ced28085159f
    CONFIG=$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_waldo_kitchen_radio_verified_pose.yaml
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth
    QUERIES='Stainless steel pots,cabinet,dark cup,frog cup,ketchup,knife,ottolenghi,plastic ladle,plate,pot,pour-over vessel,red cup,refrigerator,sink,spatula,spoon,toaster,yellow desk'
    ;;
  *) echo "unsupported scene: $SCENE" >&2; exit 2 ;;
esac

[[ "$(sha256sum "$PRIMITIVE" | cut -d' ' -f1)" == "$PRIMITIVE_SHA" ]] || {
  echo "primitive cache SHA-256 differs" >&2
  exit 3
}
MEMBERSHIP=$MEMBERSHIP_ROOT/$SCENE.pt
SCORES=$OUT_ROOT/scores/$SCENE.pt
EVAL_ROOT=$OUT_ROOT/eval/$SCENE
RESULT=$EVAL_ROOT/$SCENE/lerf_direct_3d_selection_results.json
mkdir -p "$MEMBERSHIP_ROOT/logs" "$OUT_ROOT/logs" "$OUT_ROOT/scores" "$EVAL_ROOT"
cd "$ROOT"

if [[ ! -f "$MEMBERSHIP" ]]; then
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.scripts.build_lerf_sam3_exact_mpr_memberships_v2 \
    --scene "$SCENE" \
    --responsibility-authority "$RESPONSIBILITY" \
    --sam3-cache-root "$SAM_ROOT" \
    --primitive-cache "$PRIMITIVE" \
    --min-membership 0.5 \
    --device cuda:0 \
    --output "$MEMBERSHIP" \
    >"$MEMBERSHIP_ROOT/logs/$SCENE.log" 2>&1
fi

if [[ ! -f "$SCORES" ]]; then
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.scripts.build_lerf_identity_seeded_object_topology_scores \
    --scene "$SCENE" \
    --primitive-query-cache "$PRIMITIVE" \
    --membership-cache "$MEMBERSHIP" \
    --text-embedding-cache checkpoints/siglip2_lerf_all_exact_official.pt \
    --canonical-embedding-cache checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt \
    --query-names "$QUERIES" \
    --posterior-mode legacy_seeded_residual \
    --legacy-alpha 0.25 \
    --seed-support-ratio 0.8 \
    --minimum-object-views 2 \
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
