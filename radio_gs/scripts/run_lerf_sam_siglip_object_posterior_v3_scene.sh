#!/usr/bin/env bash
set -euo pipefail

SCENE=${1:?usage: run_lerf_sam_siglip_object_posterior_v3_scene.sh SCENE}
ROOT=/root/RADIO-GS
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf_sam_siglip_object_posterior_source32_v6_relevancy_identity_extent}
SOURCE_ROOT=${SOURCE_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf_multiscale_sam3_source32}
FIELD_ROOT=${FIELD_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1}
SCORE_DEVICE=${SCORE_DEVICE:-cpu}
EVAL_GPU=${EVAL_GPU:-0}

case "$SCENE" in
  figurines)
    QUERIES='bag,green apple,green toy chair,jake,miffy,old camera,pikachu,pink ice cream,pirate hat,porcelain hand,pumpkin,red apple,red toy chair,rubber duck with buoy,rubber duck with hat,rubics cube,spatula,tesla door handle,toy cat statue,toy elephant,waldo'
    ;;
  ramen)
    QUERIES='bowl,chopsticks,corn,egg,glass of water,hand,kamaboko,napkin,nori,onion segments,plate,sake cup,spoon,wavy noodles'
    ;;
  teatime)
    QUERIES='apple,bag of cookies,bear nose,coffee,coffee mug,dall-e brand,hooves,paper napkin,plate,sheep,stuffed bear,tea in a glass,three cookies,yellow pouf'
    ;;
  waldo_kitchen)
    QUERIES='Stainless steel pots,cabinet,dark cup,frog cup,ketchup,knife,ottolenghi,plastic ladle,plate,pot,pour-over vessel,red cup,refrigerator,sink,spatula,spoon,toaster,yellow desk'
    ;;
  *)
    echo "unsupported LERF scene: $SCENE" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT/scores" "$RUN_ROOT/logs"
SCORES="$RUN_ROOT/scores/$SCENE.pt"
if [[ ! -s "$SCORES" ]]; then
  bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
    -m radio_gs.scripts.build_lerf_sam_siglip_object_posterior_scores \
    --scene "$SCENE" \
    --primitive-query-cache "$FIELD_ROOT/$SCENE/primitive_query_method_v1.pth" \
    --membership-cache "$SOURCE_ROOT/$SCENE/gaussian_multiscale_memberships.pt" \
    --proposal-teacher "$SOURCE_ROOT/$SCENE/mask_aligned_siglip2_crop_summary_teacher.pt" \
    --text-embedding-cache "$ROOT/checkpoints/siglip2_lerf_all_exact_official.pt" \
    --canonical-embedding-cache "$ROOT/checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt" \
    --query-names "$QUERIES" \
    --minimum-descriptor-score "${MINIMUM_DESCRIPTOR_SCORE:-0.55}" \
    --descriptor-gate "${DESCRIPTOR_GATE:-absolute}" \
    --descriptor-listwise-margin "${DESCRIPTOR_LISTWISE_MARGIN:-0.12}" \
    --view-identity-margin "${VIEW_IDENTITY_MARGIN:-0.12}" \
    --candidates-per-view "${CANDIDATES_PER_VIEW:-3}" \
    --maximum-proposal-area-fraction "${MAXIMUM_PROPOSAL_AREA_FRACTION:-0.25}" \
    --extent-membership-floor "${EXTENT_MEMBERSHIP_FLOOR:-0.50}" \
    --association-mode "${ASSOCIATION_MODE:-weighted_jaccard_components}" \
    --latent-logit-temperature "${LATENT_LOGIT_TEMPERATURE:-8.0}" \
    --knn-chunk-size "${KNN_CHUNK_SIZE:-8192}" \
    --device "$SCORE_DEVICE" \
    --output "$SCORES" \
    >"$RUN_ROOT/logs/${SCENE}_scores.log" 2>&1
fi

if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" RUN_ROOT="$RUN_ROOT" \
    bash "$ROOT/radio_gs/scripts/run_lerf_sam_siglip_object_posterior_eval_scene.sh" \
    "$SCENE"
fi
