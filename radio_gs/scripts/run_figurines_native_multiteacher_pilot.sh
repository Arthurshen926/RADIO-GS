#!/usr/bin/env bash
set -euo pipefail

# Close the preregistered Figurines native multi-teacher sentinel.  The two
# expensive source-only producers (native DINOv2 and official SAM3) may run in
# parallel before this launcher; all subsequent GPU stages are serialized so
# they cannot overcommit residual memory on the shared accelerator.

ROOT=${ROOT:-/root/RADIO-GS}
GPU=${GPU:-5}
POLL_SECONDS=${POLL_SECONDS:-20}
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/figurines}
CLOSURE_ROOT=${CLOSURE_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/figurines_closure}
MASK_ROOT=${MASK_ROOT:-$RUN_ROOT/native_sam3_multiscale_uniform8}
MANIFEST_NAME=${MANIFEST_NAME:-manifest_grid8_crop2.json}
MANIFEST="$MASK_ROOT/$MANIFEST_NAME"
DINO_TEACHER="$RUN_ROOT/native_dinov2_exact_mpr_trainval.pt"
SOURCE_AUTHORITY=${SOURCE_AUTHORITY:-$RUN_ROOT/source_rgb_exact_mpr_uniform8.json}
MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/exact_marginal_responsibility_authority.json
FIELD_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/figurines
BASE_FIELD="$FIELD_ROOT/generic_text_response_w005_s0_64.pth"
PRIMITIVE="$FIELD_ROOT/primitive_query_method_v1.pth"
MEMBERSHIP=${MEMBERSHIP:-$RUN_ROOT/native_sam3_multiscale_memberships.pt}
NATIVE_SIGLIP=${NATIVE_SIGLIP:-$RUN_ROOT/native_siglip2_sam_crop_teacher.pt}
ABC_ROOT="$RUN_ROOT/native_dinov2_abc_matched_v2"
SCORES="$CLOSURE_ROOT/scores/figurines.pt"
LOG="$RUN_ROOT/native_multiteacher_closure.log"

mkdir -p "$CLOSURE_ROOT/scores" "$CLOSURE_ROOT/logs"

while [[ ! -s "$MANIFEST" ]]; do
  sleep "$POLL_SECONDS"
done
if [[ "${SKIP_DINO_ABC:-0}" != 1 ]]; then
  while [[ ! -s "$DINO_TEACHER" ]]; do
    sleep "$POLL_SECONDS"
  done
fi

if [[ "${SKIP_DINO_ABC:-0}" != 1 && ! -s "$ABC_ROOT/abc_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$ROOT/radio_gs/scripts/train_native_multiteacher_abc_pilot.py" \
      --base-field "$BASE_FIELD" \
      --native-teacher "$DINO_TEACHER" \
      --output-dir "$ABC_ROOT" \
      --device cuda:0 \
      --steps 800 \
      --batch-size 1024 \
      --radio-weight 1.0 \
      --seed 0 >>"$LOG" 2>&1
fi

if [[ ! -s "$MEMBERSHIP" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.build_lerf_multiscale_sam3_exact_mpr_memberships \
      --scene figurines \
      --responsibility-authority "$MPR" \
      --primitive-cache "$PRIMITIVE" \
      --source-authority "$SOURCE_AUTHORITY" \
      --mask-root "$MASK_ROOT" \
      --manifest-name "$MANIFEST_NAME" \
      --min-membership 0.5 \
      --device cuda:0 \
      --output "$MEMBERSHIP" >>"$LOG" 2>&1
fi

if [[ ! -s "$NATIVE_SIGLIP" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher \
      --scene figurines \
      --mask-root "$MASK_ROOT" \
      --manifest-name "$MANIFEST_NAME" \
      --encoder-backend native_siglip2_vision \
      --native-siglip2-model /root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots/a713301b217d38485fb2204c808367d10bc3cc40 \
      --context-expansion 1.5 \
      --crop-resolution 384 \
      --batch-size 1 \
      --device cuda:0 \
      --output "$NATIVE_SIGLIP" >>"$LOG" 2>&1
fi

if [[ ! -s "$SCORES" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.build_lerf_sam_siglip_object_posterior_scores \
      --scene figurines \
      --primitive-query-cache "$PRIMITIVE" \
      --membership-cache "$MEMBERSHIP" \
      --proposal-teacher "$NATIVE_SIGLIP" \
      --text-embedding-cache "$ROOT/checkpoints/siglip2_lerf_all_exact_official.pt" \
      --canonical-embedding-cache "$ROOT/checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt" \
      --query-names 'bag,green apple,green toy chair,jake,miffy,old camera,pikachu,pink ice cream,pirate hat,porcelain hand,pumpkin,red apple,red toy chair,rubber duck with buoy,rubber duck with hat,rubics cube,spatula,tesla door handle,toy cat statue,toy elephant,waldo' \
      --descriptor-gate query_listwise \
      --descriptor-listwise-margin 0.12 \
      --view-identity-margin 0.12 \
      --candidates-per-view 3 \
      --maximum-proposal-area-fraction 0.25 \
      --extent-membership-floor 0.50 \
      --association-mode weighted_jaccard_components \
      --knn-chunk-size 8192 \
      --device cuda:0 \
      --output "$SCORES" >>"$LOG" 2>&1
fi

CUDA_VISIBLE_DEVICES="$GPU" RUN_ROOT="$CLOSURE_ROOT" SCORE_ROOT="$CLOSURE_ROOT" \
  bash "$ROOT/radio_gs/scripts/run_lerf_sam_siglip_object_posterior_eval_scene.sh" \
    figurines >>"$LOG" 2>&1

chmod -R o+rwX "$RUN_ROOT" "$CLOSURE_ROOT"
