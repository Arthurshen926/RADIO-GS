#!/usr/bin/env bash
set -euo pipefail

# Materialize one sealed, query-independent source-SAM grid16 carrier and then
# close the unchanged typed LERF readout. This launcher intentionally uses only
# residual memory on one explicitly assigned GPU and resumes per-frame caches.

SCENE=${1:?scene is required}
GPU=${2:?physical GPU index is required}
ROOT=${ROOT:-/root/RADIO-GS}
SOURCE_ROOT=${SOURCE_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260823/lerf_multiscale_sam3_source32_grid16}
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260823/lerf_sam_siglip_object_posterior_grid16_v1}
MINIMUM_FREE_MIB=${MINIMUM_FREE_MIB:-6500}
POLL_SECONDS=${POLL_SECONDS:-30}

case "$SCENE" in
  ramen)
    AUTHORITY="$ROOT/paper/artifacts/lerf_ramen_source32_exact_mpr_rgb_authority_20260817.json"
    AUTHORITY_SHA=7fbbaf44361b90ae3f48edb52d50bef987a61d8dc6509b51bed5205a90134fd6
    ;;
  teatime)
    AUTHORITY="$ROOT/paper/artifacts/lerf_teatime_source32_exact_mpr_rgb_authority_20260817.json"
    AUTHORITY_SHA=110e01b2c39c0e6cc02d47aa06c12d40a3f9e0a47f8704ef37683fd83ed055a0
    ;;
  waldo_kitchen)
    AUTHORITY="$ROOT/paper/artifacts/lerf_waldo_kitchen_source32_exact_mpr_rgb_authority_20260817.json"
    AUTHORITY_SHA=6f7c68363cc463be2a37da30e2c0fad215a65d8784422477bbe9595f0b0c9c27
    ;;
  *)
    echo "unsupported scene: $SCENE" >&2
    exit 2
    ;;
esac

OUTPUT="$SOURCE_ROOT/$SCENE"
LOG="$OUTPUT/grid16_dynamic.log"
mkdir -p "$OUTPUT"

free_mib() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ' ' \
    | awk -F, -v gpu="$GPU" '$1 == gpu { print $2 }'
}

while [[ ! -s "$OUTPUT/manifest.json" ]]; do
  free=$(free_mib)
  if [[ -n "$free" ]] && (( free >= MINIMUM_FREE_MIB )); then
    echo "[$(date -Is)] $SCENE grid16: trying GPU $GPU with ${free} MiB free" | tee -a "$LOG"
    if CUDA_VISIBLE_DEVICES="$GPU" \
      bash "$ROOT/radio_gs/scripts/run_official_sam3_python.sh" \
        -m radio_gs.scripts.build_sam3_multiscale_hierarchy_cache \
        --source-authority "$AUTHORITY" \
        --source-authority-sha256 "$AUTHORITY_SHA" \
        --maximum-images 32 \
        --output-root "$OUTPUT" \
        --checkpoint-path "$ROOT/checkpoints/sam3_modelscope/sam3.pt" \
        --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e \
        --device cuda:0 \
        --dtype bfloat16 \
        --resolution 1008 \
        --crop-layers 2 \
        --crop-overlap-ratio 0.25 \
        --points-per-side 16 \
        --point-grid-downscale-factor 2 \
        --minimum-quality 0.70 \
        --minimum-stability 0.0 \
        --stability-offset 1.0 \
        --minimum-crop-area-fraction 0.001 \
        --minimum-full-image-area-fraction 0.0001 \
        --maximum-full-image-area-fraction 0.90 \
        --crop-edge-tolerance-pixels 2 \
        --dedup-iou 0.85 \
        --dedup-near-equal-area-ratio 0.90 \
        --maximum-masks 0 \
        --containment-threshold 0.90 \
        --minimum-parent-area-ratio 1.05 \
        --skip-existing >>"$LOG" 2>&1; then
      break
    fi
    echo "[$(date -Is)] $SCENE grid16: attempt failed; resuming sealed frames" | tee -a "$LOG"
  fi
  sleep "$POLL_SECONDS"
done

SOURCE_ROOT="$SOURCE_ROOT" RUN_ROOT="$RUN_ROOT" GPU_SET="$GPU" \
  MINIMUM_FREE_MIB="$MINIMUM_FREE_MIB" POLL_SECONDS="$POLL_SECONDS" \
  bash "$ROOT/radio_gs/scripts/run_lerf_grid16_candidate_closure.sh" "$SCENE"

chmod -R o+rwX "$OUTPUT" "$RUN_ROOT"
