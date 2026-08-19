#!/usr/bin/env bash

set -euo pipefail

PHYSICAL_GPUS=${PHYSICAL_GPUS:?set a comma-separated list of GPUs released from SPIn}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/scannet_official_sam_instance_v1/scene0000_00/rebuilt_official_sam3_masks}
AUTH=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/scene0000_00/exact_marginal_responsibility_heldout4.json
AUTH_SHA=6ca0110b0a97a67bccd94688870a9a7e51917eff3b5c05164378806b69b2922e
FEATURE_MANIFEST=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/scene0000_00/source_only_siglip2_features/frame_manifest.json
FEATURE_MANIFEST_SHA=e6a332e04b484726b813e499d581ae333873ee1f23aca6391989d4127b2ec28c
IMAGE_ROOT=/mnt/pool/sqy/3d_understanding/scannet_og/scene0000_00/color
SAM_CHECKPOINT=/root/RADIO-GS/checkpoints/sam3_modelscope/sam3.pt
SAM_SHA=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

IFS=',' read -r -a GPUS <<< "$PHYSICAL_GPUS"
SHARD_COUNT=${#GPUS[@]}
if [[ "$SHARD_COUNT" -le 0 ]]; then
  echo "no GPUs supplied" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT/logs"

pids=()
for ((shard=0; shard<SHARD_COUNT; shard++)); do
  gpu=${GPUS[$shard]}
  CUDA_VISIBLE_DEVICES="$gpu" \
    bash radio_gs/scripts/run_official_sam3_python.sh \
    -m radio_gs.scripts.rebuild_scannet_source_sam_hierarchy \
    --exact-mpr-authority "$AUTH" \
    --expected-exact-mpr-sha256 "$AUTH_SHA" \
    --current-feature-manifest "$FEATURE_MANIFEST" \
    --expected-feature-manifest-sha256 "$FEATURE_MANIFEST_SHA" \
    --image-root "$IMAGE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --shard-index "$shard" \
    --shard-count "$SHARD_COUNT" \
    --device cuda:0 \
    --checkpoint-path "$SAM_CHECKPOINT" \
    --expected-checkpoint-sha256 "$SAM_SHA" \
    >"$OUTPUT_ROOT/logs/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "at least one official-SAM rebuild shard failed" >&2
  exit 3
fi
echo "official-SAM source hierarchy rebuilt: $OUTPUT_ROOT"
