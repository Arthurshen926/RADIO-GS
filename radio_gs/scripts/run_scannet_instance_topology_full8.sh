#!/usr/bin/env bash

set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT}
PHYSICAL_GPU=${PHYSICAL_GPU:-1}
MAX_PARALLEL=${MAX_PARALLEL:-2}
SCENES=(
  scene0000_00 scene0062_00 scene0070_00 scene0097_00
  scene0140_00 scene0347_00 scene0400_00 scene0590_00
)

running=0
for scene in "${SCENES[@]}"; do
  SCENE="$scene" \
  PHYSICAL_GPU="$PHYSICAL_GPU" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  SAM_K=12 \
  SAM_RADIUS=0.10 \
  SAM_SIMILARITY=0.70 \
  SEED_MARGIN=0.06 \
  UPDATE_MARGIN=0.01 \
  SEMANTIC_TOLERANCE=0.003 \
  CONSENSUS=0.97 \
  ITERATIONS=1 \
    bash radio_gs/scripts/run_scannet_instance_topology_scene.sh &
  running=$((running + 1))
  if [[ "$running" -ge "$MAX_PARALLEL" ]]; then
    wait -n
    running=$((running - 1))
  fi
done
wait
