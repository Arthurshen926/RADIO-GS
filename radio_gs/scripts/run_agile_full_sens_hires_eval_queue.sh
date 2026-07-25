#!/usr/bin/env bash

# Evaluate the native-official-spatial AGILE field as soon as every real field
# shard is complete.  This intentionally has no PFPR dependency: PFPR and
# AGILE share the field-side fidelity change but their evaluators are
# independent, so serializing them would only delay the AGILE promotion.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
FIELD_ROOT="${FIELD_ROOT:-output/agile3d_scannet40/full_sens_hires_official_v1/reconstruction_v1}"
RUN_ROOT="${RUN_ROOT:-output/agile3d_scannet40/full_sens_hires_official_v1}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet}"
SCENE_NAMES="${SCENE_NAMES:-scene0011_01 scene0046_00 scene0249_00}"

read -r -a SCENES <<< "$SCENE_NAMES"
if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "SCENE_NAMES must be non-empty" >&2
  exit 2
fi
for scene in "${SCENES[@]}"; do
  while test ! -s "$FIELD_ROOT/canonical_fields/$scene/canonical_mpr_v2.pt" \
    || test ! -s "$FIELD_ROOT/canonical_fields/$scene/official_dino_sam3_views.pt" \
    || test ! -s "$FIELD_ROOT/canonical_fields/$scene/shared_support_graph_k16.pt"; do
    sleep 30
  done
done

GPU="$GPU" \
REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS=1 \
ABLATION=baseline \
FIELD_ROOT="$FIELD_ROOT" \
RUN_ROOT="$RUN_ROOT" \
BENCHMARK_ROOT="$BENCHMARK_ROOT" \
SCENE_NAMES="$SCENE_NAMES" \
  bash radio_gs/scripts/run_agile_full_sens_ablation_queue.sh
