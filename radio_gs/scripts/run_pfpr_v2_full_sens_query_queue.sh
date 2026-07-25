#!/usr/bin/env bash

# Score one finished native-official PFPR v2 field shard.  The scorer opens
# only the method-visible RGB patches and public candidate geometry; its
# built-in evaluator opens private anchors only after predictions are frozen.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SCENE_NAMES="${SCENE_NAMES:?set one or more completed PFPR v2 scene IDs}"
FIELD_ROOT="${FIELD_ROOT:-output/scannet_pfpr_small_v2/full_sens_official_v2_r1/reconstruction_v1}"
RUN_ROOT="${RUN_ROOT:-output/scannet_pfpr_small_v2/full_sens_official_v2_r1}"
BENCHMARK_DIR="${BENCHMARK_DIR:-output/scannet_pfpr_small_v2/test_v2_r1}"
FIELD_CHECKPOINT_NAME="${FIELD_CHECKPOINT_NAME:-canonical_mpr_v2.pt}"
CAPABILITY_CACHE_NAME="${CAPABILITY_CACHE_NAME:-official_dino_sam3_views.pt}"

exec env \
  GPU="$GPU" \
  ABLATION=raw \
  FIELD_ROOT="$FIELD_ROOT" \
  RUN_ROOT="$RUN_ROOT" \
  BENCHMARK_DIR="$BENCHMARK_DIR" \
  SCENE_NAMES="$SCENE_NAMES" \
  EXPECTED_OBSERVATION_CONTRACT="scannet_full_observation_pfpr_queryheldout_v1" \
  REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS=1 \
  QUERY_POOLING="center3x3" \
  FIELD_CHECKPOINT_NAME="$FIELD_CHECKPOINT_NAME" \
  CAPABILITY_CACHE_NAME="$CAPABILITY_CACHE_NAME" \
  bash radio_gs/scripts/run_pfpr_full_sens_query_ablation_queue.sh
