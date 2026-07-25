#!/usr/bin/env bash

# Run one genuine, frozen-query PFPR variant after the full-.sens field exists.
# This is a dependency queue rather than a GPU holder: it sleeps while either
# artifacts or a requested physical GPU are unavailable, and then executes the
# declared scorer/evaluator exactly once.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
ABLATION="${ABLATION:?set ABLATION to raw, diagonal, or crop_context_contrastive}"
AFTER_RESULT="${AFTER_RESULT:-}"
FIELD_ROOT="${FIELD_ROOT:-output/scannet_pfpr_small_v1/full_sens_pilot_v2/reconstruction_v1}"
RUN_ROOT="${RUN_ROOT:-output/scannet_pfpr_small_v1/full_sens_pilot_v2}"
BENCHMARK_DIR="${BENCHMARK_DIR:-output/scannet_pfpr_small_v1/test_v1}"
SCENE_NAMES="${SCENE_NAMES:-scene0011_01 scene0046_00 scene0050_02}"
EXPECTED_OBSERVATION_CONTRACT="${EXPECTED_OBSERVATION_CONTRACT:-scannet_full_observation_pfpr_queryheldout_v1}"
# Match the full-observation AGILE gate: a 5 cm candidate needs meaningful
# field mass, not only a remote Gaussian tail, before it can enter retrieval.
READOUT_SUPPORT_THRESHOLD="${READOUT_SUPPORT_THRESHOLD:-0.01}"
REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS="${REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS:-0}"
# Permit an auditable field-side reconstruction variant to be scored without
# copying or overwriting the canonical directory.  These names are method
# artifacts, not evaluator parameters; defaults retain the frozen v2 field.
FIELD_CHECKPOINT_NAME="${FIELD_CHECKPOINT_NAME:-canonical_mpr_v2.pt}"
CAPABILITY_CACHE_NAME="${CAPABILITY_CACHE_NAME:-official_dino_sam3_views.pt}"
# PFPR anchors are the patch centre.  Keep the official DINO centre pooling
# explicit in queued artifacts so an unrelated scorer-default change cannot
# silently turn this into the historical all-token max matcher.
QUERY_POOLING="${QUERY_POOLING:-center3x3}"
if [[ "$QUERY_POOLING" != "center3x3" && "$QUERY_POOLING" != "center" ]]; then
  echo "QUERY_POOLING must be center3x3 or center" >&2
  exit 2
fi
for artifact_name in "$FIELD_CHECKPOINT_NAME" "$CAPABILITY_CACHE_NAME"; do
  if [[ -z "$artifact_name" || "$artifact_name" == */* || "$artifact_name" == "." || "$artifact_name" == ".." ]]; then
    echo "field/capability artifact names must be non-empty basenames" >&2
    exit 2
  fi
done

wait_for_gpu() {
  local available=0
  while (( available < 2 )); do
    local values used util
    values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU")"
    used="${values%%,*}"; util="${values##*,}"
    used="${used// /}"; util="${util// /}"
    if (( used < 1200 && util < 10 )); then
      available=$((available + 1))
    else
      available=0
    fi
    if (( available < 2 )); then
      sleep 20
    fi
  done
}

read -r -a SCENES <<< "$SCENE_NAMES"
if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "SCENE_NAMES must be non-empty" >&2
  exit 2
fi
for scene in "${SCENES[@]}"; do
  while test ! -s "$FIELD_ROOT/canonical_fields/$scene/$FIELD_CHECKPOINT_NAME" \
    || test ! -s "$FIELD_ROOT/canonical_fields/$scene/$CAPABILITY_CACHE_NAME"; do
    sleep 30
  done
done
if [[ -n "$AFTER_RESULT" ]]; then
  while test ! -s "$AFTER_RESULT"; do
    sleep 30
  done
fi

case "$ABLATION" in
  raw)
    OUTPUT_DIR="$RUN_ROOT/raw"
    ADAPTER=""
    EXTRA_ARGS=()
    ;;
  diagonal)
    OUTPUT_DIR="$RUN_ROOT/diagonal"
    ADAPTER=""
    EXTRA_ARGS=(--feature-calibration diagonal_robust --calibration-sample-size 8192)
    ;;
  crop_context_contrastive)
    OUTPUT_DIR="$RUN_ROOT/crop_context_contrastive_diagonal"
    ADAPTER="/mnt/pool/sqy/results/RADIO-GS/output/scannet_pfpr_small_v1/crop_context_contrastive_v1/context_bridge_h128_contrastive.pt"
    EXTRA_ARGS=(
      --feature-calibration diagonal_robust --calibration-sample-size 8192
      --crop-context-adapter-checkpoint "$ADAPTER"
    )
    ;;
  *)
    echo "unsupported PFPR ablation: $ABLATION" >&2
    exit 2
    ;;
esac

if [[ -n "$ADAPTER" && ! -s "$ADAPTER" ]]; then
  echo "frozen global PFPR crop adapter is missing: $ADAPTER" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
wait_for_gpu

CAPABILITY_PROVENANCE_ARGS=()
case "$REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS" in
  1|true|True|TRUE)
    CAPABILITY_PROVENANCE_ARGS=(--require-official-extracted-capability-teachers)
    ;;
  0|false|False|FALSE)
    ;;
  *)
    echo "REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS must be 0/1 or true/false" >&2
    exit 2
    ;;
esac

# The adapter was trained on scene-disjoint RGB-only crop/full-image pairs.
# The scorer still opens only manifest.method.json and public geometry; the
# private anchors appear exclusively in the subsequent evaluator process.
CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.score_dino_center \
  --benchmark-dir "$BENCHMARK_DIR" \
  --field-root "$FIELD_ROOT" \
  --geometry-cache-root "$OUTPUT_DIR/geometry" \
  --prediction-dir "$OUTPUT_DIR/predictions" \
  --scene-names "$SCENE_NAMES" \
  --expected-observation-contract "$EXPECTED_OBSERVATION_CONTRACT" \
  --require-support-gate --minimum-support-fraction 0.95 \
  --readout-support-threshold "$READOUT_SUPPORT_THRESHOLD" \
  --device cuda:0 \
  --query-pooling "$QUERY_POOLING" \
  --field-checkpoint-name "$FIELD_CHECKPOINT_NAME" \
  --capability-cache-name "$CAPABILITY_CACHE_NAME" \
  "${CAPABILITY_PROVENANCE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" >"$OUTPUT_DIR/run.log" 2>&1

bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.evaluate_predictions \
  --benchmark-dir "$BENCHMARK_DIR" \
  --prediction-dir "$OUTPUT_DIR/predictions" \
  --output "$OUTPUT_DIR/results.json" \
  --scene-names "$SCENE_NAMES" >"$OUTPUT_DIR/evaluate.log" 2>&1
