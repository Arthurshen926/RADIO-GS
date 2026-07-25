#!/usr/bin/env bash

# Run a deliberately isolated AGILE3D rapid diagnostic after a canonical
# field exists.  The released click policy and direct canonical readout are
# unchanged, but the fail-closed whole-scene support admission is omitted.
# The evaluator writes diagnostic_only/formal_comparable=false and this script
# refuses to use a non-diagnostic output path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
FIELD_ROOT="${FIELD_ROOT:?set canonical reconstruction root}"
RUN_ROOT="${RUN_ROOT:-$(dirname "$FIELD_ROOT")}" 
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet}"
SCENE_NAMES="${SCENE_NAMES:?set one or more completed AGILE scene IDs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/eval_fast_no_support_gate}"
FIELD_CHECKPOINT_NAME="${FIELD_CHECKPOINT_NAME:-canonical_mpr_v2.pt}"
CAPABILITY_CACHE_NAME="${CAPABILITY_CACHE_NAME:-official_dino_sam3_views.pt}"
SUPPORT_GRAPH_NAME="${SUPPORT_GRAPH_NAME:-shared_support_graph_k16.pt}"
OBJECT_SHARD_INDEX="${OBJECT_SHARD_INDEX:-0}"
OBJECT_SHARD_COUNT="${OBJECT_SHARD_COUNT:-1}"
BACKGROUND_CENTROIDS="${BACKGROUND_CENTROIDS:-4}"
BACKGROUND_NEGATIVE_POLICY="${BACKGROUND_NEGATIVE_POLICY:-pooled_mean}"

if [[ "$OUTPUT_DIR" != *"fast_no_support_gate"* ]]; then
  echo "fast no-gate evaluation must write to an explicitly diagnostic directory" >&2
  exit 2
fi
read -r -a SCENES <<< "$SCENE_NAMES"
if (( OBJECT_SHARD_COUNT <= 0 || OBJECT_SHARD_INDEX < 0 || OBJECT_SHARD_INDEX >= OBJECT_SHARD_COUNT )); then
  echo "invalid OBJECT_SHARD_INDEX/OBJECT_SHARD_COUNT" >&2
  exit 2
fi
for scene in "${SCENES[@]}"; do
  while [[ ! -s "$FIELD_ROOT/canonical_fields/$scene/$FIELD_CHECKPOINT_NAME" \
      || ! -s "$FIELD_ROOT/canonical_fields/$scene/raw_radio_mpr.pt.json" \
      || ! -s "$FIELD_ROOT/canonical_fields/$scene/$CAPABILITY_CACHE_NAME" \
      || ! -s "$FIELD_ROOT/canonical_fields/$scene/$SUPPORT_GRAPH_NAME" ]]; do
    sleep 30
  done
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
    if (( available < 2 )); then sleep 20; fi
  done
}

mkdir -p "$OUTPUT_DIR"
wait_for_gpu

CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field \
  --benchmark-root "$BENCHMARK_ROOT" \
  --field-root "$FIELD_ROOT" \
  --geometry-cache-root "$OUTPUT_DIR/geometry" \
  --output "$OUTPUT_DIR/results.json" \
  --scene-names "$SCENE_NAMES" \
  --object-shard-index "$OBJECT_SHARD_INDEX" \
  --object-shard-count "$OBJECT_SHARD_COUNT" \
  --device cuda:0 \
  --observation-contract scannet_full_observation_diagnostic_v1 \
  --diagnostic-no-support-gate \
  --evaluation-voxel-size-m 0.05 \
  --readout-support-threshold 0.01 \
  --seed-candidate-k 64 \
  --world-point-prototype-mode per_click_local \
  --world-point-prototype-weighting support_mass \
  --solver-type confidence_random_walker \
  --laplacian-weight 1.0 --cg-iterations 64 --support-threshold 0.5 \
  --feature-calibration none \
  --background-centroids "$BACKGROUND_CENTROIDS" \
  --background-negative-policy "$BACKGROUND_NEGATIVE_POLICY" \
  --score-calibration none \
  --score-chunk-size 8192 \
  --max-clicks 20 --click-workers 3 \
  --field-checkpoint-name "$FIELD_CHECKPOINT_NAME" \
  --capability-cache-name "$CAPABILITY_CACHE_NAME" \
  --support-graph-name "$SUPPORT_GRAPH_NAME" \
  --require-official-extracted-capability-teachers \
  >"$OUTPUT_DIR/run.log" 2>&1
