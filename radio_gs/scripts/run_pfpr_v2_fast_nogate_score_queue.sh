#!/usr/bin/env bash

# Produce a clearly isolated PFPR v2 diagnostic result once a field exists.
# This intentionally omits the fail-closed continuous-support admission check
# requested for rapid iteration.  It never overwrites a formal result and the
# scorer records support_gate_required=false in prediction_report.json.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SCENE_NAMES="${SCENE_NAMES:?set one or more completed PFPR scene IDs}"
FIELD_ROOT="${FIELD_ROOT:?set field reconstruction root}"
BENCHMARK_DIR="${BENCHMARK_DIR:-output/scannet_pfpr_small_v2/test_v2_r1}"
RUN_ROOT="${RUN_ROOT:-$(dirname "$FIELD_ROOT")}" 
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/eval_fast_no_support_gate}"
FIELD_CHECKPOINT_NAME="${FIELD_CHECKPOINT_NAME:-canonical_mpr_v2.pt}"
CAPABILITY_CACHE_NAME="${CAPABILITY_CACHE_NAME:-official_dino_sam3_views.pt}"

if [[ "$OUTPUT_DIR" != *"fast_no_support_gate"* ]]; then
  echo "fast no-gate evaluation must write to an explicitly diagnostic directory" >&2
  exit 2
fi

read -r -a SCENES <<< "$SCENE_NAMES"
for scene in "${SCENES[@]}"; do
  while [[ ! -s "$FIELD_ROOT/canonical_fields/$scene/$FIELD_CHECKPOINT_NAME" \
      || ! -s "$FIELD_ROOT/canonical_fields/$scene/$CAPABILITY_CACHE_NAME" ]]; do
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
  -m radio_gs.benchmarks.scannet_pfpr.score_dino_center \
  --benchmark-dir "$BENCHMARK_DIR" \
  --field-root "$FIELD_ROOT" \
  --geometry-cache-root "$OUTPUT_DIR/geometry" \
  --prediction-dir "$OUTPUT_DIR/predictions" \
  --scene-names "$SCENE_NAMES" \
  --expected-observation-contract scannet_full_observation_pfpr_queryheldout_v1 \
  --readout-support-threshold 0.01 \
  --device cuda:0 \
  --query-pooling center3x3 \
  --field-checkpoint-name "$FIELD_CHECKPOINT_NAME" \
  --capability-cache-name "$CAPABILITY_CACHE_NAME" \
  --require-official-extracted-capability-teachers \
  >"$OUTPUT_DIR/score.log" 2>&1

bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.evaluate_predictions \
  --benchmark-dir "$BENCHMARK_DIR" \
  --prediction-dir "$OUTPUT_DIR/predictions" \
  --output "$OUTPUT_DIR/results.json" \
  --scene-names "$SCENE_NAMES" \
  >"$OUTPUT_DIR/evaluate.log" 2>&1

OUTPUT_DIR="$OUTPUT_DIR" bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
payload = {
    "result_status": "diagnostic_only",
    "support_gate_required": False,
    "reason": "rapid_iteration_no_support_gate",
    "formal_comparable": False,
}
(root / "diagnostic_protocol.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
