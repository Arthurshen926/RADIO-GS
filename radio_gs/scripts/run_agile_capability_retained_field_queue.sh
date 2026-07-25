#!/usr/bin/env bash

# Materialize a named canonical-field reconstruction candidate whose checkpoint
# selection prioritizes the official DINO/SAM capabilities while retaining a
# fixed amount of raw RADIO MPR fidelity.  This is a field-only, label-free
# experiment: it never opens AGILE objects, clicks, labels, masks, predictions,
# or metrics.  A later evaluator must still pass the usual continuous-support
# gate and use the released AGILE interaction protocol unchanged.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
FIELD_ROOT="${FIELD_ROOT:?set FIELD_ROOT to a reconstruction root}"
SCENE_NAMES="${SCENE_NAMES:?set one or more space-separated scene names}"
AFTER_FIELD="${AFTER_FIELD:-}"
FIELD_INPUT_NAME="${FIELD_INPUT_NAME:-canonical_mpr_v1.pt}"
FIELD_OUTPUT_NAME="${FIELD_OUTPUT_NAME:-canonical_mpr_v2_capability_retained_v1.pt}"
CAPABILITY_OUTPUT_NAME="${CAPABILITY_OUTPUT_NAME:-official_dino_sam3_views_capability_retained_v1.pt}"
GRAPH_OUTPUT_NAME="${GRAPH_OUTPUT_NAME:-shared_support_graph_k16_capability_retained_v1.pt}"
# This is an absolute cosine retention budget fixed for this named
# reconstruction variant, shared by every DINO/SAM interface.  It is checked
# only on the deterministic held-out raw MPR probe, never on AGILE labels.
MAX_MPR_DROP="${MAX_MPR_DROP:-0.02}"
MAX_CAPABILITY_DROP="${MAX_CAPABILITY_DROP:-0.002}"
STEPS="${STEPS:-256}"

for artifact_name in "$FIELD_INPUT_NAME" "$FIELD_OUTPUT_NAME" "$CAPABILITY_OUTPUT_NAME" "$GRAPH_OUTPUT_NAME"; do
  if [[ -z "$artifact_name" || "$artifact_name" == */* || "$artifact_name" == "." || "$artifact_name" == ".." ]]; then
    echo "field/capability/graph artifact names must be non-empty basenames" >&2
    exit 2
  fi
done

if ! bash radio_gs/scripts/run_repo_python.sh - "$MAX_MPR_DROP" "$MAX_CAPABILITY_DROP" "$STEPS" <<'PY'
import sys

mpr_drop, capability_drop = (float(value) for value in sys.argv[1:3])
steps = int(sys.argv[3])
if not 0.0 <= mpr_drop <= 0.05:
    raise SystemExit("MAX_MPR_DROP must be in [0, 0.05]")
if not 0.0 <= capability_drop <= 0.05:
    raise SystemExit("MAX_CAPABILITY_DROP must be in [0, 0.05]")
if steps <= 0:
    raise SystemExit("STEPS must be positive")
PY
then
  exit 2
fi

if [[ -n "$AFTER_FIELD" ]]; then
  while [[ ! -s "$AFTER_FIELD" ]]; do
    sleep 30
  done
fi

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

mkdir -p "$FIELD_ROOT/logs"
for scene in "${SCENES[@]}"; do
  field_dir="$FIELD_ROOT/canonical_fields/$scene"
  config="$FIELD_ROOT/render_contracts/$scene.yaml"
  geometry="$FIELD_ROOT/render_contracts/$scene.geometry_renderer.pth"
  raw_mpr="$field_dir/raw_radio_mpr.pt"
  dino_mpr="$field_dir/dino_v3_mpr.pt"
  sam_mpr="$field_dir/sam3_mpr.pt"
  validation_plan="$field_dir/fidelity_validation_frames.json"
  field_input="$field_dir/$FIELD_INPUT_NAME"
  field_output="$field_dir/$FIELD_OUTPUT_NAME"
  capability_output="$field_dir/$CAPABILITY_OUTPUT_NAME"
  graph_output="$field_dir/$GRAPH_OUTPUT_NAME"

  while [[ ! -s "$config" || ! -s "$geometry" || ! -s "$raw_mpr" || ! -s "$dino_mpr" || ! -s "$sam_mpr" || ! -s "$validation_plan" || ! -s "$field_input" ]]; do
    sleep 30
  done

  validation_frames="$(
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/select_fidelity_validation_frames.py \
      --feature-dir "$FIELD_ROOT/radio_features/$scene" \
      --output "$validation_plan" --views 4 --print-csv
  )"

  if [[ ! -s "$field_output" || ! -s "$field_output.json" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/finetune_canonical_radio_rendering.py \
      --config "$config" \
      --geometry-checkpoint "$geometry" \
      --field-checkpoint "$field_input" \
      --mpr-cache "$raw_mpr" \
      --output "$field_output" \
      --device cuda:0 \
      --steps "$STEPS" \
      --mpr-weight 0.10 \
      --dino-render-weight 0.20 --sam3-render-weight 0.20 \
      --capability-map-source official_extracted \
      --capability-local-affinity-weight 0.25 \
      --capability-local-radius 1 --capability-local-balance-quantile 0.0 \
      --train-fusion \
      --validation-frame-ids "$validation_frames" \
      --selection-policy capability_pareto \
      --max-capability-drop "$MAX_CAPABILITY_DROP" \
      --max-mpr-drop "$MAX_MPR_DROP" \
      --seed 0 \
      >"$FIELD_ROOT/logs/${scene}.canonical_capability_retained_v1.log" 2>&1
  fi

  # Assert lineage before the derived capability bank is made available to a
  # query evaluator.  This catches accidental fallback to a raw-only or
  # benchmark-dependent selection path without inspecting any benchmark data.
  bash radio_gs/scripts/run_repo_python.sh - "$field_output" "$raw_mpr" "$MAX_MPR_DROP" <<'PY'
import sys
from pathlib import Path
import torch

field_path = Path(sys.argv[1]).resolve()
raw_path = Path(sys.argv[2]).resolve()
allowed_raw = float(sys.argv[3])
payload = torch.load(field_path, map_location="cpu")
if not isinstance(payload, dict):
    raise SystemExit("capability-retained field checkpoint is invalid")
render = payload.get("render_optimization", {})
if not isinstance(render, dict):
    raise SystemExit("capability-retained field lacks render-selection audit")
if render.get("selection_policy") != "capability_pareto":
    raise SystemExit("field did not use capability_pareto selection")
if abs(float(render.get("max_mpr_drop", -1.0)) - allowed_raw) > 1e-9:
    raise SystemExit("field raw-MPR retention budget differs from this named variant")
if Path(str(payload.get("mpr_cache", ""))).resolve() != raw_path:
    raise SystemExit("field MPR source differs from the frozen canonical source")
official = render.get("official_render_capability", {})
if not isinstance(official, dict) or official.get("teacher_map_source") != "official_extracted":
    raise SystemExit("field did not use official extracted DINO/SAM render teachers")
if any(bool(render.get(key, False)) for key in (
    "benchmark_masks_opened", "benchmark_labels_opened", "text_queries_opened",
)):
    raise SystemExit("field selection opened benchmark supervision")
PY

  if [[ ! -s "$capability_output" || ! -s "$capability_output.json" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_canonical_capability_views.py \
      --field-checkpoint "$field_output" \
      --mpr-cache "$raw_mpr" \
      --radio-checkpoint /root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar \
      --output "$capability_output" --batch-size 2048 --device cuda:0 \
      >"$FIELD_ROOT/logs/${scene}.capability_retained_v1.log" 2>&1
  fi

  if [[ ! -s "$graph_output" || ! -s "$graph_output.json" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_canonical_support_graph.py \
      --capability-cache "$capability_output" \
      --output "$graph_output" --neighbors 16 --topology-mode symmetric_union \
      >"$FIELD_ROOT/logs/${scene}.support_graph_capability_retained_v1.log" 2>&1
  fi
done
