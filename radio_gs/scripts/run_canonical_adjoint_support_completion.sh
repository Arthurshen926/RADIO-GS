#!/usr/bin/env bash

# Query-free support-completion ablation for one completed canonical field.
# Dominant top-1 MPR rows and every shared field module remain authoritative;
# raster adjoint observations only supervise rows absent from the primary MPR.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set one physical GPU index}"
SCENE_NAME="${SCENE_NAME:?set one scene ID}"
FIELD_ROOT="${FIELD_ROOT:?set the reconstruction root containing canonical_fields}"
BENCHMARK_DIR="${BENCHMARK_DIR:-output/scannet_pfpr_small_v2/test_v2_r1}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
MINIMUM_GEOMETRY_SUPPORT="${MINIMUM_GEOMETRY_SUPPORT:-0.95}"

scene="$SCENE_NAME"
field_dir="$FIELD_ROOT/canonical_fields/$scene"
config="$FIELD_ROOT/render_contracts/$scene.yaml"
geometry_checkpoint="$FIELD_ROOT/render_contracts/$scene.geometry_renderer.pth"
primary_mpr="$field_dir/raw_radio_mpr.pt"
base_field="$field_dir/canonical_mpr_v2.pt"
support_mpr="$field_dir/raw_radio_adjoint_support.pt"
completed_mpr="$field_dir/raw_radio_mpr_with_adjoint_support.pt"
completed_field="$field_dir/canonical_mpr_v3_support.pt"
completed_capability="$field_dir/official_dino_sam3_views_support.pt"
completed_graph="$field_dir/shared_support_graph_support_k16.pt"
audit="$field_dir/public_geometry_support_gate.json"
log_dir="$FIELD_ROOT/logs"
mkdir -p "$field_dir" "$log_dir"

for required in "$config" "$geometry_checkpoint" "$primary_mpr" "$base_field"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required support-completion input: $required" >&2
    exit 2
  fi
done

# This gate reads only public scene coordinates and frozen Gaussian geometry.
if [[ ! -s "$audit" ]]; then
  bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.benchmarks.scannet_pfpr.audit_geometry_support \
    --benchmark-dir "$BENCHMARK_DIR" \
    --field-root "$FIELD_ROOT" \
    --output "$audit" \
    --scene-names "$scene" \
    --device cpu \
    --readout-candidate-k 64 \
    --readout-support-threshold 0.01 \
    --minimum-support-fraction "$MINIMUM_GEOMETRY_SUPPORT" \
    --candidate-voxel-size-m 0.05 \
    >"$log_dir/${scene}.support_completion_geometry_gate.log" 2>&1
fi

AUDIT="$audit" SCENE="$scene" MINIMUM="$MINIMUM_GEOMETRY_SUPPORT" \
  bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

from radio_gs.benchmarks.scannet_pfpr.audit_geometry_support import (
    validate_geometry_support_gate,
)

payload = json.loads(Path(os.environ["AUDIT"]).read_text(encoding="utf-8"))
validate_geometry_support_gate(
    payload,
    scene_id=os.environ["SCENE"],
    minimum_support_fraction=float(os.environ["MINIMUM"]),
)
PY

validation_frames="$(
  PLAN="$field_dir/fidelity_validation_frames.json" \
    bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["PLAN"]).read_text(encoding="utf-8"))
values = payload.get("validation_frame_ids", payload.get("frame_ids", []))
print(",".join(str(int(value)) for value in values))
PY
)"
if [[ -z "$validation_frames" ]]; then
  echo "$scene: support completion requires held-out fidelity frames" >&2
  exit 2
fi

if [[ ! -s "$support_mpr" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
    --config "$config" \
    --checkpoint "$geometry_checkpoint" \
    --output "$support_mpr" \
    --device cuda:0 \
    --observation-contract legacy \
    --max-views 960 \
    --exclude-frame-ids "$validation_frames" \
    --feature-space radio \
    --aggregation-mode raster_adjoint \
    --registration-weight-mode alpha_depth \
    --raster-view-fusion contribution_mean \
    --normalize-each-view \
    >"$log_dir/${scene}.mpr_raw_adjoint_support.log" 2>&1
fi

if [[ ! -s "$completed_mpr" ]]; then
  bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/fuse_gaussian_mpr_support.py \
    --primary "$primary_mpr" \
    --support "$support_mpr" \
    --output "$completed_mpr" \
    >"$log_dir/${scene}.mpr_raw_support_fusion.log" 2>&1
fi

if [[ ! -s "$completed_field" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/complete_canonical_field_support.py \
    --base-field-checkpoint "$base_field" \
    --completed-mpr-cache "$completed_mpr" \
    --output "$completed_field" \
    --radio-checkpoint "$RADIO_CHECKPOINT" \
    --device cuda:0 \
    >"$log_dir/${scene}.canonical_support_completion.log" 2>&1
fi

if [[ ! -s "$completed_capability" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/build_canonical_capability_views.py \
    --field-checkpoint "$completed_field" \
    --mpr-cache "$completed_mpr" \
    --radio-checkpoint "$RADIO_CHECKPOINT" \
    --output "$completed_capability" \
    --batch-size 2048 \
    --device cuda:0 \
    >"$log_dir/${scene}.capability_support_completion.log" 2>&1
fi

if [[ ! -s "$completed_graph" ]]; then
  bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/build_canonical_support_graph.py \
    --capability-cache "$completed_capability" \
    --output "$completed_graph" \
    --neighbors 16 \
    --topology-mode symmetric_union \
    >"$log_dir/${scene}.support_completion_graph.log" 2>&1
fi

echo "$scene: query-free adjoint support completion is complete"
