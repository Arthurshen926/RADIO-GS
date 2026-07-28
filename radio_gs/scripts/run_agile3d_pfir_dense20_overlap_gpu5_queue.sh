#!/usr/bin/env bash

# Evaluate the exact released AGILE3D interaction on the PFIR/AGILE overlap.
#
# The official AGILE package has only point clouds and labels.  This runner
# deliberately uses the twenty official validation scenes for which the PFIR
# release also provides dense registered RGB-D observations.  It is therefore
# a dense-view overlap subset, not a replacement for the official 312-scene
# PLY-only table.  Each scene reuses its one PFIR canonical field; no AGILE
# labels, object IDs, clicks, or metrics enter field construction or export.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-5}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet}"
PFIR_FIELD_ROOT="${PFIR_FIELD_ROOT:-output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1}"
PFIR_MATERIALIZATION_REPORT="${PFIR_MATERIALIZATION_REPORT:-/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small/field_only_test_v1/materialization_report.json}"
FIELD_TERMINAL="${FIELD_TERMINAL:-$PFIR_FIELD_ROOT/canonical_mpr_v3_fields.complete}"
RUN_ROOT="${RUN_ROOT:-output/agile3d_scannet40/pfir_dense20_overlap_v1}"
RADIO_REPO="${RADIO_REPO:-/root/RADIO}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
INTERPOLATION_WORKERS="${INTERPOLATION_WORKERS:--1}"
SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-0}"

mkdir -p "$RUN_ROOT/features" "$RUN_ROOT/logs"

wait_for_gpu() {
  case "$SKIP_GPU_WAIT" in
    1|true|True|TRUE)
      return
      ;;
    0|false|False|FALSE)
      ;;
    *)
      echo "SKIP_GPU_WAIT must be 0/1 or true/false, got: $SKIP_GPU_WAIT" >&2
      exit 2
      ;;
  esac
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

while [[ ! -s "$FIELD_TERMINAL" ]]; do sleep 30; done

mapfile -t SCENES < <(
  PFIR_MATERIALIZATION_REPORT="$PFIR_MATERIALIZATION_REPORT" \
    BENCHMARK_ROOT="$BENCHMARK_ROOT" \
    bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

pfir = json.loads(Path(os.environ["PFIR_MATERIALIZATION_REPORT"]).read_text())
pfir_scenes = {str(row["scene_id"]) for row in pfir["scenes"]}
root = Path(os.environ["BENCHMARK_ROOT"])
official = {
    str(value)
    for value in np.load(root / "single" / "object_ids.npy", allow_pickle=False)[:, 0]
}
overlap = sorted(pfir_scenes & official)
if len(overlap) != len(pfir_scenes):
    raise SystemExit(
        f"PFIR scenes missing from AGILE3D official list: {sorted(pfir_scenes - official)}"
    )
for scene in overlap:
    print(scene)
PY
)

if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "PFIR/AGILE dense-view overlap is empty" >&2
  exit 2
fi
printf '%s\n' "${SCENES[@]}" >"$RUN_ROOT/scenes.txt"
echo "AGILE3D dense-view overlap: ${#SCENES[@]} official validation scenes"

for scene in "${SCENES[@]}"; do
  field_dir="$PFIR_FIELD_ROOT/canonical_fields/$scene"
  field="$field_dir/canonical_mpr_v2.pt"
  capability="$field_dir/official_dino_sam3_views.pt"
  raw_mpr="$field_dir/raw_radio_mpr.pt"
  mesh="$BENCHMARK_ROOT/scans/$scene.ply"
  output="$RUN_ROOT/features/$scene.npz"
  for required in "$field" "$capability" "$raw_mpr" "$mesh"; do
    if [[ ! -s "$required" ]]; then
      echo "missing required dense-overlap input for $scene: $required" >&2
      exit 2
    fi
  done
  if [[ ! -s "$output" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/export_canonical_field_to_agile3d_mesh.py \
      --field-checkpoint "$field" \
      --capability-cache "$capability" \
      --mpr-cache "$raw_mpr" \
      --mesh-ply "$mesh" \
      --output "$output" \
      --device cuda:0 \
      --neighbors 3 \
      --maximum-distance-m 0.10 \
      --interpolation-workers "$INTERPOLATION_WORKERS" \
      >"$RUN_ROOT/logs/$scene.export.log" 2>&1
  fi
done

SCENE_NAMES="$(IFS=,; echo "${SCENES[*]}")"
wait_for_gpu
CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.agile3d_scannet40.evaluate_feature_cache \
  --benchmark-root "$BENCHMARK_ROOT" \
  --feature-root "$RUN_ROOT/features" \
  --output "$RUN_ROOT/results.json" \
  --scene-names "$SCENE_NAMES" \
  --device cuda:0 \
  --selection-mode seeded_component \
  --observation-lift-mode observed_domain \
  --observation-lift-neighbors 3 \
  --observation-lift-maximum-distance-m 0.10 \
  >"$RUN_ROOT/logs/evaluation.log" 2>&1

date -Iseconds >"$RUN_ROOT/complete"
