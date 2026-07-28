#!/usr/bin/env bash
set -euo pipefail

# Fixed, label-free structural ablation for the three held-out AGILE scenes.
# Usage: bash radio_gs/scripts/run_capability_signed_graph_ablation.sh 3

physical_gpu="${1:?physical GPU index is required}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

benchmark_root="/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet"
field_root="output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1"
geometry_root="output/agile3d_scannet40/background4_holdout/geometry"
scenes="scene0653_01 scene0695_03 scene0704_00"
optimization_root="output/optimization_20260725"
mutual_graph="shared_support_graph_k16_mutual_surface_covis.pt"

run_python() {
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    OMP_NUM_THREADS=24 \
    MKL_NUM_THREADS=24 \
    bash radio_gs/scripts/run_repo_python.sh "$@"
}

evaluate() {
  local output="$1"
  shift
  mkdir -p "$(dirname "${output}")"
  run_python \
    -m radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field \
    --benchmark-root "${benchmark_root}" \
    --field-root "${field_root}" \
    --geometry-cache-root "${geometry_root}" \
    --scene-names "${scenes}" \
    --observation-contract dense_overlap_pilot \
    --output "${output}" \
    --device cuda:0 \
    --max-clicks 20 \
    --click-workers 3 \
    --background-centroids 4 \
    --score-chunk-size 8192 \
    "$@"
}

abstention_result="${optimization_root}/capability_abstention/agile_background4_holdout_max/results.json"
if [[ ! -s "${abstention_result}" ]]; then
  evaluate \
    "${abstention_result}" \
    --channel-confidence-mode max_affinity \
    > "$(dirname "${abstention_result}")/run.log" 2>&1
fi

mutual_root="${optimization_root}/capability_abstention/agile_background4_holdout_mutual_surface_covis_max"
mkdir -p "${mutual_root}"
for scene in ${scenes}; do
  scene_root="${field_root}/canonical_fields/${scene}"
  graph_path="${scene_root}/${mutual_graph}"
  if [[ -s "${graph_path}" ]]; then
    continue
  fi
  run_python \
    -m radio_gs.scripts.build_canonical_support_graph \
    --capability-cache "${scene_root}/official_dino_sam3_views.pt" \
    --responsibility-cache "${scene_root}/registration_responsibility.pt" \
    --output "${graph_path}" \
    --neighbors 16 \
    --topology-mode mutual_knn \
    --surface-relation local_pca_tangent_v1 \
    --surface-topology-min-affinity 0.5 \
    --covisibility-weight 0.25 \
    --require-covisibility-topology \
    --capability-affinity-mode exact_official_capability \
    --affinity-device cuda:0 \
    >> "${mutual_root}/build_graphs.log" 2>&1
done

mutual_result="${mutual_root}/results.json"
if [[ ! -s "${mutual_result}" ]]; then
  evaluate \
    "${mutual_result}" \
    --support-graph-name "${mutual_graph}" \
    --channel-confidence-mode max_affinity \
    > "${mutual_root}/run.log" 2>&1
fi

local_negative_root="${optimization_root}/signed_graph_local/agile_background4_holdout_full"
local_negative_result="${local_negative_root}/results.json"
if [[ ! -s "${local_negative_result}" ]]; then
  mkdir -p "${local_negative_root}"
  evaluate \
    "${local_negative_result}" \
    --support-graph-name "${mutual_graph}" \
    --channel-confidence-mode max_affinity \
    --negative-spatial-mode truncated_graph_decay \
    --negative-spatial-steps 4 \
    --negative-spatial-decay 0.8 \
    > "${local_negative_root}/run.log" 2>&1
fi

signed_root="${optimization_root}/signed_geodesic/agile_background4_holdout_full"
signed_result="${signed_root}/results.json"
if [[ ! -s "${signed_result}" ]]; then
  mkdir -p "${signed_root}"
  evaluate \
    "${signed_result}" \
    --support-graph-name "${mutual_graph}" \
    --channel-confidence-mode max_affinity \
    --negative-spatial-mode signed_geodesic \
    --negative-spatial-steps 12 \
    --negative-spatial-decay 0.9 \
    > "${signed_root}/run.log" 2>&1
fi
