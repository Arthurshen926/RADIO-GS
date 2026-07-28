#!/usr/bin/env bash
set -euo pipefail

# Frozen-unary PFIR readout ablation.  Both variants use one predeclared
# instance output policy and never expose test masks to the method.

physical_gpu="${1:?physical GPU index is required}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

benchmark="output/scannet_pfir_small_v1/test_v1_final"
source="${benchmark}/reconstruction_v1"
annotations="/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small/annotations"
root="output/optimization_20260725/pfir_support_readout"

evaluate() {
  local name="$1"
  local confidence="$2"
  local output="${root}/${name}"
  if [[ -s "${output}/track_b_selection.json" ]]; then
    return
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 \
    bash radio_gs/scripts/run_repo_python.sh \
      -m radio_gs.scripts.evaluate_pfir_support_redecode \
      --benchmark-dir "${benchmark}" \
      --source-run-root "${source}" \
      --output-root "${output}" \
      --annotations-root "${annotations}" \
      --device cuda:0 \
      --selection-mode top_component \
      --channel-confidence-mode "${confidence}" \
      > "${output}/run.log" 2>&1
}

evaluate "top_component" "none"
evaluate "top_component_max_affinity" "max_affinity"
