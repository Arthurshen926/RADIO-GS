#!/usr/bin/env bash

# Evaluate the predeclared four-mode scene-background scorer on the frozen
# NVOS canonical fields. This queue changes no field, prompt, target, solver,
# threshold, or evaluator; it only transfers the query-free scorer selected
# on disjoint ScanNet development/holdout scenes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SOURCE_ROOT="${SOURCE_ROOT:-output/evaluation_closeout_20260716/canonical_mpr_v3_nvos8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/optimization_20260724/background4_transfer/nvos8}"
QUEUE_PLAN="${QUEUE_PLAN:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/gaussfm_jobs/nvos_strict_unseen_v1/queue_plan.json}"
MANIFEST="${MANIFEST:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json}"
QUEUE_ROOT="$(dirname "$QUEUE_PLAN")"
SCENES=(fern flower fortress horns_center horns_left leaves orchids trex)

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

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"
for scene in "${SCENES[@]}"; do
  result="$OUTPUT_ROOT/$scene/eval_full_mask_random_walker/${scene}_evaluation.json"
  if [[ -s "$result" ]]; then
    continue
  fi
  source="$SOURCE_ROOT/$scene"
  for artifact in \
    canonical_d256_l128_capability_first.pth \
    official_dino_sam3_views.pt \
    shared_support_graph_k16.pt; do
    if [[ ! -s "$source/$artifact" ]]; then
      echo "$scene lacks frozen source artifact $artifact" >&2
      exit 2
    fi
  done
  wait_for_gpu
  exec {lock_fd}>"$OUTPUT_ROOT/locks/$scene.lock"
  flock "$lock_fd"
  if [[ -s "$result" ]]; then
    flock -u "$lock_fd"
    exec {lock_fd}>&-
    continue
  fi
  field_sha="$(sha256sum "$source/canonical_d256_l128_capability_first.pth" | awk '{print $1}')"
  mkdir -p "$(dirname "$result")"
  CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_nvos_gaussian_first.py \
    --manifest "$MANIFEST" \
    --queue-root "$QUEUE_ROOT" \
    --scene-id "$scene" \
    --output-dir "$(dirname "$result")" \
    --device cuda:0 \
    --region-space sam3 \
    --support-mode canonical_support \
    --prototype-count 4 \
    --canonical-capability-cache "$source/official_dino_sam3_views.pt" \
    --canonical-support-graph "$source/shared_support_graph_k16.pt" \
    --canonical-field-sha256 "$field_sha" \
    --graph-policy legacy \
    --component-graph-policy same \
    --feature-calibration none \
    --background-centroids 4 \
    --calibration-sample-size 8192 \
    --centroid-iterations 4 \
    --score-calibration none \
    --score-chunk-size 8192 \
    --solver-type confidence_random_walker \
    --laplacian-weight 1.0 \
    --solver-iterations 12 \
    --solver-residual 0.30 \
    --solver-support-threshold 0.50 \
    >"$OUTPUT_ROOT/logs/$scene.log" 2>&1
  flock -u "$lock_fd"
  exec {lock_fd}>&-
done

exec {aggregate_fd}>"$OUTPUT_ROOT/locks/aggregate.lock"
flock "$aggregate_fd"
if [[ ! -s "$OUTPUT_ROOT/summary.json" ]]; then
  bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/aggregate_registered_prompt_closeout.py \
    --queue-plan "$QUEUE_PLAN" \
    --result-root "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/summary.json" \
    >"$OUTPUT_ROOT/aggregate.log" 2>&1
fi
flock -u "$aggregate_fd"
exec {aggregate_fd}>&-
