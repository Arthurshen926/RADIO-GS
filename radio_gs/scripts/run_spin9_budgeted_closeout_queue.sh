#!/usr/bin/env bash

# Resume a disjoint shard of the nine available SPIn-NeRF registered-prompt
# scenes. Missing high-resolution geometry uses one fixed label-free primitive
# budget; already complete stages are reused by the protocol-locked runner.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SCENE_NAMES="${SCENE_NAMES:?set a disjoint list of SPIn scene IDs}"
QUEUE_PLAN="${QUEUE_PLAN:-output/unified_query/spin9_gaussfm_queue_20260712/queue_plan.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/evaluation_closeout_20260716/canonical_mpr_v3_spin9}"
GEOMETRY_MAX_GAUSSIANS="${GEOMETRY_MAX_GAUSSIANS:-1400000}"
GEOMETRY_PACKED="${GEOMETRY_PACKED:-0}"
AFTER_MARKERS="${AFTER_MARKERS:-}"

wait_for_dependencies() {
  local marker
  for marker in $AFTER_MARKERS; do
    while [[ ! -s "$marker" ]]; do sleep 30; done
  done
}

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

if (( GEOMETRY_MAX_GAUSSIANS <= 0 )); then
  echo "GEOMETRY_MAX_GAUSSIANS must be positive for this queue" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/queue_logs"
wait_for_dependencies
for scene in $SCENE_NAMES; do
  result="$OUTPUT_ROOT/$scene/eval_full_mask_random_walker/${scene}_evaluation.json"
  if [[ -s "$result" ]]; then
    continue
  fi
  wait_for_gpu
  geometry_args=()
  if [[ "$GEOMETRY_PACKED" == "1" ]]; then
    geometry_args+=(--geometry-packed)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/run_canonical_promptable_closeout.py \
    --queue-plan "$QUEUE_PLAN" \
    --scene-id "$scene" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --geometry-max-gaussians "$GEOMETRY_MAX_GAUSSIANS" \
    "${geometry_args[@]}" \
    >"$OUTPUT_ROOT/queue_logs/$scene.log" 2>&1
done
