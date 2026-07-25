#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

QUEUE_PLAN="${QUEUE_PLAN:-output/unified_query/spin9_gaussfm_queue_20260712/queue_plan.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/evaluation_closeout_20260716/canonical_mpr_v3_spin9}"
SCENES=(orchids leaves fern room horns fortress pinecone truck lego)

for scene in "${SCENES[@]}"; do
  result="$OUTPUT_ROOT/$scene/eval_full_mask_random_walker/${scene}_evaluation.json"
  while [[ ! -s "$result" ]]; do sleep 30; done
done

bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/aggregate_registered_prompt_closeout.py \
  --queue-plan "$QUEUE_PLAN" \
  --result-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/summary.json" \
  >"$OUTPUT_ROOT/aggregate.log" 2>&1

