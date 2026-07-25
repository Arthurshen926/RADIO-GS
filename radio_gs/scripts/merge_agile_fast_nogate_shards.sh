#!/usr/bin/env bash

# Wait for and merge a complete set of diagnostic-only AGILE object shards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SHARD_ROOT="${SHARD_ROOT:?set root containing shard_<index>/results.json}"
SHARD_COUNT="${SHARD_COUNT:?set positive shard count}"
EXPECTED_SCENES_FILE="${EXPECTED_SCENES_FILE:?set newline-delimited fixed scene list}"
OUTPUT="${OUTPUT:?set merged result path}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet}"

if (( SHARD_COUNT <= 0 )); then
  echo "SHARD_COUNT must be positive" >&2
  exit 2
fi

inputs=()
for ((index=0; index<SHARD_COUNT; index++)); do
  result="$SHARD_ROOT/shard_${index}/results.json"
  while [[ ! -s "$result" ]]; do sleep 30; done
  inputs+=("$result")
done

scene_names="$(tr '\n' ' ' <"$EXPECTED_SCENES_FILE")"
mkdir -p "$(dirname "$OUTPUT")"
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.agile3d_scannet40.merge_canonical_results \
  --benchmark-root "$BENCHMARK_ROOT" \
  --expected-scenes "$scene_names" \
  --inputs "${inputs[@]}" \
  --output "$OUTPUT"
