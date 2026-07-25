#!/usr/bin/env bash

# Merge independently evaluated AGILE3D object shards.  This is coordination,
# not a GPU reservation: each input is a completed real evaluator run that
# used the fixed released click simulator.  The merge only verifies exact
# object coverage/provenance and re-aggregates saved trajectories.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet}"
EXPECTED_SCENES="${EXPECTED_SCENES:?set the fixed comma/space-separated AGILE scenes}"
SHARD_ROOT="${SHARD_ROOT:?set the shared diagnostic root containing shard_<index>/}"
ABLATION="${ABLATION:-baseline}"
SHARD_COUNT="${SHARD_COUNT:?set the positive number of object shards}"
OUTPUT="${OUTPUT:?set the merged full-trajectory result path}"

if ! bash radio_gs/scripts/run_repo_python.sh - "$SHARD_COUNT" <<'PY'
import sys

if int(sys.argv[1]) <= 0:
    raise SystemExit("SHARD_COUNT must be positive")
PY
then
  exit 2
fi

inputs=()
for ((index=0; index<SHARD_COUNT; index++)); do
  result="$SHARD_ROOT/shard_${index}/eval_${ABLATION}/results.json"
  while [[ ! -s "$result" ]]; do
    sleep 30
  done
  inputs+=("$result")
done

# Validate shard identity before the merge opens the released object list.
# This does not inspect predictions, labels, or metrics; it only prevents an
# accidentally duplicated/mis-numbered scheduler worker from looking like a
# complete benchmark result.
bash radio_gs/scripts/run_repo_python.sh - "${inputs[@]}" "$SHARD_COUNT" <<'PY'
import json
import sys

*paths, count_raw = sys.argv[1:]
count = int(count_raw)
seen = set()
for path in paths:
    payload = json.load(open(path, encoding="utf-8"))
    shard = payload.get("shard", {})
    index = int(shard.get("object_shard_index", -1))
    actual_count = int(shard.get("object_shard_count", -1))
    if actual_count != count or index < 0 or index >= count or index in seen:
        raise SystemExit(f"invalid AGILE object shard metadata: {path}")
    if shard.get("object_assignment") != "released_object_list_position_modulo":
        raise SystemExit(f"unexpected AGILE object assignment: {path}")
    seen.add(index)
if seen != set(range(count)):
    raise SystemExit("AGILE object shards are incomplete")
PY

mkdir -p "$(dirname "$OUTPUT")"
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.agile3d_scannet40.merge_canonical_results \
  --benchmark-root "$BENCHMARK_ROOT" \
  --expected-scenes "$EXPECTED_SCENES" \
  --inputs "${inputs[@]}" \
  --output "$OUTPUT"
