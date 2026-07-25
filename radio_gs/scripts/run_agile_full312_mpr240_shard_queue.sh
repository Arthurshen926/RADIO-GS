#!/usr/bin/env bash

# Build one complete 156-scene half of the AGILE3D ScanNet40 validation set.
# The field uses 240 query-free coverage-ranked RGB-D observations per scene
# (canonical full-observation MPR v1).  Final canonical/capability/graph
# artifacts are retained; reproducible native feature and MPR training caches
# are pruned per scene so the 312-scene run fits on the shared volume.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SOURCE_ROOT="${SOURCE_ROOT:?set one materialized 156-scene source shard}"
RUN_ROOT="${RUN_ROOT:-output/agile3d_scannet40/full312_mpr240_r1/reconstruction_v1}"
AFTER_ARTIFACT="${AFTER_ARTIFACT:-}"
SCENE_START_INDEX="${SCENE_START_INDEX:-0}"
SCENE_STOP_INDEX="${SCENE_STOP_INDEX:-}"

while [[ ! -s "$SOURCE_ROOT/materialization_report.json" ]]; do sleep 30; done
if [[ -n "$AFTER_ARTIFACT" ]]; then
  while [[ ! -s "$AFTER_ARTIFACT" ]]; do sleep 30; done
fi

scene_count="$(
  SOURCE_ROOT="$SOURCE_ROOT" bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(
    (Path(os.environ["SOURCE_ROOT"]) / "materialization_report.json").read_text(
        encoding="utf-8"
    )
)
rows = payload.get("scenes", [])
if not rows:
    raise SystemExit("materialized AGILE source shard is empty")
for row in rows:
    if int(row.get("field_frame_count", 0)) != 240:
        raise SystemExit("AGILE full312 MPR-v1 shard requires exactly 240 views")
    if row.get("field_contract_version") != "scannet_full_observation_v1":
        raise SystemExit("AGILE source shard has the wrong field contract")
print(len(rows))
PY
)"

if [[ -z "$SCENE_STOP_INDEX" ]]; then
  SCENE_STOP_INDEX="$scene_count"
fi
if (( SCENE_START_INDEX < 0 || SCENE_STOP_INDEX <= SCENE_START_INDEX || SCENE_STOP_INDEX > scene_count )); then
  echo "invalid AGILE source slice [$SCENE_START_INDEX:$SCENE_STOP_INDEX] for $scene_count scenes" >&2
  exit 2
fi

exec env \
  GPU="$GPU" \
  FIELD_ROOT="$SOURCE_ROOT" \
  PFIR_MATERIALIZATION_REPORT="$SOURCE_ROOT/materialization_report.json" \
  RUN_ROOT="$RUN_ROOT" \
  GEOMETRY_ROOT="$RUN_ROOT/geometry" \
  FEATURE_ROOT="$RUN_ROOT/radio_features" \
  CONTRACT_ROOT="$RUN_ROOT/render_contracts" \
  FIELD_OUTPUT_ROOT="$RUN_ROOT/canonical_fields" \
  OBSERVATION_CONTRACT="scannet_full_observation_v1" \
  MPR_OBSERVATION_CONTRACT="canonical-full-observation-mpr-v1" \
  FIELD_SELECTION_POLICY="capability_pareto" \
  FIELD_MAX_MPR_DROP="0.02" \
  GEOMETRY_INIT_FRAMES="240" \
  GEOMETRY_MAX_POINTS="300000" \
  GEOMETRY_INIT_SELECTION_POLICY="coverage_prefix" \
  RADIO_RESOLUTION_SCALE="1.0" \
  RADIO_BATCH_SIZE="1" \
  CAPABILITY_MAP_SOURCE="official_extracted" \
  RADIO_ADAPTOR_NAMES="dino_v3_7b,sam3" \
  BUILD_SEMANTIC="0" \
  PRUNE_REGENERABLE_INTERMEDIATES="1" \
  GPU_IDLE_CONFIRMATIONS="1" \
  SCENE_START_INDEX="$SCENE_START_INDEX" \
  SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
  WRITE_TERMINAL="0" \
  bash radio_gs/scripts/run_scannet_pfir_gpu5_queue.sh
