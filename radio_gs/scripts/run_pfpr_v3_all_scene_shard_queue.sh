#!/usr/bin/env bash

# Resume-safe multi-GPU shard over all 20 ScanNet-PFPR-Small v2 scenes.
# Scene inputs are selected from the three immutable 960-view query-held-out
# materializations.  A completed scene is skipped by the underlying field
# queue; no evaluator anchor, pose/depth query, label, or metric is opened.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SHARD_INDEX="${SHARD_INDEX:?set zero-based shard index}"
SHARD_COUNT="${SHARD_COUNT:?set positive shard count}"
RUN_ROOT="${RUN_ROOT:-output/scannet_pfpr_small_v2/full_sens_official_v3_960_r1}"
BENCHMARK_DIR="${BENCHMARK_DIR:-output/scannet_pfpr_small_v2/test_v2_r1}"
AFTER_ARTIFACT="${AFTER_ARTIFACT:-}"
REQUIRE_GEOMETRY_SUPPORT_GATE="${REQUIRE_GEOMETRY_SUPPORT_GATE:-1}"

DATA_ROOT="${DATA_ROOT:-/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small}"
DEV3_ROOT="${DATA_ROOT}/field_only_full_sens_pfpr_v2_queryheldout_coverage960_dev3_v3"
SCENE0050_ROOT="${DATA_ROOT}/field_only_full_sens_pfpr_v2_queryheldout_coverage960_scene0050_v3"
REMAINING16_ROOT="${DATA_ROOT}/field_only_full_sens_pfpr_v2_queryheldout_coverage960_remaining16_v3"

if (( SHARD_COUNT <= 0 || SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "invalid SHARD_INDEX/SHARD_COUNT" >&2
  exit 2
fi

if [[ -n "$AFTER_ARTIFACT" ]]; then
  while [[ ! -s "$AFTER_ARTIFACT" ]]; do sleep 30; done
fi

mapfile -t SCENES < <(
  BENCHMARK_DIR="$BENCHMARK_DIR" bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(
    (Path(os.environ["BENCHMARK_DIR"]) / "manifest.public.json").read_text(
        encoding="utf-8"
    )
)
for row in sorted(payload.get("scene_domains", []), key=lambda item: item["scene_id"]):
    print(row["scene_id"])
PY
)

for index in "${!SCENES[@]}"; do
  if (( index % SHARD_COUNT != SHARD_INDEX )); then
    continue
  fi
  scene="${SCENES[$index]}"
  case "$scene" in
    scene0011_01|scene0046_00|scene0249_00)
      source_root="$DEV3_ROOT"
      ;;
    scene0050_02)
      source_root="$SCENE0050_ROOT"
      ;;
    *)
      source_root="$REMAINING16_ROOT"
      ;;
  esac
  env \
    GPU="$GPU" \
    SCENE_NAME="$scene" \
    SOURCE_ROOT="$source_root" \
    BENCHMARK_DIR="$BENCHMARK_DIR" \
    RUN_ROOT="$RUN_ROOT" \
    REQUIRE_GEOMETRY_SUPPORT_GATE="$REQUIRE_GEOMETRY_SUPPORT_GATE" \
    GPU_IDLE_CONFIRMATIONS="1" \
    bash radio_gs/scripts/run_pfpr_v2_mpr960_field_queue.sh
done

mkdir -p "$RUN_ROOT/queues"
date -Iseconds >"$RUN_ROOT/queues/pfpr_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.complete"
