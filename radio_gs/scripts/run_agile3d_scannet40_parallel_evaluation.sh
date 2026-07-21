#!/usr/bin/env bash

# Exact scene-disjoint AGILE3D evaluation sharding.  Every shard invokes the
# ordinary released-protocol evaluator; merge_results.py only validates and
# aggregates the completed trajectories, so this changes wall time but not a
# prediction, click, or metric definition.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet}"
RUN_ROOT="${RUN_ROOT:-output/agile3d_scannet40/formal_v1}"
GPU_LIST="${GPU_LIST:-0,1,3}"
EVAL_MIN_FREE_MEMORY_MIB="${EVAL_MIN_FREE_MEMORY_MIB:-4000}"
EVAL_CLICK_WORKERS="${EVAL_CLICK_WORKERS:-2}"
EVAL_CPU_THREADS="${EVAL_CPU_THREADS:-2}"
FIELD_TERMINAL="${FIELD_TERMINAL:-$RUN_ROOT/canonical_mpr_v3_fields.complete}"
FORMAL_TERMINAL="${FORMAL_TERMINAL:-$RUN_ROOT/formal.complete}"

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "GPU_LIST must contain at least one GPU index" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/evaluation_shards"

all_fields_ready() {
  bash radio_gs/scripts/run_repo_python.sh - "$BENCHMARK_ROOT" "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

benchmark, root = map(Path, sys.argv[1:])
scenes = sorted({str(value) for value in np.load(
    benchmark / "single" / "object_ids.npy", allow_pickle=False
)[:, 0]})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


invalid: list[str] = []
for scene in scenes:
    feature = root / "features" / f"{scene}.npz"
    feature_report = feature.with_suffix(feature.suffix + ".json")
    field = root / "scenes" / scene / "canonical_mpr_v2.pt"
    capability_report = (
        root / "scenes" / scene / "official_dino_sam3_views.pt.json"
    )
    required = (feature, feature_report, field, capability_report)
    if any(not path.is_file() or not path.stat().st_size for path in required):
        invalid.append(f"{scene}: missing field, feature, or provenance")
        continue
    try:
        field_hash = sha256(field)
        feature_hash = json.loads(feature_report.read_text())["field_checkpoint_sha256"]
        capability_hash = json.loads(capability_report.read_text())["field_checkpoint_sha256"]
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        invalid.append(f"{scene}: invalid provenance ({error})")
        continue
    if feature_hash != field_hash or capability_hash != field_hash:
        invalid.append(f"{scene}: stale field provenance")

if invalid:
    print("AGILE3D feature cache is not provenance-consistent:", *invalid[:20], sep="\n", file=sys.stderr)
raise SystemExit(1 if invalid else 0)
PY
}

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local values total used free
    values="$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$gpu")"
    total="${values%%,*}"; used="${values##*,}"
    total="${total// /}"; used="${used// /}"
    free=$(( total - used ))
    if (( free >= EVAL_MIN_FREE_MEMORY_MIB )); then
      return
    fi
    sleep 20
  done
}

while ! all_fields_ready; do sleep 60; done
date -Iseconds >"$FIELD_TERMINAL"

mapfile -t SHARDS < <(
  bash radio_gs/scripts/run_repo_python.sh - "$BENCHMARK_ROOT" "${#GPUS[@]}" <<'PY'
import sys
from collections import Counter
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
count = int(sys.argv[2])
objects = np.load(root / "single" / "object_ids.npy", allow_pickle=False)
scene_object_counts = Counter(str(value) for value in objects[:, 0])
scenes = sorted(scene_object_counts)
# Equal scene counts can leave one evaluator with far more official
# object/click trajectories.  Greedily balance the immutable scene groups by
# their released object count; this is a scheduling-only partition and never
# changes a scene, prediction, or metric row.
loads = [0] * count
shards = [[] for _ in range(count)]
for scene in sorted(scenes, key=lambda value: (-scene_object_counts[value], value)):
    index = min(range(count), key=lambda value: (loads[value], value))
    shards[index].append(scene)
    loads[index] += scene_object_counts[scene]
for index in range(count):
    values = sorted(shards[index])
    if values:
        print(f"{index}\t{','.join(values)}")
PY
)

pids=()
inputs=()
for entry in "${SHARDS[@]}"; do
  index="${entry%%$'\t'*}"
  scene_names="${entry#*$'\t'}"
  gpu="${GPUS[$index]}"
  shard="$RUN_ROOT/evaluation_shards/shard_${index}.json"
  log="$RUN_ROOT/logs/evaluation_shard_${index}.log"
  wait_for_gpu "$gpu"
  OMP_NUM_THREADS="$EVAL_CPU_THREADS" \
  MKL_NUM_THREADS="$EVAL_CPU_THREADS" \
  OPENBLAS_NUM_THREADS="$EVAL_CPU_THREADS" \
  NUMEXPR_NUM_THREADS="$EVAL_CPU_THREADS" \
  CUDA_VISIBLE_DEVICES="$gpu" bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.benchmarks.agile3d_scannet40.evaluate_feature_cache \
    --benchmark-root "$BENCHMARK_ROOT" \
    --feature-root "$RUN_ROOT/features" \
    --output "$shard" \
    --scene-names "$scene_names" \
    --device cuda:0 \
    --click-workers "$EVAL_CLICK_WORKERS" \
    --selection-mode seeded_component \
    --observation-lift-mode observed_domain \
    --observation-lift-neighbors 3 \
    --observation-lift-maximum-distance-m 0.10 \
    >"$log" 2>&1 &
  pids+=("$!")
  inputs+=("$shard")
done

for pid in "${pids[@]}"; do wait "$pid"; done

bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.agile3d_scannet40.merge_results \
  --benchmark-root "$BENCHMARK_ROOT" \
  --inputs "${inputs[@]}" \
  --output "$RUN_ROOT/results.json" \
  >"$RUN_ROOT/logs/evaluation_merge.log" 2>&1
date -Iseconds >"$FORMAL_TERMINAL"
