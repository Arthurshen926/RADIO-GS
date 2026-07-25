#!/usr/bin/env bash

# Promote one AGILE3D full-.sens field from the frozen 480-view MPR-v2 rung
# to an independently materialized 960-view MPR-v3 rung.  This queue is
# deliberately source/field-only: it opens no AGILE objects, clicks, masks,
# labels, or metrics.  It may optionally wait on the label-free v2 support
# audit and proceeds only when that fixed gate actually failed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SCENE_NAME="${SCENE_NAME:?set one full-.sens AGILE scene ID}"
FIELD_ROOT="${FIELD_ROOT:?set the independently materialized 960-view label-free source root}"
RUN_ROOT="${RUN_ROOT:?set a new v3 reconstruction root}"
AFTER_SUPPORT_PRECHECK="${AFTER_SUPPORT_PRECHECK:-}"
# A label-free all-Gaussian support audit decides this before a v3 queue is
# started.  Reusing geometry is valid only when that geometry alone already
# covers the fixed official domain; otherwise v3 must reconstruct it from the
# larger RGB-D source as well as rebuilding MPR/canonical features.
REBUILD_GEOMETRY="${REBUILD_GEOMETRY:-0}"
BASE_GEOMETRY_ROOT="${BASE_GEOMETRY_ROOT:-}"
GEOMETRY_CEILING_AUDIT="${GEOMETRY_CEILING_AUDIT:?set the label-free all-Gaussian geometry ceiling audit}"

case "$REBUILD_GEOMETRY" in
  0|false|False|FALSE)
    REBUILD_GEOMETRY=0
    ;;
  1|true|True|TRUE)
    REBUILD_GEOMETRY=1
    ;;
  *)
    echo "REBUILD_GEOMETRY must be 0/1 or true/false" >&2
    exit 2
    ;;
esac

# The semantic-vs-geometry branch must be justified by an artifact that was
# produced without object lists or labels.  This prevents a caller from
# choosing the cheaper geometry-reuse route after seeing AGILE metrics.
while [[ ! -s "$GEOMETRY_CEILING_AUDIT" ]]; do
  sleep 30
done
GEOMETRY_CEILING_AUDIT="$GEOMETRY_CEILING_AUDIT" \
SCENE_NAME="$SCENE_NAME" REBUILD_GEOMETRY="$REBUILD_GEOMETRY" \
  bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["GEOMETRY_CEILING_AUDIT"]).read_text(encoding="utf-8"))
if payload.get("mode") != "label_free_all_gaussian_geometry_support_ceiling":
    raise SystemExit("geometry admission dependency has an invalid mode")
protocol = dict(payload.get("protocol", {}))
if protocol.get("labels_opened") is not False or protocol.get("object_list_opened") is not False:
    raise SystemExit("geometry admission dependency is not label-free")
scene = str(os.environ["SCENE_NAME"])
rows = [row for row in payload.get("scene_geometry_support", []) if str(row.get("scene_id")) == scene]
if len(rows) != 1:
    raise SystemExit(f"geometry admission audit has no unique row for {scene}")
expected = bool(rows[0].get("geometry_rebuild_required", False))
requested = bool(int(os.environ["REBUILD_GEOMETRY"]))
if expected != requested:
    raise SystemExit(
        f"{scene}: geometry branch disagrees with label-free ceiling "
        f"(required={expected}, requested={requested})"
    )
PY

# If supplied, the dependency must be the label-free support record produced
# by the v2 field.  A passed v2 field exits cleanly rather than spending GPU
# time on an unneeded promotion; a failed record is sufficient because no
# released object list or label was opened to produce it.
if [[ -n "$AFTER_SUPPORT_PRECHECK" ]]; then
  while [[ ! -s "$AFTER_SUPPORT_PRECHECK" ]]; do
    sleep 30
  done
  set +e
  AFTER_SUPPORT_PRECHECK="$AFTER_SUPPORT_PRECHECK" SCENE_NAME="$SCENE_NAME" \
    bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["AFTER_SUPPORT_PRECHECK"]).read_text(encoding="utf-8"))
if payload.get("mode") != "label_free_field_support_preflight":
    raise SystemExit("v2 promotion dependency is not a label-free support audit")
protocol = dict(payload.get("protocol", {}))
if protocol.get("labels_opened") is not False or protocol.get("object_list_opened") is not False:
    raise SystemExit("v2 promotion dependency is not label-free")
scene = str(os.environ["SCENE_NAME"])
rows = [row for row in payload.get("scene_support", []) if str(row.get("scene_id")) == scene]
if len(rows) != 1:
    raise SystemExit(f"v2 support audit has no unique row for {scene}")
threshold = float(protocol.get("minimum_support_fraction", 0.0))
actual = float(rows[0].get("continuous_support_fraction", 0.0))
if threshold <= 0:
    raise SystemExit("v2 support audit has no valid fixed support threshold")
if actual >= threshold:
    print(f"{scene}: v2 support {actual:.6f} already passes {threshold:.6f}; v3 is unnecessary")
    raise SystemExit(3)
print(f"{scene}: promoting label-free v2 support {actual:.6f} below fixed gate {threshold:.6f}")
PY
  support_status=$?
  set -e
  if (( support_status == 3 )); then
    exit 0
  fi
  if (( support_status != 0 )); then
    exit "$support_status"
  fi
fi

# Resolve the source slice from the immutable source report.  The source must
# already contain a full 960-view coverage-ranked prefix; the MPR contract
# cannot manufacture extra observations from a 480-view field.
read -r SCENE_START_INDEX SCENE_STOP_INDEX < <(
  FIELD_ROOT="$FIELD_ROOT" SCENE_NAME="$SCENE_NAME" \
    bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["FIELD_ROOT"])
scene = str(os.environ["SCENE_NAME"])
report = json.loads((root / "materialization_report.json").read_text(encoding="utf-8"))
rows = sorted(report.get("scenes", []), key=lambda row: (row["field_frame_count"], row["scene_id"]))
matches = [(index, row) for index, row in enumerate(rows) if str(row["scene_id"]) == scene]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one source report row for {scene}")
index, row = matches[0]
contract = json.loads((root / scene / "field_source_contract.json").read_text(encoding="utf-8"))
if str(contract.get("field_contract_version", "")) != "scannet_full_observation_v1":
    raise SystemExit(f"{scene}: source is not scannet_full_observation_v1")
if int(row.get("field_frame_count", 0)) < 960 or int(contract.get("field_frame_count", 0)) < 960:
    raise SystemExit(f"{scene}: MPR v3 requires an independently materialized 960-view source")
for key in (
    "uses_private_anchor",
    "uses_private_depth_pixel",
    "uses_instances_or_semantic_labels",
    "contains_instance_or_label_directories",
):
    if bool(contract.get(key, False)):
        raise SystemExit(f"{scene}: source is not label/query free ({key})")
print(index, index + 1)
PY
)

if (( REBUILD_GEOMETRY )); then
  GEOMETRY_ROOT="$RUN_ROOT/geometry"
  # The earlier 480-view geometry ceiling is below the fixed support gate, so
  # a 50-frame / 200k primitive bootstrap would merely preserve its blind
  # surfaces.  These are fixed source-fidelity construction budgets shared by
  # every geometry-rebuild scene, not per-scene score tuning.
  GEOMETRY_INIT_FRAMES="${GEOMETRY_INIT_FRAMES:-240}"
  GEOMETRY_MAX_POINTS="${GEOMETRY_MAX_POINTS:-300000}"
  # The fresh geometry must clear the same fixed support definition before
  # DINO/SAM MPR begins.  This is deliberately an all-Gaussian, label-free
  # construction gate; it is not an AGILE score or object filter.
  GEOMETRY_SUPPORT_GATE_DIR="$RUN_ROOT/geometry_support_gate"
  GEOMETRY_SUPPORT_BENCHMARK_ROOT="/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet"
else
  if [[ -z "$BASE_GEOMETRY_ROOT" ]] || [[ ! -s "$BASE_GEOMETRY_ROOT/$SCENE_NAME/point_cloud/iteration_15000/point_cloud.ply" ]]; then
    echo "missing frozen base geometry for $SCENE_NAME: $BASE_GEOMETRY_ROOT" >&2
    exit 2
  fi
  GEOMETRY_ROOT="$BASE_GEOMETRY_ROOT"
  GEOMETRY_INIT_FRAMES="${GEOMETRY_INIT_FRAMES:-50}"
  GEOMETRY_MAX_POINTS="${GEOMETRY_MAX_POINTS:-200000}"
  GEOMETRY_SUPPORT_GATE_DIR=""
  GEOMETRY_SUPPORT_BENCHMARK_ROOT=""
fi

# A resumed v3 output must itself declare v3.  This prevents a lower-budget
# raw MPR cache from acquiring a v3 path name after an interrupted run.
existing_raw="$RUN_ROOT/canonical_fields/$SCENE_NAME/raw_radio_mpr.pt"
if [[ -s "$existing_raw" ]]; then
  EXISTING_RAW="$existing_raw" bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import os
import torch

payload = torch.load(os.environ["EXISTING_RAW"], map_location="cpu")
metadata = dict(payload.get("metadata", {}))
declared = dict(metadata.get("observation_lifting_contract", {}))
if declared.get("name") != "canonical-full-observation-mpr-v3":
    raise SystemExit("refusing to resume a non-v3 raw MPR cache")
PY
fi

exec env \
  GPU="$GPU" \
  FIELD_ROOT="$FIELD_ROOT" \
  PFIR_MATERIALIZATION_REPORT="$FIELD_ROOT/materialization_report.json" \
  BASE_GEOMETRY_ROOT="$GEOMETRY_ROOT" \
  RUN_ROOT="$RUN_ROOT" \
  FEATURE_ROOT="$RUN_ROOT/radio_features" \
  CONTRACT_ROOT="$RUN_ROOT/render_contracts" \
  FIELD_OUTPUT_ROOT="$RUN_ROOT/canonical_fields" \
  GEOMETRY_INIT_FRAMES="$GEOMETRY_INIT_FRAMES" \
  GEOMETRY_MAX_POINTS="$GEOMETRY_MAX_POINTS" \
  GEOMETRY_SUPPORT_GATE_DIR="$GEOMETRY_SUPPORT_GATE_DIR" \
  GEOMETRY_SUPPORT_BENCHMARK_ROOT="$GEOMETRY_SUPPORT_BENCHMARK_ROOT" \
  GEOMETRY_SUPPORT_MINIMUM_FRACTION="0.95" \
  MPR_OBSERVATION_CONTRACT="canonical-full-observation-mpr-v3" \
  FIELD_SELECTION_POLICY="capability_pareto" \
  FIELD_MAX_MPR_DROP="${FIELD_MAX_MPR_DROP:-0.02}" \
  SCENE_START_INDEX="$SCENE_START_INDEX" \
  SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
  WRITE_TERMINAL=0 \
  bash radio_gs/scripts/run_agile_full_sens_hires_queue.sh
