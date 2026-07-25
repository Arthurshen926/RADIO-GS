#!/usr/bin/env bash

# Rebuild one AGILE3D full-.sens canonical field with the explicit 480-view
# MPR contract.  Geometry and official C-RADIO maps are immutable inputs from
# the completed v1 control; only render contracts, MPR caches, and downstream
# canonical-field artifacts are written under a new root.  This is a real
# reconstruction queue, not a GPU reservation, and opens no AGILE labels,
# objects, clicks, masks, or metrics.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SCENE_NAME="${SCENE_NAME:?set one full-.sens AGILE scene ID}"
FIELD_ROOT="${FIELD_ROOT:?set the 480-view label-free field-source root}"
V1_RUN_ROOT="${V1_RUN_ROOT:?set the completed v1 reconstruction root}"
RUN_ROOT="${RUN_ROOT:?set a new v2 reconstruction root}"
AFTER_FIELD="${AFTER_FIELD:-$V1_RUN_ROOT/canonical_fields/$SCENE_NAME/canonical_mpr_v2.pt}"

if [[ "$RUN_ROOT" == "$V1_RUN_ROOT" ]]; then
  echo "RUN_ROOT must differ from V1_RUN_ROOT; v2 must not overwrite the control" >&2
  exit 2
fi

# Resolve the one-scene source slice from the immutable materialization report
# instead of trusting a hand-written index.  The audit reads only the public
# field-source manifest and proves that a 480-view coverage prefix is present.
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
if int(row.get("field_frame_count", 0)) < 480 or int(contract.get("field_frame_count", 0)) < 480:
    raise SystemExit(f"{scene}: MPR v2 requires an independently materialized 480-view source")
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

# A v2 resume is safe only when its existing raw cache itself declares v2.
# This prevents a typo in RUN_ROOT from silently treating a v1 control cache
# as the 480-view branch.
existing_raw="$RUN_ROOT/canonical_fields/$SCENE_NAME/raw_radio_mpr.pt"
if [[ -s "$existing_raw" ]]; then
  EXISTING_RAW="$existing_raw" bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import os
import torch

payload = torch.load(os.environ["EXISTING_RAW"], map_location="cpu")
metadata = dict(payload.get("metadata", {}))
declared = dict(metadata.get("observation_lifting_contract", {}))
if declared.get("name") != "canonical-full-observation-mpr-v2":
    raise SystemExit("refusing to resume a non-v2 raw MPR cache")
PY
fi

# Do not overlap a control field on the same physical GPU. While the control
# artifact is absent this queue is CPU-only; once it exists the underlying
# field queue performs its own genuine device-idle check before every GPU step.
while [[ ! -s "$AFTER_FIELD" ]]; do
  sleep 30
done

exec env \
  GPU="$GPU" \
  FIELD_ROOT="$FIELD_ROOT" \
  PFIR_MATERIALIZATION_REPORT="$FIELD_ROOT/materialization_report.json" \
  BASE_GEOMETRY_ROOT="$V1_RUN_ROOT/geometry" \
  RUN_ROOT="$RUN_ROOT" \
  FEATURE_ROOT="$V1_RUN_ROOT/radio_features" \
  CONTRACT_ROOT="$RUN_ROOT/render_contracts" \
  FIELD_OUTPUT_ROOT="$RUN_ROOT/canonical_fields" \
  MPR_OBSERVATION_CONTRACT="canonical-full-observation-mpr-v2" \
  FIELD_SELECTION_POLICY="capability_pareto" \
  FIELD_MAX_MPR_DROP="${FIELD_MAX_MPR_DROP:-0.02}" \
  SCENE_START_INDEX="$SCENE_START_INDEX" \
  SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
  WRITE_TERMINAL=0 \
  bash radio_gs/scripts/run_agile_full_sens_hires_queue.sh
