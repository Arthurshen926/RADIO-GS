#!/usr/bin/env bash

# Build one ScanNet-PFPR v2 field from a distinct 960-view, query-held-out
# full-.sens source.  This is the source-fidelity repair after a lower-budget
# field fails the fixed public-candidate continuous-support gate.  It never
# opens a private PFPR anchor, pose/depth query record, mask, label, instance,
# rank, or metric.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SCENE_NAME="${SCENE_NAME:?set one ScanNet-PFPR v2 scene ID}"
SOURCE_ROOT="${SOURCE_ROOT:?set the 960-view PFPR field-source root}"
BENCHMARK_DIR="${BENCHMARK_DIR:-output/scannet_pfpr_small_v2/test_v2_r1}"
RUN_ROOT="${RUN_ROOT:?set a distinct output root for the v3 MPR field}"
REQUIRE_GEOMETRY_SUPPORT_GATE="${REQUIRE_GEOMETRY_SUPPORT_GATE:-1}"
# A scheduling dependency is an artifact only.  It is not an evaluator score
# and performs no GPU work while absent.
AFTER_ARTIFACT="${AFTER_ARTIFACT:-${AFTER_RESULT:-}}"

while [[ ! -s "$SOURCE_ROOT/materialization_report.json" ]]; do
  sleep 30
done

# Resolve a stable one-scene shard and prove that this source has both the
# immutable PFPR held-out-frame commitment and the v3 view budget.  This reads
# only the public manifest and field-side provenance; evaluator-private anchor
# data remain unopened.
read -r SCENE_START_INDEX SCENE_STOP_INDEX < <(
  SOURCE_ROOT="$SOURCE_ROOT" BENCHMARK_DIR="$BENCHMARK_DIR" SCENE_NAME="$SCENE_NAME" \
    bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

source_root = Path(os.environ["SOURCE_ROOT"])
benchmark_dir = Path(os.environ["BENCHMARK_DIR"])
scene = str(os.environ["SCENE_NAME"])
report = json.loads((source_root / "materialization_report.json").read_text(encoding="utf-8"))
rows = sorted(report.get("scenes", []), key=lambda row: (row["field_frame_count"], row["scene_id"]))
matches = [(index, row) for index, row in enumerate(rows) if str(row["scene_id"]) == scene]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one PFPR source report row for {scene}")
index, row = matches[0]
contract = json.loads((source_root / scene / "field_source_contract.json").read_text(encoding="utf-8"))
if str(contract.get("field_contract_version", "")) != "scannet_full_observation_pfpr_queryheldout_v1":
    raise SystemExit(f"{scene}: not a PFPR query-held-out full-observation source")
if int(row.get("field_frame_count", 0)) < 960 or int(contract.get("field_frame_count", 0)) < 960:
    raise SystemExit(f"{scene}: MPR v3 requires an independently materialized 960-view source")
public = json.loads((benchmark_dir / "manifest.public.json").read_text(encoding="utf-8"))
if public.get("benchmark_version") != "scannet-pfpr-small-v2":
    raise SystemExit("PFPR v3 queue requires the immutable PFPR v2 public manifest")
expected = {
    str(item["scene_id"]): str(item.get("excluded_query_source_frame_ids_sha256", ""))
    for item in public.get("scene_domains", [])
}.get(scene, "")
actual = str(contract.get("excluded_query_source_frame_ids_sha256", ""))
if not expected or expected != actual:
    raise SystemExit(f"{scene}: public/source held-out-frame commitments differ")
for key in (
    "uses_private_anchor",
    "uses_private_depth_pixel",
    "uses_instances_or_semantic_labels",
    "contains_instance_or_label_directories",
):
    if bool(contract.get(key, False)):
        raise SystemExit(f"{scene}: source is not query/label free ({key})")
print(index, index + 1)
PY
)

# A v3 directory cannot silently resume a lower-fidelity raw MPR cache.
existing_raw="$RUN_ROOT/reconstruction_v1/canonical_fields/$SCENE_NAME/raw_radio_mpr.pt"
if [[ -s "$existing_raw" ]]; then
  EXISTING_RAW="$existing_raw" bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import os
import torch

payload = torch.load(os.environ["EXISTING_RAW"], map_location="cpu")
metadata = dict(payload.get("metadata", {}))
declared = dict(metadata.get("observation_lifting_contract", {}))
if declared.get("name") != "canonical-full-observation-mpr-v3":
    raise SystemExit("refusing to resume a non-v3 PFPR raw MPR cache")
PY
fi

if [[ -n "$AFTER_ARTIFACT" ]]; then
  while [[ ! -s "$AFTER_ARTIFACT" ]]; do
    sleep 30
  done
fi

case "$REQUIRE_GEOMETRY_SUPPORT_GATE" in
  1|true|True|TRUE)
    PFPR_GEOMETRY_GATE_DIR="$RUN_ROOT/reconstruction_v1/geometry_support_gate"
    PFPR_GEOMETRY_GATE_BENCHMARK="$BENCHMARK_DIR"
    ;;
  0|false|False|FALSE)
    PFPR_GEOMETRY_GATE_DIR=""
    PFPR_GEOMETRY_GATE_BENCHMARK=""
    ;;
  *)
    echo "REQUIRE_GEOMETRY_SUPPORT_GATE must be 0/1 or true/false" >&2
    exit 2
    ;;
esac

# The reconstruction budget is fixed for every v3 PFPR scene.  In particular,
# do not reduce a 960-view source to the legacy 50-frame/200k geometry
# bootstrap before the standard queue lifts its full coverage-ranked MPR.
exec env \
  GPU="$GPU" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  BENCHMARK_DIR="$BENCHMARK_DIR" \
  RUN_ROOT="$RUN_ROOT/reconstruction_v1" \
  MPR_OBSERVATION_CONTRACT="canonical-full-observation-mpr-v3" \
  GEOMETRY_INIT_FRAMES=240 \
  GEOMETRY_MAX_POINTS=300000 \
  GEOMETRY_INIT_SELECTION_POLICY="coverage_prefix" \
  PFPR_GEOMETRY_SUPPORT_GATE_DIR="$PFPR_GEOMETRY_GATE_DIR" \
  PFPR_GEOMETRY_SUPPORT_BENCHMARK_DIR="$PFPR_GEOMETRY_GATE_BENCHMARK" \
  PFPR_GEOMETRY_SUPPORT_MINIMUM_FRACTION=0.95 \
  FIELD_MAX_MPR_DROP="${FIELD_MAX_MPR_DROP:-0.02}" \
  GPU_IDLE_CONFIRMATIONS="${GPU_IDLE_CONFIRMATIONS:-1}" \
  SCENE_START_INDEX="$SCENE_START_INDEX" \
  SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
  WRITE_TERMINAL=0 \
  bash radio_gs/scripts/run_pfpr_v2_full_sens_field_queue.sh
