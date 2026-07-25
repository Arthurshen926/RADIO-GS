#!/usr/bin/env bash

# Build one ScanNet-PFPR v2 field from a distinct 480-view, query-held-out
# full-.sens source. This is a real reconstruction queue: it waits for source
# materialization and a genuinely idle device, then delegates to the standard
# field builder. It never opens a private PFPR anchor, depth pixel, label,
# mask, instance, or rank metric.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SCENE_NAME="${SCENE_NAME:?set one ScanNet-PFPR v2 scene ID}"
SOURCE_ROOT="${SOURCE_ROOT:?set the 480-view PFPR field-source root}"
BENCHMARK_DIR="${BENCHMARK_DIR:-output/scannet_pfpr_small_v2/test_v2_r1}"
RUN_ROOT="${RUN_ROOT:?set a distinct output root for the v2 MPR field}"
# A scheduling dependency is an artifact, not necessarily a benchmark score.
# ``AFTER_RESULT`` remains a backwards-compatible alias, but a PFPR field
# must never be made contingent on an unrelated AGILE *formal result*: a
# failed AGILE support gate intentionally emits no such result.
AFTER_ARTIFACT="${AFTER_ARTIFACT:-${AFTER_RESULT:-}}"

while [[ ! -s "$SOURCE_ROOT/materialization_report.json" ]]; do
  sleep 30
done

# Resolve a stable one-scene shard and prove that the public held-out-frame
# commitment still matches. The existing PFPR field queue repeats this audit
# immediately before construction; this early check only prevents a dormant
# queue from being misconfigured for a 240-view source.
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
if int(row.get("field_frame_count", 0)) < 480 or int(contract.get("field_frame_count", 0)) < 480:
    raise SystemExit(f"{scene}: MPR v2 requires an independently materialized 480-view source")
public = json.loads((benchmark_dir / "manifest.public.json").read_text(encoding="utf-8"))
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

# Resuming is allowed only for a cache that already declares the v2 contract;
# this prevents an old 240-view raw MPR cache from being interpreted as this
# source-fidelity branch simply because its directory was reused.
existing_raw="$RUN_ROOT/reconstruction_v1/canonical_fields/$SCENE_NAME/raw_radio_mpr.pt"
if [[ -s "$existing_raw" ]]; then
  EXISTING_RAW="$existing_raw" bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import os
import torch

payload = torch.load(os.environ["EXISTING_RAW"], map_location="cpu")
metadata = dict(payload.get("metadata", {}))
declared = dict(metadata.get("observation_lifting_contract", {}))
if declared.get("name") != "canonical-full-observation-mpr-v2":
    raise SystemExit("refusing to resume a non-v2 PFPR raw MPR cache")
PY
fi

# An optional completed artifact can serialize this field behind a prior real
# GPU job. It is a resource dependency only; no device is touched while the
# prerequisite is absent. A label-free AGILE support audit is a valid
# dependency because it is produced whether that field passes or fails.
if [[ -n "$AFTER_ARTIFACT" ]]; then
  while [[ ! -s "$AFTER_ARTIFACT" ]]; do
    sleep 30
  done
fi

exec env \
  GPU="$GPU" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  BENCHMARK_DIR="$BENCHMARK_DIR" \
  RUN_ROOT="$RUN_ROOT/reconstruction_v1" \
  MPR_OBSERVATION_CONTRACT="canonical-full-observation-mpr-v2" \
  SCENE_START_INDEX="$SCENE_START_INDEX" \
  SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
  WRITE_TERMINAL=0 \
  bash radio_gs/scripts/run_pfpr_v2_full_sens_field_queue.sh
