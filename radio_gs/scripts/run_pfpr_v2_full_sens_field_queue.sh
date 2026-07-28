#!/usr/bin/env bash

# Build a query-held-out ScanNet-PFPR v2 field shard.  This is the field-side
# half of the public pose-free patch-to-3D-point protocol: every selected
# source frame is checked against the public one-way held-out-frame commitment
# before any geometry, RADIO feature, MPR, or canonical field artifact is
# created.  No private anchors, poses, depths, masks, labels, or instance IDs
# are opened by this queue.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SCENE_START_INDEX="${SCENE_START_INDEX:?set the inclusive source-report scene index}"
SCENE_STOP_INDEX="${SCENE_STOP_INDEX:?set the exclusive source-report scene index}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small/field_only_full_sens_pfpr_v2_queryheldout_v1}"
BENCHMARK_DIR="${BENCHMARK_DIR:-output/scannet_pfpr_small_v2/test_v2_r1}"
RUN_ROOT="${RUN_ROOT:-output/scannet_pfpr_small_v2/full_sens_official_v2_r1/reconstruction_v1}"
# Optional artifact dependency for a real GPU work chain.  It is never a
# reservation: while the prerequisite is absent this process does no GPU work.
AFTER_FIELD="${AFTER_FIELD:-}"

# Fail before consuming a GPU if the materialized field source does not prove
# exclusion of exactly the public release's held-out query frames.  The check
# intentionally reads only the public manifest and the field-source contracts,
# both of which contain hashes rather than query registration information.
bash radio_gs/scripts/run_repo_python.sh - \
  "$SOURCE_ROOT" "$BENCHMARK_DIR" "$SCENE_START_INDEX" "$SCENE_STOP_INDEX" <<'PY'
import json
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
benchmark_dir = Path(sys.argv[2])
start, stop = int(sys.argv[3]), int(sys.argv[4])
public = json.loads((benchmark_dir / "manifest.public.json").read_text(encoding="utf-8"))
if public.get("benchmark_version") != "scannet-pfpr-small-v2":
    raise SystemExit("PFPR v2 field queue requires the immutable v2 public manifest")
public_hashes = {
    str(row["scene_id"]): str(row.get("excluded_query_source_frame_ids_sha256", ""))
    for row in public.get("scene_domains", [])
}
if not public_hashes or any(not value for value in public_hashes.values()):
    raise SystemExit("PFPR v2 public manifest lacks held-out-frame commitments")
report = json.loads((source_root / "materialization_report.json").read_text(encoding="utf-8"))
rows = sorted(report.get("scenes", []), key=lambda row: (row["field_frame_count"], row["scene_id"]))
if start < 0 or stop <= start or stop > len(rows):
    raise SystemExit(f"invalid PFPR source slice [{start}:{stop}]")
for row in rows[start:stop]:
    scene = str(row["scene_id"])
    contract = json.loads((source_root / scene / "field_source_contract.json").read_text(encoding="utf-8"))
    if contract.get("field_contract_version") != "scannet_full_observation_pfpr_queryheldout_v1":
        raise SystemExit(f"{scene}: field source is not the PFPR held-out full-observation contract")
    if any(bool(contract.get(key, False)) for key in (
        "uses_private_anchor", "uses_private_depth_pixel", "uses_instances_or_semantic_labels",
    )):
        raise SystemExit(f"{scene}: PFPR field source exposes private/evaluator data")
    actual = str(contract.get("excluded_query_source_frame_ids_sha256", ""))
    expected = public_hashes.get(scene, "")
    if not expected or not actual or actual != expected:
        raise SystemExit(
            f"{scene}: source/public held-out-frame commitments differ "
            f"({actual[:12]} != {expected[:12]})"
        )
PY

if [[ -n "$AFTER_FIELD" ]]; then
  while [[ ! -s "$AFTER_FIELD" ]]; do
    sleep 30
  done
fi

# PFPR evaluates DINO centre descriptors.  Reconstruct the same frozen
# official DINO/SAM capabilities at native spatial resolution, retaining SAM
# as a shared field/interface view rather than adding a PFPR-specific head.
# The generic queue waits for an actually idle GPU before each genuine stage;
# this wrapper is not an occupancy workload.
# PFPR reads the same canonical DINO capability as AGILE3D.  Use the shared,
# label-free retention budget rather than requiring a zero raw-MPR change that
# can discard every held-out capability-fidelity improvement.
exec env \
  GPU="$GPU" \
  FIELD_ROOT="$SOURCE_ROOT" \
  PFIR_MATERIALIZATION_REPORT="$SOURCE_ROOT/materialization_report.json" \
  RUN_ROOT="$RUN_ROOT" \
  OBSERVATION_CONTRACT="scannet_full_observation_pfpr_queryheldout_v1" \
  MPR_OBSERVATION_CONTRACT="${MPR_OBSERVATION_CONTRACT:-canonical-full-observation-mpr-v1}" \
  FIELD_SELECTION_POLICY="capability_pareto" \
  GEOMETRY_INIT_SELECTION_POLICY="coverage_prefix" \
  RADIO_RESOLUTION_SCALE=1.0 \
  RADIO_BATCH_SIZE=1 \
  CAPABILITY_MAP_SOURCE="official_extracted" \
  RADIO_ADAPTOR_NAMES="dino_v3_7b,sam3" \
  FIELD_MAX_MPR_DROP="${FIELD_MAX_MPR_DROP:-0.02}" \
  BUILD_SEMANTIC=0 \
  SCENE_START_INDEX="$SCENE_START_INDEX" \
  SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
  WRITE_TERMINAL=0 \
  bash radio_gs/scripts/run_scannet_pfir_gpu5_queue.sh
