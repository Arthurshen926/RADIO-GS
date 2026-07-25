#!/usr/bin/env bash

# Build one real high-spatial-fidelity full-.sens AGILE3D field shard.  This
# reuses the already query-free RGB Gaussian geometry and changes only the
# teacher reconstruction route: full-resolution C-RADIO inputs plus native
# official DINOv3/SAM3 adaptor maps before Gaussian registration.  It never
# opens AGILE objects, clicks, masks, labels, or metrics.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SCENE_START_INDEX="${SCENE_START_INDEX:?set the one-scene source slice start}"
SCENE_STOP_INDEX="${SCENE_STOP_INDEX:?set the one-scene source slice stop}"
# An optional, *same-AGILE-source* reference result can make this a strictly
# serial promotion.  The high-fidelity field itself is fully query/label-free,
# however, so it may also be predeclared and built in parallel with the
# project-raw control.  Never point this at a PFPR result or another AGILE
# scorer ablation: those dependencies add idle time without validating field
# fidelity and make the field schedule unnecessarily brittle.
AFTER_RESULT="${AFTER_RESULT:-}"

FIELD_ROOT="${FIELD_ROOT:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/ScanNet-PFIR-Small/field_only_full_sens_agile3_dev_v1}"
PFIR_MATERIALIZATION_REPORT="${PFIR_MATERIALIZATION_REPORT:-$FIELD_ROOT/materialization_report.json}"
BASE_GEOMETRY_ROOT="${BASE_GEOMETRY_ROOT:-output/agile3d_scannet40/full_sens_dev_v2/reconstruction_v1/geometry}"
RUN_ROOT="${RUN_ROOT:-output/agile3d_scannet40/full_sens_hires_official_v1/reconstruction_v1}"
MPR_OBSERVATION_CONTRACT="${MPR_OBSERVATION_CONTRACT:-canonical-full-observation-mpr-v1}"

# If requested, this is a dependency rather than a GPU reservation.  It is
# intentionally optional because both field variants use identical query-free
# source views and neither opens released clicks or labels during construction.
if [[ -n "$AFTER_RESULT" ]]; then
while test ! -s "$AFTER_RESULT"; do
  sleep 30
done
bash radio_gs/scripts/run_repo_python.sh - "$AFTER_RESULT" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("metrics"):
    raise SystemExit("reference full-source AGILE result is incomplete")
for row in payload.get("scene_support", []):
    if float(row.get("continuous_support_fraction", 0.0)) < 0.95:
        raise SystemExit("reference full-source AGILE field did not pass support gate")
PY
fi

# The generic field queue is responsible for real device-idle checks before
# every actual extraction, MPR, or field-training step.  This wrapper supplies
# only a versioned teacher source and an already built geometry root.
# Retain the shared raw canonical representation while allowing the
# capability-first held-out selection to choose a checkpoint that improves
# the official DINO/SAM maps.  This is a fixed field-fidelity budget; no
# AGILE object, click, label, mask, or metric is consulted.
GPU="$GPU" \
FIELD_ROOT="$FIELD_ROOT" \
PFIR_MATERIALIZATION_REPORT="$PFIR_MATERIALIZATION_REPORT" \
RUN_ROOT="$RUN_ROOT" \
GEOMETRY_ROOT="$BASE_GEOMETRY_ROOT" \
FEATURE_ROOT="${FEATURE_ROOT:-$RUN_ROOT/radio_features}" \
CONTRACT_ROOT="${CONTRACT_ROOT:-$RUN_ROOT/render_contracts}" \
FIELD_OUTPUT_ROOT="${FIELD_OUTPUT_ROOT:-$RUN_ROOT/canonical_fields}" \
OBSERVATION_CONTRACT=scannet_full_observation_v1 \
MPR_OBSERVATION_CONTRACT="$MPR_OBSERVATION_CONTRACT" \
FIELD_SELECTION_POLICY=capability_pareto \
BUILD_SEMANTIC=0 \
GEOMETRY_INIT_SELECTION_POLICY=coverage_prefix \
RADIO_RESOLUTION_SCALE=1.0 \
RADIO_BATCH_SIZE=1 \
CAPABILITY_MAP_SOURCE=official_extracted \
RADIO_ADAPTOR_NAMES=dino_v3_7b,sam3 \
FIELD_MAX_MPR_DROP="${FIELD_MAX_MPR_DROP:-0.02}" \
SCENE_START_INDEX="$SCENE_START_INDEX" \
SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
WRITE_TERMINAL=0 \
  bash radio_gs/scripts/run_scannet_pfir_gpu5_queue.sh
