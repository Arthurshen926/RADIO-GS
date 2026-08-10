#!/usr/bin/env bash
set -euo pipefail

V21_RUNPY=/root/RADIO-GS/radio_gs/scripts/run_repo_python.sh
V21_SEALER=/root/RADIO-GS/radio_gs/scripts/seal_full_scalar_clean_cohort_region_view_registry.py
V21_MATERIALIZER=/root/RADIO-GS/radio_gs/scripts/materialize_full_scalar_clean_training_shard.py
V21_PARENT=/root/RADIO-GS/paper/artifacts/full_scalar_scannet_clean_24train_8validation_cohort_authority_20260805.json
V21_PARENT_SHA=7f450c09d2db9f55fa8e1efc85905b29b2a7fc63a66169b6ffa123b6dd1c8463

V21_DECL_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/full_scalar_clean_cohort_region_view_registry/scene_declarations
V21_REGISTRY_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/full_scalar_clean_cohort_region_view_registry/pilot_exact4train_2validation
V21_REGISTRY=${V21_REGISTRY_ROOT}/region_view_registry_v1.json
V21_REGISTRY_RECEIPT=${V21_REGISTRY_ROOT}/registry_seal_receipt_v1.json
V21_SHARD_ROOT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/full_scalar_clean_pilot_4train_2validation_v21/shards_v1

[[ "$(sha256sum "$V21_PARENT" | awk '{print $1}')" == "$V21_PARENT_SHA" ]]

V21_REGISTRY_ARGS=()
for V21_SCENE in \
  scene0001_00 scene0002_00 scene0003_00 \
  scene0005_00 scene0004_00 scene0008_00
do
  V21_DECL=${V21_DECL_ROOT}/${V21_SCENE}.json
  [[ -f "$V21_DECL" ]]
  V21_DECL_SHA=$(sha256sum "$V21_DECL" | awk '{print $1}')
  V21_REGISTRY_ARGS+=(
    --scene-declaration "$V21_DECL"
    --expected-scene-declaration-sha256 "$V21_DECL_SHA"
  )
done

bash "$V21_RUNPY" "$V21_SEALER" registry \
  --cohort-authority "$V21_PARENT" \
  --expected-cohort-authority-sha256 "$V21_PARENT_SHA" \
  --pilot-cohort-registry \
  "${V21_REGISTRY_ARGS[@]}" \
  --output-registry "$V21_REGISTRY" \
  --output-receipt "$V21_REGISTRY_RECEIPT"

V21_REGISTRY_SHA=$(sha256sum "$V21_REGISTRY" | awk '{print $1}')

v21_materialize_scene() {
  local V21_SCENE=$1 V21_BASE=$2
  local V21_DECL=${V21_DECL_ROOT}/${V21_SCENE}.json
  local V21_ACCEPTED=${V21_BASE}/accepted_v2_source_only/${V21_SCENE}/accepted_v2_sparse_v2.pt
  local V21_STATE=${V21_BASE}/exact_state/${V21_SCENE}/factorized_primitive_state_v2.pt
  local V21_STATE_JSON=${V21_STATE}.json
  local V21_TEACHER=${V21_BASE}/accepted_v2_source_only/${V21_SCENE}/official_multiview_siglip2_teacher_sparse_v2.pt
  local V21_OUT=${V21_SHARD_ROOT}/${V21_SCENE}
  local V21_ACCEPTED_SHA V21_STATE_SHA V21_TEACHER_SHA
  local V21_FIELD_SHA V21_RADIO_SHA V21_BINDINGS

  [[ -f "$V21_DECL" && -f "$V21_ACCEPTED" && -f "$V21_STATE" ]]
  [[ -f "$V21_STATE_JSON" && -f "$V21_TEACHER" ]]

  V21_BINDINGS=$(
    bash "$V21_RUNPY" - \
      "$V21_DECL" "$V21_STATE_JSON" "$V21_SCENE" \
      "$V21_STATE" "$V21_PARENT_SHA" <<'PY'
import json
import shlex
import sys
from pathlib import Path

declaration_path, sidecar_path, scene_id, state_path, parent_sha = sys.argv[1:]
declaration = json.loads(Path(declaration_path).read_text())
sidecar = json.loads(Path(sidecar_path).read_text())
record = declaration["scene_record"]
artifacts = declaration["artifact_file_sha256"]

assert declaration["cohort_authority_file_sha256"] == parent_sha
assert record["scene_id"] == scene_id
assert record["accepted_region_authority_file_sha256"] == artifacts["accepted_region_authority"]
assert record["factorized_state_file_sha256"] == artifacts["factorized_state"]
assert record["teacher_observation_authority_file_sha256"] == artifacts["teacher_observation_authority"]
assert sidecar["output"]["sha256"] == artifacts["factorized_state"]
assert Path(sidecar["output"]["path"]).resolve() == Path(state_path).resolve()

values = {
    "V21_ACCEPTED_SHA": artifacts["accepted_region_authority"],
    "V21_STATE_SHA": artifacts["factorized_state"],
    "V21_TEACHER_SHA": artifacts["teacher_observation_authority"],
    "V21_FIELD_SHA": sidecar["field_checkpoint_sha256"],
    "V21_RADIO_SHA": sidecar["factorized_radio_cache_sha256"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
  )
  eval "$V21_BINDINGS"

  bash "$V21_RUNPY" "$V21_MATERIALIZER" \
    --cohort-authority "$V21_PARENT" \
    --expected-cohort-authority-sha256 "$V21_PARENT_SHA" \
    --cohort-region-view-registry "$V21_REGISTRY" \
    --expected-cohort-region-view-registry-sha256 "$V21_REGISTRY_SHA" \
    --pilot-cohort-registry \
    --accepted-region-authority "$V21_ACCEPTED" \
    --expected-accepted-region-authority-sha256 "$V21_ACCEPTED_SHA" \
    --factorized-state "$V21_STATE" \
    --expected-factorized-state-sha256 "$V21_STATE_SHA" \
    --expected-field-checkpoint-sha256 "$V21_FIELD_SHA" \
    --expected-factorized-radio-cache-sha256 "$V21_RADIO_SHA" \
    --teacher-observation-authority "$V21_TEACHER" \
    --expected-teacher-observation-authority-sha256 "$V21_TEACHER_SHA" \
    --output-shard "$V21_OUT/training_shard_v1.pt" \
    --output-source-state-manifest "$V21_OUT/source_state_manifest_v1.json" \
    --output-teacher-manifest "$V21_OUT/teacher_manifest_v1.json" \
    --output-receipt "$V21_OUT/materialization_receipt_v1.json"
}

v21_run_pair() {
  local V21_SCENE_A=$1 V21_BASE_A=$2
  local V21_SCENE_B=$3 V21_BASE_B=$4
  local V21_PID_A V21_PID_B V21_STATUS=0

  v21_materialize_scene "$V21_SCENE_A" "$V21_BASE_A" &
  V21_PID_A=$!
  v21_materialize_scene "$V21_SCENE_B" "$V21_BASE_B" &
  V21_PID_B=$!

  wait "$V21_PID_A" || V21_STATUS=$?
  wait "$V21_PID_B" || V21_STATUS=$?
  return "$V21_STATUS"
}

V21_BASE_1=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/full_scalar_clean_scannet_pilot_scene0001_v1/run
V21_BASE_C=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/full_scalar_clean_scannet_cohort_v3

v21_run_pair \
  scene0001_00 "$V21_BASE_1" \
  scene0002_00 "$V21_BASE_C/scene0002_00/run"

v21_run_pair \
  scene0003_00 "$V21_BASE_C/scene0003_00/run" \
  scene0005_00 "$V21_BASE_C/scene0005_00/run"

v21_run_pair \
  scene0004_00 "$V21_BASE_C/scene0004_00/run" \
  scene0008_00 "$V21_BASE_C/scene0008_00/run"
