#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-5}"
FIELD_ROOT="${FIELD_ROOT:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/ScanNet-PFIR-Small/field_only_test_v1}"
PFIR_MATERIALIZATION_REPORT="${PFIR_MATERIALIZATION_REPORT:-$FIELD_ROOT/materialization_report.json}"
RUN_ROOT="${RUN_ROOT:-output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1}"
GEOMETRY_ROOT="${GEOMETRY_ROOT:-$RUN_ROOT/geometry}"
FEATURE_ROOT="${FEATURE_ROOT:-$RUN_ROOT/radio_features}"
CONTRACT_ROOT="${CONTRACT_ROOT:-$RUN_ROOT/render_contracts}"
FIELD_OUTPUT_ROOT="${FIELD_OUTPUT_ROOT:-$RUN_ROOT/canonical_fields}"
GS_ITERS="${GS_ITERS:-15000}"
GEOMETRY_INIT_FRAMES="${GEOMETRY_INIT_FRAMES:-50}"
GEOMETRY_INIT_STRIDE="${GEOMETRY_INIT_STRIDE:-8}"
GEOMETRY_MAX_POINTS="${GEOMETRY_MAX_POINTS:-200000}"
# A full-.sens materializer writes a query-free greedy coverage ranking.  The
# RGB-D folders remain numeric-sorted for normal training, but geometry
# initialization can explicitly restore this ranking so it does not collapse
# a dense source back into the first temporal subset.
GEOMETRY_INIT_SELECTION_POLICY="${GEOMETRY_INIT_SELECTION_POLICY:-uniform}"
RADIO_REPO="${RADIO_REPO:-/root/RADIO}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
RADIO_RESOLUTION_SCALE="${RADIO_RESOLUTION_SCALE:-0.5}"
RADIO_BATCH_SIZE="${RADIO_BATCH_SIZE:-2}"
# The default preserves the frozen low-resolution baseline.  The explicit
# official_extracted route materializes C-RADIO's native DINO/SAM maps before
# they are resampled for Gaussian registration; it is a versioned field
# fidelity experiment, not an evaluator or query-protocol change.
CAPABILITY_MAP_SOURCE="${CAPABILITY_MAP_SOURCE:-project_raw}"
RADIO_ADAPTOR_NAMES="${RADIO_ADAPTOR_NAMES:-dino_v3_7b,sam3}"
# DINO and SAM cache lifting share an immutable registration-responsibility
# sidecar but otherwise write disjoint artifacts.  A caller with a second
# genuinely idle device can opt into concurrent capability lifting.  Keep the
# default empty so historic single-GPU queues retain their exact ordering.
CAPABILITY_MPR_PARALLEL_GPU="${CAPABILITY_MPR_PARALLEL_GPU:-}"
READOUT_CHECKPOINT="${READOUT_CHECKPOINT:-output/scannet_pfir_small_v1/readout_v3/global_surface_region_readout_h256_clean.pt}"
FIDELITY_VALIDATION_VIEWS="${FIDELITY_VALIDATION_VIEWS:-4}"
OBSERVATION_CONTRACT="${OBSERVATION_CONTRACT:-field_only_dense_rgbd_v1}"
# Preserve the historical MPR policy for legacy/dense sources.  A full ScanNet
# source receives a distinct, auditable MPR contract that consumes only its
# own query-free coverage ranking.
if [[ -z "${MPR_OBSERVATION_CONTRACT+x}" ]]; then
  case "$OBSERVATION_CONTRACT" in
    scannet_full_observation_v1|scannet_full_observation_pfpr_queryheldout_v1)
      MPR_OBSERVATION_CONTRACT="canonical-full-observation-mpr-v1"
      ;;
    *)
      MPR_OBSERVATION_CONTRACT="canonical-mpr-v1"
      ;;
  esac
fi
# Canonical v2 is normally selected on raw RADIO fidelity.  Capability-first
# selection is a named, label-free promotion option for interfaces (such as
# AGILE3D/PFPR) whose prediction only reads official DINO/SAM capability views.
# Keep the historic default unless an experiment declares the alternate policy.
FIELD_SELECTION_POLICY="${FIELD_SELECTION_POLICY:-validation}"
FIELD_MAX_CAPABILITY_DROP="${FIELD_MAX_CAPABILITY_DROP:-0.002}"
FIELD_MAX_MPR_DROP="${FIELD_MAX_MPR_DROP:-0.0}"
# Optional, label-free geometry admission for a full-ScanNet construction
# ladder.  It is intentionally disabled for ordinary PFIR/ScanNet queues;
# when supplied by AGILE v3 it runs immediately after raw MPR has bound the
# fresh geometry to its source contract and before expensive DINO/SAM lifting.
GEOMETRY_SUPPORT_GATE_DIR="${GEOMETRY_SUPPORT_GATE_DIR:-}"
GEOMETRY_SUPPORT_BENCHMARK_ROOT="${GEOMETRY_SUPPORT_BENCHMARK_ROOT:-}"
GEOMETRY_SUPPORT_MINIMUM_FRACTION="${GEOMETRY_SUPPORT_MINIMUM_FRACTION:-0.95}"
# PFPR exposes a different, public geometry-only candidate domain.  This
# optional gate therefore has its own typed inputs rather than reusing the
# AGILE evaluator's scene PLY path or silently opening PFPR private anchors.
PFPR_GEOMETRY_SUPPORT_GATE_DIR="${PFPR_GEOMETRY_SUPPORT_GATE_DIR:-}"
PFPR_GEOMETRY_SUPPORT_BENCHMARK_DIR="${PFPR_GEOMETRY_SUPPORT_BENCHMARK_DIR:-}"
PFPR_GEOMETRY_SUPPORT_MINIMUM_FRACTION="${PFPR_GEOMETRY_SUPPORT_MINIMUM_FRACTION:-0.95}"
# These are field-reconstruction parameters, shared by all DINO/SAM query
# interfaces.  The zero-balance default preserves the frozen baseline; a
# named nonzero value can test whether teacher-defined boundary relations need
# equal tail weight rather than being overwhelmed by smooth interior pairs.
CAPABILITY_LOCAL_AFFINITY_WEIGHT="${CAPABILITY_LOCAL_AFFINITY_WEIGHT:-0.25}"
CAPABILITY_LOCAL_RADIUS="${CAPABILITY_LOCAL_RADIUS:-1}"
CAPABILITY_LOCAL_BALANCE_QUANTILE="${CAPABILITY_LOCAL_BALANCE_QUANTILE:-0.0}"
# Direct AGILE3D and PFPR only require canonical DINO/SAM capability banks and
# the shared support graph.  A semantic readout may be skipped for a focused
# field-promotion run without changing field geometry or capability training.
BUILD_SEMANTIC="${BUILD_SEMANTIC:-1}"
# A second worker may take a disjoint scene suffix when another real GPU
# becomes available. Defaults preserve the original complete serial queue.
SCENE_START_INDEX="${SCENE_START_INDEX:-0}"
SCENE_STOP_INDEX="${SCENE_STOP_INDEX:-}"
WRITE_TERMINAL="${WRITE_TERMINAL:-1}"
# A full 312-scene run cannot retain every native RADIO map and training MPR
# tensor on the shared volume.  This opt-in mode preserves every artifact used
# by direct canonical inference plus the raw-MPR provenance sidecar, then
# removes only reproducible field-training intermediates after a scene has
# completed.  A per-scene terminal makes the queue resume-safe after pruning.
PRUNE_REGENERABLE_INTERMEDIATES="${PRUNE_REGENERABLE_INTERMEDIATES:-0}"
GPU_IDLE_CONFIRMATIONS="${GPU_IDLE_CONFIRMATIONS:-2}"

case "$PRUNE_REGENERABLE_INTERMEDIATES" in
  0|false|False|FALSE)
    PRUNE_REGENERABLE_INTERMEDIATES=0
    ;;
  1|true|True|TRUE)
    PRUNE_REGENERABLE_INTERMEDIATES=1
    if [[ "$BUILD_SEMANTIC" == "1" || "$BUILD_SEMANTIC" == "true" || "$BUILD_SEMANTIC" == "TRUE" ]]; then
      echo "intermediate pruning is supported only for BUILD_SEMANTIC=0" >&2
      exit 2
    fi
    ;;
  *)
    echo "PRUNE_REGENERABLE_INTERMEDIATES must be 0/1 or true/false" >&2
    exit 2
    ;;
esac
if (( GPU_IDLE_CONFIRMATIONS <= 0 )); then
  echo "GPU_IDLE_CONFIRMATIONS must be positive" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$GEOMETRY_ROOT" "$FEATURE_ROOT" "$CONTRACT_ROOT" "$FIELD_OUTPUT_ROOT"

wait_for_gpu_on() {
  local device="${1:?set a physical GPU index}"
  local available=0
  while (( available < GPU_IDLE_CONFIRMATIONS )); do
    local values used util
    values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$device")"
    used="${values%%,*}"
    util="${values##*,}"
    used="${used// /}"
    util="${util// /}"
    if (( used < 1200 && util < 10 )); then
      available=$((available + 1))
    else
      available=0
    fi
    if (( available < GPU_IDLE_CONFIRMATIONS )); then
      sleep 20
    fi
  done
}

wait_for_gpu() {
  wait_for_gpu_on "$GPU"
}

validate_feature_manifest() {
  FEATURE_DIR="$1" \
  RADIO_RESOLUTION_SCALE="$RADIO_RESOLUTION_SCALE" \
  CAPABILITY_MAP_SOURCE="$CAPABILITY_MAP_SOURCE" \
  RADIO_ADAPTOR_NAMES="$RADIO_ADAPTOR_NAMES" \
    bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["FEATURE_DIR"])
manifest_path = root / "frame_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"missing RADIO feature manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_scale = float(os.environ["RADIO_RESOLUTION_SCALE"])
actual_scale = float(manifest.get("resolution_scale", float("nan")))
if not math.isfinite(actual_scale) or abs(actual_scale - expected_scale) > 1e-8:
    raise SystemExit(
        f"RADIO feature scale mismatch: cache={actual_scale}, requested={expected_scale}"
    )
if str(os.environ["CAPABILITY_MAP_SOURCE"]) != "official_extracted":
    raise SystemExit(0)
features = manifest.get("features", {})
adaptors = features.get("adaptors", []) if isinstance(features, dict) else []
if not isinstance(adaptors, list):
    raise SystemExit("official capability maps are absent from RADIO manifest")
by_name = {
    str(item.get("name")): item
    for item in adaptors
    if isinstance(item, dict)
}
for name in ("dino_v3_7b", "sam3"):
    item = by_name.get(name)
    if item is None:
        raise SystemExit(f"missing official {name} adaptor maps in RADIO manifest")
    subdir = str(item.get("subdir", ""))
    if not subdir or not (root / subdir).is_dir():
        raise SystemExit(f"official {name} adaptor directory is missing")
PY
}

mapfile -t SCENES < <(
  PFIR_MATERIALIZATION_REPORT="$PFIR_MATERIALIZATION_REPORT" \
  SCENE_START_INDEX="$SCENE_START_INDEX" \
    SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
  bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path
p = Path(os.environ["PFIR_MATERIALIZATION_REPORT"])
rows = json.loads(p.read_text())["scenes"]
rows = sorted(rows, key=lambda x: (x["field_frame_count"], x["scene_id"]))
start = int(os.environ.get("SCENE_START_INDEX", "0"))
stop_raw = os.environ.get("SCENE_STOP_INDEX", "").strip()
stop = len(rows) if not stop_raw else int(stop_raw)
if start < 0 or stop < start or stop > len(rows):
    raise SystemExit(f"invalid PFIR scene slice [{start}:{stop}] for {len(rows)} scenes")
for row in rows[start:stop]:
    print(row["scene_id"])
PY
)

if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "No PFIR scenes selected by slice [${SCENE_START_INDEX}:${SCENE_STOP_INDEX:-end}]" >&2
  exit 2
fi
if [[ "$GEOMETRY_INIT_SELECTION_POLICY" != "uniform" && "$GEOMETRY_INIT_SELECTION_POLICY" != "coverage_prefix" ]]; then
  echo "GEOMETRY_INIT_SELECTION_POLICY must be uniform or coverage_prefix" >&2
  exit 2
fi
if [[ "$CAPABILITY_MAP_SOURCE" != "project_raw" && "$CAPABILITY_MAP_SOURCE" != "official_extracted" ]]; then
  echo "CAPABILITY_MAP_SOURCE must be project_raw or official_extracted" >&2
  exit 2
fi
if [[ -n "$CAPABILITY_MPR_PARALLEL_GPU" ]]; then
  if [[ ! "$CAPABILITY_MPR_PARALLEL_GPU" =~ ^[0-9]+$ ]]; then
    echo "CAPABILITY_MPR_PARALLEL_GPU must be an integer physical GPU index" >&2
    exit 2
  fi
  if [[ "$CAPABILITY_MPR_PARALLEL_GPU" == "$GPU" ]]; then
    echo "CAPABILITY_MPR_PARALLEL_GPU must differ from GPU" >&2
    exit 2
  fi
fi
if [[ "$RADIO_BATCH_SIZE" -le 0 ]]; then
  echo "RADIO_BATCH_SIZE must be positive" >&2
  exit 2
fi
if [[ -n "$GEOMETRY_SUPPORT_GATE_DIR" || -n "$GEOMETRY_SUPPORT_BENCHMARK_ROOT" ]]; then
  if [[ -z "$GEOMETRY_SUPPORT_GATE_DIR" || -z "$GEOMETRY_SUPPORT_BENCHMARK_ROOT" ]]; then
    echo "GEOMETRY_SUPPORT_GATE_DIR and GEOMETRY_SUPPORT_BENCHMARK_ROOT must be set together" >&2
    exit 2
  fi
  if ! bash radio_gs/scripts/run_repo_python.sh - "$GEOMETRY_SUPPORT_MINIMUM_FRACTION" <<'PY'
import sys

value = float(sys.argv[1])
if not 0.0 < value <= 1.0:
    raise SystemExit("GEOMETRY_SUPPORT_MINIMUM_FRACTION must be in (0, 1]")
PY
  then
    exit 2
  fi
fi
if [[ -n "$PFPR_GEOMETRY_SUPPORT_GATE_DIR" || -n "$PFPR_GEOMETRY_SUPPORT_BENCHMARK_DIR" ]]; then
  if [[ -z "$PFPR_GEOMETRY_SUPPORT_GATE_DIR" || -z "$PFPR_GEOMETRY_SUPPORT_BENCHMARK_DIR" ]]; then
    echo "PFPR_GEOMETRY_SUPPORT_GATE_DIR and PFPR_GEOMETRY_SUPPORT_BENCHMARK_DIR must be set together" >&2
    exit 2
  fi
  if ! bash radio_gs/scripts/run_repo_python.sh - "$PFPR_GEOMETRY_SUPPORT_MINIMUM_FRACTION" <<'PY'
import sys

value = float(sys.argv[1])
if not 0.0 < value <= 1.0:
    raise SystemExit("PFPR_GEOMETRY_SUPPORT_MINIMUM_FRACTION must be in (0, 1]")
PY
  then
    exit 2
  fi
fi
if ! bash radio_gs/scripts/run_repo_python.sh - \
  "$CAPABILITY_LOCAL_AFFINITY_WEIGHT" "$CAPABILITY_LOCAL_RADIUS" "$CAPABILITY_LOCAL_BALANCE_QUANTILE" <<'PY'
import sys

weight = float(sys.argv[1])
radius = int(sys.argv[2])
balance = float(sys.argv[3])
if weight < 0 or radius <= 0 or not 0.0 <= balance < 0.5:
    raise SystemExit(
        "CAPABILITY_LOCAL_AFFINITY_WEIGHT/RADIUS/BALANCE_QUANTILE are invalid"
    )
PY
then
  exit 2
fi
echo "PFIR field queue: ${#SCENES[@]} scenes from slice [${SCENE_START_INDEX}:${SCENE_STOP_INDEX:-end}]"

for scene in "${SCENES[@]}"; do
  scene_root="$FIELD_ROOT/$scene"
  geometry_dir="$GEOMETRY_ROOT/$scene"
  ply="$geometry_dir/point_cloud/iteration_${GS_ITERS}/point_cloud.ply"
  feature_dir="$FEATURE_ROOT/$scene"
  config="$CONTRACT_ROOT/$scene.yaml"
  checkpoint="$CONTRACT_ROOT/$scene.geometry_renderer.pth"
  field_dir="$FIELD_OUTPUT_ROOT/$scene"
  responsibility="$field_dir/registration_responsibility.pt"
  raw_mpr="$field_dir/raw_radio_mpr.pt"
  dino_mpr="$field_dir/dino_v3_mpr.pt"
  sam3_mpr="$field_dir/sam3_mpr.pt"
  field_v1="$field_dir/canonical_mpr_v1.pt"
  field_v2="$field_dir/canonical_mpr_v2.pt"
  capability="$field_dir/official_dino_sam3_views.pt"
  graph="$field_dir/shared_support_graph_k16.pt"
  semantic="$field_dir/global_region_summary_semantic.pt"
  semantic_query="$field_dir/global_region_summary_semantic_query.pt"
  validation_plan="$field_dir/fidelity_validation_frames.json"
  compact_terminal="$field_dir/compact_direct_field.complete"
  mkdir -p "$field_dir"

  if (( PRUNE_REGENERABLE_INTERMEDIATES )) \
    && [[ -s "$compact_terminal" \
      && -s "$field_v2" \
      && -s "$capability" \
      && -s "$graph" \
      && -s "$raw_mpr.json" ]]; then
    echo "$scene: compact direct field already complete"
    continue
  fi

  if [[ ! -s "$ply" || ! -s "$geometry_dir/final.pth" ]]; then
    wait_for_gpu
    geometry_args=(
      radio_gs/scripts/train_scannet_gs.py
      --scene_root "$scene_root"
      --scene "$scene"
      --output_dir "$GEOMETRY_ROOT"
      --iters "$GS_ITERS"
      --frame_stride 1
      --init_frames "$GEOMETRY_INIT_FRAMES"
      --init_stride "$GEOMETRY_INIT_STRIDE"
      --max_points "$GEOMETRY_MAX_POINTS"
      --init-selection-policy "$GEOMETRY_INIT_SELECTION_POLICY"
    )
    if [[ "$GEOMETRY_INIT_SELECTION_POLICY" == "coverage_prefix" ]]; then
      geometry_args+=(--field-source-contract "$scene_root/field_source_contract.json")
    fi
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      "${geometry_args[@]}" >"$RUN_ROOT/logs/${scene}.geometry.log" 2>&1
  fi

  if [[ ! -s "$feature_dir/frame_manifest.json" ]]; then
    wait_for_gpu
    radio_args=(
      radio_gs/scripts/extract_radio_features.py
      --scene "$scene"
      --image_dir "$scene_root/color"
      --output_dir "$feature_dir"
      --radio_repo "$RADIO_REPO"
      --radio_version c-radio_v4-h
      --batch_size "$RADIO_BATCH_SIZE"
      --resolution_scale "$RADIO_RESOLUTION_SCALE"
      --skip_pca_stats
    )
    if [[ "$CAPABILITY_MAP_SOURCE" == "official_extracted" ]]; then
      radio_args+=(--extract_adaptors --adaptor_names "$RADIO_ADAPTOR_NAMES")
    fi
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      "${radio_args[@]}" >"$RUN_ROOT/logs/${scene}.radio.log" 2>&1
  fi
  validate_feature_manifest "$feature_dir"

  validation_frames="$(
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/select_fidelity_validation_frames.py \
      --feature-dir "$feature_dir" \
      --output "$validation_plan" \
      --views "$FIDELITY_VALIDATION_VIEWS" \
      --print-csv
  )"

  if [[ ! -s "$config" || ! -s "$checkpoint" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_geometry_render_contract.py \
      --ply-path "$ply" \
      --scene-root "$scene_root" \
      --feature-dir "$feature_dir" \
      --output-config "$config" \
      --output-checkpoint "$checkpoint" \
      --observation-contract "$OBSERVATION_CONTRACT" \
      >"$RUN_ROOT/logs/${scene}.render_contract.log" 2>&1
  fi

  # PFPR's fixed 5 cm candidate domain is public and geometry-only.  Audit
  # the all-Gaussian ceiling after the render contract exists but before any
  # raw/DINO/SAM MPR cache is built; private anchors and crop pixels remain
  # inaccessible to this construction gate.
  if [[ -n "$PFPR_GEOMETRY_SUPPORT_GATE_DIR" ]]; then
    pfpr_geometry_support_audit="$PFPR_GEOMETRY_SUPPORT_GATE_DIR/${scene}.json"
    if [[ ! -s "$pfpr_geometry_support_audit" ]]; then
      mkdir -p "$PFPR_GEOMETRY_SUPPORT_GATE_DIR"
      bash radio_gs/scripts/run_repo_python.sh \
        -m radio_gs.benchmarks.scannet_pfpr.audit_geometry_support \
        --benchmark-dir "$PFPR_GEOMETRY_SUPPORT_BENCHMARK_DIR" \
        --field-root "$RUN_ROOT" \
        --output "$pfpr_geometry_support_audit" \
        --scene-names "$scene" \
        --device cpu \
        --readout-candidate-k 64 --readout-support-threshold 0.01 \
        --minimum-support-fraction "$PFPR_GEOMETRY_SUPPORT_MINIMUM_FRACTION" \
        --candidate-voxel-size-m 0.05 \
        >"$RUN_ROOT/logs/${scene}.pfpr_geometry_support_gate.log" 2>&1
    fi
    PFPR_GEOMETRY_SUPPORT_AUDIT="$pfpr_geometry_support_audit" \
    PFPR_GEOMETRY_SUPPORT_SCENE="$scene" \
    PFPR_GEOMETRY_SUPPORT_MINIMUM_FRACTION="$PFPR_GEOMETRY_SUPPORT_MINIMUM_FRACTION" \
      bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

from radio_gs.benchmarks.scannet_pfpr.audit_geometry_support import validate_geometry_support_gate

payload = json.loads(Path(os.environ["PFPR_GEOMETRY_SUPPORT_AUDIT"]).read_text(encoding="utf-8"))
scene = os.environ["PFPR_GEOMETRY_SUPPORT_SCENE"]
minimum = float(os.environ["PFPR_GEOMETRY_SUPPORT_MINIMUM_FRACTION"])
actual = validate_geometry_support_gate(payload, scene_id=scene, minimum_support_fraction=minimum)
print(f"{scene}: rebuilt public-candidate geometry support {actual:.6f} passes fixed gate {minimum:.6f}")
PY
  fi

  if [[ ! -s "$raw_mpr" || ! -s "$responsibility" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output "$raw_mpr" \
      --device cuda:0 \
      --observation-contract "$MPR_OBSERVATION_CONTRACT" \
      --feature-space radio \
      --exclude-frame-ids "$validation_frames" \
      --save-responsibility-cache "$responsibility" \
      >"$RUN_ROOT/logs/${scene}.mpr_raw.log" 2>&1
  fi

  # A geometry rebuild must prove that all of its Gaussian support can cover
  # the released 5 cm scene domain *before* we spend GPU time reconstructing
  # DINO/SAM features.  The audit parses only official x/y/z/R/G/B and the
  # query-free raw-MPR/geometry artifacts; it has no object list, label,
  # click, mask, or metric access.  A failure is an explicit construction
  # rejection, never a coverage-based evaluator subgroup.
  if [[ -n "$GEOMETRY_SUPPORT_GATE_DIR" ]]; then
    geometry_support_audit="$GEOMETRY_SUPPORT_GATE_DIR/${scene}.json"
    if [[ ! -s "$geometry_support_audit" ]]; then
      mkdir -p "$GEOMETRY_SUPPORT_GATE_DIR"
      wait_for_gpu
      CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
        -m radio_gs.benchmarks.agile3d_scannet40.audit_geometry_support \
        --benchmark-root "$GEOMETRY_SUPPORT_BENCHMARK_ROOT" \
        --field-root "$RUN_ROOT" \
        --output "$geometry_support_audit" \
        --scene-names "$scene" \
        --device cuda:0 \
        --readout-candidate-k 64 --readout-support-threshold 0.01 \
        --minimum-support-fraction "$GEOMETRY_SUPPORT_MINIMUM_FRACTION" \
        --evaluation-voxel-size-m 0.05 \
        >"$RUN_ROOT/logs/${scene}.geometry_support_gate.log" 2>&1
    fi
    GEOMETRY_SUPPORT_AUDIT="$geometry_support_audit" \
    GEOMETRY_SUPPORT_SCENE="$scene" \
    GEOMETRY_SUPPORT_MINIMUM_FRACTION="$GEOMETRY_SUPPORT_MINIMUM_FRACTION" \
      bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["GEOMETRY_SUPPORT_AUDIT"]).read_text(encoding="utf-8"))
if payload.get("mode") != "label_free_all_gaussian_geometry_support_ceiling":
    raise SystemExit("geometry support gate has an invalid audit mode")
protocol = dict(payload.get("protocol", {}))
if protocol.get("labels_opened") is not False or protocol.get("object_list_opened") is not False:
    raise SystemExit("geometry support gate is not label free")
scene = os.environ["GEOMETRY_SUPPORT_SCENE"]
rows = [row for row in payload.get("scene_geometry_support", []) if row.get("scene_id") == scene]
if len(rows) != 1:
    raise SystemExit(f"geometry support gate has no unique row for {scene}")
actual = float(rows[0].get("geometry_only_support_fraction", 0.0))
minimum = float(os.environ["GEOMETRY_SUPPORT_MINIMUM_FRACTION"])
if actual < minimum:
    raise SystemExit(
        f"{scene}: rebuilt all-Gaussian geometry support {actual:.6f} < fixed gate {minimum:.6f}"
    )
print(f"{scene}: rebuilt all-Gaussian geometry support {actual:.6f} passes fixed gate {minimum:.6f}")
PY
  fi

  # Both capability spaces use the exact same immutable raw-radio
  # responsibility cache.  When a second device is explicitly supplied, lift
  # the two independent teacher spaces concurrently to avoid making native
  # official DINO/SAM reconstruction artificially serial.  The cache paths,
  # source frames, held-out frames, and all field-side losses remain exactly
  # the same as the sequential path below.
  if [[ -n "$CAPABILITY_MPR_PARALLEL_GPU" && ! -s "$dino_mpr" && ! -s "$sam3_mpr" ]]; then
    wait_for_gpu
    wait_for_gpu_on "$CAPABILITY_MPR_PARALLEL_GPU"
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output "$dino_mpr" \
      --device cuda:0 \
      --observation-contract "$MPR_OBSERVATION_CONTRACT" \
      --feature-space dino_v3 \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --capability-map-source "$CAPABILITY_MAP_SOURCE" \
      --exclude-frame-ids "$validation_frames" \
      --responsibility-cache "$responsibility" \
      >"$RUN_ROOT/logs/${scene}.mpr_dino.log" 2>&1 &
    dino_pid=$!
    CUDA_VISIBLE_DEVICES="$CAPABILITY_MPR_PARALLEL_GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output "$sam3_mpr" \
      --device cuda:0 \
      --observation-contract "$MPR_OBSERVATION_CONTRACT" \
      --feature-space sam3 \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --capability-map-source "$CAPABILITY_MAP_SOURCE" \
      --exclude-frame-ids "$validation_frames" \
      --responsibility-cache "$responsibility" \
      >"$RUN_ROOT/logs/${scene}.mpr_sam3.log" 2>&1 &
    sam3_pid=$!
    dino_status=0
    sam3_status=0
    wait "$dino_pid" || dino_status=$?
    wait "$sam3_pid" || sam3_status=$?
    if (( dino_status != 0 || sam3_status != 0 )); then
      echo "parallel DINO/SAM MPR cache build failed: dino=${dino_status}, sam3=${sam3_status}" >&2
      exit 1
    fi
  fi

  if [[ ! -s "$dino_mpr" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output "$dino_mpr" \
      --device cuda:0 \
      --observation-contract "$MPR_OBSERVATION_CONTRACT" \
      --feature-space dino_v3 \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --capability-map-source "$CAPABILITY_MAP_SOURCE" \
      --exclude-frame-ids "$validation_frames" \
      --responsibility-cache "$responsibility" \
      >"$RUN_ROOT/logs/${scene}.mpr_dino.log" 2>&1
  fi

  if [[ ! -s "$sam3_mpr" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output "$sam3_mpr" \
      --device cuda:0 \
      --observation-contract "$MPR_OBSERVATION_CONTRACT" \
      --feature-space sam3 \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --capability-map-source "$CAPABILITY_MAP_SOURCE" \
      --exclude-frame-ids "$validation_frames" \
      --responsibility-cache "$responsibility" \
      >"$RUN_ROOT/logs/${scene}.mpr_sam3.log" 2>&1
  fi

  if [[ ! -s "$field_v1" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/train_canonical_radio_field.py \
      --mpr-cache "$raw_mpr" \
      --observation-contract "$MPR_OBSERVATION_CONTRACT" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --output "$field_v1" \
      --device cuda:0 \
      --coefficient-dim 256 \
      --local-dim 128 \
      --primitive-fusion \
      --official-capability-loss \
      --dino-mpr-cache "$dino_mpr" \
      --sam3-mpr-cache "$sam3_mpr" \
      --epochs 20 \
      --min-epochs 5 \
      --target-cosine 0.985 \
      --seed 0 \
      >"$RUN_ROOT/logs/${scene}.canonical_v1.log" 2>&1
  fi

  if [[ ! -s "$field_v2" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/finetune_canonical_radio_rendering.py \
      --config "$config" \
      --geometry-checkpoint "$checkpoint" \
      --field-checkpoint "$field_v1" \
      --mpr-cache "$raw_mpr" \
      --output "$field_v2" \
      --device cuda:0 \
      --steps 256 \
      --mpr-weight 0.10 \
      --dino-render-weight 0.20 \
      --sam3-render-weight 0.20 \
      --capability-map-source "$CAPABILITY_MAP_SOURCE" \
      --capability-local-affinity-weight "$CAPABILITY_LOCAL_AFFINITY_WEIGHT" \
      --capability-local-radius "$CAPABILITY_LOCAL_RADIUS" \
      --capability-local-balance-quantile "$CAPABILITY_LOCAL_BALANCE_QUANTILE" \
      --train-fusion \
      --validation-frame-ids "$validation_frames" \
      --selection-policy "$FIELD_SELECTION_POLICY" \
      --max-capability-drop "$FIELD_MAX_CAPABILITY_DROP" \
      --max-mpr-drop "$FIELD_MAX_MPR_DROP" \
      --seed 0 \
      >"$RUN_ROOT/logs/${scene}.canonical_v2.log" 2>&1
  fi

  if [[ ! -s "$capability" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_canonical_capability_views.py \
      --field-checkpoint "$field_v2" \
      --mpr-cache "$raw_mpr" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --output "$capability" \
      --batch-size 2048 \
      --device cuda:0 \
      >"$RUN_ROOT/logs/${scene}.capability.log" 2>&1
  fi

  if [[ ! -s "$graph" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_canonical_support_graph.py \
      --capability-cache "$capability" \
      --output "$graph" \
      --neighbors 16 \
      --topology-mode symmetric_union \
      >"$RUN_ROOT/logs/${scene}.support_graph.log" 2>&1
  fi

  if [[ "$BUILD_SEMANTIC" == "1" || "$BUILD_SEMANTIC" == "true" || "$BUILD_SEMANTIC" == "TRUE" ]]; then
  if [[ ! -s "$semantic" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_surface_region_semantic_cache.py \
      --field-checkpoint "$field_v2" \
      --support-graph "$graph" \
      --readout-checkpoint "$READOUT_CHECKPOINT" \
      --output "$semantic" \
      --query-output "$semantic_query" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --device cuda:0 \
      >"$RUN_ROOT/logs/${scene}.semantic.log" 2>&1
  fi

  if [[ ! -s "$semantic_query" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/materialize_surface_region_query_cache.py \
      --semantic-cache "$semantic" \
      --output "$semantic_query" \
      >"$RUN_ROOT/logs/${scene}.semantic_query.log" 2>&1
  fi
  fi

  if (( PRUNE_REGENERABLE_INTERMEDIATES )); then
    for required in "$field_v2" "$capability" "$graph" "$raw_mpr.json"; do
      if [[ ! -s "$required" ]]; then
        echo "$scene: refusing to prune before the compact direct field is complete: $required" >&2
        exit 2
      fi
    done
    case "$feature_dir" in
      "$FEATURE_ROOT"/"$scene")
        ;;
      *)
        echo "$scene: refusing to prune an unexpected feature directory: $feature_dir" >&2
        exit 2
        ;;
    esac
    pruning_audit="$field_dir/compact_direct_field_pruning.txt"
    {
      echo "scene=$scene"
      echo "mode=regenerable_training_intermediates_only"
      for artifact in "$raw_mpr" "$dino_mpr" "$sam3_mpr" "$responsibility" "$field_v1"; do
        if [[ -e "$artifact" ]]; then
          stat -c '%s %n' "$artifact"
        fi
      done
      if [[ -d "$feature_dir" ]]; then
        du -sb "$feature_dir"
      fi
    } >"$pruning_audit"
    rm -f -- "$raw_mpr" "$dino_mpr" "$sam3_mpr" "$responsibility" "$field_v1"
    rm -rf -- "$feature_dir"
    date -Iseconds >"$compact_terminal"
  fi
done

case "$WRITE_TERMINAL" in
  1|true|True|TRUE)
    date -Iseconds >"$RUN_ROOT/canonical_mpr_v3_fields.complete"
    ;;
  0|false|False|FALSE)
    echo "PFIR shard complete; the designated full queue will write the field terminal."
    ;;
  *)
    echo "WRITE_TERMINAL must be 0/1 or true/false, got: $WRITE_TERMINAL" >&2
    exit 2
    ;;
esac
