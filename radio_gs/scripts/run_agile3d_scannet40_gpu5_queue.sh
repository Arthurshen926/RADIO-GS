#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-5}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet}"
FRAME_ROOT="${FRAME_ROOT:-/mnt/pool/Datasets/ScanNet/data/tasks/scannet_frames_25k}"
RUN_ROOT="${RUN_ROOT:-output/agile3d_scannet40/formal_v1}"
PFIR_TERMINAL="${PFIR_TERMINAL:-output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1/canonical_mpr_v3_fields.complete}"
WAIT_FOR_PFIR="${WAIT_FOR_PFIR:-1}"
RADIO_REPO="${RADIO_REPO:-/root/RADIO}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
FIDELITY_VALIDATION_VIEWS="${FIDELITY_VALIDATION_VIEWS:-4}"
# A second worker may take a disjoint suffix after another real job frees a
# GPU.  Defaults preserve the original full, serial benchmark run.
SCENE_START_INDEX="${SCENE_START_INDEX:-0}"
SCENE_STOP_INDEX="${SCENE_STOP_INDEX:-}"
RUN_EVALUATOR="${RUN_EVALUATOR:-1}"
FIELD_TERMINAL="${FIELD_TERMINAL:-$RUN_ROOT/canonical_mpr_v3_fields.complete}"
WRITE_FIELD_TERMINAL="${WRITE_FIELD_TERMINAL:-1}"
# Leave an explicit amount of headroom when sharing a GPU with another real
# workload.  The default retains the conservative idle-GPU behavior below.
GPU_MIN_FREE_MEMORY_MIB="${GPU_MIN_FREE_MEMORY_MIB:-0}"
# More than one disjoint scene shard can share a physical GPU.  Serialize
# only their GPU stages so that two RADIO/MPR loads cannot pass a free-memory
# check concurrently and overcommit the card.  This changes scheduling, not
# any field, query, or evaluation computation.
GPU_SERIALIZE_STAGES="${GPU_SERIALIZE_STAGES:-1}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-$RUN_ROOT/.gpu_stage_locks}"
# Usually this is the physical GPU index.  A deliberately provisioned GPU
# with ample headroom may expose a second independent scheduling slot; this
# only controls the lock namespace and is never used by the model itself.
GPU_LOCK_KEY="${GPU_LOCK_KEY:-$GPU}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/scenes" "$RUN_ROOT/features"

wait_for_gpu() {
  if (( GPU_MIN_FREE_MEMORY_MIB > 0 )); then
    while true; do
      local values total used free
      values="$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$GPU")"
      total="${values%%,*}"; used="${values##*,}"
      total="${total// /}"; used="${used// /}"
      free=$(( total - used ))
      if (( free >= GPU_MIN_FREE_MEMORY_MIB )); then
        return
      fi
      sleep 20
    done
  fi
  local available=0
  while (( available < 2 )); do
    local values used util
    values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU")"
    used="${values%%,*}"; util="${values##*,}"
    used="${used// /}"; util="${util// /}"
    if (( used < 1200 && util < 10 )); then
      available=$((available + 1))
    else
      available=0
    fi
    if (( available < 2 )); then sleep 20; fi
  done
}

run_gpu_stage() {
  case "$GPU_SERIALIZE_STAGES" in
    1|true|True|TRUE)
      mkdir -p "$GPU_LOCK_DIR"
      (
        flock -x 9
        # Check after taking the per-device lock: a sibling shard may have
        # released memory while this shard was waiting for its turn.
        wait_for_gpu
        CUDA_VISIBLE_DEVICES="$GPU" "$@"
      ) 9>"$GPU_LOCK_DIR/gpu_${GPU_LOCK_KEY}.lock"
      ;;
    0|false|False|FALSE)
      wait_for_gpu
      CUDA_VISIBLE_DEVICES="$GPU" "$@"
      ;;
    *)
      echo "GPU_SERIALIZE_STAGES must be 0/1 or true/false, got: $GPU_SERIALIZE_STAGES" >&2
      exit 2
      ;;
  esac
}

# Preserve the original serial schedule by default.  A field trained for
# AGILE3D has no data or checkpoint dependency on PFIR, however, so an
# explicitly requested parallel run may set WAIT_FOR_PFIR=0 and use a second
# otherwise idle GPU.
case "$WAIT_FOR_PFIR" in
  0|false|False|FALSE)
    ;;
  1|true|True|TRUE)
    while [[ ! -s "$PFIR_TERMINAL" ]]; do sleep 30; done
    ;;
  *)
    echo "WAIT_FOR_PFIR must be 0/1 or true/false, got: $WAIT_FOR_PFIR" >&2
    exit 2
    ;;
esac

mapfile -t SCENES < <(
  BENCHMARK_ROOT="$BENCHMARK_ROOT" \
    SCENE_START_INDEX="$SCENE_START_INDEX" \
    SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
    bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import os
from pathlib import Path
import numpy as np
root = Path(os.environ["BENCHMARK_ROOT"])
objects = np.load(root / "single" / "object_ids.npy", allow_pickle=False)
scenes = sorted({str(value) for value in objects[:, 0]})
start = int(os.environ.get("SCENE_START_INDEX", "0"))
stop_raw = os.environ.get("SCENE_STOP_INDEX", "").strip()
stop = len(scenes) if not stop_raw else int(stop_raw)
if start < 0 or stop < start or stop > len(scenes):
    raise SystemExit(
        f"invalid scene slice [{start}:{stop}] for {len(scenes)} AGILE3D scenes"
    )
for scene in scenes[start:stop]:
    print(scene)
PY
)

if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "No AGILE3D scenes selected by slice [${SCENE_START_INDEX}:${SCENE_STOP_INDEX:-end}]" >&2
  exit 2
fi
echo "AGILE3D queue: ${#SCENES[@]} scenes from slice [${SCENE_START_INDEX}:${SCENE_STOP_INDEX:-end}]"

for scene in "${SCENES[@]}"; do
  scene_out="$RUN_ROOT/scenes/$scene"
  scene_frames="$FRAME_ROOT/$scene"
  official_ply="$BENCHMARK_ROOT/scans/$scene.ply"
  geometry="$scene_out/official_5cm_gaussians.ply"
  mapping="$scene_out/official_5cm_mapping.npz"
  feature_dir="$scene_out/radio_features"
  config="$scene_out/geometry_renderer.yaml"
  checkpoint="$scene_out/geometry_renderer.pth"
  responsibility="$scene_out/registration_responsibility.pt"
  raw_mpr="$scene_out/raw_radio_mpr.pt"
  dino_mpr="$scene_out/dino_v3_mpr.pt"
  sam3_mpr="$scene_out/sam3_mpr.pt"
  field_v1="$scene_out/canonical_mpr_v1.pt"
  field_v2="$scene_out/canonical_mpr_v2.pt"
  capability="$scene_out/official_dino_sam3_views.pt"
  exported="$RUN_ROOT/features/$scene.npz"
  validation_plan="$scene_out/fidelity_validation_frames.json"
  mkdir -p "$scene_out"
  # Scene shards may overlap after an interrupted run is resumed.  Hold one
  # scene-level lock across the whole dependency chain so a later v2 field
  # can never invalidate a capability cache or mesh feature export published
  # by another worker.
  mkdir -p "$RUN_ROOT/.scene_locks"
  (
    flock -x 9

  if [[ ! -s "$geometry" || ! -s "$mapping" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_agile3d_gaussian_geometry.py \
      --input-ply "$official_ply" \
      --output-ply "$geometry" \
      --output-mapping "$mapping" \
      >"$RUN_ROOT/logs/${scene}.geometry.log" 2>&1
  fi

  if [[ ! -s "$feature_dir/frame_manifest.json" ]]; then
    run_gpu_stage bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/extract_radio_features.py \
      --scene "$scene" \
      --image_dir "$scene_frames/color" \
      --output_dir "$feature_dir" \
      --radio_repo "$RADIO_REPO" \
      --radio_version c-radio_v4-h \
      --batch_size 2 \
      --resolution_scale 0.5 \
      --skip_pca_stats \
      >"$RUN_ROOT/logs/${scene}.radio.log" 2>&1
  fi

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
      --ply-path "$geometry" \
      --scene-root "$scene_frames" \
      --feature-dir "$feature_dir" \
      --output-config "$config" \
      --output-checkpoint "$checkpoint" \
      >"$RUN_ROOT/logs/${scene}.render_contract.log" 2>&1
  fi

  if [[ ! -s "$raw_mpr" || ! -s "$responsibility" ]]; then
    run_gpu_stage bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      --config "$config" --checkpoint "$checkpoint" \
      --output "$raw_mpr" --device cuda:0 \
      --observation-contract canonical-mpr-v1 \
      --feature-space radio \
      --exclude-frame-ids "$validation_frames" \
      --save-responsibility-cache "$responsibility" \
      >"$RUN_ROOT/logs/${scene}.mpr_raw.log" 2>&1
  fi

  for space in dino_v3 sam3; do
    if [[ "$space" == "dino_v3" ]]; then target="$dino_mpr"; else target="$sam3_mpr"; fi
    if [[ ! -s "$target" ]]; then
      run_gpu_stage bash radio_gs/scripts/run_repo_python.sh \
        radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
        --config "$config" --checkpoint "$checkpoint" \
        --output "$target" --device cuda:0 \
        --observation-contract canonical-mpr-v1 \
        --feature-space "$space" \
        --radio-checkpoint "$RADIO_CHECKPOINT" \
        --exclude-frame-ids "$validation_frames" \
        --responsibility-cache "$responsibility" \
        >"$RUN_ROOT/logs/${scene}.mpr_${space}.log" 2>&1
    fi
  done

  if [[ ! -s "$field_v1" ]]; then
    run_gpu_stage bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/train_canonical_radio_field.py \
      --mpr-cache "$raw_mpr" \
      --observation-contract canonical-mpr-v1 \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --output "$field_v1" --device cuda:0 \
      --coefficient-dim 256 --local-dim 128 --primitive-fusion \
      --official-capability-loss \
      --dino-mpr-cache "$dino_mpr" --sam3-mpr-cache "$sam3_mpr" \
      --epochs 20 --min-epochs 5 --target-cosine 0.985 --seed 0 \
      >"$RUN_ROOT/logs/${scene}.canonical_v1.log" 2>&1
  fi

  if [[ ! -s "$field_v2" ]]; then
    run_gpu_stage bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/finetune_canonical_radio_rendering.py \
      --config "$config" --geometry-checkpoint "$checkpoint" \
      --field-checkpoint "$field_v1" --mpr-cache "$raw_mpr" \
      --output "$field_v2" --device cuda:0 \
      --steps 256 --mpr-weight 0.10 \
      --dino-render-weight 0.20 --sam3-render-weight 0.20 \
      --capability-local-affinity-weight 0.25 --train-fusion \
      --validation-frame-ids "$validation_frames" --seed 0 \
      >"$RUN_ROOT/logs/${scene}.canonical_v2.log" 2>&1
  fi

  if [[ ! -s "$capability" ]]; then
    run_gpu_stage bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_canonical_capability_views.py \
      --field-checkpoint "$field_v2" --mpr-cache "$raw_mpr" \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --output "$capability" --batch-size 2048 --device cuda:0 \
      >"$RUN_ROOT/logs/${scene}.capability.log" 2>&1
  fi

  if [[ ! -s "$exported" ]]; then
    run_gpu_stage bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/export_canonical_field_to_agile3d_mesh.py \
      --field-checkpoint "$field_v2" --capability-cache "$capability" \
      --mpr-cache "$raw_mpr" --mesh-ply "$official_ply" \
      --quantization-map "$mapping" --store-quantized \
      --output "$exported" --device cuda:0 \
      >"$RUN_ROOT/logs/${scene}.export.log" 2>&1
  fi
  ) 9>"$RUN_ROOT/.scene_locks/${scene}.lock"
done

case "$WRITE_FIELD_TERMINAL" in
  1|true|True|TRUE)
    date -Iseconds >"$FIELD_TERMINAL"
    ;;
  0|false|False|FALSE)
    echo "AGILE3D field shard complete; the designated full queue writes the field terminal."
    ;;
  *)
    echo "WRITE_FIELD_TERMINAL must be 0/1 or true/false, got: $WRITE_FIELD_TERMINAL" >&2
    exit 2
    ;;
esac

case "${RUN_EVALUATOR:-1}" in
  1|true|True|TRUE)
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      -m radio_gs.benchmarks.agile3d_scannet40.evaluate_feature_cache \
      --benchmark-root "$BENCHMARK_ROOT" \
      --feature-root "$RUN_ROOT/features" \
      --output "$RUN_ROOT/results.json" \
      --device cuda:0 \
      --observation-lift-mode observed_domain \
      --observation-lift-neighbors 3 \
      --observation-lift-maximum-distance-m 0.10 \
      >"$RUN_ROOT/logs/evaluation.log" 2>&1
    date -Iseconds >"$RUN_ROOT/formal.complete"
    ;;
  0|false|False|FALSE)
    echo "AGILE3D shard complete; the designated full queue will run the only formal evaluator."
    ;;
  *)
    echo "RUN_EVALUATOR must be 0/1 or true/false, got: ${RUN_EVALUATOR:-}" >&2
    exit 2
    ;;
esac
