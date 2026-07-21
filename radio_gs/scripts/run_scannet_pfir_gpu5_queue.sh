#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-5}"
FIELD_ROOT="${FIELD_ROOT:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/ScanNet-PFIR-Small/field_only_test_v1}"
RUN_ROOT="${RUN_ROOT:-output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1}"
GEOMETRY_ROOT="${GEOMETRY_ROOT:-$RUN_ROOT/geometry}"
FEATURE_ROOT="${FEATURE_ROOT:-$RUN_ROOT/radio_features}"
CONTRACT_ROOT="${CONTRACT_ROOT:-$RUN_ROOT/render_contracts}"
FIELD_OUTPUT_ROOT="${FIELD_OUTPUT_ROOT:-$RUN_ROOT/canonical_fields}"
GS_ITERS="${GS_ITERS:-15000}"
RADIO_REPO="${RADIO_REPO:-/root/RADIO}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
READOUT_CHECKPOINT="${READOUT_CHECKPOINT:-output/scannet_pfir_small_v1/readout_v3/global_surface_region_readout_h256_clean.pt}"
FIDELITY_VALIDATION_VIEWS="${FIDELITY_VALIDATION_VIEWS:-4}"
# A second worker may take a disjoint scene suffix when another real GPU
# becomes available. Defaults preserve the original complete serial queue.
SCENE_START_INDEX="${SCENE_START_INDEX:-0}"
SCENE_STOP_INDEX="${SCENE_STOP_INDEX:-}"
WRITE_TERMINAL="${WRITE_TERMINAL:-1}"

mkdir -p "$RUN_ROOT/logs" "$GEOMETRY_ROOT" "$FEATURE_ROOT" "$CONTRACT_ROOT" "$FIELD_OUTPUT_ROOT"

wait_for_gpu() {
  local available=0
  while (( available < 2 )); do
    local values used util
    values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU")"
    used="${values%%,*}"
    util="${values##*,}"
    used="${used// /}"
    util="${util// /}"
    if (( used < 1200 && util < 10 )); then
      available=$((available + 1))
    else
      available=0
    fi
    if (( available < 2 )); then
      sleep 20
    fi
  done
}

mapfile -t SCENES < <(
  SCENE_START_INDEX="$SCENE_START_INDEX" \
    SCENE_STOP_INDEX="$SCENE_STOP_INDEX" \
  bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json
import os
from pathlib import Path
p = Path("/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/ScanNet-PFIR-Small/field_only_test_v1/materialization_report.json")
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
  mkdir -p "$field_dir"

  if [[ ! -s "$ply" || ! -s "$geometry_dir/final.pth" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/train_scannet_gs.py \
      --scene_root "$scene_root" \
      --scene "$scene" \
      --output_dir "$GEOMETRY_ROOT" \
      --iters "$GS_ITERS" \
      --frame_stride 1 \
      --init_frames 50 \
      --init_stride 8 \
      --max_points 200000 \
      >"$RUN_ROOT/logs/${scene}.geometry.log" 2>&1
  fi

  if [[ ! -s "$feature_dir/frame_manifest.json" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/extract_radio_features.py \
      --scene "$scene" \
      --image_dir "$scene_root/color" \
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
      --ply-path "$ply" \
      --scene-root "$scene_root" \
      --feature-dir "$feature_dir" \
      --output-config "$config" \
      --output-checkpoint "$checkpoint" \
      >"$RUN_ROOT/logs/${scene}.render_contract.log" 2>&1
  fi

  if [[ ! -s "$raw_mpr" || ! -s "$responsibility" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output "$raw_mpr" \
      --device cuda:0 \
      --observation-contract canonical-mpr-v1 \
      --feature-space radio \
      --exclude-frame-ids "$validation_frames" \
      --save-responsibility-cache "$responsibility" \
      >"$RUN_ROOT/logs/${scene}.mpr_raw.log" 2>&1
  fi

  if [[ ! -s "$dino_mpr" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --output "$dino_mpr" \
      --device cuda:0 \
      --observation-contract canonical-mpr-v1 \
      --feature-space dino_v3 \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
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
      --observation-contract canonical-mpr-v1 \
      --feature-space sam3 \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --exclude-frame-ids "$validation_frames" \
      --responsibility-cache "$responsibility" \
      >"$RUN_ROOT/logs/${scene}.mpr_sam3.log" 2>&1
  fi

  if [[ ! -s "$field_v1" ]]; then
    wait_for_gpu
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/train_canonical_radio_field.py \
      --mpr-cache "$raw_mpr" \
      --observation-contract canonical-mpr-v1 \
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
      --capability-local-affinity-weight 0.25 \
      --train-fusion \
      --validation-frame-ids "$validation_frames" \
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
