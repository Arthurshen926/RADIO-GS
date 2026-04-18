#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-replica}"

RADIO_REPO="${RADIO_REPO:-/root/RADIO}"
REPLICA_ROOT="${REPLICA_ROOT:-/mnt/pool/sqy/dataset}"
LERF_ROOT="${LERF_ROOT:-/mnt/pool/sqy/lerf_ovs}"
REPLICA_SCENES="${REPLICA_SCENES:-room_0,room_2}"
LERF_SCENES="${LERF_SCENES:-figurines,ramen,teatime,waldo_kitchen}"
REPLICA_SEQUENCES="${REPLICA_SEQUENCES:-Sequence_1,Sequence_2}"
GEOM_ITERS="${GEOM_ITERS:-30000}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
GEOM_GPU="${GEOM_GPU:-}"
FEATURE_GPU="${FEATURE_GPU:-}"
GEOM_TAG="${GEOM_TAG:-v8_fixed_poses_3dgs}"

run_python() {
  local gpu="$1"
  shift
  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$@"
  else
    "$@"
  fi
}

log_skip() {
  local kind="$1"
  local path="$2"
  echo "[skip] ${kind}: ${path}"
}

extract_replica_scene() {
  local scene="$1"
  IFS=',' read -r -a sequences <<< "$REPLICA_SEQUENCES"
  local ply_path="output/3dgs_models/$scene/$GEOM_TAG/point_cloud/iteration_${GEOM_ITERS}/point_cloud.ply"

  if [[ -f "$ply_path" ]]; then
    log_skip "replica geometry" "$ply_path"
  else
    run_python "$GEOM_GPU" python radio_gs/scripts/train_rgb_gs.py \
      --scene "$scene" \
      --dataset_root "$REPLICA_ROOT" \
      --sequences "$REPLICA_SEQUENCES" \
      --output_dir output/3dgs_models \
      --tag "$GEOM_TAG" \
      --iters "$GEOM_ITERS"
  fi

  for sequence in "${sequences[@]}"; do
    local image_dir="$REPLICA_ROOT/$scene/$sequence/rgb"
    local output_dir="output/radio_features_1280d_reextract_20260407/$scene/$sequence"
    local manifest_path="$output_dir/frame_manifest.json"
    if [[ -f "$manifest_path" ]]; then
      log_skip "replica features" "$manifest_path"
    else
      run_python "$FEATURE_GPU" python radio_gs/scripts/extract_radio_features.py \
        --scene "$scene" \
        --image_dir "$image_dir" \
        --output_dir "$output_dir" \
        --radio_repo "$RADIO_REPO" \
        --radio_version c-radio_v4-h \
        --batch_size "$FEATURE_BATCH_SIZE"
    fi
  done
}

extract_lerf_scene() {
  local scene="$1"
  local scene_root="$LERF_ROOT/$scene"
  local feat_root="output/radio_features_lerf/$scene"
  local ply_path="output/3dgs_models/$scene/point_cloud/iteration_${GEOM_ITERS}/point_cloud.ply"
  local manifest_path="$feat_root/frame_manifest.json"
  local traj_path="$feat_root/traj_w_c.txt"

  if [[ -f "$ply_path" ]]; then
    log_skip "lerf geometry" "$ply_path"
  else
    run_python "$GEOM_GPU" python radio_gs/scripts/train_colmap_gs.py \
      --scene_root "$scene_root" \
      --output_dir "output/3dgs_models/$scene" \
      --iters "$GEOM_ITERS"
  fi

  if [[ -f "$manifest_path" ]]; then
    log_skip "lerf features" "$manifest_path"
  else
    run_python "$FEATURE_GPU" python radio_gs/scripts/extract_radio_features.py \
      --scene "$scene" \
      --image_dir "$scene_root/images" \
      --output_dir "$feat_root" \
      --radio_repo "$RADIO_REPO" \
      --radio_version c-radio_v4-h \
      --batch_size "$FEATURE_BATCH_SIZE"
  fi

  if [[ -f "$traj_path" ]]; then
    log_skip "lerf poses" "$traj_path"
  else
    SCENE_ROOT="$scene_root" FEAT_ROOT="$feat_root" python - <<'PY'
from pathlib import Path
import os
import numpy as np
from radio_gs.data.lerf_dataset import _parse_colmap_sparse

scene = Path(os.environ["SCENE_ROOT"])
out_path = Path(os.environ["FEAT_ROOT"]) / "traj_w_c.txt"
colmap = _parse_colmap_sparse(scene)
c2w = np.stack(colmap["c2w_list"], axis=0).reshape(-1, 16)
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(out_path, c2w, fmt="%.8f")
print(f"wrote {out_path}")
PY
  fi
}

case "$MODE" in
  replica)
    IFS=',' read -r -a scenes <<< "$REPLICA_SCENES"
    for scene in "${scenes[@]}"; do
      extract_replica_scene "$scene"
    done
    ;;
  lerf)
    IFS=',' read -r -a scenes <<< "$LERF_SCENES"
    for scene in "${scenes[@]}"; do
      extract_lerf_scene "$scene"
    done
    ;;
  all)
    "$0" replica
    "$0" lerf
    ;;
  *)
    echo "Usage: bash radio_gs/scripts/rebuild_3dgs_assets.sh [replica|lerf|all]" >&2
    exit 1
    ;;
esac
