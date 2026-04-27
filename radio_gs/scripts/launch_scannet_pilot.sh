#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SCENE="${SCENE:-scene0000_00}"
SCANNET_ROOT="${SCANNET_ROOT:-/mnt/pool/sqy/results/ScanNetPilot/scans/${SCENE}}"
FEATURE_OUT="${FEATURE_OUT:-output/radio_features_scannet/${SCENE}}"
GS_OUT="${GS_OUT:-output/3dgs_models/scannet}"
TRAIN_OUT="${TRAIN_OUT:-output/radio_gs/scannet_${SCENE}_v14_pilot}"
CONFIG_PATH="${CONFIG_PATH:-radio_gs/configs/generated/scannet_${SCENE}_pilot.yaml}"
RADIO_REPO="${RADIO_REPO:-/root/RADIO}"
TEMPLATE_PATH="${TEMPLATE_PATH:-radio_gs/configs/scannet_hybrid_v14_template.yaml}"
GEOM_GPU="${GEOM_GPU:-4}"
FEAT_GPU="${FEAT_GPU:-4}"
TRAIN_GPU="${TRAIN_GPU:-5}"
FRAME_SKIP_PREP="${FRAME_SKIP_PREP:-10}"
MAX_PREP_FRAMES="${MAX_PREP_FRAMES:-250}"
FEATURE_FRAME_STRIDE="${FEATURE_FRAME_STRIDE:-1}"
FEATURE_MAX_FRAMES="${FEATURE_MAX_FRAMES:-200}"
GS_MAX_FRAMES="${GS_MAX_FRAMES:-120}"
GS_ITERS="${GS_ITERS:-15000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-120}"

echo "[scannet] scene root: $SCANNET_ROOT"

if [[ ! -f "$SCANNET_ROOT/${SCENE}.sens" ]]; then
  echo "[scannet] missing $SCANNET_ROOT/${SCENE}.sens" >&2
  exit 1
fi

mkdir -p "$(dirname "$CONFIG_PATH")"

bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/prepare_scannet_scene.py \
  --scene_root "$SCANNET_ROOT" \
  --frame_skip "$FRAME_SKIP_PREP" \
  --max_frames "$MAX_PREP_FRAMES"

CUDA_VISIBLE_DEVICES="$FEAT_GPU" bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/extract_radio_features.py \
  --scene "$SCENE" \
  --image_dir "$SCANNET_ROOT/color" \
  --output_dir "$FEATURE_OUT" \
  --radio_repo "$RADIO_REPO" \
  --radio_version c-radio_v4-h \
  --batch_size 4 \
  --frame_stride "$FEATURE_FRAME_STRIDE" \
  --max_frames "$FEATURE_MAX_FRAMES"

CUDA_VISIBLE_DEVICES="$GEOM_GPU" bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/train_scannet_gs.py \
  --scene_root "$SCANNET_ROOT" \
  --scene "$SCENE" \
  --output_dir "$GS_OUT" \
  --iters "$GS_ITERS" \
  --frame_stride 2 \
  --max_frames "$GS_MAX_FRAMES"

SCENE="$SCENE" \
SCANNET_ROOT="$SCANNET_ROOT" \
FEATURE_OUT="$FEATURE_OUT" \
GS_OUT="$GS_OUT" \
TRAIN_OUT="$TRAIN_OUT" \
CONFIG_PATH="$CONFIG_PATH" \
RADIO_REPO="$RADIO_REPO" \
TEMPLATE_PATH="$TEMPLATE_PATH" \
GS_ITERS="$GS_ITERS" \
python - <<'PY'
import os
from pathlib import Path


def repo_abs(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((Path.cwd() / path).resolve())


scene = os.environ["SCENE"]
scannet_root = repo_abs(os.environ["SCANNET_ROOT"])
feature_out = repo_abs(os.environ["FEATURE_OUT"])
gs_out = repo_abs(os.environ["GS_OUT"])
train_out = repo_abs(os.environ["TRAIN_OUT"])
config_path = Path(os.environ["CONFIG_PATH"]).resolve()
radio_repo = repo_abs(os.environ["RADIO_REPO"])
template_path = Path(os.environ["TEMPLATE_PATH"]).resolve()
gs_iters = os.environ["GS_ITERS"]

text = template_path.read_text(encoding="utf-8")
replacements = {
    'exp_name: "radio_gs_scannet_scene0000_00_v14_template"': f'exp_name: "radio_gs_scannet_{scene}_v14_pilot"',
    'output_dir: "/root/RADIO-GS/output/radio_gs/scannet_scene0000_00_v14_template"': f'output_dir: "{train_out}"',
    'scene: "scene0000_00"': f'scene: "{scene}"',
    'scene_root: "/mnt/pool/Datasets/ScanNet/data/scans/scene0000_00"': f'scene_root: "{scannet_root}"',
    'ply_path: "/root/RADIO-GS/output/3dgs_models/scannet/scene0000_00/point_cloud/iteration_30000/point_cloud.ply"': f'ply_path: "{gs_out}/{scene}/point_cloud/iteration_{gs_iters}/point_cloud.ply"',
    'radio_repo: "/root/RADIO"': f'radio_repo: "{radio_repo}"',
    'feature_dir: "/root/RADIO-GS/output/radio_features_scannet/scene0000_00"': f'feature_dir: "{feature_out}"',
    'rgb_dir: "/mnt/pool/Datasets/ScanNet/data/scans/scene0000_00/color"': f'rgb_dir: "{scannet_root}/color"',
    'depth_dir: "/mnt/pool/Datasets/ScanNet/data/scans/scene0000_00/depth"': f'depth_dir: "{scannet_root}/depth"',
    'semantics_dir: "/mnt/pool/Datasets/ScanNet/data/scans/scene0000_00/label-filt"': f'semantics_dir: "{scannet_root}/label-filt"',
    'pose_dir: "/mnt/pool/Datasets/ScanNet/data/scans/scene0000_00/pose"': f'pose_dir: "{scannet_root}/pose"',
    'frozen_depth_head_weight: 0.1': 'frozen_depth_head_weight: 0.0',
}
for src, dst in replacements.items():
    if src not in text:
        raise SystemExit(f"Expected template snippet missing: {src}")
    text = text.replace(src, dst)

config_path.write_text(text, encoding="utf-8")
print(f"wrote {config_path}")
PY

CUDA_VISIBLE_DEVICES="$TRAIN_GPU" bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/train_feature_field.py \
  --config "$CONFIG_PATH"
