#!/usr/bin/env bash

set -euo pipefail

VARIANT="${VARIANT:-v67_dino_cv001_b4_s32768_ft20}"
CONFIG_ROOT="${CONFIG_ROOT:-radio_gs/configs/generated/scannet_dino_cv}"
TEXT_CACHE="${TEXT_CACHE:-checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt}"
if [[ -z "${PROMPT_TEMPLATES:-}" ]]; then
  PROMPT_TEMPLATES='{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}'
fi
GPU=""
QUEUE_NAME="scannet_dino_cv"
DRY_RUN="${DRY_RUN:-0}"

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

config_for() {
  echo "${CONFIG_ROOT}/scannet_og_hybrid_${VARIANT}_$1.yaml"
}

checkpoint_for() {
  echo "output/radio_gs/scannet_og_$1_${VARIANT}/checkpoints/best.pth"
}

result_dir_for() {
  echo "output/scannet_pointcloud_eval/$1_${VARIANT}_gidx_labelpoint"
}

warmstart_for() {
  echo "output/radio_gs/scannet_og_$1_v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63/checkpoints/best.pth"
}

log() {
  printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$QUEUE_NAME" "$*"
}

print_cmd() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

usage() {
  cat <<'EOF'
Usage:
  launch_scannet_dino_cv_queue.sh --gpu GPU [--name NAME] SCENE...
  launch_scannet_dino_cv_queue.sh --print-config SCENE
  launch_scannet_dino_cv_queue.sh --print-prompts

Run generated ScanNet v67 DINO cross-view fine-tuning and evaluation jobs.
Set DRY_RUN=1 to print commands without executing them.
EOF
}

if [[ "${1:-}" == "--print-config" ]]; then
  if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
  fi
  config_for "$2"
  exit 0
fi

if [[ "${1:-}" == "--print-prompts" ]]; then
  printf '%s\n' "$PROMPT_TEMPLATES"
  exit 0
fi

SCENES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="${2:?missing value for --gpu}"
      shift 2
      ;;
    --name)
      QUEUE_NAME="${2:?missing value for --name}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      SCENES+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$GPU" || ${#SCENES[@]} -eq 0 ]]; then
  usage >&2
  exit 2
fi

cd "$(repo_root)"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

for scene in "${SCENES[@]}"; do
  cfg="$(config_for "$scene")"
  ckpt="$(checkpoint_for "$scene")"
  warmstart="$(warmstart_for "$scene")"
  result_dir="$(result_dir_for "$scene")"
  result_json="${result_dir}/scannet_pointcloud_radio_gs_results.json"

  if [[ ! -f "$cfg" ]]; then
    log "missing config: $cfg"
    exit 3
  fi
  if [[ ! -f "$warmstart" ]]; then
    log "missing warmstart: $warmstart"
    exit 3
  fi
  if [[ -f "$result_json" ]]; then
    log "skip completed scene=$scene result=$result_json"
    continue
  fi

  log "train start scene=$scene gpu=$CUDA_VISIBLE_DEVICES cfg=$cfg"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/train_feature_field.py --config "$cfg" --warmstart "$warmstart"
  else
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/train_feature_field.py \
      --config "$cfg" \
      --warmstart "$warmstart"
  fi

  log "eval start scene=$scene"
  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py --scene "$scene" --prepared_root dataset/scannet_og --config "$cfg" --checkpoint "$ckpt" --output_dir "$result_dir" --class_splits 19,15,10 --query_mode gaussian_index --gaussian_index_position_mode label_point --opacity_filter_mode label_index --opacity_threshold 0.1 --save_logits_npz --save_feature_rgb_ply --save_ply --text_embedding_cache "$TEXT_CACHE" --prompt_templates "$PROMPT_TEMPLATES"
  else
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py \
      --scene "$scene" \
      --prepared_root dataset/scannet_og \
      --config "$cfg" \
      --checkpoint "$ckpt" \
      --output_dir "$result_dir" \
      --class_splits 19,15,10 \
      --query_mode gaussian_index \
      --gaussian_index_position_mode label_point \
      --opacity_filter_mode label_index \
      --opacity_threshold 0.1 \
      --save_logits_npz \
      --save_feature_rgb_ply \
      --save_ply \
      --text_embedding_cache "$TEXT_CACHE" \
      --prompt_templates "$PROMPT_TEMPLATES"
  fi

  log "done scene=$scene"
done

log "queue complete"
