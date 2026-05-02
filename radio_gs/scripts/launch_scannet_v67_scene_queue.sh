#!/usr/bin/env bash

set -euo pipefail

VARIANT="${VARIANT:-v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63}"
TEXT_CACHE="${TEXT_CACHE:-checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt}"
if [[ -z "${PROMPT_TEMPLATES:-}" ]]; then
  PROMPT_TEMPLATES='{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}'
fi
GPU=""
WAIT_PIDFILE=""
QUEUE_NAME="scannet_v67_queue"
FORCE=0

repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

warmstart_for() {
  case "$1" in
    scene0000_00) echo "output/radio_gs/scannet_og_scene0000_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s20480_b4_long20_fromv60/checkpoints/best.pth" ;;
    scene0062_00) echo "output/radio_gs/scannet_og_scene0062_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s20480_b4_long20_fromv60/checkpoints/best.pth" ;;
    scene0070_00) echo "output/radio_gs/scannet_og_scene0070_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s24576_b4_long20_fromv60/checkpoints/best.pth" ;;
    scene0097_00) echo "output/radio_gs/scannet_og_scene0097_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s20480_b4_long20_fromv60/checkpoints/best.pth" ;;
    scene0140_00) echo "output/radio_gs/scannet_og_scene0140_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s20480_b4_long20_fromv60/checkpoints/best.pth" ;;
    scene0200_00) echo "output/radio_gs/scannet_og_scene0200_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s20480_b4_long20_fromv60/checkpoints/best.pth" ;;
    scene0347_00) echo "output/radio_gs/scannet_og_scene0347_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s20480_b4_long20_fromv60/checkpoints/best.pth" ;;
    scene0400_00) echo "output/radio_gs/scannet_og_scene0400_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s24576_b4_long30_fromv61/checkpoints/best.pth" ;;
    scene0590_00) echo "output/radio_gs/scannet_og_scene0590_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s20480_b4_long20_fromv60/checkpoints/best.pth" ;;
    scene0645_00) echo "output/radio_gs/scannet_og_scene0645_00_v63fair_teacherpce_gidx_labelpoint_dp080_pce20_s24576_b4_long20_fromv60/checkpoints/best.pth" ;;
    *) echo "unknown scene: $1" >&2; return 2 ;;
  esac
}

log() {
  printf '[%s] [%s] %s\n' "$(date '+%F %T')" "$QUEUE_NAME" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  launch_scannet_v67_scene_queue.sh --gpu GPU [--wait-pidfile PATH] [--name NAME] [--force] SCENE...
  launch_scannet_v67_scene_queue.sh --print-warmstart SCENE

Run the v67 ScanNet direct point training/evaluation queue in the foreground.
Launch it with setsid from the caller when it needs to survive terminal cleanup.
EOF
}

if [[ "${1:-}" == "--print-warmstart" ]]; then
  if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
  fi
  warmstart_for "$2"
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
    --wait-pidfile)
      WAIT_PIDFILE="${2:?missing value for --wait-pidfile}"
      shift 2
      ;;
    --name)
      QUEUE_NAME="${2:?missing value for --name}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
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

if [[ -n "$WAIT_PIDFILE" && -f "$WAIT_PIDFILE" ]]; then
  wait_pid="$(cat "$WAIT_PIDFILE")"
  if [[ -n "$wait_pid" ]] && kill -0 "$wait_pid" 2>/dev/null; then
    log "waiting for pid=$wait_pid from $WAIT_PIDFILE"
    while kill -0 "$wait_pid" 2>/dev/null; do
      sleep 20
    done
  fi
fi

for scene in "${SCENES[@]}"; do
  cfg="radio_gs/configs/generated/scannet_og/scannet_og_hybrid_${VARIANT}_${scene}.yaml"
  ckpt="output/radio_gs/scannet_og_${scene}_${VARIANT}/checkpoints/best.pth"
  result_dir="output/scannet_pointcloud_eval/${scene}_v67_teacherbalanced_fromv63_best_gidx_labelpoint"
  result_json="${result_dir}/scannet_pointcloud_radio_gs_results.json"
  warmstart="$(warmstart_for "$scene")"

  if [[ "$FORCE" -eq 0 && -f "$result_json" ]]; then
    log "skip completed scene=$scene result=$result_json"
    continue
  fi
  if [[ ! -f "$cfg" ]]; then
    log "missing config: $cfg"
    exit 3
  fi
  if [[ ! -f "$warmstart" ]]; then
    log "missing warmstart: $warmstart"
    exit 3
  fi

  log "train start scene=$scene gpu=$CUDA_VISIBLE_DEVICES warmstart=$warmstart"
  bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/train_feature_field.py \
    --config "$cfg" \
    --warmstart "$warmstart"

  log "eval start scene=$scene"
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

  for split in 19 15 10; do
    pred="${result_dir}/visualizations/${scene}/pred_split_${split}.ply"
    if [[ -f "$pred" ]]; then
      bash radio_gs/scripts/run_repo_python.sh \
        radio_gs/scripts/make_scannet_gt_error_vis.py \
        --pred_ply "$pred" \
        --split "$split" \
        --output_dir "${result_dir}/visualizations/${scene}" || true
    fi
  done
  log "done scene=$scene"
done

log "queue complete"
