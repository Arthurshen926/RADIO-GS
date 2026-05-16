#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/RADIO-GS"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="output/radio_gs/logs/sam3_object_boundary_${STAMP}"
DIRECT_OUT="output/radio_gs/lerf_direct3d_sam3_object_boundary_${STAMP}"
GROUND_OUT="output/radio_gs/lerf_grounding_sam3_object_boundary_${STAMP}"
mkdir -p "$LOG_DIR" "$DIRECT_OUT" "$GROUND_OUT"

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local apps
    apps="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d '\r' || true)"
    if [[ -z "$apps" ]]; then
      printf '[%s] GPU%s free\n' "$(date '+%F %T')" "$gpu" | tee -a "$LOG_DIR/queue.log"
      return
    fi
    printf '[%s] GPU%s busy: %s\n' "$(date '+%F %T')" "$gpu" "$apps" | tee -a "$LOG_DIR/queue.log"
    sleep "${GPU_WAIT_SECONDS:-60}"
  done
}

scene_config() {
  case "$1" in
    figurines) echo "radio_gs/configs/lerf_hybrid_v14_figurines_sam3_object_boundary_ft.yaml" ;;
    ramen) echo "radio_gs/configs/lerf_hybrid_v14_ramen_sam3_object_boundary_ft.yaml" ;;
    teatime) echo "radio_gs/configs/lerf_hybrid_v14_teatime_sam3_object_boundary_ft.yaml" ;;
    waldo_kitchen) echo "radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_sam3_object_boundary_ft.yaml" ;;
    *) echo "unknown scene: $1" >&2; return 1 ;;
  esac
}

scene_output() {
  case "$1" in
    figurines) echo "output/radio_gs/lerf_figurines_sam3_object_boundary_ft" ;;
    ramen) echo "output/radio_gs/lerf_ramen_sam3_object_boundary_ft" ;;
    teatime) echo "output/radio_gs/lerf_teatime_sam3_object_boundary_ft" ;;
    waldo_kitchen) echo "output/radio_gs/lerf_waldo_kitchen_sam3_object_boundary_ft" ;;
    *) echo "unknown scene: $1" >&2; return 1 ;;
  esac
}

scene_grounding_temp() {
  case "$1" in
    figurines) echo "50" ;;
    ramen) echo "40" ;;
    teatime) echo "25" ;;
    waldo_kitchen) echo "25" ;;
    *) echo "50" ;;
  esac
}

scene_threshold_sweep() {
  case "$1" in
    figurines) echo "0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.18,0.20,0.24" ;;
    ramen) echo "0.08,0.10,0.12,0.14,0.15,0.16,0.18,0.20,0.22,0.24,0.26,0.30" ;;
    teatime) echo "0.30,0.32,0.34,0.36,0.38,0.40,0.42,0.44,0.46,0.48,0.50" ;;
    waldo_kitchen) echo "0.20,0.25,0.30,0.32,0.34,0.35,0.36,0.38,0.40,0.45,0.50,0.55" ;;
    *) echo "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" ;;
  esac
}

scene_max_ratio() {
  case "$1" in
    waldo_kitchen) echo "0.07" ;;
    ramen) echo "0.03" ;;
    *) echo "0.06" ;;
  esac
}

scene_mask_refinement() {
  case "$1" in
    figurines) echo "rgb_grabcut_largest_component" ;;
    ramen) echo "rgb_grabcut_largest_component" ;;
    *) echo "rgb_grabcut" ;;
  esac
}

checkpoint_for_scene() {
  local scene="$1"
  local out_dir
  out_dir="$(scene_output "$scene")"
  if [[ -f "$out_dir/checkpoints/best.pth" ]]; then
    echo "$out_dir/checkpoints/best.pth"
  else
    echo "$out_dir/checkpoints/latest.pth"
  fi
}

run_scene() {
  local gpu="$1"
  local scene="$2"
  local cfg
  cfg="$(scene_config "$scene")"
  local train_log="$LOG_DIR/${scene}_train_gpu${gpu}.log"
  local direct_log="$LOG_DIR/${scene}_direct3d_gpu${gpu}.log"
  local grounding_log="$LOG_DIR/${scene}_grounding_gpu${gpu}.log"

  wait_for_gpu "$gpu"
  printf '[%s] Start train %s on GPU%s\n' "$(date '+%F %T')" "$scene" "$gpu" | tee -a "$LOG_DIR/queue.log"
  CUDA_VISIBLE_DEVICES="$gpu" bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.scripts.train_feature_field \
    --config "$cfg" 2>&1 | tee "$train_log"

  local ckpt
  ckpt="$(checkpoint_for_scene "$scene")"
  if [[ ! -f "$ckpt" ]]; then
    printf '[%s] Missing checkpoint after %s: %s\n' "$(date '+%F %T')" "$scene" "$ckpt" | tee -a "$LOG_DIR/queue.log"
    return 1
  fi

  printf '[%s] Eval direct3d %s on GPU%s\n' "$(date '+%F %T')" "$scene" "$gpu" | tee -a "$LOG_DIR/queue.log"
  CUDA_VISIBLE_DEVICES="$gpu" bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.scripts.eval_lerf_direct_3d_selection \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --scene "$scene" \
    --output_dir "$DIRECT_OUT" \
    --score_source direct \
    --use_point_summary_adapter \
    --scoring softmax_scene \
    --softmax_temperature 50 \
    --selection_mode score_threshold \
    --threshold_sweep "$(scene_threshold_sweep "$scene")" \
    --score_threshold 0.25 \
    --selection_min_ratio 0.003 \
    --selection_max_ratio "$(scene_max_ratio "$scene")" \
    --silhouette_threshold 0.55 \
    --mask_refinement "$(scene_mask_refinement "$scene")" \
    --mask_refinement_iters 1 \
    --save_masks \
    --gpu 0 2>&1 | tee "$direct_log"

  printf '[%s] Eval rendered grounding %s on GPU%s\n' "$(date '+%F %T')" "$scene" "$gpu" | tee -a "$LOG_DIR/queue.log"
  CUDA_VISIBLE_DEVICES="$gpu" bash radio_gs/scripts/run_repo_python.sh \
    -m radio_gs.scripts.eval_lerf_grounding \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --scene "$scene" \
    --output_dir "$GROUND_OUT" \
    --text_embedding_cache checkpoints/siglip2_lerf_text_embeddings.pt \
    --scoring softmax_scene \
    --relevancy_temp "$(scene_grounding_temp "$scene")" \
    --iou_threshold 0.60 \
    --mask_refinement rgb_grabcut \
    --mask_refinement_iters 2 \
    --gpu 0 2>&1 | tee "$grounding_log"
}

run_lane() {
  local gpu="$1"
  shift
  local scene
  for scene in "$@"; do
    run_scene "$gpu" "$scene"
  done
}

run_lane 4 figurines ramen &
pid4="$!"
run_lane 5 teatime waldo_kitchen &
pid5="$!"
wait "$pid4" "$pid5"

printf '[%s] SAM3 object-boundary queue complete. Logs: %s\n' "$(date '+%F %T')" "$LOG_DIR" | tee -a "$LOG_DIR/queue.log"
