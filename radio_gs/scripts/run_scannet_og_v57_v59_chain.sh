#!/usr/bin/env bash
# Run the current direct RADIO-GS ScanNet v57 -> teacher cache -> v59 chain
# on one GPU in a single persistent process.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <scene> [scene ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
SCENES=("$@")

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PY_WRAPPER="${PY_WRAPPER:-radio_gs/scripts/run_repo_python.sh}"
PREPARED_ROOT="${PREPARED_ROOT:-dataset/scannet_og}"
CONFIG_ROOT="${CONFIG_ROOT:-radio_gs/configs/generated/scannet_og}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/radio_gs}"
QUEUE_ROOT="${QUEUE_ROOT:-output/radio_gs/pipeline_queue}"
EVAL_ROOT="${EVAL_ROOT:-output/scannet_pointcloud_eval}"
TEACHER_EVAL_ROOT="${TEACHER_EVAL_ROOT:-output/scannet_pointcloud_teacher_cache_norm}"
TEXT_CACHE="${TEXT_CACHE:-checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt}"
CLASS_SPLITS="${CLASS_SPLITS:-19,15,10}"
OPACITY_THRESHOLD="${OPACITY_THRESHOLD:-0.1}"
TEACHER_MAX_VIEWS="${TEACHER_MAX_VIEWS:-64}"
TEACHER_VIEW_CHUNK_SIZE="${TEACHER_VIEW_CHUNK_SIZE:-8}"
TEACHER_CHUNK_SIZE="${TEACHER_CHUNK_SIZE:-4096}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_CLEAR_STAGE_LOCKS="${FORCE_CLEAR_STAGE_LOCKS:-1}"

V44_VARIANT="${V44_VARIANT:-v44fair_nolabel_pointdecodefix_gidx_tkl6_dp020_s16384_b5_long30}"
V57_VARIANT="${V57_VARIANT:-v57fair_basepce_gidx_dp010_s16384_b5_ft12}"
V59_VARIANT="${V59_VARIANT:-v59fair_cache_centerpce_gidx_dp010_s16384_b5_ft8}"

if [[ -z "${TEACHER_CACHE_PATH+x}" ]]; then
  TEACHER_CACHE_PATH='output/scannet_teacher_cache_norm/{scene}_radio_teacher_features.pt'
fi
if [[ -z "${PROMPT_TEMPLATES+x}" ]]; then
  PROMPT_TEMPLATES='{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}'
fi

LOCK_PATH=""

log() {
  echo "[$(date '+%F %T')] $*"
}

cleanup() {
  if [[ -n "${LOCK_PATH}" ]]; then
    rm -rf "${LOCK_PATH}"
  fi
}

print_cmd() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

acquire_gpu_lock() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1; not acquiring GPU${GPU_ID} lock"
    return 0
  fi

  local lock_root="${GPU_LOCK_DIR:-/tmp/radio_gs_gpu_locks}"
  mkdir -p "${lock_root}"
  LOCK_PATH="${lock_root}/gpu_${GPU_ID}.lock"
  if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
    echo "GPU${GPU_ID} lock already exists: ${LOCK_PATH}" >&2
    echo "Remove it only after confirming the recorded pid is not running." >&2
    exit 1
  fi
  echo "$$" > "${LOCK_PATH}/pid"
  trap cleanup EXIT
}

run_step() {
  local name="$1"
  shift
  local log_path="$1"
  shift
  local marker="$1"
  shift

  log "START ${name}"
  mkdir -p "$(dirname "${log_path}")" "$(dirname "${marker}")"

  if [[ -f "${marker}" ]]; then
    log "SKIP ${name}: marker exists"
    return 0
  fi

  if [[ "${FORCE_CLEAR_STAGE_LOCKS}" == "1" ]]; then
    rm -rf "${marker}.lock"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN command for ${name}:"
    print_cmd bash radio_gs/scripts/run_and_mark_success.sh "${marker}" "$@"
    return 0
  fi

  bash radio_gs/scripts/run_and_mark_success.sh "${marker}" "$@" >> "${log_path}" 2>&1
  log "DONE ${name}"
}

cd "${REPO_ROOT}"
acquire_gpu_lock
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SELECTED_GPU="${GPU_ID}"
export PYTHONUNBUFFERED=1

for scene in "${SCENES[@]}"; do
  scene_key="${scene%_00}"
  v44_checkpoint="${OUTPUT_ROOT}/scannet_og_${scene}_${V44_VARIANT}/checkpoints/best.pth"
  v57_config="${CONFIG_ROOT}/scannet_og_hybrid_${V57_VARIANT}_${scene}.yaml"
  v57_checkpoint="${OUTPUT_ROOT}/scannet_og_${scene}_${V57_VARIANT}/checkpoints/best.pth"
  v59_config="${CONFIG_ROOT}/scannet_og_hybrid_${V59_VARIANT}_${scene}.yaml"
  v59_checkpoint="${OUTPUT_ROOT}/scannet_og_${scene}_${V59_VARIANT}/checkpoints/best.pth"

  run_step "${scene_key}_v57_train" \
    "${QUEUE_ROOT}/scannet_${scene_key}_v57s16_b5_train_gpu${GPU_ID}_direct.log" \
    "${QUEUE_ROOT}/markers/scannet_${scene_key}_v57s16_b5_train.done" \
    bash "${PY_WRAPPER}" radio_gs/scripts/train_feature_field.py \
      --config "${v57_config}" \
      --warmstart "${v44_checkpoint}"

  run_step "${scene_key}_v57_eval" \
    "${QUEUE_ROOT}/scannet_${scene_key}_v57s16_b5_eval_gpu${GPU_ID}_direct.log" \
    "${QUEUE_ROOT}/markers/scannet_${scene_key}_v57s16_b5_eval.done" \
    bash "${PY_WRAPPER}" radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py \
      --scene "${scene}" \
      --prepared_root "${PREPARED_ROOT}" \
      --config "${v57_config}" \
      --checkpoint "${v57_checkpoint}" \
      --output_dir "${EVAL_ROOT}/${scene}_v57_basepce_noadapter_gidx_labelpoint_s16384_b5" \
      --class_splits "${CLASS_SPLITS}" \
      --query_mode gaussian_index \
      --gaussian_index_position_mode label_point \
      --opacity_filter_mode label_index \
      --opacity_threshold "${OPACITY_THRESHOLD}" \
      --save_logits_npz \
      --save_feature_rgb_ply \
      --save_ply \
      --text_embedding_cache "${TEXT_CACHE}" \
      --prompt_templates "${PROMPT_TEMPLATES}"

  run_step "${scene_key}_teacher_cache_norm" \
    "${QUEUE_ROOT}/scannet_${scene_key}_teacher_cache_norm_gpu${GPU_ID}.log" \
    "${QUEUE_ROOT}/markers/scannet_${scene_key}_teacher_cache_norm.done" \
    bash "${PY_WRAPPER}" radio_gs/scripts/eval_scannet_pointcloud_radio_teacher.py \
      --scene "${scene}" \
      --prepared_root "${PREPARED_ROOT}" \
      --config "${v57_config}" \
      --output_dir "${TEACHER_EVAL_ROOT}/${scene}" \
      --class_splits "${CLASS_SPLITS}" \
      --max_views "${TEACHER_MAX_VIEWS}" \
      --teacher_split train \
      --view_chunk_size "${TEACHER_VIEW_CHUNK_SIZE}" \
      --chunk_size "${TEACHER_CHUNK_SIZE}" \
      --text_embedding_cache "${TEXT_CACHE}" \
      --prompt_templates "${PROMPT_TEMPLATES}" \
      --save_teacher_cache \
      --teacher_cache_path "${TEACHER_CACHE_PATH}" \
      --normalize_teacher_features \
      --save_logits_npz \
      --save_feature_rgb_ply \
      --save_ply

  run_step "${scene_key}_v59_train" \
    "${QUEUE_ROOT}/scannet_${scene_key}_v59s16_b5_centerpce_train_gpu${GPU_ID}_direct.log" \
    "${QUEUE_ROOT}/markers/scannet_${scene_key}_v59s16_b5_centerpce_train.done" \
    bash "${PY_WRAPPER}" radio_gs/scripts/train_feature_field.py \
      --config "${v59_config}" \
      --warmstart "${v57_checkpoint}"

  run_step "${scene_key}_v59_eval" \
    "${QUEUE_ROOT}/scannet_${scene_key}_v59s16_b5_centerpce_eval_gpu${GPU_ID}_direct.log" \
    "${QUEUE_ROOT}/markers/scannet_${scene_key}_v59s16_b5_centerpce_eval.done" \
    bash "${PY_WRAPPER}" radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py \
      --scene "${scene}" \
      --prepared_root "${PREPARED_ROOT}" \
      --config "${v59_config}" \
      --checkpoint "${v59_checkpoint}" \
      --output_dir "${EVAL_ROOT}/${scene}_v59_cache_centerpce_noadapter_gidx_labelpoint_s16384_b5" \
      --class_splits "${CLASS_SPLITS}" \
      --query_mode gaussian_index \
      --gaussian_index_position_mode label_point \
      --opacity_filter_mode label_index \
      --opacity_threshold "${OPACITY_THRESHOLD}" \
      --save_logits_npz \
      --save_feature_rgb_ply \
      --save_ply \
      --text_embedding_cache "${TEXT_CACHE}" \
      --prompt_templates "${PROMPT_TEMPLATES}"

  log "DONE ${scene} chain"
done

log "all requested scene chains complete"
