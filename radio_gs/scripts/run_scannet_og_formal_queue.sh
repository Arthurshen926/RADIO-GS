#!/usr/bin/env bash
# Run the formal OpenGaussian ScanNet RADIO-GS queue on one GPU.
#
# Each scene runs:
#   prepare/check inputs -> RGB 3DGS geometry -> RADIO extraction -> config
#   generation -> RADIO-GS feature-field training -> direct point-cloud eval.
#
# Usage:
#   CUDA_VISIBLE_DEVICES is set by this script from the first argument.
#   bash radio_gs/scripts/run_scannet_og_formal_queue.sh 4 scene0000_00 scene0070_00

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu_id> <scene> [scene ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
SCENES=("$@")

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PY_WRAPPER="${PY_WRAPPER:-radio_gs/scripts/run_repo_python.sh}"
PREPARED_ROOT="${PREPARED_ROOT:-dataset/scannet_og}"
GEOM_TAG="${GEOM_TAG:-og_rgb_3dgs}"
GEOM_ITERS="${GEOM_ITERS:-30000}"
GEOM_OUTPUT_ROOT="${GEOM_OUTPUT_ROOT:-output/3dgs_models/scannet_og}"
FEATURE_ROOT="${FEATURE_ROOT:-output/radio_features_scannet_og}"
CONFIG_ROOT="${CONFIG_ROOT:-radio_gs/configs/generated/scannet_og}"
EVAL_ROOT="${EVAL_ROOT:-output/scannet_pointcloud_eval}"
QUEUE_ROOT="${QUEUE_ROOT:-output/queues/scannet_og_formal_${RUN_ID}}"

GEOM_EXTRA_ARGS="${GEOM_EXTRA_ARGS:-}"
FEATURE_EXTRA_ARGS="${FEATURE_EXTRA_ARGS:-}"
TRAIN_FEATURE_EXTRA_ARGS="${TRAIN_FEATURE_EXTRA_ARGS:-}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
SAVE_EVAL_PLY="${SAVE_EVAL_PLY:-1}"
SAVE_FEATURE_RGB_PLY="${SAVE_FEATURE_RGB_PLY:-1}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-1}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-4096}"
TEXT_CACHE="${TEXT_CACHE:-checkpoints/siglip2_scannet_og_text_embeddings.pt}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-3}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
SCANNET_VARIANT="${SCANNET_VARIANT:-v14}"
WARMSTART_VARIANT="${WARMSTART_VARIANT:-}"
WARMSTART_PATH="${WARMSTART_PATH:-}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"
SIGLIP_SPATIAL_ALIGNMENT_WEIGHT="${SIGLIP_SPATIAL_ALIGNMENT_WEIGHT:-}"
DIRECT_POINT_LOSS_WEIGHT="${DIRECT_POINT_LOSS_WEIGHT:-0.0}"
DIRECT_POINT_SAMPLE_COUNT="${DIRECT_POINT_SAMPLE_COUNT:-2048}"
DIRECT_POINT_QUERY_MODE="${DIRECT_POINT_QUERY_MODE:-gaussian_index}"
DIRECT_POINT_K="${DIRECT_POINT_K:-8}"
EVAL_QUERY_MODE="${EVAL_QUERY_MODE:-${DIRECT_POINT_QUERY_MODE}}"
EVAL_OPACITY_FILTER_MODE="${EVAL_OPACITY_FILTER_MODE:-label_index}"

mkdir -p "${QUEUE_ROOT}"
STATUS_FILE="${QUEUE_ROOT}/gpu${GPU_ID}_status.tsv"
PROGRESS_FILE="${QUEUE_ROOT}/progress.tsv"
LATEST_LINK="${REPO_ROOT}/output/queues/scannet_og_formal_latest"

if [[ "${UPDATE_LATEST:-1}" == "1" ]]; then
  ln -sfn "$(basename "${QUEUE_ROOT}")" "${LATEST_LINK}"
fi
touch "${PROGRESS_FILE}"

log() {
  echo "[$(date '+%F %T')] [gpu${GPU_ID}] $*"
}

run_gpu() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "$@"
}

mark_status() {
  local scene="$1"
  local stage="$2"
  local status="$3"
  local line
  line="$(printf '%s\tgpu%s\t%s\t%s\t%s' "$(date '+%F %T')" "${GPU_ID}" "${scene}" "${stage}" "${status}")"
  printf '%s\n' "${line}" >> "${STATUS_FILE}"
  printf '%s\n' "${line}" >> "${PROGRESS_FILE}"
}

for scene in "${SCENES[@]}"; do
  scene_root="${PREPARED_ROOT}/${scene}"
  geom_ply="${GEOM_OUTPUT_ROOT}/${scene}/${GEOM_TAG}/point_cloud/iteration_${GEOM_ITERS}/point_cloud.ply"
  feature_manifest="${FEATURE_ROOT}/${scene}/frame_manifest.json"
  config_path="${CONFIG_ROOT}/scannet_og_hybrid_${SCANNET_VARIANT}_${scene}.yaml"
  checkpoint_path="output/radio_gs/scannet_og_${scene}_${SCANNET_VARIANT}/checkpoints/best.pth"
  eval_scene_dir="${EVAL_ROOT}/${scene}_${SCANNET_VARIANT}"
  if [[ "${SCANNET_VARIANT}" == "v14" ]]; then
    eval_scene_dir="${EVAL_ROOT}/${scene}"
  fi
  eval_json="${eval_scene_dir}/scannet_pointcloud_radio_gs_results.json"

  log "scene ${scene}: starting"
  mark_status "${scene}" "scene" "start"

  if [[ ! -d "${scene_root}" ]]; then
    log "scene ${scene}: prepared root missing, preparing from zip"
    mark_status "${scene}" "prepare" "start"
    bash "${PY_WRAPPER}" radio_gs/scripts/prepare_opengaussian_scannet_scene.py \
      --scene "${scene}" \
      --output_root "${PREPARED_ROOT}" \
      --copy_mode copy
    mark_status "${scene}" "prepare" "done"
  fi

  if [[ ! -f "${geom_ply}" ]]; then
    log "scene ${scene}: training RGB 3DGS geometry"
    mark_status "${scene}" "geometry" "start"
    # shellcheck disable=SC2086
    run_gpu bash "${PY_WRAPPER}" radio_gs/scripts/train_opengaussian_scannet_gs.py \
      --scene_root "${scene_root}" \
      --scene "${scene}" \
      --tag "${GEOM_TAG}" \
      --iters "${GEOM_ITERS}" \
      --output_dir "${GEOM_OUTPUT_ROOT}" \
      --device cuda \
      ${GEOM_EXTRA_ARGS}
    mark_status "${scene}" "geometry" "done"
  else
    log "scene ${scene}: geometry exists, skipping ${geom_ply}"
    mark_status "${scene}" "geometry" "skip"
  fi

  if [[ ! -f "${feature_manifest}" ]]; then
    log "scene ${scene}: extracting RADIO features"
    mark_status "${scene}" "features" "start"
    # shellcheck disable=SC2086
    run_gpu bash "${PY_WRAPPER}" -m radio_gs.scripts.extract_radio_features \
      --scene "${scene}" \
      --image_dir "${scene_root}/color" \
      --output_dir "${FEATURE_ROOT}/${scene}" \
      --radio_repo /root/RADIO \
      --radio_version c-radio_v4-h \
      --batch_size "${FEATURE_BATCH_SIZE}" \
      --device cuda \
      ${FEATURE_EXTRA_ARGS}
    mark_status "${scene}" "features" "done"
  else
    log "scene ${scene}: features exist, skipping ${feature_manifest}"
    mark_status "${scene}" "features" "skip"
  fi

  log "scene ${scene}: generating config"
  mark_status "${scene}" "config" "start"
  config_extra_args=()
  if [[ -n "${TRAIN_EPOCHS}" ]]; then
    config_extra_args+=(--epochs "${TRAIN_EPOCHS}")
  fi
  if [[ -n "${SIGLIP_SPATIAL_ALIGNMENT_WEIGHT}" ]]; then
    config_extra_args+=(--siglip_spatial_alignment_weight "${SIGLIP_SPATIAL_ALIGNMENT_WEIGHT}")
  fi
  bash "${PY_WRAPPER}" radio_gs/scripts/generate_scannet_og_configs.py \
    --scenes "${scene}" \
    --prepared_root "${PREPARED_ROOT}" \
    --output_root "${CONFIG_ROOT}" \
    --repo_root "${REPO_ROOT}" \
    --geom_tag "${GEOM_TAG}" \
    --iters "${GEOM_ITERS}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --num_workers "${TRAIN_NUM_WORKERS}" \
    --variant "${SCANNET_VARIANT}" \
    --direct_point_loss_weight "${DIRECT_POINT_LOSS_WEIGHT}" \
    --direct_point_sample_count "${DIRECT_POINT_SAMPLE_COUNT}" \
    --direct_point_query_mode "${DIRECT_POINT_QUERY_MODE}" \
    --direct_point_k "${DIRECT_POINT_K}" \
    "${config_extra_args[@]}"
  mark_status "${scene}" "config" "done"

  if [[ ! -f "${checkpoint_path}" ]]; then
    log "scene ${scene}: training RADIO-GS feature field"
    mark_status "${scene}" "radio_gs" "start"
    warmstart_args=()
    scene_warmstart_path=""
    if [[ -n "${WARMSTART_PATH}" ]]; then
      scene_warmstart_path="${WARMSTART_PATH}"
    elif [[ -n "${WARMSTART_VARIANT}" ]]; then
      scene_warmstart_path="output/radio_gs/scannet_og_${scene}_${WARMSTART_VARIANT}/checkpoints/best.pth"
    fi
    if [[ -n "${scene_warmstart_path}" ]]; then
      if [[ -f "${scene_warmstart_path}" ]]; then
        log "scene ${scene}: warmstarting RADIO-GS from ${scene_warmstart_path}"
        warmstart_args+=(--warmstart "${scene_warmstart_path}")
      else
        log "scene ${scene}: requested warmstart missing, training without it: ${scene_warmstart_path}"
      fi
    fi
    # shellcheck disable=SC2086
    run_gpu bash "${PY_WRAPPER}" radio_gs/scripts/train_feature_field.py \
      --config "${config_path}" \
      "${warmstart_args[@]}" \
      ${TRAIN_FEATURE_EXTRA_ARGS}
    mark_status "${scene}" "radio_gs" "done"
  else
    log "scene ${scene}: RADIO-GS checkpoint exists, skipping ${checkpoint_path}"
    mark_status "${scene}" "radio_gs" "skip"
  fi

  log "scene ${scene}: evaluating direct point cloud understanding"
  mark_status "${scene}" "eval" "start"
  mkdir -p "${eval_scene_dir}"
  eval_output_args=()
  if [[ "${SAVE_EVAL_PLY}" == "1" ]]; then
    eval_output_args+=(--save_ply)
  fi
  if [[ "${SAVE_FEATURE_RGB_PLY}" == "1" ]]; then
    eval_output_args+=(--save_feature_rgb_ply)
  fi
  # shellcheck disable=SC2086
  run_gpu bash "${PY_WRAPPER}" radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py \
    --scene "${scene}" \
    --prepared_root "${PREPARED_ROOT}" \
    --config "${config_path}" \
    --checkpoint "${checkpoint_path}" \
    --output_dir "${eval_scene_dir}" \
    --text_embedding_cache "${TEXT_CACHE}" \
    --chunk_size "${EVAL_CHUNK_SIZE}" \
    --query_mode "${EVAL_QUERY_MODE}" \
    --opacity_filter_mode "${EVAL_OPACITY_FILTER_MODE}" \
    "${eval_output_args[@]}" \
    ${EVAL_EXTRA_ARGS}
  mark_status "${scene}" "eval" "done"

  if [[ -f "${eval_json}" ]]; then
    log "scene ${scene}: complete -> ${eval_json}"
  else
    log "scene ${scene}: complete, eval JSON path not found at expected ${eval_json}"
  fi
  mark_status "${scene}" "scene" "done"
done

log "all assigned scenes complete"
