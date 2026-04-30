#!/usr/bin/env bash
# Run v60 multi-split direct point pseudoCE train/eval for prepared ScanNet scenes.

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
TEXT_CACHE="${TEXT_CACHE:-checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt}"
CLASS_SPLITS="${CLASS_SPLITS:-19,15,10}"
OPACITY_THRESHOLD="${OPACITY_THRESHOLD:-0.1}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_CLEAR_STAGE_LOCKS="${FORCE_CLEAR_STAGE_LOCKS:-1}"
SKIP_GPU_LOCK="${SKIP_GPU_LOCK:-0}"

V59_VARIANT_S16384="${V59_VARIANT_S16384:-v59fair_cache_centerpce_gidx_dp010_s16384_b5_ft8}"
V59_VARIANT_S8192="${V59_VARIANT_S8192:-v59fair_cache_centerpce_gidx_dp010_s8192_b4_ft8}"
V60_VARIANT_S16384="${V60_VARIANT_S16384:-v60fair_cache_multisplit_centerpce_gidx_dp010_s16384_b5_ft6}"
V60_VARIANT_S8192="${V60_VARIANT_S8192:-v60fair_cache_multisplit_centerpce_gidx_dp010_s8192_b4_ft6}"

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
  if [[ "${SKIP_GPU_LOCK}" == "1" ]]; then
    log "SKIP_GPU_LOCK=1; assuming caller owns GPU${GPU_ID} lock"
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

resolve_variants() {
  local scene="$1"
  local config_s16384="${CONFIG_ROOT}/scannet_og_hybrid_${V60_VARIANT_S16384}_${scene}.yaml"
  local config_s8192="${CONFIG_ROOT}/scannet_og_hybrid_${V60_VARIANT_S8192}_${scene}.yaml"

  if [[ -f "${config_s16384}" ]]; then
    echo "${V60_VARIANT_S16384} ${V59_VARIANT_S16384}"
  elif [[ -f "${config_s8192}" ]]; then
    echo "${V60_VARIANT_S8192} ${V59_VARIANT_S8192}"
  else
    echo "No v60 config found for ${scene}: checked ${config_s16384} and ${config_s8192}" >&2
    exit 1
  fi
}

cd "${REPO_ROOT}"
acquire_gpu_lock
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export SELECTED_GPU="${GPU_ID}"
export PYTHONUNBUFFERED=1

for scene in "${SCENES[@]}"; do
  scene_key="${scene%_00}"
  read -r v60_variant v59_variant < <(resolve_variants "${scene}")
  v60_config="${CONFIG_ROOT}/scannet_og_hybrid_${v60_variant}_${scene}.yaml"
  v60_checkpoint="${OUTPUT_ROOT}/scannet_og_${scene}_${v60_variant}/checkpoints/best.pth"
  v59_checkpoint="${OUTPUT_ROOT}/scannet_og_${scene}_${v59_variant}/checkpoints/best.pth"

  if [[ ! -f "${v59_checkpoint}" ]]; then
    echo "Warmstart checkpoint missing for ${scene}: ${v59_checkpoint}" >&2
    exit 1
  fi

  run_step "${scene_key}_v60_train" \
    "${QUEUE_ROOT}/scannet_${scene_key}_v60_multisplit_train_gpu${GPU_ID}.log" \
    "${QUEUE_ROOT}/markers/scannet_${scene_key}_v60_multisplit_train.done" \
    bash "${PY_WRAPPER}" radio_gs/scripts/train_feature_field.py \
      --config "${v60_config}" \
      --warmstart "${v59_checkpoint}"

  run_step "${scene_key}_v60_eval" \
    "${QUEUE_ROOT}/scannet_${scene_key}_v60_multisplit_eval_gpu${GPU_ID}.log" \
    "${QUEUE_ROOT}/markers/scannet_${scene_key}_v60_multisplit_eval.done" \
    bash "${PY_WRAPPER}" radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py \
      --scene "${scene}" \
      --prepared_root "${PREPARED_ROOT}" \
      --config "${v60_config}" \
      --checkpoint "${v60_checkpoint}" \
      --output_dir "${EVAL_ROOT}/${scene}_v60_multisplit_centerpce_noadapter_gidx_labelpoint" \
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

  log "DONE ${scene} v60 queue"
done

log "all requested v60 scenes complete"
