#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/RADIO-GS"
cd "${REPO_ROOT}"

GPU="${1:-1}"
VARIANT="${2:-dino_maskprop_relcv_textheat_ft35_b4_ramenteatime_20260528_bestreadout}"
TRAIN_VARIANT="${3:-dino_maskprop_relcv_textheat_ft35_b4_ramenteatime_20260528}"
shift 3 || true
SCENES=("$@")
if [[ ${#SCENES[@]} -eq 0 ]]; then
  SCENES=(ramen teatime)
fi

for SCENE in "${SCENES[@]}"; do
  CFG="tmp/lerf_${TRAIN_VARIANT}/lerf_${SCENE}_${TRAIN_VARIANT}.yaml"
  CKPT="output/radio_gs/lerf_${SCENE}_${TRAIN_VARIANT}/checkpoints/best.pth"
  OUT="output/lerf_sam_dino_tasks/${VARIANT}/${SCENE}"
  if [[ ! -f "${CFG}" || ! -f "${CKPT}" ]]; then
    echo "missing DINO trained config/checkpoint for ${SCENE}: ${CFG} ${CKPT}" >&2
    exit 3
  fi
  echo "[dino-bestreadout-eval] gpu=${GPU} scene=${SCENE}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_lerf_sam_dino_tasks.py \
    --config "${CFG}" \
    --checkpoint "${CKPT}" \
    --scene "${SCENE}" \
    --output_dir "${OUT}" \
    --max_visuals 0 \
    --dino_background_contrast 1.1 \
    --dino_foreground_pool topk_mean \
    --dino_area_scale 2.0 \
    --dino_component_cleanup peak \
    --dino_match_mutual \
    --dino_match_ransac_model homography \
    --dino_match_ransac_threshold 4.0 \
    --dino_match_ransac_min_inliers 4 \
    --dino_propagation_seed_prior \
    --dino_propagation_seed_weight 2.0 \
    --dino_propagation_seed_radius 0
done

result_files=()
for SCENE in "${SCENES[@]}"; do
  result_files+=("output/lerf_sam_dino_tasks/${VARIANT}/${SCENE}/lerf_sam_dino_task_results.json")
done
bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/aggregate_lerf_sam_dino_tasks.py \
  "${result_files[@]}" \
  --output_dir "output/lerf_sam_dino_tasks/${VARIANT}"
