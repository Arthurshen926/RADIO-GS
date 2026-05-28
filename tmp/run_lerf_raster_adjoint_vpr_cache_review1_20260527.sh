#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/RADIO-GS"
cd "${REPO_ROOT}"

GPU="${1:-4}"
SCENE="${2:-figurines}"
MAX_FRAMES="${3:-16}"
STAMP="${4:-raster_adjoint_vpr_mf${MAX_FRAMES}_20260527}"

base_config_for_scene() {
  case "$1" in
    figurines) echo "radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml" ;;
    ramen) echo "radio_gs/configs/lerf_hybrid_v14_ramen_fdh_ws240_240ep.yaml" ;;
    teatime) echo "radio_gs/configs/lerf_hybrid_v14_teatime_fdh_ws240_240ep.yaml" ;;
    waldo_kitchen) echo "radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep.yaml" ;;
    *) return 1 ;;
  esac
}

checkpoint_for_scene() {
  case "$1" in
    figurines) echo "output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth" ;;
    ramen) echo "output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth" ;;
    teatime) echo "output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep/checkpoints/best.pth" ;;
    waldo_kitchen) echo "output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth" ;;
    *) return 1 ;;
  esac
}

base="$(base_config_for_scene "${SCENE}")"
ckpt="$(checkpoint_for_scene "${SCENE}")"
cache_dir="output/radio_gs/vpr_feature_cache/raster_adjoint_alpha_20260527"
score_dir="output/radio_gs/score_cache/raster_adjoint_alpha_20260527"
out="output/radio_gs/lerf_${SCENE}_${STAMP}_registered_eval"
mkdir -p "${cache_dir}" "${score_dir}" "${out}"

echo "[raster-adjoint-vpr] gpu=${GPU} scene=${SCENE} max_frames=${MAX_FRAMES}"
CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/eval_lerf_direct_3d_selection.py \
  --config "${base}" \
  --checkpoint "${ckpt}" \
  --scene "${SCENE}" \
  --output_dir "${out}" \
  --score_source registered_view \
  --score_cache "${score_dir}/${SCENE}_mf${MAX_FRAMES}.pt" \
  --registered_feature_cache "${cache_dir}/${SCENE}_mf${MAX_FRAMES}.pt" \
  --scoring softmax_scene \
  --softmax_temperature 50 \
  --registration_frame_mode all_poses \
  --registration_max_frames "${MAX_FRAMES}" \
  --registration_assignment_mode raster_adjoint \
  --registration_alpha_threshold 0.02 \
  --registered_view_fallback direct \
  --selection_mode score_threshold \
  --threshold_sweep 0.15,0.25,0.35,0.45,0.55 \
  --selection_min_ratio 0.005 \
  --silhouette_threshold 0.6 \
  --mask_refinement rgb_grabcut_largest_component
