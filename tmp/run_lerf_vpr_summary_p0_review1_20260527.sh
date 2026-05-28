#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/RADIO-GS"
cd "${REPO_ROOT}"

GPU="${1:-5}"
VARIANT="${2:-vpr_summary_p0_ft20_20260527}"
CFG_ROOT="tmp/lerf_${VARIANT}"
SCENES=(figurines ramen teatime waldo_kitchen)

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

mkdir -p "${CFG_ROOT}"

for scene in "${SCENES[@]}"; do
  base="$(base_config_for_scene "${scene}")"
  cache="output/radio_gs/vpr_feature_cache/max128_softmax_20260515/${scene}.pt"
  run="output/radio_gs/lerf_${scene}_${VARIANT}"
  cfg="${CFG_ROOT}/lerf_${scene}_${VARIANT}.yaml"
  bash radio_gs/scripts/run_repo_python.sh - "${base}" "${cfg}" "${cache}" "${run}" "${scene}" <<'PY'
import sys
import yaml

base, out, cache, run, scene = sys.argv[1:]
with open(base, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
root = f"/mnt/pool/sqy/3d_understanding/lerf_ovs/{scene}"
cfg.update(
    {
        "scene_root": root,
        "rgb_dir": f"{root}/images",
        "output_dir": run,
        "exp_name": run.rsplit("/", 1)[-1],
        "epochs": 20,
        "save_every": 5,
        "eval_every": 5,
        "log_every": 20,
        "direct_point_loss_weight": 0.5,
        "direct_point_sample_count": 32768,
        "direct_point_sample_strategy": "uniform",
        "direct_point_source": "gaussian",
        "direct_point_query_mode": "gaussian_index",
        "direct_point_gaussian_position_mode": "gaussian_center",
        "direct_point_teacher_cache": cache,
        "direct_point_teacher_cache_feature_key": "summary_features",
        "direct_point_teacher_cache_feature_space": "siglip_summary",
        "direct_point_teacher_cache_require_xyz_alignment": True,
        "direct_point_teacher_cache_fail_max_l2": 1.0e-5,
        "direct_point_summary_adapter_weight": 1.0,
        "direct_point_view_count_weighting": "clipped_log",
        "direct_point_view_count_min_weight": 0.25,
        "direct_point_proposal_consistency_weight": 0.05,
        "direct_point_proposal_voxel_size": 0.05,
        "direct_point_proposal_min_count": 2,
        "direct_point_proposal_space": "adapter",
        "lr_point_summary_adapter": 1.0e-4,
    }
)
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out)
PY

  echo "[lerf-vpr-summary-p0] gpu=${GPU} scene=${scene} cfg=${cfg}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/train_feature_field.py \
    --config "${cfg}" \
    --warmstart "$(checkpoint_for_scene "${scene}")"
done

for scene in "${SCENES[@]}"; do
  cfg="${CFG_ROOT}/lerf_${scene}_${VARIANT}.yaml"
  run="output/radio_gs/lerf_${scene}_${VARIANT}"
  out="output/radio_gs/lerf_${scene}_${VARIANT}_direct_eval"
  echo "[lerf-vpr-summary-p0-eval] gpu=${GPU} scene=${scene}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_lerf_direct_3d_selection.py \
    --config "${cfg}" \
    --checkpoint "${run}/checkpoints/best.pth" \
    --scene "${scene}" \
    --output_dir "${out}" \
    --score_source direct \
    --scoring softmax_scene \
    --softmax_temperature 50 \
    --selection_mode score_threshold \
    --threshold_sweep 0.15,0.25,0.35,0.45,0.55 \
    --selection_min_ratio 0.005 \
    --use_point_summary_adapter \
    --point_summary_adapter_blend_alpha 1.0 \
    --point_summary_adapter_valid_mask_mode teacher_cache \
    --direct_primitive_confidence_mode teacher_cache_valid \
    --direct_primitive_confidence_blend 1.0 \
    --direct_primitive_opacity_threshold 0.02 \
    --silhouette_threshold 0.6 \
    --mask_refinement rgb_grabcut_largest_component
done
