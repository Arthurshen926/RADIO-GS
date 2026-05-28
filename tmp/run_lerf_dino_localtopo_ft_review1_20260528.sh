#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/RADIO-GS"
cd "${REPO_ROOT}"

GPU="${1:-0}"
VARIANT="${2:-dino_localtopo_distinctive_maskprop_ft25_b4_20260528}"
BATCH_SIZE="${3:-4}"
SCENES=("${@:4}")
if [[ ${#SCENES[@]} -eq 0 ]]; then
  SCENES=(figurines waldo_kitchen)
fi

EPOCHS="${EPOCHS:-25}"
ALIGN_WEIGHT="${ALIGN_WEIGHT:-0.015}"
RELATION_WEIGHT="${RELATION_WEIGHT:-0.04}"
LOCAL_AFFINITY_WEIGHT="${LOCAL_AFFINITY_WEIGHT:-0.02}"
LOCAL_AFFINITY_DOWNSAMPLE="${LOCAL_AFFINITY_DOWNSAMPLE:-1}"
LOCAL_AFFINITY_RADIUS="${LOCAL_AFFINITY_RADIUS:-1}"
TOKEN_CONTRAST_WEIGHT="${TOKEN_CONTRAST_WEIGHT:-0.01}"
PEAK_BACKGROUND_WEIGHT="${PEAK_BACKGROUND_WEIGHT:-0.012}"
PEAK_BACKGROUND_ANCHOR_STRATEGY="${PEAK_BACKGROUND_ANCHOR_STRATEGY:-distinctive}"
CROSS_VIEW_WEIGHT="${CROSS_VIEW_WEIGHT:-0.005}"
CROSS_VIEW_PROP_WEIGHT="${CROSS_VIEW_PROP_WEIGHT:-0.03}"
CROSS_VIEW_MASK_PROP_WEIGHT="${CROSS_VIEW_MASK_PROP_WEIGHT:-0.04}"
CROSS_VIEW_ANCHOR_STRATEGY="${CROSS_VIEW_ANCHOR_STRATEGY:-distinctive}"

CFG_ROOT="tmp/lerf_${VARIANT}"
mkdir -p "${CFG_ROOT}"

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

for scene in "${SCENES[@]}"; do
  base="$(base_config_for_scene "${scene}")"
  run="output/radio_gs/lerf_${scene}_${VARIANT}"
  cfg="${CFG_ROOT}/lerf_${scene}_${VARIANT}.yaml"
  train_ids="${CFG_ROOT}/${scene}_train_ids.txt"
  val_ids="${CFG_ROOT}/${scene}_val_ids.txt"

  bash radio_gs/scripts/run_repo_python.sh - "${scene}" "${train_ids}" "${val_ids}" <<'PY'
import sys
from pathlib import Path
from radio_gs.data.benchmark_paths import extract_feature_frame_index, list_feature_paths

scene, train_out, val_out = sys.argv[1:]
feature_dir = Path("output/radio_features_lerf") / scene
frame_ids = [extract_feature_frame_index(path) for path in list_feature_paths(feature_dir)]
if len(frame_ids) < 4:
    raise SystemExit(f"too few frames for {scene}: {len(frame_ids)}")
Path(train_out).write_text("\n".join(str(i) for i in frame_ids) + "\n", encoding="utf-8")
Path(val_out).write_text("\n".join(str(i) for i in frame_ids) + "\n", encoding="utf-8")
print(scene, len(frame_ids))
PY

  bash radio_gs/scripts/run_repo_python.sh - \
    "${base}" "${cfg}" "${run}" "${scene}" "${train_ids}" "${val_ids}" "${BATCH_SIZE}" \
    "${EPOCHS}" "${ALIGN_WEIGHT}" "${RELATION_WEIGHT}" "${LOCAL_AFFINITY_WEIGHT}" \
    "${LOCAL_AFFINITY_DOWNSAMPLE}" "${LOCAL_AFFINITY_RADIUS}" \
    "${TOKEN_CONTRAST_WEIGHT}" "${PEAK_BACKGROUND_WEIGHT}" \
    "${CROSS_VIEW_WEIGHT}" "${CROSS_VIEW_PROP_WEIGHT}" \
    "${CROSS_VIEW_MASK_PROP_WEIGHT}" "${CROSS_VIEW_ANCHOR_STRATEGY}" \
    "${PEAK_BACKGROUND_ANCHOR_STRATEGY}" <<'PY'
import sys
import yaml

(
    base,
    out,
    run,
    scene,
    train_ids,
    val_ids,
    batch_size,
    epochs,
    align_weight,
    relation_weight,
    local_affinity_weight,
    local_affinity_downsample,
    local_affinity_radius,
    token_contrast_weight,
    peak_background_weight,
    cross_view_weight,
    cross_view_prop_weight,
    cross_view_mask_prop_weight,
    cross_view_anchor_strategy,
    peak_background_anchor_strategy,
) = sys.argv[1:]
with open(base, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
root = f"/mnt/pool/sqy/3d_understanding/lerf_ovs/{scene}"
cfg.update(
    {
        "scene_root": root,
        "rgb_dir": f"{root}/images",
        "val_rgb_dir": f"{root}/images",
        "pose_file": f"output/radio_features_lerf/{scene}/traj_w_c.txt",
        "val_pose_file": f"output/radio_features_lerf/{scene}/traj_w_c.txt",
        "output_dir": run,
        "exp_name": run.rsplit("/", 1)[-1],
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "train_shuffle": False,
        "train_frame_ids_path": train_ids,
        "val_frame_ids_path": val_ids,
        "lr_features": 1.8e-5,
        "lr_hash": 2.7e-5,
        "lr_decoder": 1.3e-5,
        "lr_heads": 1.3e-5,
        "lr_refiner": 3.5e-5,
        "warmup_epochs": 2,
        "save_every": 5,
        "eval_every": 5,
        "log_every": 20,
        "radio_adaptor_alignment_names": "dino_v3",
        "radio_adaptor_alignment_weight": float(align_weight),
        "radio_adaptor_relation_names": "dino_v3",
        "radio_adaptor_relation_weight": float(relation_weight),
        "radio_adaptor_relation_downsample": 2,
        "radio_adaptor_relation_max_tokens": 384,
        "radio_adaptor_local_affinity_names": "dino_v3",
        "radio_adaptor_local_affinity_weight": float(local_affinity_weight),
        "radio_adaptor_local_affinity_downsample": int(local_affinity_downsample),
        "radio_adaptor_local_affinity_radius": int(local_affinity_radius),
        "radio_adaptor_token_contrast_names": "dino_v3",
        "radio_adaptor_token_contrast_weight": float(token_contrast_weight),
        "radio_adaptor_token_contrast_downsample": 2,
        "radio_adaptor_token_contrast_max_tokens": 256,
        "radio_adaptor_token_contrast_temperature": 0.07,
        "radio_adaptor_peak_background_names": "dino_v3",
        "radio_adaptor_peak_background_weight": float(peak_background_weight),
        "radio_adaptor_peak_background_downsample": 2,
        "radio_adaptor_peak_background_max_tokens": 256,
        "radio_adaptor_peak_background_num_anchors": 16,
        "radio_adaptor_peak_background_temperature": 0.2,
        "radio_adaptor_peak_background_anchor_strategy": str(peak_background_anchor_strategy),
        "radio_adaptor_cross_view_names": "dino_v3",
        "radio_adaptor_cross_view_weight": float(cross_view_weight),
        "radio_adaptor_cross_view_downsample": 2,
        "radio_adaptor_cross_view_max_tokens": 192,
        "radio_adaptor_cross_view_objective": "transport_cycle",
        "radio_adaptor_cross_view_propagation_names": "dino_v3",
        "radio_adaptor_cross_view_propagation_weight": float(cross_view_prop_weight),
        "radio_adaptor_cross_view_propagation_downsample": 2,
        "radio_adaptor_cross_view_propagation_max_tokens": 192,
        "radio_adaptor_cross_view_propagation_num_anchors": 16,
        "radio_adaptor_cross_view_propagation_temperature": 0.2,
        "radio_adaptor_cross_view_propagation_anchor_strategy": str(cross_view_anchor_strategy),
        "radio_adaptor_cross_view_mask_propagation_names": "dino_v3",
        "radio_adaptor_cross_view_mask_propagation_weight": float(cross_view_mask_prop_weight),
        "radio_adaptor_cross_view_mask_propagation_downsample": 2,
        "radio_adaptor_cross_view_mask_propagation_max_tokens": 192,
        "radio_adaptor_cross_view_mask_propagation_num_anchors": 16,
        "radio_adaptor_cross_view_mask_propagation_temperature": 0.2,
        "radio_adaptor_cross_view_mask_propagation_anchor_strategy": str(cross_view_anchor_strategy),
        "radio_adaptor_alignment_checkpoint": "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar",
        "text_heatmap_distill_weight": 0.001,
        "text_heatmap_distill_embeddings": "checkpoints/siglip2_lerf_text_embeddings.pt",
        "text_heatmap_distill_mode": "spatial",
        "text_heatmap_distill_downsample": 2,
    }
)
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out)
PY

  echo "[lerf-dino-localtopo] gpu=${GPU} scene=${scene} batch=${BATCH_SIZE} cfg=${cfg}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/train_feature_field.py \
    --config "${cfg}" \
    --warmstart "$(checkpoint_for_scene "${scene}")"

  out="output/lerf_sam_dino_tasks/${VARIANT}/${scene}"
  echo "[lerf-dino-localtopo-eval] gpu=${GPU} scene=${scene}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_lerf_sam_dino_tasks.py \
    --config "${cfg}" \
    --checkpoint "${run}/checkpoints/best.pth" \
    --scene "${scene}" \
    --output_dir "${out}" \
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
    --dino_propagation_seed_weight 0.35 \
    --dino_propagation_seed_radius 1 \
    --dino_transport_match_weight 0.25 \
    --dino_transport_match_radius 1 \
    --dino_feature_boundary_refinement \
    --dino_feature_boundary_background_weight 0.5 \
    --dino_feature_boundary_seed_topk_ratio 0.25
done

result_files=()
for scene in "${SCENES[@]}"; do
  result_files+=("output/lerf_sam_dino_tasks/${VARIANT}/${scene}/lerf_sam_dino_task_results.json")
done
bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/aggregate_lerf_sam_dino_tasks.py \
  "${result_files[@]}" \
  --output_dir "output/lerf_sam_dino_tasks/${VARIANT}"
