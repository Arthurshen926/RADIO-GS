#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/RADIO-GS"
cd "${REPO_ROOT}"

GPU="${1:-2}"
VARIANT="${2:-v75_propcontrast02_from_v67dinocv_ft8}"
BATCH_SIZE="${3:-2}"
DIRECT_POINT_SAMPLE_COUNT="${4:-32768}"
CFG_ROOT="tmp/scannet_propcontrast_${VARIANT}"
SCENES=(
  scene0000_00
  scene0062_00
  scene0070_00
  scene0097_00
  scene0140_00
  scene0347_00
  scene0400_00
  scene0590_00
)

if ! [[ "${GPU}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU must be a numeric CUDA_VISIBLE_DEVICES list, got: ${GPU}" >&2
  exit 2
fi

mkdir -p "${CFG_ROOT}"

bash radio_gs/scripts/run_repo_python.sh - "${CFG_ROOT}" "${VARIANT}" "${BATCH_SIZE}" "${DIRECT_POINT_SAMPLE_COUNT}" "${SCENES[@]}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_root = Path(sys.argv[1])
variant = sys.argv[2]
batch_size = int(sys.argv[3])
direct_point_sample_count = int(sys.argv[4])
scenes = sys.argv[5:]
template = "radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_{scene}.yaml"

for scene in scenes:
    base_path = Path(template.format(scene=scene))
    with base_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update(
        {
            "exp_name": f"radio_gs_scannet_og_{scene}_{variant}",
            "output_dir": f"/root/RADIO-GS/output/radio_gs/scannet_og_{scene}_{variant}",
            "epochs": 8,
            "batch_size": batch_size,
            "train_shuffle": False,
            "save_every": 4,
            "eval_every": 4,
            "log_every": 25,
            "direct_point_sample_count": direct_point_sample_count,
            "direct_point_proposal_consistency_weight": 0.03,
            "direct_point_proposal_contrast_weight": 0.02,
            "direct_point_proposal_contrast_temperature": 0.07,
            "direct_point_proposal_voxel_size": 0.05,
            "direct_point_proposal_min_count": 2,
            "direct_point_proposal_space": "auto",
        }
    )
    out = cfg_root / f"scannet_og_hybrid_{variant}_{scene}.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(out)
PY

for scene in "${SCENES[@]}"; do
  cfg="${CFG_ROOT}/scannet_og_hybrid_${VARIANT}_${scene}.yaml"
  warmstart="output/radio_gs/scannet_og_${scene}_v67_dino_cv001_b2_s32768_ft20/checkpoints/best.pth"
  echo "[scannet-propcontrast] gpu=${GPU} scene=${scene} cfg=${cfg}"
  CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/train_feature_field.py \
    --config "${cfg}" \
    --warmstart "${warmstart}"
done

eval_out="output/scannet_pointcloud_eval/vala8_${VARIANT}_knn16_cand80_scene_mean_a045_smoothk12a1"
CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py \
  --scene_list "$(IFS=,; echo "${SCENES[*]}")" \
  --prepared_root dataset/scannet_og \
  --config "${CFG_ROOT}/scannet_og_hybrid_${VARIANT}_{scene}.yaml" \
  --checkpoint "output/radio_gs/scannet_og_{scene}_${VARIANT}/checkpoints/best.pth" \
  --output_dir "${eval_out}" \
  --class_splits 19,15,10 \
  --query_mode knn \
  --k 16 \
  --candidate_k 80 \
  --chunk_size 32768 \
  --opacity_filter_mode auto \
  --logit_calibration scene_mean \
  --logit_calibration_alpha 0.45 \
  --logit_smoothing spatial_knn \
  --logit_smoothing_k 12 \
  --logit_smoothing_alpha 1.0 \
  --logit_smoothing_iterations 1 \
  --prompt_templates "{query}" \
  --text_embedding_cache checkpoints/siglip2_scannet_text_embeddings_v67_knn.pt

bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/build_scannet_vala8_report.py \
  --input "${eval_out}/scannet_pointcloud_radio_gs_results.json" \
  --output_json "paper/artifacts/scannet_pointcloud_radio_gs_vala8_${VARIANT}_results.json" \
  --output_md "paper/artifacts/scannet_pointcloud_radio_gs_vala8_${VARIANT}_results.md" \
  --label "RADIO-GS VALA8 region-prototype contrast training + DINO-CV contextual kNN16/cand80 + spatial readout" \
  --require_exact_scene_set \
  --expect_arg query_mode=knn \
  --expect_arg k=16 \
  --expect_arg candidate_k=80 \
  --expect_arg logit_calibration=scene_mean \
  --expect_arg logit_calibration_alpha=0.45 \
  --expect_arg logit_smoothing=spatial_knn \
  --expect_arg logit_smoothing_k=12
