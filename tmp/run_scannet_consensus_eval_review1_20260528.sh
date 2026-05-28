#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/RADIO-GS"
cd "${REPO_ROOT}"

GPU="${1:-2}"
VARIANT="${2:-v67_dino_cv001_b2_s32768_ft20}"
TAG="${3:-consensus_vxl005_a02_c060_cons080}"

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

if [[ "${VARIANT}" == v67_* ]]; then
  CONFIG_PATTERN="radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_${VARIANT}_{scene}.yaml"
else
  CONFIG_PATTERN="tmp/scannet_propcons_${VARIANT}/scannet_og_hybrid_${VARIANT}_{scene}.yaml"
fi
CHECKPOINT_PATTERN="output/radio_gs/scannet_og_{scene}_${VARIANT}/checkpoints/best.pth"
EVAL_OUT="output/scannet_pointcloud_eval/vala8_${VARIANT}_knn16_cand80_scene_mean_a045_smoothk12a1_${TAG}"

CUDA_VISIBLE_DEVICES="${GPU}" bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py \
  --scene_list "$(IFS=,; echo "${SCENES[*]}")" \
  --prepared_root dataset/scannet_og \
  --config "${CONFIG_PATTERN}" \
  --checkpoint "${CHECKPOINT_PATTERN}" \
  --output_dir "${EVAL_OUT}" \
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
  --proposal_smoothing voxel \
  --proposal_voxel_size 0.05 \
  --proposal_smoothing_alpha 0.2 \
  --proposal_min_count 3 \
  --proposal_smoothing_gate low_confidence_and_proposal_consensus \
  --proposal_confidence_threshold 0.6 \
  --proposal_consensus_threshold 0.8 \
  --prompt_templates "{query}" \
  --text_embedding_cache checkpoints/siglip2_scannet_text_embeddings_v67_knn.pt

bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/build_scannet_vala8_report.py \
  --input "${EVAL_OUT}/scannet_pointcloud_radio_gs_results.json" \
  --output_json "paper/artifacts/scannet_pointcloud_radio_gs_vala8_${VARIANT}_${TAG}_results.json" \
  --output_md "paper/artifacts/scannet_pointcloud_radio_gs_vala8_${VARIANT}_${TAG}_results.md" \
  --label "RADIO-GS VALA8 contextual kNN16/cand80 + spatial smoothing + consensus-gated proposal readout" \
  --require_exact_scene_set \
  --expect_arg query_mode=knn \
  --expect_arg k=16 \
  --expect_arg candidate_k=80 \
  --expect_arg logit_calibration=scene_mean \
  --expect_arg logit_calibration_alpha=0.45 \
  --expect_arg logit_smoothing=spatial_knn \
  --expect_arg logit_smoothing_k=12 \
  --expect_arg proposal_smoothing=voxel \
  --expect_arg proposal_smoothing_gate=low_confidence_and_proposal_consensus \
  --expect_arg proposal_confidence_threshold=0.6 \
  --expect_arg proposal_consensus_threshold=0.8
