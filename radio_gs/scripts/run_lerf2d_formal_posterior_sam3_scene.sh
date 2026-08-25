#!/usr/bin/env bash
set -euo pipefail

SCENE=${1:?usage: run_lerf2d_formal_posterior_sam3_scene.sh SCENE GPU}
GPU=${2:?usage: run_lerf2d_formal_posterior_sam3_scene.sh SCENE GPU}
ROOT=/root/RADIO-GS
RESULT_ROOT=/mnt/pool/sqy/results/RADIO-GS/output
OUT_ROOT=${OUT_ROOT:-$RESULT_ROOT/optimization_20260825/lerf_formal_sam3_v1}
SCORE_ROOT=$RESULT_ROOT/optimization_20260817/lerf_sam_siglip_object_posterior_source32_v6_relevancy_identity_extent/scores
AUTHORITY=$RESULT_ROOT/optimization_20260802/lerf2d_ours_native_multiscale_scalar_maps_v1/manifest.json
AUTHORITY_SHA=cf90415031620fcadb120a8fef05c6b9a008f9d76836b860b6660e62125b5f4f
RGB_AUTHORITY=$ROOT/paper/artifacts/lerf2d_target_rgb_authority_20260825.json
RGB_AUTHORITY_SHA=59a8c54e37602051253c72fc518921282cc9f1dd1ccc30a3bf38ce8ce1b50a98

case "$SCENE" in
  figurines)
    CONFIG=lerf_figurines_radio_verified_pose.yaml
    CHECKPOINT=$RESULT_ROOT/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth ;;
  ramen)
    CONFIG=lerf_ramen_radio_verified_pose.yaml
    CHECKPOINT=$RESULT_ROOT/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth ;;
  teatime)
    CONFIG=lerf_teatime_radio_verified_pose.yaml
    CHECKPOINT=$RESULT_ROOT/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth ;;
  waldo_kitchen)
    CONFIG=lerf_waldo_kitchen_radio_verified_pose.yaml
    CHECKPOINT=$RESULT_ROOT/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth ;;
  *) echo "unsupported LERF scene: $SCENE" >&2; exit 2 ;;
esac

SCENE_OUT=$OUT_ROOT/$SCENE
CUDA_VISIBLE_DEVICES=$GPU bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$ROOT/radio_gs/scripts/materialize_lerf2d_formal_posterior_coarse_receipt.py" \
  --scene "$SCENE" \
  --config "$ROOT/radio_gs/configs/generated/frozen_eval_20260802/$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --posterior-cache "$SCORE_ROOT/$SCENE.pt" \
  --query-authority-manifest "$AUTHORITY" \
  --query-authority-manifest-sha256 "$AUTHORITY_SHA" \
  --output-dir "$SCENE_OUT/coarse" --device cuda:0

COARSE_RECEIPT=$SCENE_OUT/coarse/coarse_prediction_receipt.json
COARSE_SHA=$(sha256sum "$COARSE_RECEIPT" | awk '{print $1}')
CUDA_VISIBLE_DEVICES=$GPU bash "$ROOT/radio_gs/scripts/run_official_sam3_python.sh" \
  "$ROOT/radio_gs/scripts/refine_lerf2d_coarse_receipt_official_sam3.py" \
  --coarse-receipt "$COARSE_RECEIPT" --coarse-receipt-sha256 "$COARSE_SHA" \
  --rgb-authority "$RGB_AUTHORITY" --rgb-authority-sha256 "$RGB_AUTHORITY_SHA" \
  --checkpoint "$ROOT/checkpoints/sam3_modelscope/sam3.pt" \
  --output-dir "$SCENE_OUT/sam3" --device cuda:0

PREDICTION_RECEIPT=$SCENE_OUT/sam3/prediction_receipt.json
PREDICTION_SHA=$(sha256sum "$PREDICTION_RECEIPT" | awk '{print $1}')
bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$ROOT/radio_gs/scripts/score_lerf2d_official_sam3_box_receipt.py" \
  --prediction-receipt "$PREDICTION_RECEIPT" \
  --prediction-receipt-sha256 "$PREDICTION_SHA" \
  --label-root /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
  --scenes "$SCENE" --lerf-contract-only \
  --output-json "$SCENE_OUT/evaluation_current_raster.json"
