#!/usr/bin/env bash

set -euo pipefail

[[ "$#" -eq 2 ]] || { echo "usage: $0 ramen|teatime|waldo_kitchen legacy|v2" >&2; exit 2; }
SCENE="$1"
ARM="$2"
[[ "$ARM" == legacy || "$ARM" == v2 ]] || { echo "unsupported arm: $ARM" >&2; exit 2; }

REPO=/root/RADIO-GS
ROOT="$REPO/local_ssd_results/lerf_full4_prior_preserving_v2_20260810"
PREREG="$REPO/paper/artifacts/lerf_full4_prior_preserving_text_likelihood_v2_preregistration_20260810.json"
[[ "$(sha256sum "$PREREG" | awk '{print $1}')" == 843517902b5094803475192d386c437be255a19271d118bba01ba4f872fbe90b ]]
[[ "$(sha256sum "$REPO/radio_gs/scripts/eval_lerf_direct_3d_selection.py" | awk '{print $1}')" == c72b03ff0f2c21c7c2b59a4e9e2111854d97da525194739b501a6dda3b4f8903 ]]

case "$SCENE" in
  ramen)
    CONFIG="$REPO/radio_gs/configs/generated/frozen_eval_20260802/lerf_ramen_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth
    POSITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_o1_o2_streaming_v1_fix2/ramen/ramen_o2_positive.pt
    NEGATIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_o1_o2_streaming_v1_fix2/ramen/ramen_o2_negative.pt
    LIKELIHOOD="$ROOT/ramen_prior_preserving_v2.pt"
    LIKELIHOOD_SHA=f5177cce50cf43adb64466220d32c7c3e61ab2da4ad9aa1b6b0a0645f6b0a605
    INVENTORY=/root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_dual_v1/ramen/sanitized_prediction_inventory.json
    ;;
  teatime)
    CONFIG="$REPO/radio_gs/configs/generated/frozen_eval_20260802/lerf_teatime_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep/checkpoints/best.pth
    POSITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_o1_o2_frozen_text_rebind_v1/teatime_o2/teatime_o2_positive.pt
    NEGATIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_o1_o2_frozen_text_rebind_v1/teatime_o2/teatime_o2_negative.pt
    LIKELIHOOD="$ROOT/teatime_prior_preserving_v2.pt"
    LIKELIHOOD_SHA=5e3b3d60aedc4b66717f974a58689382f6565c9b5457abf93b706e1830172f0d
    INVENTORY=/root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_dual_v1/teatime/sanitized_prediction_inventory.json
    ;;
  waldo_kitchen)
    CONFIG="$REPO/radio_gs/configs/generated/frozen_eval_20260802/lerf_waldo_kitchen_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth
    POSITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_waldo_kitchen_o1_o2_streaming_unpaced_gpu1_lowmem_v3/waldo_kitchen_o2_positive.pt
    NEGATIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_waldo_kitchen_o1_o2_streaming_unpaced_gpu1_lowmem_v3/waldo_kitchen_o2_negative.pt
    LIKELIHOOD="$ROOT/waldo_kitchen_prior_preserving_v2.pt"
    LIKELIHOOD_SHA=92e5526559b734aa1ca38c36625650e72dae224651171368833a786f40e1a509
    INVENTORY=/root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_full4_v1/waldo_kitchen/sanitized_prediction_inventory.json
    ;;
  *) echo "unsupported scene: $SCENE" >&2; exit 2 ;;
esac

[[ "$(sha256sum "$LIKELIHOOD" | awk '{print $1}')" == "$LIKELIHOOD_SHA" ]]
OUTPUT="$ROOT/pred_${SCENE}_${ARM}"
RECEIPT="$ROOT/pred_${SCENE}_${ARM}.receipt.json"
LOG="$ROOT/pred_${SCENE}_${ARM}.log"
[[ ! -e "$OUTPUT" && ! -e "$RECEIPT" && ! -e "$LOG" ]]

LIKELIHOOD_ARGS=()
if [[ "$ARM" == v2 ]]; then
  LIKELIHOOD_ARGS=(--ours_source_text_likelihood_cache "$LIKELIHOOD")
fi

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/root/RADIO-GS/local_ssd_results/nvidia_driver_535_runtime:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"
cd "$REPO"
/root/miniconda3/envs/cybersim_agent/bin/python \
  radio_gs/scripts/eval_lerf_direct_3d_selection.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --scene "$SCENE" \
  --protocol_preset vala_repo_3d \
  --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
  --output_dir "$OUTPUT" \
  --summary_head_weights "$REPO/checkpoints/siglip2_summary_head.pth" \
  --text_embedding_cache "$REPO/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
  --canonical_embedding_cache "$REPO/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
  --ours_multiscale_query_score_cache "$POSITIVE" \
  --ours_multiscale_negative_score_cache "$NEGATIVE" \
  "${LIKELIHOOD_ARGS[@]}" \
  --prediction_only \
  --prediction_receipt "$RECEIPT" \
  --prediction_inventory "$INVENTORY" \
  --save_masks \
  --gpu 0 \
  >"$LOG" 2>&1

sha256sum "$RECEIPT"
