#!/usr/bin/env bash

set -euo pipefail

[[ "$#" -eq 1 ]] || { echo "usage: $0 figurines|ramen|teatime|waldo_kitchen" >&2; exit 2; }
SCENE="$1"
REPO=/root/RADIO-GS
ROOT="$REPO/local_ssd_results/lerf_full4_post_readout_v3_20260810"
PREREG="$REPO/paper/artifacts/lerf_full4_post_readout_probability_mixture_v3_preregistration_20260810.json"
[[ "$(sha256sum "$PREREG" | awk '{print $1}')" == aaf13a34c91045505bcb343290736d816ae8b6f96fe950dd5382682d1d436623 ]]
[[ "$(sha256sum "$REPO/radio_gs/querying/lerf_source_text_likelihood.py" | awk '{print $1}')" == 10eeb5d805bf3543e8f4305be0e43df6b57459f59407276bad201200f38c6aa9 ]]
[[ "$(sha256sum "$REPO/radio_gs/scripts/eval_lerf_direct_3d_selection.py" | awk '{print $1}')" == b89cad76e753874293b0f23606026d82a91fb4067f903edd615fcd25d39a77c9 ]]

case "$SCENE" in
  figurines)
    CONFIG="$REPO/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
    POSITIVE=/root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/positive_fp32.pt
    NEGATIVE=/root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/negative_fp32.pt
    LIKELIHOOD=/root/RADIO-GS/local_ssd_results/figurines_prior_preserving_text_likelihood_v2_20260810/stage_b_source_text_likelihood_prior_preserving_v2.pt
    LIKELIHOOD_SHA=8edbbdd05c19060b74b231566ea03997eed1cd414c29e8431a9e16813ef69863
    INVENTORY=/root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_full4_v1/figurines/sanitized_prediction_inventory.json
    ;;
  ramen)
    CONFIG="$REPO/radio_gs/configs/generated/frozen_eval_20260802/lerf_ramen_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth
    POSITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_o1_o2_streaming_v1_fix2/ramen/ramen_o2_positive.pt
    NEGATIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_o1_o2_streaming_v1_fix2/ramen/ramen_o2_negative.pt
    LIKELIHOOD=/root/RADIO-GS/local_ssd_results/lerf_full4_prior_preserving_v2_20260810/ramen_prior_preserving_v2.pt
    LIKELIHOOD_SHA=f5177cce50cf43adb64466220d32c7c3e61ab2da4ad9aa1b6b0a0645f6b0a605
    INVENTORY=/root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_dual_v1/ramen/sanitized_prediction_inventory.json
    ;;
  teatime)
    CONFIG="$REPO/radio_gs/configs/generated/frozen_eval_20260802/lerf_teatime_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep/checkpoints/best.pth
    POSITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_o1_o2_frozen_text_rebind_v1/teatime_o2/teatime_o2_positive.pt
    NEGATIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_o1_o2_frozen_text_rebind_v1/teatime_o2/teatime_o2_negative.pt
    LIKELIHOOD=/root/RADIO-GS/local_ssd_results/lerf_full4_prior_preserving_v2_20260810/teatime_prior_preserving_v2.pt
    LIKELIHOOD_SHA=5e3b3d60aedc4b66717f974a58689382f6565c9b5457abf93b706e1830172f0d
    INVENTORY=/root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_dual_v1/teatime/sanitized_prediction_inventory.json
    ;;
  waldo_kitchen)
    CONFIG="$REPO/radio_gs/configs/generated/frozen_eval_20260802/lerf_waldo_kitchen_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth
    POSITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_waldo_kitchen_o1_o2_streaming_unpaced_gpu1_lowmem_v3/waldo_kitchen_o2_positive.pt
    NEGATIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf_waldo_kitchen_o1_o2_streaming_unpaced_gpu1_lowmem_v3/waldo_kitchen_o2_negative.pt
    LIKELIHOOD=/root/RADIO-GS/local_ssd_results/lerf_full4_prior_preserving_v2_20260810/waldo_kitchen_prior_preserving_v2.pt
    LIKELIHOOD_SHA=92e5526559b734aa1ca38c36625650e72dae224651171368833a786f40e1a509
    INVENTORY=/root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_full4_v1/waldo_kitchen/sanitized_prediction_inventory.json
    ;;
  *) echo "unsupported scene: $SCENE" >&2; exit 2 ;;
esac

[[ "$(sha256sum "$LIKELIHOOD" | awk '{print $1}')" == "$LIKELIHOOD_SHA" ]]
mkdir -p "$ROOT"
OUTPUT="$ROOT/pred_${SCENE}_v3"
RECEIPT="$ROOT/pred_${SCENE}_v3.receipt.json"
LOG="$ROOT/pred_${SCENE}_v3.log"
[[ ! -e "$OUTPUT" && ! -e "$RECEIPT" && ! -e "$LOG" ]]

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
  --ours_source_text_likelihood_post_readout_cache "$LIKELIHOOD" \
  --prediction_only \
  --prediction_receipt "$RECEIPT" \
  --prediction_inventory "$INVENTORY" \
  --save_masks \
  --gpu 0 \
  >"$LOG" 2>&1

sha256sum "$RECEIPT"
