#!/usr/bin/env bash

set -euo pipefail

[[ "$#" -eq 1 ]] || { echo "usage: $0 control|stage_b" >&2; exit 2; }
ARM="$1"
REPO=/root/RADIO-GS
ROOT="$REPO/local_ssd_results/figurines_prior_preserving_text_likelihood_v2_20260810"
PREREG="$REPO/paper/artifacts/figurines_prior_preserving_text_likelihood_v2_preregistration_20260810.json"
CORRECTION="$REPO/paper/artifacts/figurines_prior_preserving_text_likelihood_v2_execution_correction_20260810.json"
[[ "$(sha256sum "$PREREG" | awk '{print $1}')" == 0e13d258fbfcb7c23f3f521e447c33fb6044478af78c4d9f2022cfc22b37131c ]]
[[ "$(sha256sum "$CORRECTION" | awk '{print $1}')" == d01305e0cfa4f4313ce07f3d1917c52170808a171fc05546e7e671bdcee3ef33 ]]

case "$ARM" in
  control)
    POSITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/figurines_positive_fp32.pt
    NEGATIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/figurines_negative_fp32.pt
    LIKELIHOOD="$ROOT/control_source_text_likelihood_prior_preserving_v2.pt"
    ;;
  stage_b)
    POSITIVE=/root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/positive_fp32.pt
    NEGATIVE=/root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/negative_fp32.pt
    LIKELIHOOD="$ROOT/stage_b_source_text_likelihood_prior_preserving_v2.pt"
    ;;
  *) echo "unsupported arm: $ARM" >&2; exit 2 ;;
esac

OUTPUT="$ROOT/pred_${ARM}_prior_preserving_v2"
RECEIPT="$ROOT/pred_${ARM}_prior_preserving_v2.receipt.json"
LOG="$ROOT/pred_${ARM}_prior_preserving_v2.log"
[[ ! -e "$OUTPUT" && ! -e "$RECEIPT" && ! -e "$LOG" ]]

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="/root/RADIO-GS/local_ssd_results/nvidia_driver_535_runtime:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"
cd "$REPO"
/root/miniconda3/envs/cybersim_agent/bin/python \
  radio_gs/scripts/eval_lerf_direct_3d_selection.py \
  --config radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml \
  --checkpoint /mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth \
  --scene figurines \
  --protocol_preset vala_repo_3d \
  --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
  --output_dir "$OUTPUT" \
  --summary_head_weights "$REPO/checkpoints/siglip2_summary_head.pth" \
  --text_embedding_cache "$REPO/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
  --canonical_embedding_cache "$REPO/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
  --ours_multiscale_query_score_cache "$POSITIVE" \
  --ours_multiscale_negative_score_cache "$NEGATIVE" \
  --ours_source_text_likelihood_cache "$LIKELIHOOD" \
  --prediction_only \
  --prediction_receipt "$RECEIPT" \
  --prediction_inventory /root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_full4_v1/figurines/sanitized_prediction_inventory.json \
  --save_masks \
  --gpu 0 \
  >"$LOG" 2>&1

sha256sum "$RECEIPT"
