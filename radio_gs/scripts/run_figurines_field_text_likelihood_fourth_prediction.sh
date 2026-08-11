#!/usr/bin/env bash

set -euo pipefail

REPO=/root/RADIO-GS
ROOT="$REPO/local_ssd_results/figurines_field_text_likelihood_2x2_20260810"
CONTINUATION="$REPO/paper/artifacts/figurines_field_text_likelihood_2x2_fourth_arm_continuation_v4_20260810.json"
EXPECTED_CONTINUATION_SHA=749323dba658f32754edc009266f9f093c9f06c7e5368abc02c18cfcfd027d49
if [[ "$(sha256sum "$CONTINUATION" | awk '{print $1}')" != "$EXPECTED_CONTINUATION_SHA" ]]; then
  echo "fourth-arm continuation changed" >&2
  exit 2
fi
for receipt in pred_control_legacy pred_control_learned pred_stage_b_v2_legacy; do
  [[ -f "$ROOT/$receipt.receipt.json" ]] || {
    echo "missing already-sealed receipt: $receipt" >&2
    exit 2
  }
done
[[ ! -e "$ROOT/pred_stage_b_v2_learned" ]]
[[ ! -e "$ROOT/pred_stage_b_v2_learned.receipt.json" ]]
[[ ! -e "$ROOT/pred_stage_b_v2_learned.log" ]]

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
  --output_dir "$ROOT/pred_stage_b_v2_learned" \
  --summary_head_weights "$REPO/checkpoints/siglip2_summary_head.pth" \
  --text_embedding_cache "$REPO/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
  --canonical_embedding_cache "$REPO/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
  --ours_multiscale_query_score_cache /root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/positive_fp32.pt \
  --ours_multiscale_negative_score_cache /root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/negative_fp32.pt \
  --ours_source_text_likelihood_cache "$ROOT/stage_b_v2_source_text_likelihood.pt" \
  --prediction_only \
  --prediction_receipt "$ROOT/pred_stage_b_v2_learned.receipt.json" \
  --prediction_inventory /root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_full4_v1/figurines/sanitized_prediction_inventory.json \
  --save_masks \
  --gpu 0 \
  >"$ROOT/pred_stage_b_v2_learned.log" 2>&1

sha256sum "$ROOT/pred_stage_b_v2_learned.receipt.json"
