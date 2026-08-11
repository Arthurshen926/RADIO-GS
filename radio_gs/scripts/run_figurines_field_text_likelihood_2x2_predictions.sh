#!/usr/bin/env bash

set -euo pipefail

REPO=/root/RADIO-GS
PYTHON=/root/miniconda3/envs/cybersim_agent/bin/python
ROOT="$REPO/local_ssd_results/figurines_field_text_likelihood_2x2_20260810"
RECEIPT="$REPO/paper/artifacts/figurines_field_text_likelihood_2x2_one_shot_execution_receipt_20260810.json"
EXPECTED_RECEIPT_SHA=f17d98b1aa9c2b31cca49eee3920f855bd36a6c25eee157646d254f96ab8c0bc
CORRECTION="$REPO/paper/artifacts/figurines_field_text_likelihood_2x2_one_shot_execution_correction_v2_20260810.json"
EXPECTED_CORRECTION_SHA=04e2d97b5386da9698482248f1910a391ba5470c9444b087f50b66b0b8d65c82
RUNTIME_ADDENDUM="$REPO/paper/artifacts/figurines_field_text_likelihood_2x2_gpu1_runtime_addendum_v3_20260810.json"
EXPECTED_RUNTIME_ADDENDUM_SHA=f28ca4f7244d225a5a0c21e43802977d8b709acf01ce7e0f650990633b74dd8a
EVALUATOR="$REPO/radio_gs/scripts/eval_lerf_direct_3d_selection.py"
CONFIG="$REPO/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"
CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
INVENTORY=/root/radio_gs_local_cache/optimization_20260810/lerf_target_rgb_sam3_box_o2_full4_v1/figurines/sanitized_prediction_inventory.json
SUMMARY_HEAD="$REPO/checkpoints/siglip2_summary_head.pth"
POSITIVE_TEXT="$REPO/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt"
NEGATIVE_TEXT="$REPO/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt"

cd "$REPO"
observed_receipt_sha="$(sha256sum "$RECEIPT" | awk '{print $1}')"
if [[ "$observed_receipt_sha" != "$EXPECTED_RECEIPT_SHA" ]]; then
  echo "execution receipt changed: $observed_receipt_sha" >&2
  exit 2
fi
observed_correction_sha="$(sha256sum "$CORRECTION" | awk '{print $1}')"
if [[ "$observed_correction_sha" != "$EXPECTED_CORRECTION_SHA" ]]; then
  echo "execution correction changed: $observed_correction_sha" >&2
  exit 2
fi
observed_runtime_addendum_sha="$(sha256sum "$RUNTIME_ADDENDUM" | awk '{print $1}')"
if [[ "$observed_runtime_addendum_sha" != "$EXPECTED_RUNTIME_ADDENDUM_SHA" ]]; then
  echo "GPU runtime addendum changed: $observed_runtime_addendum_sha" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH="/root/RADIO-GS/local_ssd_results/nvidia_driver_535_runtime:/root/miniconda3/envs/cybersim_agent/lib:${LD_LIBRARY_PATH:-}"

run_arm() {
  local name="$1"
  local positive="$2"
  local negative="$3"
  local likelihood="$4"
  local output="$ROOT/pred_$name"
  local prediction_receipt="$ROOT/pred_$name.receipt.json"
  local log="$ROOT/pred_$name.log"
  if [[ -e "$output" || -e "$prediction_receipt" || -e "$log" ]]; then
    echo "refusing to clobber arm $name" >&2
    exit 2
  fi
  local -a command=(
    "$PYTHON" "$EVALUATOR"
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --scene figurines
    --protocol_preset vala_repo_3d
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label
    --output_dir "$output"
    --summary_head_weights "$SUMMARY_HEAD"
    --text_embedding_cache "$POSITIVE_TEXT"
    --canonical_embedding_cache "$NEGATIVE_TEXT"
    --ours_multiscale_query_score_cache "$positive"
    --ours_multiscale_negative_score_cache "$negative"
    --prediction_only
    --prediction_receipt "$prediction_receipt"
    --prediction_inventory "$INVENTORY"
    --save_masks
    --gpu 0
  )
  if [[ -n "$likelihood" ]]; then
    command+=(--ours_source_text_likelihood_cache "$likelihood")
  fi
  "${command[@]}" >"$log" 2>&1
  sha256sum "$prediction_receipt"
}

run_arm \
  control_legacy \
  /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/figurines_positive_fp32.pt \
  /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/figurines_negative_fp32.pt \
  ""
run_arm \
  control_learned \
  /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/figurines_positive_fp32.pt \
  /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/figurines_negative_fp32.pt \
  "$ROOT/control_source_text_likelihood.pt"
run_arm \
  stage_b_v2_legacy \
  /root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/positive_fp32.pt \
  /root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/negative_fp32.pt \
  ""
run_arm \
  stage_b_v2_learned \
  /root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/positive_fp32.pt \
  /root/radio_gs_local_cache/stage_b_20260810/figurines_lerf3d_one_shot_v2/negative_fp32.pt \
  "$ROOT/stage_b_v2_source_text_likelihood.pt"

sha256sum "$ROOT"/pred_*.receipt.json > "$ROOT/four_arm_prediction_receipts.sha256"
