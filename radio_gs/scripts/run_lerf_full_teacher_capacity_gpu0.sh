#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/root/RADIO-GS
OUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260803/lerf_full_teacher_capacity_v1
READOUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260717/surface_region_contract_v2/readout_v2_clean_h256.pth
READOUT_SHA=5b2d123a7827d9ab79aa4aa5a70077f00a656beebcf4c95ea5a3c9efdbe13ccb
RADIO=/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
RADIO_SHA=bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9
REGISTRATION="$REPO_ROOT/paper/artifacts/evidence_to_support_v1_experiment_registration_20260803.json"

guard() {
  local tag="$1"
  shift
  mkdir -p "$OUT/logs"
  GPU=0 \
  GPU_TELEMETRY_LOG="$OUT/logs/${tag}.telemetry.csv" \
  GPU_OWNER_AUDIT_LOG="$OUT/logs/${tag}.owner_audit.csv" \
  GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
  GPU_POLL_SECONDS=20 \
  GPU_START_MAX_TEMP_C=78 \
  GPU_SOFT_PAUSE_TEMP_C=81 \
  GPU_SOFT_RESUME_TEMP_C=76 \
  GPU_MAX_TEMP_C=84 \
  CUDA_VISIBLE_DEVICES=0 \
    bash "$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh" -- "$@"
}

run_scene() {
  local scene="$1"
  local field="$2"
  local field_sha="$3"
  local mpr_sha="$4"
  local graph="$5"
  local graph_sha="$6"
  local text="$7"
  local text_sha="$8"
  local renderer="$9"
  local renderer_sha="${10}"
  local config="${11}"
  local pacing=0.25
  local semantic_batch_size=256
  local resume_tag="$scene"
  if [[ "$scene" == "waldo_kitchen" ]]; then
    pacing=0.05
    semantic_batch_size=1024
    resume_tag="${scene}_bs1024"
  fi
  local descriptor="$OUT/descriptors/${scene}_full_mpr_teacher.pt"
  local score_cache="$OUT/query_scores/${scene}_full_mpr_teacher.pt"
  mkdir -p "$OUT/descriptors" "$OUT/query_scores"

  if [[ ! -f "$descriptor" ]]; then
    guard "${scene}_descriptor" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$REPO_ROOT/radio_gs/scripts/build_surface_region_semantic_cache.py" \
        --field-checkpoint "$field" \
        --field-checkpoint-sha256 "$field_sha" \
        --canonical-radio-source mpr_teacher \
        --experiment-registration "$REGISTRATION" \
        --support-graph "$graph" \
        --support-graph-sha256 "$graph_sha" \
        --readout-checkpoint "$READOUT" \
        --readout-checkpoint-sha256 "$READOUT_SHA" \
        --mpr-cache-sha256 "$mpr_sha" \
        --radio-checkpoint "$RADIO" \
        --radio-checkpoint-sha256 "$RADIO_SHA" \
        --output "$descriptor" \
        --resume-dir "$OUT/resume_${resume_tag}" \
        --radio-batch-size 4096 \
        --semantic-batch-size "$semantic_batch_size" \
        --thermal-pacing-seconds-per-batch "$pacing" \
        --device cuda:0
  fi

  if [[ ! -f "$score_cache" ]]; then
    local descriptor_sha
    descriptor_sha="$(sha256sum "$descriptor" | awk '{print $1}')"
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$REPO_ROOT/radio_gs/scripts/materialize_lerf_multiscale_query_score_cache.py" \
      --descriptor-cache "$descriptor" \
      --descriptor-cache-sha256 "$descriptor_sha" \
      --text-query-cache "$text" \
      --text-query-cache-sha256 "$text_sha" \
      --field-checkpoint "$field" \
      --field-checkpoint-sha256 "$field_sha" \
      --readout-checkpoint "$READOUT" \
      --readout-checkpoint-sha256 "$READOUT_SHA" \
      --renderer-geometry-checkpoint "$renderer" \
      --renderer-geometry-checkpoint-sha256 "$renderer_sha" \
      --output "$score_cache" \
      --chunk-size 4096
  fi

  if [[ ! -f "$OUT/fixed/$scene/$scene/lerf_direct_3d_selection_results.json" ]]; then
    guard "${scene}_fixed" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$REPO_ROOT/radio_gs/scripts/eval_lerf_direct_3d_selection.py" \
        --config "$config" \
        --checkpoint "$renderer" \
        --scene "$scene" \
        --protocol_preset vala_repo_3d \
        --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
        --output_dir "$OUT/fixed/$scene" \
        --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
        --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
        --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
        --ours_multiscale_query_score_cache "$score_cache" \
        --gpu 0
  fi

  if [[ ! -f "$OUT/otsu3/$scene/$scene/lerf_direct_3d_selection_results.json" ]]; then
    guard "${scene}_otsu3" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$REPO_ROOT/radio_gs/scripts/eval_lerf_adaptive_support_diagnostic.py" \
        --config "$config" \
        --checkpoint "$renderer" \
        --scene "$scene" \
        --ours_multiscale_query_score_cache "$score_cache" \
        --output_dir "$OUT/otsu3/$scene" \
        --calibration_mode recursive_upper_otsu3 \
        --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
        --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
        --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
        --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
        --gpu 0
  fi

  if [[ ! -f "$OUT/score_quality/$scene/$scene/lerf_direct_3d_selection_results.json" ]]; then
    guard "${scene}_score_quality" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$REPO_ROOT/radio_gs/scripts/eval_lerf_score_quality_diagnostic.py" \
        --config "$config" \
        --checkpoint "$renderer" \
        --scene "$scene" \
        --ours_multiscale_query_score_cache "$score_cache" \
        --frozen_formal_result "$OUT/fixed/$scene/$scene/lerf_direct_3d_selection_results.json" \
        --target_blind_adaptive_result "$OUT/otsu3/$scene/$scene/lerf_direct_3d_selection_results.json" \
        --output_dir "$OUT/score_quality/$scene" \
        --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
        --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
        --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
        --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
        --gpu 0
  fi
}

run_scene figurines \
  /mnt/pool/sqy/results/RADIO-GS/output/canonical_fields/figurines_compact_d256_l128_primary_frozen_adjoint16_fallback_caploss_seed0.pth \
  328ba9f9f19f69f02a118462cbb427fac7670cbc83e4d4eade7e66902943aa66 \
  df01507d65b6a6e6ad75e001fd926b30e18482dd64cb065f3c58710c17969f81 \
  /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260716/global_3d_readout/figurines_support_graph_v3.pt \
  abcdd466fbbd726f277b59b137a59ac93b0a2c270a7557fc9916a478a66a1451 \
  /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260715/text/siglip2_lerf_figurines_officialcanonical_prompt5.pt \
  08fa6f870c824fde212b302d6c88cf74487f1122055fac709708399fb480578b \
  /mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth \
  6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2 \
  "$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"

if [[ "${RUN_WALDO_CONFIRMATION:-0}" == "1" ]]; then
  run_scene waldo_kitchen \
    /mnt/pool/sqy/results/RADIO-GS/output/canonical_fields/waldo_kitchen_compact_d256_l128_primary_frozen_adjoint16_fallback_caploss_seed0.pth \
    3f5a8892c47985f1f4312f104e110b9f57b76ba6b95801906a0bb230b61c8861 \
    fc4b5a31841e569d2da1e12a073be82f991cfaf924f097043f9cebfb8e47760a \
    /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260716/global_3d_readout/waldo_kitchen_support_graph_v3.pt \
    1e25ba9ed3ffd4dd0d80733abebcf1cef32d7653739e1ab18aa37a827d0ce7d1 \
    /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260715/text/siglip2_lerf_waldo_kitchen_officialcanonical_prompt5.pt \
    3607f6661d46d71f8a0b92ccdcf8d16cd922f89f7a9dcae69553ef67d4801fb1 \
    /mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth \
    16a47d24f83744efced0830cbef226ead3c124535e242de9de7f0cbc752ff95d \
    "$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_waldo_kitchen_radio_verified_pose.yaml"
fi
