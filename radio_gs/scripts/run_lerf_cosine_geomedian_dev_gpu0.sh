#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=/root/RADIO-GS
OUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260804/lerf3d_cosine_geomedian_v1/dev_gate
READOUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260804/lerf3d_cosine_geomedian_v1/readout_seed0.pt
READOUT_SHA=26493b5262f801b306c65f7e3564fa28652064c0089b07ebeb65212321da4d1b
RADIO=/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
RADIO_SHA=bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9
REGISTRATION="$REPO_ROOT/paper/artifacts/lerf3d_cosine_geomedian_readout_registration_20260804.json"
REGISTRATION_SHA=c2d65f4585644e4ee5a9afbbbffab7f4dcf2463a8872f4c4c8718c3b3bf0c724

mkdir -p "$OUT/logs" "$OUT/streamed_scores" "$OUT/query_scores"
printf '%s  %s\n' "$REGISTRATION_SHA" "$REGISTRATION" | sha256sum --check --status
printf '%s  %s\n' "$READOUT_SHA" "$READOUT" | sha256sum --check --status

guard() {
  local tag="$1"
  shift
  GPU=0 \
  GPU_TELEMETRY_LOG="$OUT/logs/${tag}.telemetry.csv" \
  GPU_OWNER_AUDIT_LOG="$OUT/logs/${tag}.owner_audit.csv" \
  GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
  GPU_POLL_SECONDS=180 \
  GPU_START_MAX_TEMP_C=83 \
  GPU_MAX_TEMP_C=87 \
  GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS=2 \
  GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=2 \
  GPU_SOFT_PAUSE_TEMP_C=0 \
  GPU_SOFT_RESUME_TEMP_C=0 \
  GPU_MAX_POWER_LIMIT_W=300.5 \
  CUDA_VISIBLE_DEVICES=0 \
    bash "$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh" -- "$@"
}

scene_values() {
  local scene="$1"
  case "$scene" in
    figurines)
      FIELD=/mnt/pool/sqy/results/RADIO-GS/output/canonical_fields/figurines_compact_d256_l128_primary_frozen_adjoint16_fallback_caploss_seed0.pth
      FIELD_SHA=328ba9f9f19f69f02a118462cbb427fac7670cbc83e4d4eade7e66902943aa66
      MPR_SHA=df01507d65b6a6e6ad75e001fd926b30e18482dd64cb065f3c58710c17969f81
      GRAPH=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260716/global_3d_readout/figurines_support_graph_v3.pt
      GRAPH_SHA=abcdd466fbbd726f277b59b137a59ac93b0a2c270a7557fc9916a478a66a1451
      TEXT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260715/text/siglip2_lerf_figurines_officialcanonical_prompt5.pt
      TEXT_SHA=08fa6f870c824fde212b302d6c88cf74487f1122055fac709708399fb480578b
      RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
      RENDERER_SHA=6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2
      CONFIG="$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"
      QUERIES='bag,green apple,green toy chair,jake,miffy,old camera,pikachu,pink ice cream,pirate hat,porcelain hand,pumpkin,red apple,red toy chair,rubber duck with buoy,rubber duck with hat,rubics cube,spatula,tesla door handle,toy cat statue,toy elephant,waldo'
      SEMANTIC_BATCH=512
      ;;
    waldo_kitchen)
      FIELD=/mnt/pool/sqy/results/RADIO-GS/output/canonical_fields/waldo_kitchen_compact_d256_l128_primary_frozen_adjoint16_fallback_caploss_seed0.pth
      FIELD_SHA=3f5a8892c47985f1f4312f104e110b9f57b76ba6b95801906a0bb230b61c8861
      MPR_SHA=fc4b5a31841e569d2da1e12a073be82f991cfaf924f097043f9cebfb8e47760a
      GRAPH=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260716/global_3d_readout/waldo_kitchen_support_graph_v3.pt
      GRAPH_SHA=1e25ba9ed3ffd4dd0d80733abebcf1cef32d7653739e1ab18aa37a827d0ce7d1
      TEXT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260715/text/siglip2_lerf_waldo_kitchen_officialcanonical_prompt5.pt
      TEXT_SHA=3607f6661d46d71f8a0b92ccdcf8d16cd922f89f7a9dcae69553ef67d4801fb1
      RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth
      RENDERER_SHA=16a47d24f83744efced0830cbef226ead3c124535e242de9de7f0cbc752ff95d
      CONFIG="$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_waldo_kitchen_radio_verified_pose.yaml"
      QUERIES='Stainless steel pots,cabinet,dark cup,frog cup,ketchup,knife,ottolenghi,plastic ladle,plate,pot,pour-over vessel,red cup,refrigerator,sink,spatula,spoon,toaster,yellow desk'
      SEMANTIC_BATCH=2048
      ;;
    *)
      echo "unsupported scene $scene" >&2
      exit 2
      ;;
  esac
}

seal_scene_scores() {
  local scene="$1"
  scene_values "$scene"
  local streamed="$OUT/streamed_scores/${scene}.pt"
  local score_cache="$OUT/query_scores/${scene}.pt"
  if [[ ! -f "$streamed" ]]; then
    guard "${scene}_stream_scores" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$REPO_ROOT/radio_gs/scripts/build_surface_region_semantic_cache.py" \
        --field-checkpoint "$FIELD" \
        --field-checkpoint-sha256 "$FIELD_SHA" \
        --support-graph "$GRAPH" \
        --support-graph-sha256 "$GRAPH_SHA" \
        --readout-checkpoint "$READOUT" \
        --readout-checkpoint-sha256 "$READOUT_SHA" \
        --mpr-cache-sha256 "$MPR_SHA" \
        --radio-checkpoint "$RADIO" \
        --radio-checkpoint-sha256 "$RADIO_SHA" \
        --output "$streamed" \
        --resume-dir "$OUT/resume/${scene}_bs${SEMANTIC_BATCH}" \
        --radio-batch-size 4096 \
        --semantic-batch-size "$SEMANTIC_BATCH" \
        --thermal-pacing-seconds-per-batch 0.05 \
        --stream-text-queries "$QUERIES" \
        --text-embedding-cache "$TEXT" \
        --preserve-streamed-text-scales \
        --device cuda:0
  fi
  if [[ ! -f "$score_cache" ]]; then
    local streamed_sha
    streamed_sha="$(sha256sum "$streamed" | awk '{print $1}')"
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$REPO_ROOT/radio_gs/scripts/materialize_lerf_streamed_multiscale_query_score_cache.py" \
      --streamed-score-cache "$streamed" \
      --streamed-score-cache-sha256 "$streamed_sha" \
      --text-query-cache "$TEXT" \
      --text-query-cache-sha256 "$TEXT_SHA" \
      --field-checkpoint "$FIELD" \
      --field-checkpoint-sha256 "$FIELD_SHA" \
      --readout-checkpoint "$READOUT" \
      --readout-checkpoint-sha256 "$READOUT_SHA" \
      --renderer-geometry-checkpoint "$RENDERER" \
      --renderer-geometry-checkpoint-sha256 "$RENDERER_SHA" \
      --output "$score_cache"
  fi
}

evaluate_scene() {
  local scene="$1"
  scene_values "$scene"
  local score_cache="$OUT/query_scores/${scene}.pt"
  local fixed="$OUT/fixed/$scene/$scene/lerf_direct_3d_selection_results.json"
  local otsu3="$OUT/otsu3/$scene/$scene/lerf_direct_3d_selection_results.json"
  local quality="$OUT/score_quality/$scene/$scene/lerf_direct_3d_selection_results.json"
  if [[ ! -f "$fixed" ]]; then
    guard "${scene}_fixed" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$REPO_ROOT/radio_gs/scripts/eval_lerf_direct_3d_selection.py" \
        --config "$CONFIG" \
        --checkpoint "$RENDERER" \
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
  if [[ ! -f "$otsu3" ]]; then
    guard "${scene}_otsu3" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$REPO_ROOT/radio_gs/scripts/eval_lerf_adaptive_support_diagnostic.py" \
        --config "$CONFIG" \
        --checkpoint "$RENDERER" \
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
  if [[ ! -f "$quality" ]]; then
    guard "${scene}_score_quality" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$REPO_ROOT/radio_gs/scripts/eval_lerf_score_quality_diagnostic.py" \
        --config "$CONFIG" \
        --checkpoint "$RENDERER" \
        --scene "$scene" \
        --ours_multiscale_query_score_cache "$score_cache" \
        --frozen_formal_result "$fixed" \
        --target_blind_adaptive_result "$otsu3" \
        --output_dir "$OUT/score_quality/$scene" \
        --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
        --summary_head_weights "$REPO_ROOT/checkpoints/siglip2_summary_head.pth" \
        --text_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
        --canonical_embedding_cache "$REPO_ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
        --experiment-registration "$REGISTRATION" \
        --experiment-registration-sha256 "$REGISTRATION_SHA" \
        --gpu 0
  fi
}

# Freeze both scene score tensors before the first benchmark label is opened.
seal_scene_scores figurines
seal_scene_scores waldo_kitchen
sha256sum "$OUT/query_scores/figurines.pt" "$OUT/query_scores/waldo_kitchen.pt" \
  > "$OUT/sealed_scores_before_labels.sha256"

evaluate_scene figurines
evaluate_scene waldo_kitchen
