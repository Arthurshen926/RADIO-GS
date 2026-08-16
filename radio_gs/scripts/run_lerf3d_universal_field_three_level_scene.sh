#!/usr/bin/env bash

set -euo pipefail

ROOT=/root/RADIO-GS
OUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260816/lerf3d_universal_field_three_level_2x2_v1
SCENE=${SCENE:?set SCENE to figurines, ramen, teatime, or waldo_kitchen}
MODE=${MODE:-materialize}
PHYSICAL_GPU=${PHYSICAL_GPU:-0}
PYTHON=(bash "$ROOT/radio_gs/scripts/run_repo_python.sh")
RADIO=/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
RADIO_SHA=bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9
READOUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260717/surface_region_contract_v2/readout_v2_clean_h256.pth
READOUT_SHA=5b2d123a7827d9ab79aa4aa5a70077f00a656beebcf4c95ea5a3c9efdbe13ccb
READOUT_AUTHORITY="$ROOT/paper/artifacts/surface_region_accepted_v2_legacy_radio_authority_20260805.json"
READOUT_AUTHORITY_SHA=2d72ed54eef69378c665ddf83067838d79e42d6b72545d89aab12f95cb781948

case "$SCENE" in
  figurines)
    FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/figurines/generic_text_response_w005_s0_64_lineage.pth
    FIELD_SHA=9beeb9db4f91055ee17eaee4b85c60f790417fb9cc109772fea853b1c5b86e8b
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/factorized_raw_radio_exact_marginal.pt
    MPR_SHA=4bad5345f6721f7fb2fab5a234a93ae80c0e5ce39217d1bd841e29559fabbf4b
    TEXT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260715/text/siglip2_lerf_figurines_officialcanonical_prompt5.pt
    TEXT_SHA=08fa6f870c824fde212b302d6c88cf74487f1122055fac709708399fb480578b
    QUERIES='bag,green apple,green toy chair,jake,miffy,old camera,pikachu,pink ice cream,pirate hat,porcelain hand,pumpkin,red apple,red toy chair,rubber duck with buoy,rubber duck with hat,rubics cube,spatula,tesla door handle,toy cat statue,toy elephant,waldo'
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
    RENDERER_SHA=6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2
    ;;
  ramen)
    FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/ramen/generic_text_response_w005_s0_64.pth
    FIELD_SHA=9b469fb1732235f9f81db1609fc0e2cf710f7211f74410b895bc1edcf8971e8f
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/ramen/fix4c_exact_marginal_v1/factorized_raw_radio_exact_marginal.pt
    MPR_SHA=1b3c6c5ac01b6dea3e993db64217b60a942ba33adbed2e34e8357846854b3f8e
    TEXT="$OUT/assets/ramen/exact_query_text_cache.pt"
    TEXT_SHA=60781c9a9e2db6c18427d5496da330068e3665303e396deea1f3f377b801721c
    QUERIES='bowl,chopsticks,corn,egg,glass of water,hand,kamaboko,napkin,nori,onion segments,plate,sake cup,spoon,wavy noodles'
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth
    RENDERER_SHA=e18ca7ef06009953048b364a379585b378381df1c350c8d14643d5656cc5246d
    ;;
  teatime)
    FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/teatime/generic_text_response_w005_s0_64.pth
    FIELD_SHA=6fccff587374e5d6a9ba7cb9f381065bdd1d91615f6331e942e06248992f4d55
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/teatime/exact_marginal_target_v1/factorized_raw_radio_exact_marginal.pt
    MPR_SHA=c9d903172972c982d24e6062416b2a01136b4e1c5e7cc0fc2694fa5e4dcc1af3
    TEXT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260715/text/siglip2_lerf_teatime_officialcanonical_prompt5.pt
    TEXT_SHA=ecc942ae3ac2e43d1a503a4bb606baf35d7d1b3fa3d49b85ed2641117ef9c6a0
    QUERIES='apple,bag of cookies,bear nose,coffee,coffee mug,dall-e brand,hooves,paper napkin,plate,sheep,stuffed bear,tea in a glass,three cookies,yellow pouf'
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth
    RENDERER_SHA=f5be0dea030dba6b69906e8b49da740d2722ebb589fe1f9d6bce4731de877f53
    ;;
  waldo_kitchen)
    FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/generic_text_response_w005_s0_64.pth
    FIELD_SHA=c6604f4bb69771d41452b28fb28cdfac8c6934ed22a47312322143dde3c4f4c1
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/factorized_raw_radio_exact_marginal.pt
    MPR_SHA=677c47a915528d48085fb29f1697cc77dff92cfd7281ae66210db62e2cf71787
    TEXT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260715/text/siglip2_lerf_waldo_kitchen_officialcanonical_prompt5.pt
    TEXT_SHA=3607f6661d46d71f8a0b92ccdcf8d16cd922f89f7a9dcae69553ef67d4801fb1
    QUERIES='Stainless steel pots,cabinet,dark cup,frog cup,ketchup,knife,ottolenghi,plastic ladle,plate,pot,pour-over vessel,red cup,refrigerator,sink,spatula,spoon,toaster,yellow desk'
    RENDERER=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth
    RENDERER_SHA=16a47d24f83744efced0830cbef226ead3c124535e242de9de7f0cbc752ff95d
    ;;
  *)
    echo "unsupported scene: $SCENE" >&2
    exit 2
    ;;
esac

SCENE_ROOT="$OUT/assets/$SCENE"
STATE="$SCENE_ROOT/factorized_primitive_state_v2.pt"
CAPABILITY="$SCENE_ROOT/current_field_official_capability_bank.pt"
GRAPH="$SCENE_ROOT/current_field_support_graph_k16.pt"
if [[ "$SCENE" == ramen ]]; then
  STREAMED="$SCENE_ROOT/current_field_three_level_streamed_scores_exact_text_v2.pt"
  RESUME_STREAMED="$SCENE_ROOT/resume_streamed_scores_exact_text_v2"
else
  STREAMED="$SCENE_ROOT/current_field_three_level_streamed_scores.pt"
  RESUME_STREAMED="$SCENE_ROOT/resume_streamed_scores"
fi
SCORES="$OUT/query_scores/$SCENE.pt"
CONFIG="$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_${SCENE}_radio_verified_pose.yaml"
RESULT="$OUT/results/$SCENE/$SCENE/lerf_direct_3d_selection_results.json"
mkdir -p "$SCENE_ROOT" "$OUT/query_scores" "$OUT/logs" "$OUT/results"

guard() {
  local tag=$1
  shift
  GPU="$PHYSICAL_GPU" \
  GPU_TELEMETRY_LOG="$OUT/logs/${SCENE}_${tag}.telemetry.csv" \
  GPU_OWNER_AUDIT_LOG="$OUT/logs/${SCENE}_${tag}.owner_audit.csv" \
  GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
  GPU_POLL_SECONDS=30 \
  GPU_START_MAX_TEMP_C=81 \
  GPU_SOFT_PAUSE_TEMP_C=84 \
  GPU_SOFT_RESUME_TEMP_C=80 \
  GPU_MAX_TEMP_C=87 \
  GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS=2 \
  GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=2 \
  GPU_MAX_POWER_LIMIT_W=300.5 \
  CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
    bash "$ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh" -- "$@"
}

if [[ "$MODE" == materialize ]]; then
  if [[ ! -f "$STATE" ]]; then
    "${PYTHON[@]}" "$ROOT/radio_gs/scripts/build_factorized_primitive_state.py" \
      --field-checkpoint "$FIELD" \
      --expected-field-checkpoint-sha256 "$FIELD_SHA" \
      --factorized-radio-cache "$MPR" \
      --expected-factorized-radio-cache-sha256 "$MPR_SHA" \
      --output "$STATE" \
      --chunk-size 4096
  fi
  if [[ ! -f "$CAPABILITY" ]]; then
    guard capability "${PYTHON[@]}" \
      "$ROOT/radio_gs/scripts/build_canonical_capability_views.py" \
      --field-checkpoint "$FIELD" \
      --field-checkpoint-schema factorized-v2 \
      --expected-field-checkpoint-sha256 "$FIELD_SHA" \
      --output "$CAPABILITY" \
      --mpr-cache "$MPR" \
      --expected-mpr-cache-sha256 "$MPR_SHA" \
      --observation-contract canonical \
      --radio-checkpoint "$RADIO" \
      --expected-radio-checkpoint-sha256 "$RADIO_SHA" \
      --batch-size 2048 \
      --device cuda:0
  fi
  CAPABILITY_SHA=$(sha256sum "$CAPABILITY" | awk '{print $1}')
  if [[ ! -f "$GRAPH" ]]; then
    guard support_graph "${PYTHON[@]}" \
      "$ROOT/radio_gs/scripts/build_canonical_support_graph.py" \
      --capability-cache "$CAPABILITY" \
      --expected-capability-cache-sha256 "$CAPABILITY_SHA" \
      --output "$GRAPH" \
      --neighbors 16 \
      --spatial-scale 2.0 \
      --appearance-temperature 0.1 \
      --boundary-temperature 0.1 \
      --normal-temperature 0.2 \
      --surface-tangent-temperature 0.2 \
      --surface-relation none \
      --surface-topology-min-affinity 0.0 \
      --covisibility-weight 0.0 \
      --affinity-dim 256 \
      --hash-batch-size 8192 \
      --capability-affinity-mode signed_hash \
      --affinity-device cuda:0 \
      --affinity-chunk-size 65536 \
      --topology-mode symmetric_union
  fi
  STATE_SHA=$(sha256sum "$STATE" | awk '{print $1}')
  GRAPH_SHA=$(sha256sum "$GRAPH" | awk '{print $1}')
  if [[ ! -f "$STREAMED" ]]; then
    guard streamed_scores "${PYTHON[@]}" \
      "$ROOT/radio_gs/scripts/build_surface_region_semantic_cache.py" \
      --field-checkpoint "$FIELD" \
      --field-checkpoint-schema factorized-v2 \
      --factorized-primitive-state "$STATE" \
      --factorized-primitive-state-sha256 "$STATE_SHA" \
      --field-checkpoint-sha256 "$FIELD_SHA" \
      --support-graph "$GRAPH" \
      --support-graph-sha256 "$GRAPH_SHA" \
      --readout-checkpoint "$READOUT" \
      --readout-checkpoint-sha256 "$READOUT_SHA" \
      --readout-legacy-radio-authority "$READOUT_AUTHORITY" \
      --readout-legacy-radio-authority-sha256 "$READOUT_AUTHORITY_SHA" \
      --mpr-cache-sha256 "$MPR_SHA" \
      --output "$STREAMED" \
      --region-radii 0.25,0.45,0.7 \
      --graph-neighbors 16 \
      --radio-batch-size 4096 \
      --semantic-batch-size 512 \
      --resume-dir "$RESUME_STREAMED" \
      --radio-feature-normalization legacy_raw \
      --thermal-pacing-seconds-per-batch 0.05 \
      --stream-text-queries "$QUERIES" \
      --text-embedding-cache "$TEXT" \
      --preserve-streamed-text-scales \
      --device cuda:0 \
      --radio-checkpoint "$RADIO" \
      --radio-checkpoint-sha256 "$RADIO_SHA"
  fi
  STREAMED_SHA=$(sha256sum "$STREAMED" | awk '{print $1}')
  if [[ ! -f "$SCORES" ]]; then
    "${PYTHON[@]}" \
      "$ROOT/radio_gs/scripts/materialize_lerf_streamed_multiscale_query_score_cache.py" \
      --streamed-score-cache "$STREAMED" \
      --streamed-score-cache-sha256 "$STREAMED_SHA" \
      --text-query-cache "$TEXT" \
      --text-query-cache-sha256 "$TEXT_SHA" \
      --field-checkpoint "$FIELD" \
      --field-checkpoint-sha256 "$FIELD_SHA" \
      --readout-checkpoint "$READOUT" \
      --readout-checkpoint-sha256 "$READOUT_SHA" \
      --renderer-geometry-checkpoint "$RENDERER" \
      --renderer-geometry-checkpoint-sha256 "$RENDERER_SHA" \
      --output "$SCORES"
  fi
  sha256sum "$STATE" "$CAPABILITY" "$GRAPH" "$STREAMED" "$SCORES"
elif [[ "$MODE" == evaluate ]]; then
  test -f "$SCORES"
  if [[ ! -f "$RESULT" ]]; then
    guard evaluate "${PYTHON[@]}" \
      "$ROOT/radio_gs/scripts/eval_lerf_direct_3d_selection.py" \
      --config "$CONFIG" \
      --checkpoint "$RENDERER" \
      --scene "$SCENE" \
      --protocol_preset vala_repo_3d \
      --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
      --output_dir "$OUT/results/$SCENE" \
      --summary_head_weights "$ROOT/checkpoints/siglip2_summary_head.pth" \
      --text_embedding_cache "$ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
      --canonical_embedding_cache "$ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
      --ours_multiscale_query_score_cache "$SCORES" \
      --gpu 0
  fi
  sha256sum "$RESULT"
else
  echo "unsupported MODE: $MODE" >&2
  exit 2
fi
