#!/usr/bin/env bash

set -euo pipefail

ROOT=/root/RADIO-GS
OUT=${RUN_OUTPUT_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf3d_peak_retention_guard_full4_v2_bound}
SCENE=${SCENE:?set SCENE to figurines, ramen, teatime, or waldo_kitchen}
PHYSICAL_GPU=${PHYSICAL_GPU:?set PHYSICAL_GPU to an authorized free GPU}
SAM_EXACT_CACHE=${SAM_EXACT_CACHE:-}
EXPECTED_SAM_EXACT_CACHE_SHA256=${EXPECTED_SAM_EXACT_CACHE_SHA256:-}
SAM_SEED_EXTENT_ALPHA=${SAM_SEED_EXTENT_ALPHA:-0}
SAM_SEED_EXTENT_PROPOSAL_MEAN_RATIO=${SAM_SEED_EXTENT_PROPOSAL_MEAN_RATIO:-0.5}
SAM_SEED_EXTENT_SEED_SUPPORT_RATIO=${SAM_SEED_EXTENT_SEED_SUPPORT_RATIO:-0.8}
SAM_SEED_EXTENT_MINIMUM_VIEWS=${SAM_SEED_EXTENT_MINIMUM_VIEWS:-2}
SAM_SEED_EXTENT_QUERY_CONDITIONED=${SAM_SEED_EXTENT_QUERY_CONDITIONED:-0}
VALA_POST_MASK_REFINEMENT=${VALA_POST_MASK_REFINEMENT:-peak_component_retention_guard}
SAM_PROMPT_MASK_HEAD_CHECKPOINT=${SAM_PROMPT_MASK_HEAD_CHECKPOINT:-}
EXPECTED_SAM_PROMPT_MASK_HEAD_SHA256=${EXPECTED_SAM_PROMPT_MASK_HEAD_SHA256:-}
SAM_PROMPT_MASK_HEAD_FEATURE_DIR=${SAM_PROMPT_MASK_HEAD_FEATURE_DIR:-}
EXPECTED_SAM_PROMPT_MASK_HEAD_FEATURE_MANIFEST_SHA256=${EXPECTED_SAM_PROMPT_MASK_HEAD_FEATURE_MANIFEST_SHA256:-}

case "$SCENE" in
  figurines)
    CONFIG="$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_figurines_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
    CHECKPOINT_SHA=6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2
    CONFIG_SHA=e3e213f6551aaa339683b8de18459ba87427f8146d2709ff6173cd58b46be8bd
    CACHE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/figurines/primitive_query_method_v1.pth
    CACHE_SHA=acc0b8b4cbf429d92e2f9df05865898066349fb79bcbe0bd3933ae1e504f1e18
    FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/figurines/generic_text_response_w005_s0_64_lineage.pth
    FIELD_SHA=9beeb9db4f91055ee17eaee4b85c60f790417fb9cc109772fea853b1c5b86e8b
    ;;
  ramen)
    CONFIG="$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_ramen_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth
    CHECKPOINT_SHA=e18ca7ef06009953048b364a379585b378381df1c350c8d14643d5656cc5246d
    CONFIG_SHA=9fdb2cc5ba5ff9c8786700753cac4e1b5fd0d3f8e089b2af3fb764592fd8ec93
    CACHE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/ramen/primitive_query_method_v1.pth
    CACHE_SHA=893fda2a90142f71ee8175e666f12353a93e08a8125d8d5bdaf26d3a95dc54b5
    FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/ramen/generic_text_response_w005_s0_64.pth
    FIELD_SHA=9b469fb1732235f9f81db1609fc0e2cf710f7211f74410b895bc1edcf8971e8f
    ;;
  teatime)
    CONFIG="$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_teatime_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth
    CHECKPOINT_SHA=f5be0dea030dba6b69906e8b49da740d2722ebb589fe1f9d6bce4731de877f53
    CONFIG_SHA=9c92cfbfd3d901134d23b3e4fde88e27cc8a352f4034c213b3de89d10569e775
    CACHE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/teatime/primitive_query_method_v1.pth
    CACHE_SHA=3938c13cd5f2c78cc2522aeff26cb0f77ba08cbeb519288b4b564dffd629b96b
    FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/teatime/generic_text_response_w005_s0_64.pth
    FIELD_SHA=6fccff587374e5d6a9ba7cb9f381065bdd1d91615f6331e942e06248992f4d55
    ;;
  waldo_kitchen)
    CONFIG="$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_waldo_kitchen_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth
    CHECKPOINT_SHA=16a47d24f83744efced0830cbef226ead3c124535e242de9de7f0cbc752ff95d
    CONFIG_SHA=c270ffc84d197a3e5dbadeb00f7e9a23cc32947d09332889401c93459b92c264
    CACHE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/primitive_query_method_v1.pth
    CACHE_SHA=01ffe08e54466dc0da720bcc2e25029ae2b085e24e78f8ac5ad9ced28085159f
    FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/generic_text_response_w005_s0_64.pth
    FIELD_SHA=c6604f4bb69771d41452b28fb28cdfac8c6934ed22a47312322143dde3c4f4c1
    ;;
  *)
    echo "unsupported scene: $SCENE" >&2
    exit 2
    ;;
esac

RESULT="$OUT/$SCENE/$SCENE/lerf_direct_3d_selection_results.json"
if [[ -f "$RESULT" ]]; then
  echo "existing result must be audited explicitly; refusing to reuse: $RESULT" >&2
  exit 3
fi

SCENE_OUT="$OUT/$SCENE"
if [[ -d "$SCENE_OUT" ]] && [[ -n "$(find "$SCENE_OUT" -mindepth 1 -print -quit)" ]]; then
  echo "partial output must be audited before resume: $SCENE_OUT" >&2
  exit 3
fi
mkdir -p "$OUT/.locks" "$OUT/logs"
LOG="$OUT/logs/${SCENE}.log"
if [[ -e "$LOG" ]]; then
  echo "existing log must be audited before rerun: $LOG" >&2
  exit 3
fi
LOCK="$OUT/.locks/$SCENE"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "scene is already running: $SCENE ($LOCK)" >&2
  exit 4
fi
RECEIPT_TMP=""
cleanup() {
  if [[ -n "$RECEIPT_TMP" ]]; then
    rm -f "$RECEIPT_TMP"
  fi
  rmdir "$LOCK" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$SCENE_OUT"
RECEIPT="$OUT/input_receipts/$SCENE.json"
mkdir -p "$OUT/input_receipts"
if [[ -e "$RECEIPT" ]]; then
  echo "existing input receipt must be audited before rerun: $RECEIPT" >&2
  exit 3
fi
RECEIPT_TMP="$OUT/input_receipts/.${SCENE}.json.$$"
bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$ROOT/radio_gs/scripts/validate_lerf3d_peak_retention_inputs.py" \
  --scene "$SCENE" \
  --primitive-query-cache "$CACHE" \
  --expected-primitive-query-cache-sha256 "$CACHE_SHA" \
  --field "$FIELD" \
  --expected-field-sha256 "$FIELD_SHA" \
  --renderer "$CHECKPOINT" \
  --expected-renderer-sha256 "$CHECKPOINT_SHA" \
  --config "$CONFIG" \
  --expected-config-sha256 "$CONFIG_SHA" \
  --method-authority "$ROOT/paper/artifacts/five_benchmark_method_v1_authority_20260815.json" \
  --summary-head "$ROOT/checkpoints/siglip2_summary_head.pth" \
  --text-cache "$ROOT/checkpoints/siglip2_lerf_all_exact_official.pt" \
  --canonical-cache "$ROOT/checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt" \
  >"$RECEIPT_TMP"
mv -n "$RECEIPT_TMP" "$RECEIPT"
RECEIPT_TMP=""
SAM_ARGS=()
SAM_VALIDATE_ARGS=()
if [[ -n "$SAM_EXACT_CACHE" ]]; then
  if [[ -z "$EXPECTED_SAM_EXACT_CACHE_SHA256" ]]; then
    echo "EXPECTED_SAM_EXACT_CACHE_SHA256 is required with SAM_EXACT_CACHE" >&2
    exit 5
  fi
  if [[ ! -r "$SAM_EXACT_CACHE" ]] || \
     [[ "$(sha256sum "$SAM_EXACT_CACHE" | cut -d' ' -f1)" != "$EXPECTED_SAM_EXACT_CACHE_SHA256" ]]; then
    echo "SAM exact-MPR membership cache SHA-256 differs" >&2
    exit 5
  fi
  SAM_ARGS+=(
    --sam3_exact_mpr_membership_cache "$SAM_EXACT_CACHE"
    --sam3_seed_extent_alpha "$SAM_SEED_EXTENT_ALPHA"
    --sam3_seed_extent_proposal_mean_ratio "$SAM_SEED_EXTENT_PROPOSAL_MEAN_RATIO"
    --sam3_seed_extent_seed_support_ratio "$SAM_SEED_EXTENT_SEED_SUPPORT_RATIO"
    --sam3_seed_extent_minimum_views "$SAM_SEED_EXTENT_MINIMUM_VIEWS"
  )
  if [[ "$SAM_SEED_EXTENT_QUERY_CONDITIONED" == 1 ]]; then
    SAM_ARGS+=(--sam3_seed_extent_query_conditioned)
  fi
  SAM_VALIDATE_ARGS+=(
    --sam-membership-cache "$SAM_EXACT_CACHE"
    --expected-sam-membership-cache-sha256 "$EXPECTED_SAM_EXACT_CACHE_SHA256"
    --sam-seed-alpha "$SAM_SEED_EXTENT_ALPHA"
    --sam-proposal-mean-ratio "$SAM_SEED_EXTENT_PROPOSAL_MEAN_RATIO"
    --sam-seed-support-ratio "$SAM_SEED_EXTENT_SEED_SUPPORT_RATIO"
    --sam-minimum-views "$SAM_SEED_EXTENT_MINIMUM_VIEWS"
  )
  if [[ "$SAM_SEED_EXTENT_QUERY_CONDITIONED" == 1 ]]; then
    SAM_VALIDATE_ARGS+=(--sam-query-conditioned)
  fi
fi
PROMPT_HEAD_ARGS=()
if [[ "$VALA_POST_MASK_REFINEMENT" == sam3_prompt_mask_head ]]; then
  if [[ -z "$SAM_PROMPT_MASK_HEAD_CHECKPOINT" ]] || \
     [[ -z "$EXPECTED_SAM_PROMPT_MASK_HEAD_SHA256" ]] || \
     [[ -z "$SAM_PROMPT_MASK_HEAD_FEATURE_DIR" ]] || \
     [[ -z "$EXPECTED_SAM_PROMPT_MASK_HEAD_FEATURE_MANIFEST_SHA256" ]]; then
    echo "prompt-mask-head checkpoint, feature directory, and expected SHA-256 values are required" >&2
    exit 6
  fi
  FEATURE_MANIFEST="$SAM_PROMPT_MASK_HEAD_FEATURE_DIR/canonical_render_manifest.json"
  if [[ ! -r "$SAM_PROMPT_MASK_HEAD_CHECKPOINT" ]] || \
     [[ "$(sha256sum "$SAM_PROMPT_MASK_HEAD_CHECKPOINT" | cut -d' ' -f1)" != "$EXPECTED_SAM_PROMPT_MASK_HEAD_SHA256" ]] || \
     [[ ! -r "$FEATURE_MANIFEST" ]] || \
     [[ "$(sha256sum "$FEATURE_MANIFEST" | cut -d' ' -f1)" != "$EXPECTED_SAM_PROMPT_MASK_HEAD_FEATURE_MANIFEST_SHA256" ]]; then
    echo "prompt-mask-head checkpoint or current-field feature manifest differs" >&2
    exit 6
  fi
  PROMPT_HEAD_ARGS+=(
    --sam3_prompt_mask_head_checkpoint "$SAM_PROMPT_MASK_HEAD_CHECKPOINT"
    --sam3_prompt_mask_head_feature_dir "$SAM_PROMPT_MASK_HEAD_FEATURE_DIR"
    --sam3_prompt_mask_head_logit_threshold 0.0
    --sam3_prompt_mask_head_min_initial_iou 0.5
    --sam3_prompt_mask_head_max_initial_area_fraction 1.0
    --sam3_prompt_mask_head_min_refined_area_ratio 0.7
    --sam3_prompt_mask_head_max_refined_area_ratio 1.3
    --sam3_prompt_mask_head_support_dilate 12
    --sam3_prompt_mask_head_coarse_dilate 1
    --sam3_prompt_mask_head_coarse_threshold 0.5
    --sam3_prompt_mask_head_min_heatmap_mean_ratio 0.85
    --sam3_prompt_mask_head_min_heatmap_mass_ratio 0.25
    --sam3_prompt_mask_head_require_peak_in_refined
    --sam3_prompt_mask_head_initial_refinement peak_component
  )
fi
CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
  bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$ROOT/radio_gs/scripts/eval_lerf_direct_3d_selection.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --scene "$SCENE" \
  --protocol_preset vala_repo_3d \
  --vala_post_mask_refinement "$VALA_POST_MASK_REFINEMENT" \
  --external_query_feature_cache "$CACHE" \
  --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
  --output_dir "$SCENE_OUT" \
  --summary_head_weights "$ROOT/checkpoints/siglip2_summary_head.pth" \
  --text_embedding_cache "$ROOT/checkpoints/siglip2_lerf_all_exact_official.pt" \
  --canonical_embedding_cache "$ROOT/checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt" \
  "${SAM_ARGS[@]}" \
  "${PROMPT_HEAD_ARGS[@]}" \
  --gpu 0 \
  >"$LOG" 2>&1

sha256sum "$RESULT"
bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$ROOT/radio_gs/scripts/validate_lerf3d_peak_retention_result.py" \
  --scene "$SCENE" \
  --result "$RESULT" \
  --input-receipt "$RECEIPT" \
  --primitive-query-cache "$CACHE" \
  --text-cache "$ROOT/checkpoints/siglip2_lerf_all_exact_official.pt" \
  --canonical-cache "$ROOT/checkpoints/siglip2_lerf_all_generic_negatives_exact_official.pt" \
  --expected-post-mask-refinement "$VALA_POST_MASK_REFINEMENT" \
  "${SAM_VALIDATE_ARGS[@]}"
