#!/usr/bin/env bash
set -euo pipefail

# Scene0000 source-heldout native-DINO A/B/C sentinel and unchanged ScanNet
# readout confirmation.  This is a development sentinel, not a paper8 rollout.

ROOT=${ROOT:-/root/RADIO-GS}
GPU=${GPU:-4}
POLL_SECONDS=${POLL_SECONDS:-20}
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/scene0000_00}
CORE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/scene0000_00
TEACHER="$RUN_ROOT/native_dinov2_exact_mpr_trainval.pt"
ABC="$RUN_ROOT/native_dinov2_abc_matched_v2"
BASE_FIELD="$CORE/generic_text_response_w005_s0_64.pth"
GEOMETRY=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/scannet_og_scene0000_00_v14/checkpoints/best.pth
LOG="$RUN_ROOT/native_dinov2_abc_matched_v2_closure.log"

while [[ ! -s "$TEACHER" ]]; do
  sleep "$POLL_SECONDS"
done

if [[ ! -s "$ABC/abc_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$ROOT/radio_gs/scripts/train_native_multiteacher_abc_pilot.py" \
      --base-field "$BASE_FIELD" \
      --native-teacher "$TEACHER" \
      --output-dir "$ABC" \
      --device cuda:0 \
      --steps 800 \
      --batch-size 1024 \
      --radio-weight 1.0 \
      --seed 0 >>"$LOG" 2>&1
fi

# The benchmark readout is opened only after the source-heldout gate admits
# arm B.  This keeps a native-teacher regression from turning into
# metric-guided candidate selection.
MIN_NATIVE_GAIN=${MIN_NATIVE_GAIN:-0.0}
MIN_RADIO_COSINE=${MIN_RADIO_COSINE:-0.995}
if ! bash "$ROOT/radio_gs/scripts/run_repo_python.sh" - \
  "$ABC/abc_summary.json" "$MIN_NATIVE_GAIN" "$MIN_RADIO_COSINE" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
arm_a = summary["arms"]["A"]["heldout_native"]["mean_cosine"]
arm_b = summary["arms"]["B"]["heldout_native"]["mean_cosine"]
radio = summary["arms"]["B"]["radio_preservation"]["mean_cosine"]
passed = arm_b >= arm_a + float(sys.argv[2]) and radio >= float(sys.argv[3])
print(json.dumps({"source_gate_passed": passed, "A": arm_a, "B": arm_b, "radio": radio}))
raise SystemExit(0 if passed else 1)
PY
then
  echo "source-heldout gate rejected arm B; benchmark readout remains closed" >>"$LOG"
  chmod -R o+rwX "$RUN_ROOT"
  exit 0
fi

for ARM in b c; do
  FIELD="$ABC/arm_${ARM}_$( [[ "$ARM" == b ]] && echo radio_anchored || echo native_only )_field.pth"
  CACHE="$ABC/arm_${ARM}_primitive_query_method_v1.pth"
  if [[ ! -s "$CACHE" ]]; then
    FIELD_SHA=$(sha256sum "$FIELD" | awk '{print $1}')
    CUDA_VISIBLE_DEVICES="$GPU" \
      bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
        -m radio_gs.scripts.materialize_method_v1_primitive_query_cache \
        --field "$FIELD" \
        --expected-field-sha256 "$FIELD_SHA" \
        --geometry-checkpoint "$GEOMETRY" \
        --expected-geometry-sha256 d5ce0a13264ee2bb5a638a2eab6c51cd8a81e87ec1b03d8a70b956c1d08c40fa \
        --summary-head-weights "$ROOT/checkpoints/siglip2_summary_head.pth" \
        --expected-summary-head-sha256 41ccc47b2da9b1aed3ee1e80397dc721ec625e083054175c27698e8840b6263c \
        --authority "$ROOT/paper/artifacts/five_benchmark_method_v1_authority_20260815.json" \
        --output "$CACHE" \
        --device cuda:0 >>"$LOG" 2>&1
  fi
  RESULT_ROOT="$RUN_ROOT/scannet_eval_arm_${ARM}"
  RESULT="$RESULT_ROOT/scannet_vala_gaussian_protocol_results.json"
  if [[ ! -s "$RESULT" ]]; then
    CACHE_SHA=$(sha256sum "$CACHE" | awk '{print $1}')
    CUDA_VISIBLE_DEVICES="$GPU" \
      bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
        "$ROOT/radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py" \
        --scene_list scene0000_00 \
        --prepared_root /mnt/pool/sqy/3d_understanding/scannet_og \
        --config "$ROOT/radio_gs/configs/generated/frozen_eval_20260802/scannet_scene0000_canonical_mpr_v3_paper8.yaml" \
        --checkpoint "$GEOMETRY" \
        --label_ply /mnt/pool/sqy/3d_understanding/scannet_og/scene0000_00/scene0000_00_vh_clean_2.labels.ply \
        --output_dir "$RESULT_ROOT" \
        --class_splits 19,15,10 \
        --feature_chunk_size 8192 \
        --pseudo_chunk_size 512 \
        --pseudo_gt_cache_dir "$CORE/scannet_vala_method_v1/pseudo_gt" \
        --radius_factor 5.0 \
        --candidate_k 1000 \
        --fallback_k 1 \
        --row_opacity_threshold 0.1 \
        --compact_feature_key features \
        --prompt_templates '{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}' \
        --text_embedding_cache "$ROOT/checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt" \
        --text_encoder siglip2 \
        --class_aliases none \
        --projection_weights "$ROOT/checkpoints/siglip2_feat_projection.pth" \
        --summary_head_weights "$ROOT/checkpoints/siglip2_summary_head.pth" \
        --external_query_feature_cache "$CACHE" \
        --expected_external_query_feature_cache_sha256 "$CACHE_SHA" \
        --device cuda >>"$LOG" 2>&1
  fi
done

chmod -R o+rwX "$RUN_ROOT"
