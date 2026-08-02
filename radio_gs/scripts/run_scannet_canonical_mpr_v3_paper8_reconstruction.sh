#!/usr/bin/env bash

# Reconstruct one missing ScanNet paper-8 canonical-mpr-v3 field from a
# completed, provenance-bound raw C-RADIO feature bundle.  This is a
# query-free construction queue: no benchmark labels, masks, objects, text
# queries, legacy semantic tensors, or metric outputs are opened here.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set one physical GPU index (0 or 1)}"
SCENE="${SCENE:?set one scene, e.g. scene0000_00 or 0000}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/optimization_20260802/scannet_canonical_mpr_v3_paper8}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
RADIO_CHECKPOINT_SHA256="${RADIO_CHECKPOINT_SHA256:-bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9}"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
START_STAGE="${START_STAGE:-validation_frames}"
STOP_STAGE="${STOP_STAGE:-support_graph}"
STATUS_ONLY="${STATUS_ONLY:-0}"

# These defaults keep both 300 W cards productive in the warm chassis.  A
# stage is paused at 81 C and resumed at 76 C; only 84 C, PCIe loss, a foreign
# owner, or three consecutive telemetry failures aborts it.  The 20 s sample
# period avoids turning telemetry into a scheduling bottleneck.
GPU_MAX_POWER_LIMIT_W="${GPU_MAX_POWER_LIMIT_W:-300.5}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-20}"
GPU_START_MAX_TEMP_C="${GPU_START_MAX_TEMP_C:-78}"
GPU_SOFT_PAUSE_TEMP_C="${GPU_SOFT_PAUSE_TEMP_C:-81}"
GPU_SOFT_RESUME_TEMP_C="${GPU_SOFT_RESUME_TEMP_C:-76}"
GPU_MAX_TEMP_C="${GPU_MAX_TEMP_C:-84}"
GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES="${GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES:-3}"

if [[ ! "$GPU" =~ ^[01]$ ]]; then
  echo "GPU must be physical index 0 or 1" >&2
  exit 2
fi
case "$SCENE" in
  scene????_00) SCENE_ID="${SCENE#scene}"; SCENE_ID="${SCENE_ID%_00}" ;;
  ???? ) SCENE_ID="$SCENE" ;;
  *) echo "SCENE must be one paper-8 ID such as 0000 or scene0000_00" >&2; exit 2 ;;
esac
SCENE_NAME="scene${SCENE_ID}_00"

# Geometry checkpoints are frozen carriers.  In particular, scene0400's v67
# checkpoint is permitted only for its Gaussian geometry; none of its learned
# semantic tensors is named, copied, or consumed by this queue.
case "$SCENE_ID" in
  0000)
    EXPECTED_FRAMES=279
    CONFIG="$REPO_ROOT/radio_gs/configs/generated/frozen_eval_20260802/scannet_scene0000_canonical_mpr_v3_paper8.yaml"
    GEOMETRY_CHECKPOINT="$REPO_ROOT/output/radio_gs/scannet_og_scene0000_00_v14/checkpoints/best.pth"
    GEOMETRY_CHECKPOINT_SHA256="d5ce0a13264ee2bb5a638a2eab6c51cd8a81e87ec1b03d8a70b956c1d08c40fa"
    EXPECTED_FEATURE_BUNDLE_SHA256="1b56ee37ad05045c6b97bc24ee34dce59a06b84486abb0735a87f887af843fbc"
    EXPECTED_EXCLUDED_STEMS=""
    ;;
  0070)
    EXPECTED_FRAMES=67
    CONFIG="$REPO_ROOT/radio_gs/configs/generated/frozen_eval_20260802/scannet_scene0070_canonical_mpr_v3_paper8.yaml"
    GEOMETRY_CHECKPOINT="$REPO_ROOT/output/radio_gs/scannet_og_scene0070_00_v14_b1/checkpoints/best.pth"
    GEOMETRY_CHECKPOINT_SHA256="2e083e662b8be14dcb5c02c58e3b34ea893294f82c9f583e1f1dad571e7437d1"
    EXPECTED_FEATURE_BUNDLE_SHA256="0b13a36666b63d4d47abc2d841065c98ed4d3229eb6de6833b69aa8ea7f83068"
    EXPECTED_EXCLUDED_STEMS=""
    ;;
  0097)
    EXPECTED_FRAMES=38
    CONFIG="$REPO_ROOT/radio_gs/configs/generated/frozen_eval_20260802/scannet_scene0097_canonical_mpr_v3_paper8.yaml"
    GEOMETRY_CHECKPOINT="$REPO_ROOT/output/radio_gs/scannet_og_scene0097_00_v14/checkpoints/best.pth"
    GEOMETRY_CHECKPOINT_SHA256="28e00ac88cf6cb1a8687d0ac44e9b3cc025910c24a7060a6850b6f62e669079c"
    EXPECTED_FEATURE_BUNDLE_SHA256="cb8d41739f663c807c81ee99e6e2f78411a3a09e3f79b3ac130514b6b6f76d96"
    EXPECTED_EXCLUDED_STEMS=""
    ;;
  0347)
    EXPECTED_FRAMES=54
    CONFIG="$REPO_ROOT/radio_gs/configs/generated/frozen_eval_20260802/scannet_scene0347_canonical_mpr_v3_paper8.yaml"
    GEOMETRY_CHECKPOINT="$REPO_ROOT/output/radio_gs/scannet_og_scene0347_00_v14/checkpoints/best.pth"
    GEOMETRY_CHECKPOINT_SHA256="d2b33c0ebabeba9e245c6fae358762a5e9c9172e964f8331890b2cc098564103"
    EXPECTED_FEATURE_BUNDLE_SHA256="5406f3c9e8b74966d17a54d581aeec6ecf03a08e640e40c4b12400fe1f23508f"
    EXPECTED_EXCLUDED_STEMS=""
    ;;
  0400)
    EXPECTED_FRAMES=61
    CONFIG="$REPO_ROOT/radio_gs/configs/generated/frozen_eval_20260802/scannet_scene0400_canonical_mpr_v3_paper8.yaml"
    GEOMETRY_CHECKPOINT="$REPO_ROOT/output/radio_gs/scannet_og_scene0400_00_v67_dino_cv001_b2_s32768_ft20/checkpoints/best.pth"
    GEOMETRY_CHECKPOINT_SHA256="e3bb13d1ea1e7baade004873e0db2f261b7a4c7eb2e56c6636e1e3ca11113db4"
    EXPECTED_FEATURE_BUNDLE_SHA256="dad197a9fb6ad1c104e16b8cd0a2879f337d2029f1c0167cf813f902a9084d7c"
    EXPECTED_EXCLUDED_STEMS="60,80,1260"
    ;;
  0590)
    EXPECTED_FRAMES=135
    CONFIG="$REPO_ROOT/radio_gs/configs/generated/frozen_eval_20260802/scannet_scene0590_canonical_mpr_v3_paper8.yaml"
    GEOMETRY_CHECKPOINT="$REPO_ROOT/output/radio_gs/scannet_og_scene0590_00_v14/checkpoints/best.pth"
    GEOMETRY_CHECKPOINT_SHA256="f6aef423b88a5a13c9e58687448616a2bbbca03ad080c27ba2e212b78ead587b"
    EXPECTED_FEATURE_BUNDLE_SHA256="c514cbdc09657a6ba865d45c1aca8d2f7b0b70994ea644359b5cc2e62749831c"
    EXPECTED_EXCLUDED_STEMS=""
    ;;
  *)
    echo "unsupported missing paper-8 scene: $SCENE_ID" >&2
    exit 2
    ;;
esac

SCENE_ROOT="/mnt/pool/sqy/3d_understanding/scannet_og/$SCENE_NAME"
IMAGE_DIR="$SCENE_ROOT/color"
FEATURE_DIR="$RUN_ROOT/feature_bundles/$SCENE_NAME"
FIELD_DIR="$RUN_ROOT/canonical_v2_validation/scannet_scene${SCENE_ID}"
LOG_DIR="$RUN_ROOT/logs"
VALIDATION_PLAN="$FIELD_DIR/fidelity_validation_frames.json"
RESPONSIBILITY="$FIELD_DIR/responsibility_heldout4.pt"
RAW_MPR="$FIELD_DIR/raw_radio_heldout4.pt"
DINO_MPR="$FIELD_DIR/dino_v3_heldout4.pt"
SAM3_MPR="$FIELD_DIR/sam3_heldout4.pt"
FIELD_V1="$FIELD_DIR/canonical_v1_d256_l128_fusion_fixedsig.pth"
FIELD_V2="$FIELD_DIR/canonical_mpr_v2_d256_l128_fusion.pth"
CAPABILITY="$FIELD_DIR/v2_official_dino_sam3_views.pt"
GRAPH="$FIELD_DIR/v2_shared_support_graph_k16.pt"
VALIDATOR="$REPO_ROOT/radio_gs/scripts/validate_scannet_canonical_mpr_v3_paper8_stage.py"
TELEMETRY_LOG="$LOG_DIR/gpu${GPU}_canonical_mpr_v3_telemetry.csv"
OWNER_LOG="$LOG_DIR/gpu${GPU}_canonical_mpr_v3_owner.csv"
mkdir -p "$FIELD_DIR" "$LOG_DIR"

STAGES=(validation_frames raw_mpr dino_mpr sam3_mpr field_v1 field_v2 capability support_graph)
stage_index() {
  local wanted="$1" index
  for index in "${!STAGES[@]}"; do
    if [[ "${STAGES[$index]}" == "$wanted" ]]; then
      printf '%s\n' "$index"
      return 0
    fi
  done
  echo "unknown stage: $wanted" >&2
  return 2
}
START_INDEX="$(stage_index "$START_STAGE")"
STOP_INDEX="$(stage_index "$STOP_STAGE")"
if (( START_INDEX > STOP_INDEX )); then
  echo "START_STAGE must not follow STOP_STAGE" >&2
  exit 2
fi

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

validated_artifact_sha256() {
  local artifact="$1" stamp="${1}.paper8_validation.json" digest
  [[ -s "$artifact" && -s "$stamp" ]] || {
    echo "validated artifact or stamp is missing: $artifact" >&2
    return 3
  }
  digest="$(
    sed -n 's/^[[:space:]]*"artifact_sha256": "\([0-9a-f]\{64\}\)",\{0,1\}$/\1/p' "$stamp"
  )"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "validation stamp has no canonical artifact digest: $stamp" >&2
    return 3
  }
  printf '%s\n' "$digest"
}

assert_digest() {
  local path="$1" expected="$2" label="$3" actual
  [[ -s "$path" ]] || { echo "missing $label: $path" >&2; exit 2; }
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA-256 mismatch: $actual != $expected" >&2
    exit 2
  fi
}

validate_feature_and_config() {
  FEATURE_BUNDLE_SHA256="$(
    CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh "$VALIDATOR" feature \
      --scene "$SCENE_NAME" \
      --feature-dir "$FEATURE_DIR" \
      --config "$CONFIG" \
      --scene-root "$SCENE_ROOT" \
      --expected-frames "$EXPECTED_FRAMES" \
      --expected-excluded-stems "$EXPECTED_EXCLUDED_STEMS" \
      --expected-output-bundle-sha256 "$EXPECTED_FEATURE_BUNDLE_SHA256" \
      --print-sha256
  )"
  [[ "$FEATURE_BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "feature validator returned an invalid bundle digest" >&2
    exit 2
  }
}

validation_args() {
  local kind="$1" path="$2"
  printf '%s\0' "$VALIDATOR" artifact \
    --kind "$kind" \
    --path "$path" \
    --scene "$SCENE_NAME" \
    --config "$CONFIG" \
    --geometry-checkpoint "$GEOMETRY_CHECKPOINT" \
    --geometry-checkpoint-sha256 "$GEOMETRY_CHECKPOINT_SHA256" \
    --feature-output-bundle-sha256 "$FEATURE_BUNDLE_SHA256" \
    --validation-plan "$VALIDATION_PLAN" \
    --responsibility-cache "$RESPONSIBILITY" \
    --raw-mpr "$RAW_MPR" \
    --dino-mpr "$DINO_MPR" \
    --sam3-mpr "$SAM3_MPR" \
    --field-v1 "$FIELD_V1" \
    --field-v2 "$FIELD_V2" \
    --capability-cache "$CAPABILITY" \
    --radio-checkpoint "$RADIO_CHECKPOINT" \
    --radio-checkpoint-sha256 "$RADIO_CHECKPOINT_SHA256"
  if (( ! STATUS_ONLY )); then
    printf '%s\0' --write-stamp
  fi
}

validate_artifact() {
  local kind="$1" path="$2"
  local -a args
  mapfile -d '' -t args < <(validation_args "$kind" "$path")
  CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh "${args[@]}"
}

artifact_is_current() {
  local kind="$1" path="$2"
  [[ -s "$path" ]] || return 1
  validate_artifact "$kind" "$path" >/dev/null
}

run_guarded() {
  local stage="$1"
  shift
  echo "$SCENE_NAME: running $stage on physical GPU$GPU"
  GPU="$GPU" \
  GPU_TELEMETRY_LOG="$TELEMETRY_LOG" \
  GPU_OWNER_AUDIT_LOG="$OWNER_LOG" \
  GPU_OWNER_PID_NAMESPACE_MODE="exclusive-singleton-after-clear-v1" \
  GPU_MAX_POWER_LIMIT_W="$GPU_MAX_POWER_LIMIT_W" \
  GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
  GPU_START_MAX_TEMP_C="$GPU_START_MAX_TEMP_C" \
  GPU_SOFT_PAUSE_TEMP_C="$GPU_SOFT_PAUSE_TEMP_C" \
  GPU_SOFT_RESUME_TEMP_C="$GPU_SOFT_RESUME_TEMP_C" \
  GPU_MAX_TEMP_C="$GPU_MAX_TEMP_C" \
  GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES="$GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES" \
    bash "$THERMAL_GUARD" -- \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      bash radio_gs/scripts/run_repo_python.sh "$@" \
      >"$LOG_DIR/${SCENE_NAME}.${stage}.log" 2>&1
}

run_or_validate() {
  local index="$1" kind="$2" path="$3" stage="$4"
  shift 4
  if artifact_is_current "$kind" "$path"; then
    echo "$SCENE_NAME: $stage is current"
    return 0
  fi
  if [[ -e "$path" ]]; then
    echo "$SCENE_NAME: existing $stage failed provenance/structure validation: $path" >&2
    exit 3
  fi
  if (( STATUS_ONLY )); then
    echo "$SCENE_NAME: missing $stage: $path" >&2
    exit 4
  fi
  if (( index < START_INDEX )); then
    echo "$SCENE_NAME: START_STAGE=$START_STAGE requires valid prior $stage" >&2
    exit 4
  fi
  run_guarded "$stage" "$@"
  validate_artifact "$kind" "$path" >/dev/null
}

assert_digest "$RADIO_CHECKPOINT" "$RADIO_CHECKPOINT_SHA256" "official RADIO checkpoint"
assert_digest "$GEOMETRY_CHECKPOINT" "$GEOMETRY_CHECKPOINT_SHA256" "geometry checkpoint"
validate_feature_and_config

# Always regenerate this tiny query-free plan.  It is deterministic and binds
# itself to the current feature manifest digest.
if (( ! STATUS_ONLY )); then
  CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/select_fidelity_validation_frames.py \
    --feature-dir "$FEATURE_DIR" \
    --output "$VALIDATION_PLAN" \
    --views 4 \
    >"$LOG_DIR/${SCENE_NAME}.validation_frames.log" 2>&1
fi
if ! artifact_is_current validation_frames "$VALIDATION_PLAN"; then
  echo "$SCENE_NAME: validation-frame plan is absent or stale" >&2
  exit 4
fi
VALIDATION_FRAMES="$(
  CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh "$VALIDATOR" plan \
    --path "$VALIDATION_PLAN" --print-csv
)"
echo "$SCENE_NAME: held-out fidelity frames = $VALIDATION_FRAMES"
if (( STOP_INDEX == 0 )); then exit 0; fi

run_or_validate 1 raw_mpr "$RAW_MPR" raw_mpr \
  radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
  --config "$CONFIG" \
  --checkpoint "$GEOMETRY_CHECKPOINT" \
  --output "$RAW_MPR" \
  --device cuda:0 \
  --observation-contract canonical-mpr-v1 \
  --feature-space radio \
  --exclude-frame-ids "$VALIDATION_FRAMES" \
  --expected-feature-scene "$SCENE_NAME" \
  --expected-feature-image-dir "$IMAGE_DIR" \
  --expected-geometry-checkpoint-sha256 "$GEOMETRY_CHECKPOINT_SHA256" \
  --expected-feature-output-bundle-sha256 "$FEATURE_BUNDLE_SHA256" \
  --save-responsibility-cache "$RESPONSIBILITY"
if (( STOP_INDEX == 1 )); then exit 0; fi

RESPONSIBILITY_SHA256="$(validate_artifact responsibility "$RESPONSIBILITY")"
[[ "$RESPONSIBILITY_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 3

run_or_validate 2 dino_mpr "$DINO_MPR" dino_mpr \
  radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
  --config "$CONFIG" \
  --checkpoint "$GEOMETRY_CHECKPOINT" \
  --output "$DINO_MPR" \
  --device cuda:0 \
  --observation-contract canonical-mpr-v1 \
  --feature-space dino_v3 \
  --radio-checkpoint "$RADIO_CHECKPOINT" \
  --capability-map-source project_raw \
  --exclude-frame-ids "$VALIDATION_FRAMES" \
  --expected-feature-scene "$SCENE_NAME" \
  --expected-feature-image-dir "$IMAGE_DIR" \
  --expected-geometry-checkpoint-sha256 "$GEOMETRY_CHECKPOINT_SHA256" \
  --expected-feature-output-bundle-sha256 "$FEATURE_BUNDLE_SHA256" \
  --responsibility-cache "$RESPONSIBILITY" \
  --expected-responsibility-cache-sha256 "$RESPONSIBILITY_SHA256"
if (( STOP_INDEX == 2 )); then exit 0; fi

run_or_validate 3 sam3_mpr "$SAM3_MPR" sam3_mpr \
  radio_gs/scripts/build_gaussian_multiview_teacher_cache.py \
  --config "$CONFIG" \
  --checkpoint "$GEOMETRY_CHECKPOINT" \
  --output "$SAM3_MPR" \
  --device cuda:0 \
  --observation-contract canonical-mpr-v1 \
  --feature-space sam3 \
  --radio-checkpoint "$RADIO_CHECKPOINT" \
  --capability-map-source project_raw \
  --exclude-frame-ids "$VALIDATION_FRAMES" \
  --expected-feature-scene "$SCENE_NAME" \
  --expected-feature-image-dir "$IMAGE_DIR" \
  --expected-geometry-checkpoint-sha256 "$GEOMETRY_CHECKPOINT_SHA256" \
  --expected-feature-output-bundle-sha256 "$FEATURE_BUNDLE_SHA256" \
  --responsibility-cache "$RESPONSIBILITY" \
  --expected-responsibility-cache-sha256 "$RESPONSIBILITY_SHA256"
if (( STOP_INDEX == 3 )); then exit 0; fi

# Each run_or_validate call immediately above reopened the artifact, checked
# all of its dependencies, and atomically wrote a validation stamp.  Reuse the
# stamped digest here; the downstream trainers independently enforce the same
# SHA while loading, so re-reading multi-gigabyte caches a second time would
# add latency without weakening fail-closed behavior.
RAW_MPR_SHA256="$(validated_artifact_sha256 "$RAW_MPR")"
DINO_MPR_SHA256="$(validated_artifact_sha256 "$DINO_MPR")"
SAM3_MPR_SHA256="$(validated_artifact_sha256 "$SAM3_MPR")"

run_or_validate 4 field_v1 "$FIELD_V1" field_v1 \
  radio_gs/scripts/train_canonical_radio_field.py \
  --mpr-cache "$RAW_MPR" \
  --expected-mpr-cache-sha256 "$RAW_MPR_SHA256" \
  --observation-contract canonical-mpr-v1 \
  --radio-checkpoint "$RADIO_CHECKPOINT" \
  --expected-radio-checkpoint-sha256 "$RADIO_CHECKPOINT_SHA256" \
  --expected-feature-output-bundle-sha256 "$FEATURE_BUNDLE_SHA256" \
  --output "$FIELD_V1" \
  --device cuda:0 \
  --coefficient-dim 256 \
  --local-dim 128 \
  --primitive-fusion \
  --official-capability-loss \
  --dino-mpr-cache "$DINO_MPR" \
  --expected-dino-v3-mpr-cache-sha256 "$DINO_MPR_SHA256" \
  --sam3-mpr-cache "$SAM3_MPR" \
  --expected-sam3-mpr-cache-sha256 "$SAM3_MPR_SHA256" \
  --epochs 20 \
  --min-epochs 5 \
  --target-cosine 0.985 \
  --seed 0
if (( STOP_INDEX == 4 )); then exit 0; fi

run_or_validate 5 field_v2 "$FIELD_V2" field_v2 \
  radio_gs/scripts/finetune_canonical_radio_rendering.py \
  --config "$CONFIG" \
  --geometry-checkpoint "$GEOMETRY_CHECKPOINT" \
  --field-checkpoint "$FIELD_V1" \
  --mpr-cache "$RAW_MPR" \
  --output "$FIELD_V2" \
  --device cuda:0 \
  --steps 256 \
  --mpr-weight 0.10 \
  --max-mpr-drop 0.005 \
  --dino-render-weight 0.20 \
  --sam3-render-weight 0.20 \
  --radio-checkpoint "$RADIO_CHECKPOINT" \
  --capability-map-source project_raw \
  --capability-local-affinity-weight 0.25 \
  --capability-local-radius 1 \
  --capability-local-balance-quantile 0.0 \
  --validation-frame-ids "$VALIDATION_FRAMES" \
  --selection-policy capability_pareto \
  --max-capability-drop 0.002 \
  --seed 0
if (( STOP_INDEX == 5 )); then exit 0; fi

run_or_validate 6 capability "$CAPABILITY" capability \
  radio_gs/scripts/build_canonical_capability_views.py \
  --field-checkpoint "$FIELD_V2" \
  --mpr-cache "$RAW_MPR" \
  --radio-checkpoint "$RADIO_CHECKPOINT" \
  --output "$CAPABILITY" \
  --batch-size 2048 \
  --device cuda:0
if (( STOP_INDEX == 6 )); then exit 0; fi

# Graph construction is CPU-only, but remains a formally validated stage.
if artifact_is_current support_graph "$GRAPH"; then
  echo "$SCENE_NAME: support_graph is current"
elif [[ -e "$GRAPH" ]]; then
  echo "$SCENE_NAME: existing support graph failed validation: $GRAPH" >&2
  exit 3
elif (( STATUS_ONLY || 7 < START_INDEX )); then
  echo "$SCENE_NAME: missing support graph: $GRAPH" >&2
  exit 4
else
  CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/build_canonical_support_graph.py \
    --capability-cache "$CAPABILITY" \
    --output "$GRAPH" \
    --neighbors 16 \
    --capability-affinity-mode signed_hash \
    --affinity-dim 256 \
    --topology-mode symmetric_union \
    >"$LOG_DIR/${SCENE_NAME}.support_graph.log" 2>&1
  validate_artifact support_graph "$GRAPH" >/dev/null
fi

echo "$SCENE_NAME: canonical-mpr-v3 paper-8 reconstruction complete"
