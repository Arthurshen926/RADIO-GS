#!/usr/bin/env bash

# Materialize and score the frozen SPIn-NeRF local9 full-reference-mask
# diagnostic.  Every GPU stage is isolated behind a physical-GPU guard.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-1}"
SCENE_NAMES="${SCENE_NAMES:-orchids leaves fern room horns fortress pinecone truck lego}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/optimization_20260802/spin9_exact_local9}"
MANIFEST="$REPO_ROOT/output/unified_query/manifests/spin_nerf_full_reference_mask_9scene_diagnostic_v1.json"
QUEUE_ROOT="$REPO_ROOT/output/unified_query/spin9_gaussfm_queue_20260712/scenes"
FEATURE_ROOT="$RUN_ROOT/rendered_features"
PREDICTION_ROOT="$RUN_ROOT/predictions"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
read -r -a SCENES <<<"$SCENE_NAMES"

[[ "$GPU" =~ ^[01]$ ]] || { echo "GPU must be physical index 0 or 1" >&2; exit 2; }
mkdir -p "$RUN_ROOT/logs" "$FEATURE_ROOT" "$PREDICTION_ROOT"

run_guarded() {
  local stage="$1"
  shift
  env CUDA_VISIBLE_DEVICES="$GPU" \
    GPU="$GPU" \
    GPU_MAX_POWER_LIMIT_W=300.5 \
    GPU_POLL_SECONDS=20 \
    GPU_START_MAX_TEMP_C=78 \
    GPU_SOFT_PAUSE_TEMP_C=81 \
    GPU_SOFT_RESUME_TEMP_C=76 \
    GPU_MAX_TEMP_C=84 \
    GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=3 \
    GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
    GPU_TELEMETRY_LOG="$RUN_ROOT/logs/gpu${GPU}_telemetry.csv" \
    GPU_OWNER_AUDIT_LOG="$RUN_ROOT/logs/gpu${GPU}_owner.csv" \
    bash "$THERMAL_GUARD" -- \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      bash radio_gs/scripts/run_repo_python.sh "$@" \
      >"$RUN_ROOT/logs/${stage}.log" 2>&1
}

for scene in "${SCENES[@]}"; do
  case "$scene" in
    room|horns|truck)
      config="$RUN_ROOT/carriers/$scene/config.yaml"
      geometry="$RUN_ROOT/carriers/$scene/geometry_carrier.pth"
      field="$RUN_ROOT/fields/$scene/canonical_d256_l128_capability_first.pth"
      ;;
    *)
      config="$QUEUE_ROOT/$scene/gaussfm_main_track.yaml"
      geometry="$QUEUE_ROOT/$scene/feature_field/checkpoints/best.pth"
      field="$REPO_ROOT/output/evaluation_closeout_20260716/canonical_mpr_v3_spin9/$scene/canonical_d256_l128_capability_first.pth"
      ;;
  esac
  camera_map="$QUEUE_ROOT/$scene/rgb_to_colmap_camera_mapping.json"
  if [[ ! -s "$config" || ! -s "$geometry" || ! -s "$field" ]]; then
    if [[ "${AVAILABLE_ONLY:-0}" == "1" ]]; then
      echo "$scene: field is not ready; skipped by AVAILABLE_ONLY=1"
      continue
    fi
    echo "$scene canonical render assets are missing" >&2
    exit 3
  fi
  if [[ ! -s "$FEATURE_ROOT/$scene/render_manifest.json" ]]; then
    run_guarded "render_${scene}" \
      radio_gs/scripts/render_promptable_nvs_features.py \
      --manifest "$MANIFEST" \
      --scene-id "$scene" \
      --camera-map "$camera_map" \
      --config "$config" \
      --checkpoint "$geometry" \
      --canonical-field-checkpoint "$field" \
      --output-dir "$FEATURE_ROOT" \
      --device cuda \
      --overwrite
  fi
done

for scene in "${SCENES[@]}"; do
  [[ -s "$FEATURE_ROOT/$scene/render_manifest.json" ]] || {
    if [[ "${AVAILABLE_ONLY:-0}" == "1" ]]; then exit 0; fi
    echo "$scene render manifest is missing" >&2
    exit 3
  }
done

if [[ "${RENDER_ONLY:-0}" == "1" ]]; then
  exit 0
fi

# Prediction/evaluation consume the complete frozen cohort even when rendering
# was split across GPUs through SCENE_NAMES.
for scene in orchids leaves fern room horns fortress pinecone truck lego; do
  [[ -s "$FEATURE_ROOT/$scene/render_manifest.json" ]] || {
    echo "$scene render manifest is missing; cannot score the full local9 cohort" >&2
    exit 3
  }
done

if [[ ! -s "$PREDICTION_ROOT/prediction_manifest.json" ]]; then
  run_guarded predict_raw_cosine_margin \
    radio_gs/scripts/predict_promptable_nvs_feature_readout.py \
    --manifest "$MANIFEST" \
    --feature-root "$FEATURE_ROOT" \
    --feature-pattern '{scene_id}/{camera_name}.pt' \
    --feature-layout chw \
    --radio-sam3-adaptor-checkpoint /root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar \
    --adaptor-device cuda:0 \
    --method-name 'RADIO-GS canonical-mpr-v3 SPIn local9 full-mask raw cosine margin' \
    --output-dir "$PREDICTION_ROOT" \
    --overwrite
fi

if [[ ! -s "$RUN_ROOT/evaluation.json" ]]; then
  CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_promptable_nvs_segmentation.py \
    --manifest "$MANIFEST" \
    --prediction-manifest "$PREDICTION_ROOT/prediction_manifest.json" \
    --output "$RUN_ROOT/evaluation.json" \
    >"$RUN_ROOT/logs/evaluation.log" 2>&1
fi
