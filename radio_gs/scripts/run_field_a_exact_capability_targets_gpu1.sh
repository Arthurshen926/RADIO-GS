#!/usr/bin/env bash

# Registered Field-A target construction for figurines.  This builds only the
# query-free DINOv3/SAM3 primitive teachers; it never starts field training.

set -euo pipefail

REPO_ROOT=/root/RADIO-GS
RUN_PYTHON="$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
BUILDER="$REPO_ROOT/radio_gs/scripts/build_gaussian_multiview_teacher_cache.py"
GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
CONFIG="$REPO_ROOT/radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"
GEOMETRY=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
GEOMETRY_SHA=6900e08d2380d1f1563a2caf01b5846a9ab8df049b2a3c6a3f710452fa96eff2
RADIO=/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
FEATURE_BUNDLE_SHA=2fc5c37e1790afba62ccb18175c1a853d624b5bd9f039a8701fbdb7d783b08e3
OUT_ROOT=/mnt/pool/sqy/results/RADIO-GS/output
RUN_ROOT="$OUT_ROOT/optimization_20260803/field_a"
DINO_OUT="$OUT_ROOT/canonical_mpr/figurines_dino_v3_adjoint_train16_mean_resultant_field_a.pt"
SAM_OUT="$OUT_ROOT/canonical_mpr/figurines_sam3_adjoint_train16_mean_resultant_field_a.pt"

mkdir -p "$RUN_ROOT/logs"

run_target() {
  local space="$1"
  local output="$2"
  if [[ -e "$output" || -e "$output.json" ]]; then
    echo "refusing to overwrite existing Field-A target: $output" >&2
    return 2
  fi
  GPU=1 \
  GPU_TELEMETRY_LOG="$RUN_ROOT/logs/${space}_target_gpu1.telemetry.csv" \
  GPU_OWNER_AUDIT_LOG="$RUN_ROOT/logs/${space}_target_gpu1.owner_audit.csv" \
  GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
  GPU_POLL_SECONDS=20 \
  GPU_START_MAX_TEMP_C=78 \
  GPU_SOFT_PAUSE_TEMP_C=81 \
  GPU_SOFT_RESUME_TEMP_C=76 \
  GPU_MAX_TEMP_C=84 \
  GPU_MAX_POWER_LIMIT_W=300.5 \
  CUDA_VISIBLE_DEVICES=1 \
    bash "$GUARD" -- \
      bash "$RUN_PYTHON" "$BUILDER" \
        --config "$CONFIG" \
        --checkpoint "$GEOMETRY" \
        --expected-geometry-checkpoint-sha256 "$GEOMETRY_SHA" \
        --expected-feature-output-bundle-sha256 "$FEATURE_BUNDLE_SHA" \
        --radio-checkpoint "$RADIO" \
        --output "$output" \
        --device cuda:0 \
        --observation-contract legacy \
        --max-views 16 \
        --exclude-frame-ids 41,105,152,195 \
        --render-batch-size 4 \
        --point-chunk-size 4096 \
        --max-estimated-cpu-memory-fraction 0.85 \
        --depth-tolerance 0.08 \
        --relative-depth-tolerance 0.02 \
        --alpha-threshold 0.02 \
        --normalize-each-view \
        --no-robust-mpr \
        --feature-space "$space" \
        --capability-map-source project_raw \
        --capability-storage dense \
        --projection-batch-size 2 \
        --aggregation-mode raster_adjoint \
        --registration-weight-mode alpha_depth \
        --adjoint-channel-chunk-size 32 \
        --raster-channel-chunk-size 32 \
        --raster-view-fusion contribution_mean \
        --raster-reliability-mode mean_resultant
}

run_target dino_v3 "$DINO_OUT"
run_target sam3 "$SAM_OUT"
