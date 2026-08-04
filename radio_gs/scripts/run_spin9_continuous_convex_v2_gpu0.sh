#!/usr/bin/env bash

# Registered difficult-scene SPIn component replacement.  The frozen scoring
# protocol is unchanged; only release reachability is replaced by the single
# continuous convex v2 configuration sealed on 2026-08-04.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SCENE_NAMES="${SCENE_NAMES:-lego truck room}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/optimization_20260804/spin9_continuous_convex_v2}"
REFERENCE_CALIBRATION_ONLY="${REFERENCE_CALIBRATION_ONLY:-0}"
ASSET_ROOT="$REPO_ROOT/output/optimization_20260802/spin9_exact_local9"
REFERENCE_FIELD_ROOT="$REPO_ROOT/output/evaluation_closeout_20260716/canonical_mpr_v3_spin9"
EVAL_QUEUE_ROOT="$REPO_ROOT/output/optimization_20260803/spin9_exact_adjoint_ladder/eval_queue"
QUERY_CACHE_ROOT="$REPO_ROOT/output/optimization_20260803/spin9_query_conditioned_diffusion_v1"
MANIFEST="$REPO_ROOT/output/unified_query/manifests/spin_nerf_full_reference_mask_9scene_diagnostic_v1.json"
REGISTRATION="$REPO_ROOT/paper/artifacts/evidence_to_support_v2_spin_component_registration_20260804.json"
EXPECTED_REGISTRATION_SHA256="30cd48eb7abe1ba3f003e9a8aa5425ebc1fbde873c4374bebd5b706df40775a0"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"

if [[ "$REFERENCE_CALIBRATION_ONLY" == "1" ]]; then
  [[ "$GPU" =~ ^[01]$ ]] || { echo "calibration receipt GPU must be 0 or 1" >&2; exit 2; }
else
  [[ "$GPU" == "0" ]] || { echo "registered SPIn v2 run requires physical GPU0" >&2; exit 2; }
fi
actual_registration_sha="$(sha256sum "$REGISTRATION" | awk '{print $1}')"
[[ "$actual_registration_sha" == "$EXPECTED_REGISTRATION_SHA256" ]] || {
  echo "SPIn v2 registration changed: $actual_registration_sha" >&2
  exit 3
}
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/reference_calibrated"

run_guarded() {
  local stage="$1"
  shift
  env CUDA_VISIBLE_DEVICES="$GPU" \
    GPU="$GPU" GPU_MAX_POWER_LIMIT_W=300.5 GPU_POLL_SECONDS=180 \
    GPU_START_MAX_TEMP_C=82 GPU_MAX_TEMP_C=87 \
    GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=2 \
    GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS=2 \
    GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
    GPU_TELEMETRY_LOG="$RUN_ROOT/logs/gpu${GPU}_telemetry.csv" \
    GPU_OWNER_AUDIT_LOG="$RUN_ROOT/logs/gpu${GPU}_owner.csv" \
    bash "$THERMAL_GUARD" -- env CUDA_VISIBLE_DEVICES="$GPU" \
      bash radio_gs/scripts/run_repo_python.sh "$@" \
      >"$RUN_ROOT/logs/${stage}.log" 2>&1
}

for scene in $SCENE_NAMES; do
  case "$scene" in
    room|horns|truck) field_dir="$ASSET_ROOT/fields/$scene" ;;
    *) field_dir="$REFERENCE_FIELD_ROOT/$scene" ;;
  esac
  capability="$field_dir/official_dino_sam3_views.pt"
  graph="$field_dir/shared_support_graph_k16.pt"
  knn="$QUERY_CACHE_ROOT/knn/${scene}_euclidean_k200_self.pt"
  relation="$QUERY_CACHE_ROOT/features/${scene}_dino_signed_hash256.pt"
  for asset in "$capability" "$graph" "$knn" "$relation"; do
    [[ -s "$asset" ]] || { echo "$scene missing asset: $asset" >&2; exit 3; }
  done
  field_sha="$(
    bash radio_gs/scripts/run_repo_python.sh - "$capability.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["field_checkpoint_sha256"])
PY
  )"
  output="$RUN_ROOT/reference_calibrated/$scene"
  if [[ "$REFERENCE_CALIBRATION_ONLY" == "1" ]]; then
    report="$output/query_compatibility_reference_threshold.json"
    calibration_only_args=(
      --query-diffusion-reference-threshold-source query_compatibility
      --query-diffusion-reference-calibration-only
    )
  else
    report="$output/${scene}_evaluation.json"
    calibration_only_args=()
  fi
  if [[ -s "$report" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$scene: retained completed continuous-v2 report"
    continue
  fi
  mkdir -p "$output"
  echo "[$(date --iso-8601=seconds)] $scene: continuous convex v2 on GPU$GPU"
  run_guarded "${scene}.continuous_v2" \
    radio_gs/scripts/eval_nvos_gaussian_first.py \
    --manifest "$MANIFEST" --queue-root "$EVAL_QUEUE_ROOT" \
    --scene-id "$scene" --output-dir "$output" --device cuda:0 \
    --region-space sam3 --support-mode canonical_support --prototype-count 4 \
    --canonical-capability-cache "$capability" \
    --canonical-support-graph "$graph" --canonical-field-sha256 "$field_sha" \
    --prompt-registration-mode raster_adjoint --prompt-registration-scale 1.0 \
    --alpha-threshold 0.0 --registered-seed-unary-weight 0.0 \
    --registered-observation-fusion direct_raster_adjoint \
    --registered-seed-construction joint_signed --registered-forward-unary none \
    --registered-selection-mode all_components --registered-readout-stage propagated \
    --graph-policy instance_mix --component-graph-policy same \
    --feature-calibration none --score-calibration none \
    --solver-type confidence_random_walker --laplacian-weight 1.0 \
    --solver-iterations 12 --solver-residual 0.30 --solver-support-threshold 0.50 \
    --query-conditioned-diffusion-kernel continuous_convex_v2 \
    --query-diffusion-knn-cache "$knn" \
    --query-diffusion-feature-cache "$relation" \
    --query-diffusion-reference-calibration --query-diffusion-logistic-c 0.01 \
    --query-diffusion-feature-bandwidth 1.0 \
    --query-diffusion-regularizer-bandwidth 1.0 \
    --query-diffusion-iterations 100 --query-diffusion-edge-binarize-threshold 1e-5 \
    --query-diffusion-max-positive-fraction 0.1 \
    --query-diffusion-distance-chunk-size 64 \
    "${calibration_only_args[@]}"
done
