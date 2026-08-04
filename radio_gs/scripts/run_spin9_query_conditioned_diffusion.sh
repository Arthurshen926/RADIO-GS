#!/usr/bin/env bash

# Resumable SPIn query-conditioned support queue.  Each scene completes its
# reference-only calibration before this evaluator opens any target mask.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-1}"
SCENE_NAMES="${SCENE_NAMES:-leaves lego orchids}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/optimization_20260803/spin9_query_conditioned_diffusion_v1}"
ASSET_ROOT="${ASSET_ROOT:-$REPO_ROOT/output/optimization_20260802/spin9_exact_local9}"
REFERENCE_FIELD_ROOT="$REPO_ROOT/output/evaluation_closeout_20260716/canonical_mpr_v3_spin9"
EVAL_QUEUE_ROOT="$REPO_ROOT/output/optimization_20260803/spin9_exact_adjoint_ladder/eval_queue"
MANIFEST="$REPO_ROOT/output/unified_query/manifests/spin_nerf_full_reference_mask_9scene_diagnostic_v1.json"
REGISTRATION="$REPO_ROOT/paper/artifacts/evidence_to_support_v1_experiment_registration_20260803.json"
EXPECTED_REGISTRATION_SHA256="7c539fb523c7152446bdc5f28325986a9162baa6c85a5608a66552023aa869c4"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"

[[ "$GPU" =~ ^[01]$ ]] || { echo "GPU must be physical index 0 or 1" >&2; exit 2; }
actual_registration_sha="$(sha256sum "$REGISTRATION" | awk '{print $1}')"
[[ "$actual_registration_sha" == "$EXPECTED_REGISTRATION_SHA256" ]] || {
  echo "experiment registration changed: $actual_registration_sha" >&2
  exit 3
}
mkdir -p "$RUN_ROOT/knn" "$RUN_ROOT/features" "$RUN_ROOT/logs" "$RUN_ROOT/reference_calibrated"

run_guarded() {
  local stage="$1"
  shift
  env CUDA_VISIBLE_DEVICES="$GPU" \
    GPU="$GPU" GPU_MAX_POWER_LIMIT_W=300.5 GPU_POLL_SECONDS=20 \
    GPU_START_MAX_TEMP_C=78 GPU_SOFT_PAUSE_TEMP_C=81 \
    GPU_SOFT_RESUME_TEMP_C=76 GPU_MAX_TEMP_C=84 \
    GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=3 \
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
  field="$field_dir/canonical_d256_l128_capability_first.pth"
  knn="$RUN_ROOT/knn/${scene}_euclidean_k200_self.pt"
  relation="$RUN_ROOT/features/${scene}_dino_signed_hash256.pt"
  for asset in "$capability" "$graph" "$field"; do
    [[ -s "$asset" ]] || { echo "$scene missing asset: $asset" >&2; exit 3; }
  done
  field_sha="$(
    bash radio_gs/scripts/run_repo_python.sh - "$capability.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["field_checkpoint_sha256"])
PY
  )"
  if [[ ! -s "$knn" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_query_diffusion_knn_cache.py \
      --support-graph "$graph" --output "$knn" --num-neighbors 200 \
      --experiment-registration "$REGISTRATION" \
      >"$RUN_ROOT/logs/${scene}.knn.log" 2>&1
  fi
  if [[ ! -s "$relation" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_query_diffusion_feature_cache.py \
      --capability-cache "$capability" --support-graph "$graph" \
      --output "$relation" --hash-dimension 256 --hash-batch-size 1024 \
      --experiment-registration "$REGISTRATION" \
      >"$RUN_ROOT/logs/${scene}.relation.log" 2>&1
  fi
  output="$RUN_ROOT/reference_calibrated/$scene"
  report="$output/${scene}_evaluation.json"
  if [[ -s "$report" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$scene: retained completed reference-calibrated report"
    continue
  fi
  mkdir -p "$output"
  echo "[$(date --iso-8601=seconds)] $scene: reference-only grid on GPU$GPU"
  run_guarded "${scene}.reference_calibrated" \
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
    --query-conditioned-diffusion-kernel ludvig_release_compat \
    --query-diffusion-knn-cache "$knn" \
    --query-diffusion-feature-cache "$relation" \
    --query-diffusion-reference-calibration --query-diffusion-logistic-c 0.01 \
    --query-diffusion-iterations 100 --query-diffusion-edge-binarize-threshold 1e-5 \
    --query-diffusion-max-positive-fraction 0.1 \
    --query-diffusion-distance-chunk-size 64
done
