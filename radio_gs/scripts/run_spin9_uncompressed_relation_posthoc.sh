#!/usr/bin/env bash

# Post-hoc matched relation-capacity diagnostic.  This is intentionally not a
# preregistered promotion queue: lego/orchids were selected after full9.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
SCENE_NAMES="${SCENE_NAMES:-lego orchids}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/optimization_20260803/spin9_relation_capacity_posthoc_v1}"
FORMAL_ROOT="$REPO_ROOT/output/optimization_20260803/spin9_query_conditioned_diffusion_v1"
FIELD_ROOT="$REPO_ROOT/output/evaluation_closeout_20260716/canonical_mpr_v3_spin9"
EVAL_QUEUE_ROOT="$REPO_ROOT/output/optimization_20260803/spin9_exact_adjoint_ladder/eval_queue"
MANIFEST="$REPO_ROOT/output/unified_query/manifests/spin_nerf_full_reference_mask_9scene_diagnostic_v1.json"
DECLARATION="$REPO_ROOT/paper/artifacts/spin_relation_capacity_posthoc_diagnostic_declaration_20260803.json"
EXPECTED_DECLARATION_SHA256="5944a9f049786d28bc526c37c3a9ce0183c284ce75a9145ce633c865344e5af1"
EXPECTED_EVALUATOR_SHA256="3a9f781687bf61916e3ff139dfba769f415b811ec88de19d02d552ba4a647477"
EXPECTED_QUERY_KERNEL_SHA256="e46c624bd204d776afc5d6455df1f122a416e0cb24db83ca0d6860eb98c860f6"
EXPECTED_CACHE_INTERFACE_SHA256="01fe1a52fd5e42e76cb2cbf0d0f8f32a7a58c2fcd25c26868a09dc7e0f131cb3"
EXPECTED_CACHE_BUILDER_SHA256="b02704fca65816d8ae5189fee3448dcac7ff64c60ecc2736e38b5dae8986ebc6"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"

[[ "$GPU" == "0" ]] || { echo "post-hoc SPIn diagnostic is assigned to physical GPU0" >&2; exit 2; }
actual_declaration_sha="$(sha256sum "$DECLARATION" | awk '{print $1}')"
[[ "$actual_declaration_sha" == "$EXPECTED_DECLARATION_SHA256" ]] || {
  echo "post-hoc diagnostic declaration changed: $actual_declaration_sha" >&2
  exit 3
}
verify_implementation() {
  local binding expected source actual
  for binding in \
    "$EXPECTED_EVALUATOR_SHA256:radio_gs/scripts/eval_nvos_gaussian_first.py" \
    "$EXPECTED_QUERY_KERNEL_SHA256:radio_gs/querying/query_conditioned_diffusion.py" \
    "$EXPECTED_CACHE_INTERFACE_SHA256:radio_gs/interfaces/query_diffusion_cache.py" \
    "$EXPECTED_CACHE_BUILDER_SHA256:radio_gs/scripts/build_query_diffusion_uncompressed_feature_cache.py"
  do
    expected="${binding%%:*}"
    source="${binding#*:}"
    actual="$(sha256sum "$source" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || {
      echo "diagnostic implementation changed: $source $actual" >&2
      exit 3
    }
  done
}
verify_implementation
mkdir -p "$RUN_ROOT/features" "$RUN_ROOT/logs" "$RUN_ROOT/reference_calibrated"

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
  verify_implementation
  case "$scene" in
    lego|orchids) ;;
    *) echo "undeclared post-hoc diagnostic scene: $scene" >&2; exit 3 ;;
  esac
  field_dir="$FIELD_ROOT/$scene"
  capability="$field_dir/official_dino_sam3_views.pt"
  graph="$field_dir/shared_support_graph_k16.pt"
  knn="$FORMAL_ROOT/knn/${scene}_euclidean_k200_self.pt"
  relation="$RUN_ROOT/features/${scene}_official_dino4096_uncompressed.pt"
  for asset in "$capability" "$graph" "$knn"; do
    [[ -s "$asset" ]] || { echo "$scene missing asset: $asset" >&2; exit 3; }
  done
  field_sha="$(
    bash radio_gs/scripts/run_repo_python.sh - "$capability.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["field_checkpoint_sha256"])
PY
  )"
  if [[ ! -s "$relation" || "${OVERWRITE_CACHE:-0}" == "1" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_query_diffusion_uncompressed_feature_cache.py \
      --capability-cache "$capability" --support-graph "$graph" \
      --output "$relation" --diagnostic-declaration "$DECLARATION" \
      >"$RUN_ROOT/logs/${scene}.uncompressed_relation.log" 2>&1
  fi
  output="$RUN_ROOT/reference_calibrated/$scene"
  report="$output/${scene}_evaluation.json"
  if [[ -s "$report" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$scene: retained completed post-hoc reference-calibrated report"
    continue
  fi
  mkdir -p "$output"
  echo "[$(date --iso-8601=seconds)] $scene: post-hoc uncompressed4096 reference-only grid on GPU$GPU"
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
