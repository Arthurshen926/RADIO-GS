#!/usr/bin/env bash

# Evaluate the SPIn-NeRF full-reference-mask diagnostic with one exact
# foreground/background raster adjoint.  A single evaluator pass persists the
# unary, propagated, and connected stage metrics, so it covers ladder B-D
# without re-opening the prompt or changing the frozen target protocol.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-1}"
SCENE_NAMES="${SCENE_NAMES:-room horns truck}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/optimization_20260803/spin9_exact_adjoint_ladder}"
ASSET_ROOT="${ASSET_ROOT:-$REPO_ROOT/output/optimization_20260802/spin9_exact_local9}"
SOURCE_QUEUE="$REPO_ROOT/output/unified_query/spin9_gaussfm_queue_20260712"
REFERENCE_FIELD_ROOT="$REPO_ROOT/output/evaluation_closeout_20260716/canonical_mpr_v3_spin9"
MANIFEST="$REPO_ROOT/output/unified_query/manifests/spin_nerf_full_reference_mask_9scene_diagnostic_v1.json"
EVAL_QUEUE_ROOT="$RUN_ROOT/eval_queue"
GRAPH_POLICY="${GRAPH_POLICY:-instance_mix}"
SELECTION_MODE="${SELECTION_MODE:-all_components}"
READOUT_STAGE="${READOUT_STAGE:-propagated}"
OBSERVATION_FUSION="${OBSERVATION_FUSION:-direct_raster_adjoint}"
PROMPT_CYCLE_DIAGNOSTIC="${PROMPT_CYCLE_DIAGNOSTIC:-0}"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"

[[ "$GPU" =~ ^[01]$ ]] || { echo "GPU must be physical index 0 or 1" >&2; exit 2; }
case "$GRAPH_POLICY" in
  legacy|typed|geometry|appearance|boundary|category_mix|instance_mix) ;;
  *) echo "unsupported GRAPH_POLICY: $GRAPH_POLICY" >&2; exit 2 ;;
esac
case "$SELECTION_MODE" in
  seeded_component|all_components) ;;
  *) echo "unsupported SELECTION_MODE: $SELECTION_MODE" >&2; exit 2 ;;
esac
case "$READOUT_STAGE" in
  unary_prior|propagated|connected) ;;
  *) echo "unsupported READOUT_STAGE: $READOUT_STAGE" >&2; exit 2 ;;
esac
case "$OBSERVATION_FUSION" in
  direct_raster_adjoint|raster_adjoint_bernoulli_poe|dual_registration_bernoulli_poe) ;;
  *) echo "unsupported OBSERVATION_FUSION: $OBSERVATION_FUSION" >&2; exit 2 ;;
esac
case "$PROMPT_CYCLE_DIAGNOSTIC" in
  0|1) ;;
  *) echo "PROMPT_CYCLE_DIAGNOSTIC must be 0 or 1" >&2; exit 2 ;;
esac

prompt_cycle_args=()
if [[ "$PROMPT_CYCLE_DIAGNOSTIC" == "1" ]]; then
  prompt_cycle_args+=(--export-registered-prompt-cycle-diagnostic)
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/evaluations" "$EVAL_QUEUE_ROOT/scenes"

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

link_required() {
  local source="$1" destination="$2"
  [[ -s "$source" ]] || { echo "missing required asset: $source" >&2; exit 3; }
  if [[ -e "$destination" && ! -L "$destination" ]]; then
    echo "refusing to replace non-symlink evaluation asset: $destination" >&2
    exit 3
  fi
  mkdir -p "$(dirname "$destination")"
  ln -sfn "$(readlink -f "$source")" "$destination"
}

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

for scene in $SCENE_NAMES; do
  source_scene="$SOURCE_QUEUE/scenes/$scene"
  eval_scene="$EVAL_QUEUE_ROOT/scenes/$scene"
  case "$scene" in
    room|horns|truck)
      config="$ASSET_ROOT/carriers/$scene/config.yaml"
      checkpoint="$ASSET_ROOT/carriers/$scene/geometry_carrier.pth"
      field_dir="$ASSET_ROOT/fields/$scene"
      ;;
    *)
      config="$source_scene/gaussfm_main_track.yaml"
      checkpoint="$source_scene/feature_field/checkpoints/best.pth"
      field_dir="$REFERENCE_FIELD_ROOT/$scene"
      ;;
  esac
  capability="$field_dir/official_dino_sam3_views.pt"
  graph="$field_dir/shared_support_graph_k16.pt"
  field="$field_dir/canonical_d256_l128_capability_first.pth"
  camera_map="$source_scene/rgb_to_colmap_camera_mapping.json"
  for asset in "$config" "$checkpoint" "$capability" "$graph" "$field" "$camera_map"; do
    [[ -s "$asset" ]] || { echo "$scene missing required asset: $asset" >&2; exit 3; }
  done

  # The evaluator intentionally resolves the immutable queue layout.  This
  # diagnostic queue only redirects the three missing geometry checkpoints;
  # protocol manifest, camera map, config contents, and target assets stay the
  # frozen originals.
  link_required "$config" "$eval_scene/gaussfm_main_track.yaml"
  link_required "$camera_map" "$eval_scene/rgb_to_colmap_camera_mapping.json"
  link_required "$checkpoint" "$eval_scene/feature_field/checkpoints/best.pth"

  output_dir="$RUN_ROOT/evaluations/$scene"
  report="$output_dir/${scene}_evaluation.json"
  if [[ -s "$report" ]]; then
    echo "$scene: existing exact-adjoint report retained"
    continue
  fi
  mkdir -p "$output_dir"
  field_sha="$(sha256_file "$field")"
  echo "[$(date --iso-8601=seconds)] $scene: $OBSERVATION_FUSION + $GRAPH_POLICY"
  run_guarded "${scene}.${OBSERVATION_FUSION}" \
    radio_gs/scripts/eval_nvos_gaussian_first.py \
    --manifest "$MANIFEST" \
    --queue-root "$EVAL_QUEUE_ROOT" \
    --scene-id "$scene" \
    --output-dir "$output_dir" \
    --device cuda:0 \
    --region-space sam3 \
    --support-mode canonical_support \
    --prototype-count 4 \
    --canonical-capability-cache "$capability" \
    --canonical-support-graph "$graph" \
    --canonical-field-sha256 "$field_sha" \
    --prompt-registration-mode raster_adjoint \
    --prompt-registration-scale 1.0 \
    --alpha-threshold 0.0 \
    --registered-seed-unary-weight 0.0 \
    --registered-observation-fusion "$OBSERVATION_FUSION" \
    --registered-seed-construction joint_signed \
    --registered-forward-unary none \
    --registered-selection-mode "$SELECTION_MODE" \
    --registered-readout-stage "$READOUT_STAGE" \
    "${prompt_cycle_args[@]}" \
    --graph-policy "$GRAPH_POLICY" \
    --component-graph-policy same \
    --feature-calibration none \
    --score-calibration none \
    --solver-type confidence_random_walker \
    --laplacian-weight 1.0 \
    --solver-iterations 12 \
    --solver-residual 0.30 \
    --solver-support-threshold 0.50
done
