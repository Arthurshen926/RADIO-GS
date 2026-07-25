#!/usr/bin/env bash

# Run one real, fixed-protocol AGILE3D full-.sens ablation when a requested
# GPU becomes free.  This is deliberately a work queue, not an occupancy job:
# it waits only for required field artifacts and then executes the declared
# field-quality or interaction evaluation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
ABLATION="${ABLATION:?set ABLATION to baseline, background4, background8, reliability, unary_contrast1, cell_seed, seed_top1, equal_click, seed_conflict_exclusive, exact_capability_graph, exact_capability_manifold_covisibility, surface_relation, surface_manifold, or surface_manifold_covisibility}"
AFTER_RESULT="${AFTER_RESULT:-}"
FIELD_ROOT="${FIELD_ROOT:-output/agile3d_scannet40/full_sens_dev_v2/reconstruction_v1}"
RUN_ROOT="${RUN_ROOT:-output/agile3d_scannet40/full_sens_dev_v2}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet}"
SCENE_NAMES="${SCENE_NAMES:-scene0011_01 scene0046_00 scene0249_00}"
# These three names make a field-side reconstruction variant explicit without
# changing released clicks, the official point domain, or any evaluator
# setting.  The defaults preserve the frozen canonical-v2 route.  A named
# variant must materialize all three matching artifacts under the same field
# directory before this queue will ever open AGILE labels.
FIELD_CHECKPOINT_NAME="${FIELD_CHECKPOINT_NAME:-canonical_mpr_v2.pt}"
CAPABILITY_CACHE_NAME="${CAPABILITY_CACHE_NAME:-official_dino_sam3_views.pt}"
SUPPORT_GRAPH_NAME="${SUPPORT_GRAPH_NAME:-shared_support_graph_k16.pt}"
OBJECT_SHARD_INDEX="${OBJECT_SHARD_INDEX:-0}"
OBJECT_SHARD_COUNT="${OBJECT_SHARD_COUNT:-1}"
REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS="${REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS:-0}"
# Full-.sens fields use a meaningful cell-support gate: 0.01 is approximately
# the contribution of a unit-opacity Gaussian at three standard deviations,
# rather than a permissive long-tail reachability test.
READOUT_SUPPORT_THRESHOLD="${READOUT_SUPPORT_THRESHOLD:-0.01}"

read -r -a SCENES <<< "$SCENE_NAMES"
if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "SCENE_NAMES must be non-empty" >&2
  exit 2
fi
for artifact_name in "$FIELD_CHECKPOINT_NAME" "$CAPABILITY_CACHE_NAME" "$SUPPORT_GRAPH_NAME"; do
  if [[ -z "$artifact_name" || "$artifact_name" == */* || "$artifact_name" == "." || "$artifact_name" == ".." ]]; then
    echo "field/capability/graph artifact names must be non-empty basenames" >&2
    exit 2
  fi
done
if ! bash radio_gs/scripts/run_repo_python.sh - \
  "$OBJECT_SHARD_INDEX" "$OBJECT_SHARD_COUNT" <<'PY'
import sys

index, count = int(sys.argv[1]), int(sys.argv[2])
if count <= 0 or index < 0 or index >= count:
    raise SystemExit("OBJECT_SHARD_INDEX/OBJECT_SHARD_COUNT are invalid")
PY
then
  exit 2
fi

wait_for_gpu() {
  local available=0
  while (( available < 2 )); do
    local values used util
    values="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU")"
    used="${values%%,*}"; util="${values##*,}"
    used="${used// /}"; util="${util// /}"
    if (( used < 1200 && util < 10 )); then
      available=$((available + 1))
    else
      available=0
    fi
    if (( available < 2 )); then
      sleep 20
    fi
  done
}

wait_for_common_fields() {
  local scene
  for scene in "${SCENES[@]}"; do
    while test ! -s "$FIELD_ROOT/canonical_fields/$scene/$FIELD_CHECKPOINT_NAME" \
      || test ! -s "$FIELD_ROOT/canonical_fields/$scene/raw_radio_mpr.pt" \
      || test ! -s "$FIELD_ROOT/canonical_fields/$scene/$CAPABILITY_CACHE_NAME" \
      || test ! -s "$FIELD_ROOT/canonical_fields/$scene/$SUPPORT_GRAPH_NAME"; do
      sleep 30
    done
  done
}

wait_for_common_fields

CAPABILITY_PROVENANCE_ARGS=()
case "$REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS" in
  1|true|True|TRUE)
    CAPABILITY_PROVENANCE_ARGS=(--require-official-extracted-capability-teachers)
    ;;
  0|false|False|FALSE)
    ;;
  *)
    echo "REQUIRE_OFFICIAL_EXTRACTED_CAPABILITY_TEACHERS must be 0/1 or true/false" >&2
    exit 2
    ;;
esac

FIELD_VARIANT_ARGS=(
  --field-checkpoint-name "$FIELD_CHECKPOINT_NAME"
  --capability-cache-name "$CAPABILITY_CACHE_NAME"
  --support-graph-name "$SUPPORT_GRAPH_NAME"
)

# A dependent ablation may share a physical GPU with an earlier real run.  It
# never reserves the device: it simply stays dormant until that prior result
# exists, then applies the same genuine-free-GPU check immediately before its
# own evaluator execution.
if [[ -n "$AFTER_RESULT" ]]; then
  while test ! -s "$AFTER_RESULT"; do
    sleep 30
  done
fi

case "$ABLATION" in
  baseline)
    OUTPUT_DIR="$RUN_ROOT/eval_baseline"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    wait_for_gpu
    EXTRA_ARGS=(--click-seed-kernel native_gaussian --unary-edge-contrast 0)
    ;;
  background4)
    OUTPUT_DIR="$RUN_ROOT/eval_background4"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    wait_for_gpu
    # A smaller query-free scene-mode bank is retained because it is a
    # capacity ablation of the same generic scorer, not a target-aware
    # selection.  It was useful on an earlier disjoint dense pilot and must
    # now be compared under the identical full-observation source.
    EXTRA_ARGS=(
      --click-seed-kernel native_gaussian --unary-edge-contrast 0
      --background-centroids 4 --calibration-sample-size 8192 --centroid-iterations 4
    )
    ;;
  background8)
    OUTPUT_DIR="$RUN_ROOT/eval_background8"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    wait_for_gpu
    # Fixed scene-only modes are fitted without query/object labels.  This is
    # a named scoring ablation, not a test-set calibration.
    EXTRA_ARGS=(
      --click-seed-kernel native_gaussian --unary-edge-contrast 0
      --background-centroids 8 --calibration-sample-size 8192 --centroid-iterations 4
    )
    ;;
  reliability)
    OUTPUT_DIR="$RUN_ROOT/eval_reliability"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    wait_for_gpu
    for scene in "${SCENES[@]}"; do
      CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
        radio_gs/scripts/build_canonical_reliability_cache.py \
        --field-checkpoint "$FIELD_ROOT/canonical_fields/$scene/$FIELD_CHECKPOINT_NAME" \
        --mpr-cache "$FIELD_ROOT/canonical_fields/$scene/raw_radio_mpr.pt" \
        --output "$FIELD_ROOT/canonical_fields/$scene/canonical_reliability_v1.pt" \
        --device cuda:0 >>"$OUTPUT_DIR/run.log" 2>&1
    done
    EXTRA_ARGS=(--click-seed-kernel native_gaussian --unary-edge-contrast 0 --reliability-cache-name canonical_reliability_v1.pt)
    ;;
  unary_contrast1)
    OUTPUT_DIR="$RUN_ROOT/eval_unary_contrast1"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    wait_for_gpu
    EXTRA_ARGS=(--click-seed-kernel native_gaussian --unary-edge-contrast 1.0)
    ;;
  cell_seed)
    OUTPUT_DIR="$RUN_ROOT/eval_cell_seed"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    wait_for_gpu
    EXTRA_ARGS=(--click-seed-kernel evaluator_voxel_convolved --unary-edge-contrast 0)
    ;;
  seed_top1)
    OUTPUT_DIR="$RUN_ROOT/eval_seed_top1"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    wait_for_gpu
    # Continuous covariance weights still form per-click local prototypes;
    # this only localizes the hard Laplacian constraints to one best-supported
    # primitive per released click.
    EXTRA_ARGS=(--click-seed-kernel native_gaussian --unary-edge-contrast 0 --seed-topk 1)
    ;;
  equal_click)
    OUTPUT_DIR="$RUN_ROOT/eval_equal_click"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    wait_for_gpu
    # Treat each accumulated released correction as one interaction event,
    # independent of the local Gaussian footprint's total support mass.
    EXTRA_ARGS=(--click-seed-kernel native_gaussian --unary-edge-contrast 0 --world-point-prototype-weighting equal_click)
    ;;
  seed_conflict_exclusive)
    OUTPUT_DIR="$RUN_ROOT/eval_seed_conflict_exclusive"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    wait_for_gpu
    # When broad Gaussian supports overlap opposite accumulated world clicks,
    # retain only sign-dominant hard constraints and leave exact ties to the
    # same DINO/SAM unary plus seeded Laplacian.  No scene/object/GT data is
    # used to choose this policy.
    EXTRA_ARGS=(
      --click-seed-kernel native_gaussian --unary-edge-contrast 0
      --hard-seed-conflict-policy exclusive_relative
    )
    ;;
  exact_capability_graph)
    OUTPUT_DIR="$RUN_ROOT/eval_exact_capability_graph"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    # The default graph hashes DINO/SAM to 256 dimensions before local edge
    # scoring.  This named variant keeps the frozen official capability rows
    # at their native dimensions and computes only those edge cosines in GPU
    # chunks.  It still uses the same field, topology, released clicks, 5 cm
    # readout, and evaluator; the graph construction opens no labels/query.
    wait_for_gpu
    for scene in "${SCENES[@]}"; do
      EXACT_GRAPH="$FIELD_ROOT/canonical_fields/$scene/shared_support_graph_k16_exact_capability_v1.pt"
      if [[ ! -s "$EXACT_GRAPH" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
          radio_gs/scripts/build_canonical_support_graph.py \
          --capability-cache "$FIELD_ROOT/canonical_fields/$scene/official_dino_sam3_views.pt" \
          --output "$EXACT_GRAPH" \
          --neighbors 16 --topology-mode symmetric_union \
          --capability-affinity-mode exact_official_capability \
          --affinity-device cuda:0 --affinity-chunk-size 4096 \
          >>"$OUTPUT_DIR/run.log" 2>&1
      fi
      AFFINITY_AUDIT="$OUTPUT_DIR/${scene}.capability_affinity_audit.json"
      if [[ ! -s "$AFFINITY_AUDIT" ]]; then
        bash radio_gs/scripts/run_repo_python.sh \
          radio_gs/scripts/audit_canonical_support_graph_affinity.py \
          --hashed-graph "$FIELD_ROOT/canonical_fields/$scene/shared_support_graph_k16.pt" \
          --native-graph "$EXACT_GRAPH" \
          --output "$AFFINITY_AUDIT" \
          >>"$OUTPUT_DIR/run.log" 2>&1
      fi
    done
    EXTRA_ARGS=(
      --click-seed-kernel native_gaussian --unary-edge-contrast 0
      --support-graph-name shared_support_graph_k16_exact_capability_v1.pt
    )
    ;;
  exact_capability_manifold_covisibility)
    OUTPUT_DIR="$RUN_ROOT/eval_exact_capability_manifold_covisibility"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    # This is the explicitly named compositional candidate, not an
    # evaluator-specific fallback: it combines only frozen field relations
    # that were independently declared above. Native official DINO/SAM edge
    # affinities avoid signed-hash boundary blur; tangent continuity and MPR
    # co-visibility prevent Euclidean shortcuts between distinct surfaces;
    # four scene modes supply a fixed query-independent background bank.
    # It uses no object, label, click outcome, mask, or metric to build the
    # graph or the background reference.
    wait_for_gpu
    for scene in "${SCENES[@]}"; do
      EXACT_MANIFOLD_GRAPH="$FIELD_ROOT/canonical_fields/$scene/shared_support_graph_k16_exact_capability_surface_pca_tangent_covisibility_v1.pt"
      if [[ ! -s "$EXACT_MANIFOLD_GRAPH" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
          radio_gs/scripts/build_canonical_support_graph.py \
          --capability-cache "$FIELD_ROOT/canonical_fields/$scene/official_dino_sam3_views.pt" \
          --responsibility-cache "$FIELD_ROOT/canonical_fields/$scene/registration_responsibility.pt" \
          --output "$EXACT_MANIFOLD_GRAPH" \
          --neighbors 16 --topology-mode symmetric_union \
          --capability-affinity-mode exact_official_capability \
          --affinity-device cuda:0 --affinity-chunk-size 4096 \
          --surface-relation local_pca_tangent_v1 \
          --surface-normal-neighbors 24 --surface-normal-min-planarity 0.0 \
          --covisibility-weight 0.25 \
          >>"$OUTPUT_DIR/run.log" 2>&1
      fi
    done
    EXTRA_ARGS=(
      --click-seed-kernel native_gaussian --unary-edge-contrast 0
      --background-centroids 4 --calibration-sample-size 8192 --centroid-iterations 4
      --support-graph-name shared_support_graph_k16_exact_capability_surface_pca_tangent_covisibility_v1.pt
    )
    ;;
  surface_relation)
    OUTPUT_DIR="$RUN_ROOT/eval_surface_relation"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    # Build a second, query-independent graph from the same frozen field. Its
    # local-PCA normals are derived only from canonical primitive centres; an
    # uncertain corner/line relation becomes neutral rather than a learned or
    # target-aware boundary. This CPU stage deliberately runs before waiting
    # for a real GPU so it never reserves one while preparing the graph.
    for scene in "${SCENES[@]}"; do
      SURFACE_GRAPH="$FIELD_ROOT/canonical_fields/$scene/shared_support_graph_k16_surface_pca_v1.pt"
      if [[ ! -s "$SURFACE_GRAPH" ]]; then
        bash radio_gs/scripts/run_repo_python.sh \
          radio_gs/scripts/build_canonical_support_graph.py \
          --capability-cache "$FIELD_ROOT/canonical_fields/$scene/official_dino_sam3_views.pt" \
          --output "$SURFACE_GRAPH" \
          --neighbors 16 --topology-mode symmetric_union \
          --surface-relation local_pca_v1 \
          --surface-normal-neighbors 24 --surface-normal-min-planarity 0.0 \
          >>"$OUTPUT_DIR/run.log" 2>&1
      fi
    done
    wait_for_gpu
    EXTRA_ARGS=(
      --click-seed-kernel native_gaussian --unary-edge-contrast 0
      --support-graph-name shared_support_graph_k16_surface_pca_v1.pt
    )
    ;;
  surface_manifold)
    OUTPUT_DIR="$RUN_ROOT/eval_surface_manifold"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    # This supersets the normal-only relation with a point-to-local-tangent
    # continuity channel.  It uses only the frozen canonical primitive
    # geometry, so graph construction remains query/label free and can happen
    # before a GPU is actually available for the evaluator.
    for scene in "${SCENES[@]}"; do
      SURFACE_GRAPH="$FIELD_ROOT/canonical_fields/$scene/shared_support_graph_k16_surface_pca_tangent_v1.pt"
      if [[ ! -s "$SURFACE_GRAPH" ]]; then
        bash radio_gs/scripts/run_repo_python.sh \
          radio_gs/scripts/build_canonical_support_graph.py \
          --capability-cache "$FIELD_ROOT/canonical_fields/$scene/official_dino_sam3_views.pt" \
          --output "$SURFACE_GRAPH" \
          --neighbors 16 --topology-mode symmetric_union \
          --surface-relation local_pca_tangent_v1 \
          --surface-normal-neighbors 24 --surface-normal-min-planarity 0.0 \
          >>"$OUTPUT_DIR/run.log" 2>&1
      fi
    done
    wait_for_gpu
    EXTRA_ARGS=(
      --click-seed-kernel native_gaussian --unary-edge-contrast 0
      --support-graph-name shared_support_graph_k16_surface_pca_tangent_v1.pt
    )
    ;;
  surface_manifold_covisibility)
    OUTPUT_DIR="$RUN_ROOT/eval_surface_manifold_covisibility"
    OUTPUT="$OUTPUT_DIR/results.json"
    mkdir -p "$OUTPUT_DIR"
    # MPR already records a sparse top-1 responsibility per registered RGB-D
    # view.  Reuse that query-free audit to add a primitive co-visibility
    # relation, bound by digest to the same raw-MPR cache as the capability
    # bank.  It is an explicit graph variant, not an evaluator-side bridge.
    for scene in "${SCENES[@]}"; do
      SURFACE_GRAPH="$FIELD_ROOT/canonical_fields/$scene/shared_support_graph_k16_surface_pca_tangent_covisibility_v1.pt"
      if [[ ! -s "$SURFACE_GRAPH" ]]; then
        bash radio_gs/scripts/run_repo_python.sh \
          radio_gs/scripts/build_canonical_support_graph.py \
          --capability-cache "$FIELD_ROOT/canonical_fields/$scene/official_dino_sam3_views.pt" \
          --responsibility-cache "$FIELD_ROOT/canonical_fields/$scene/registration_responsibility.pt" \
          --output "$SURFACE_GRAPH" \
          --neighbors 16 --topology-mode symmetric_union \
          --surface-relation local_pca_tangent_v1 \
          --surface-normal-neighbors 24 --surface-normal-min-planarity 0.0 \
          --covisibility-weight 0.25 \
          >>"$OUTPUT_DIR/run.log" 2>&1
      fi
    done
    wait_for_gpu
    EXTRA_ARGS=(
      --click-seed-kernel native_gaussian --unary-edge-contrast 0
      --support-graph-name shared_support_graph_k16_surface_pca_tangent_covisibility_v1.pt
    )
    ;;
  *)
    echo "unsupported ABLATION: $ABLATION" >&2
    exit 2
    ;;
esac

# Persist a label-free quality record for each baseline field before any
# released AGILE object or label is opened.  A failed record is a field/source
# reconstruction failure (for example, one that needs a larger query-free
# coverage budget), never a low-coverage result stratum.  Other scorer
# ablations reuse the same field preflight inside their formal evaluator.
if [[ "$ABLATION" == "baseline" ]]; then
  SUPPORT_AUDIT="$OUTPUT_DIR/support_preflight.json"
  if [[ ! -s "$SUPPORT_AUDIT" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      -m radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field \
      --benchmark-root "$BENCHMARK_ROOT" \
      --field-root "$FIELD_ROOT" \
      --geometry-cache-root "$OUTPUT_DIR/geometry" \
      --output "$SUPPORT_AUDIT" \
      --scene-names "$SCENE_NAMES" \
      --device cuda:0 \
      --observation-contract scannet_full_observation_pilot \
      --require-support-gate --minimum-support-fraction 0.95 \
      --evaluation-voxel-size-m 0.05 \
      --readout-support-threshold "$READOUT_SUPPORT_THRESHOLD" \
      --support-only \
      --seed-candidate-k 64 \
      --world-point-prototype-mode per_click_local \
      --world-point-prototype-weighting support_mass \
      --solver-type confidence_random_walker \
      --laplacian-weight 1.0 --cg-iterations 64 --support-threshold 0.5 \
      --feature-calibration none --background-centroids 0 --score-calibration none \
      "${FIELD_VARIANT_ARGS[@]}" \
      "${CAPABILITY_PROVENANCE_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" >>"$OUTPUT_DIR/run.log" 2>&1
  fi
  bash radio_gs/scripts/run_repo_python.sh - "$SUPPORT_AUDIT" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("mode") != "label_free_field_support_preflight":
    raise SystemExit("AGILE support audit has an invalid mode")
if payload.get("protocol", {}).get("labels_opened") is not False:
    raise SystemExit("AGILE support audit is not label-free")
if not payload.get("support_gate_passed", False):
    failures = [
        (row.get("scene_id"), row.get("continuous_support_fraction"))
        for row in payload.get("scene_support", [])
        if not row.get("support_gate_passed", False)
    ]
    raise SystemExit(f"AGILE full-observation support gate failed: {failures}")
PY
  # The audit itself is a real GPU readout. Recheck physical availability
  # before starting the released interactive evaluator.
  wait_for_gpu
fi

CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field \
  --benchmark-root "$BENCHMARK_ROOT" \
  --field-root "$FIELD_ROOT" \
  --geometry-cache-root "$OUTPUT_DIR/geometry" \
  --output "$OUTPUT" \
  --scene-names "$SCENE_NAMES" \
  --object-shard-index "$OBJECT_SHARD_INDEX" \
  --object-shard-count "$OBJECT_SHARD_COUNT" \
  --device cuda:0 \
  --observation-contract scannet_full_observation_pilot \
  --require-support-gate --minimum-support-fraction 0.95 \
  --evaluation-voxel-size-m 0.05 \
  --readout-support-threshold "$READOUT_SUPPORT_THRESHOLD" \
  --seed-candidate-k 64 \
  --world-point-prototype-mode per_click_local \
  --world-point-prototype-weighting support_mass \
  --solver-type confidence_random_walker \
  --laplacian-weight 1.0 --cg-iterations 64 --support-threshold 0.5 \
  --feature-calibration none --background-centroids 0 --score-calibration none \
  --max-clicks 20 --click-workers 3 \
  "${FIELD_VARIANT_ARGS[@]}" \
  "${CAPABILITY_PROVENANCE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" >>"$OUTPUT_DIR/run.log" 2>&1
