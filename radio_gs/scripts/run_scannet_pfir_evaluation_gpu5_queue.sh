#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-5}"
BENCHMARK_DIR="${BENCHMARK_DIR:-output/scannet_pfir_small_v1/test_v1_final}"
RUN_ROOT="${RUN_ROOT:-$BENCHMARK_DIR/reconstruction_v1}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-/mnt/pool/sqy/3d_understanding/ScanNet-PFIR-Small/annotations}"
FIELD_TERMINAL="${FIELD_TERMINAL:-$RUN_ROOT/canonical_mpr_v3_fields.complete}"
# The default waits for the all-scene field audit.  An explicitly selected
# shard whose referenced scenes are already complete may set this to 0 to use
# an otherwise idle GPU; the per-query compiler still validates every field,
# capability bank, graph, and semantic sidecar before it reads a crop.
WAIT_FOR_FIELDS="${WAIT_FOR_FIELDS:-1}"
RADIO_REPO="${RADIO_REPO:-/root/RADIO}"
RADIO_CHECKPOINT="${RADIO_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar}"
# Query shards are deliberately disjoint and only affect execution order.  A
# shard always writes the same per-query immutable cache/predictions as the
# complete queue, while the designated aggregation invocation evaluates the
# two tracks once every manifest query is present.
QUERY_START_INDEX="${QUERY_START_INDEX:-0}"
QUERY_STOP_INDEX="${QUERY_STOP_INDEX:-}"
RUN_EVALUATOR="${RUN_EVALUATOR:-1}"
WRITE_TERMINAL="${WRITE_TERMINAL:-1}"

QUERY_ROOT="$RUN_ROOT/query_caches"
RANKING_ROOT="$RUN_ROOT/predictions/ranking"
SELECTION_ROOT="$RUN_ROOT/predictions/selection"
mkdir -p "$QUERY_ROOT" "$RANKING_ROOT" "$SELECTION_ROOT" "$RUN_ROOT/logs"

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
    if (( available < 2 )); then sleep 20; fi
  done
}

case "$WAIT_FOR_FIELDS" in
  1|true|True|TRUE)
    while [[ ! -s "$FIELD_TERMINAL" ]]; do sleep 30; done
    ;;
  0|false|False|FALSE)
    ;;
  *)
    echo "WAIT_FOR_FIELDS must be 0/1 or true/false, got: $WAIT_FOR_FIELDS" >&2
    exit 2
    ;;
esac

mapfile -t QUERIES < <(
  MANIFEST="$BENCHMARK_DIR/manifest.method.json" \
    QUERY_START_INDEX="$QUERY_START_INDEX" \
    QUERY_STOP_INDEX="$QUERY_STOP_INDEX" \
  bash radio_gs/scripts/run_repo_python.sh - <<'PY'
import json, os
rows = json.load(open(os.environ["MANIFEST"]))["queries"]
start = int(os.environ.get("QUERY_START_INDEX", "0"))
stop_raw = os.environ.get("QUERY_STOP_INDEX", "").strip()
stop = len(rows) if not stop_raw else int(stop_raw)
if start < 0 or stop < start or stop > len(rows):
    raise SystemExit(f"invalid PFIR query slice [{start}:{stop}] for {len(rows)} queries")
for row in rows[start:stop]:
    print("\t".join((row["query_id"], row["scene_id"], row["crop_rgb_path"])))
PY
)

echo "PFIR evaluation queue: ${#QUERIES[@]} queries from slice [${QUERY_START_INDEX}:${QUERY_STOP_INDEX:-end}]"

for record in "${QUERIES[@]}"; do
  IFS=$'\t' read -r query_id scene crop <<<"$record"
  field_dir="$RUN_ROOT/canonical_fields/$scene"
  field="$field_dir/canonical_mpr_v2.pt"
  capability="$field_dir/official_dino_sam3_views.pt"
  graph="$field_dir/shared_support_graph_k16.pt"
  semantic="$field_dir/global_region_summary_semantic.pt"
  semantic_query="$field_dir/global_region_summary_semantic_query.pt"
  cache="$QUERY_ROOT/$query_id.pt"
  ranking="$RANKING_ROOT/$query_id.npy"
  selection="$SELECTION_ROOT/$query_id.npy"
  mesh="$ANNOTATIONS_ROOT/$scene/${scene}_vh_clean_2.ply"

  if [[ ! -s "$semantic_query" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/materialize_surface_region_query_cache.py \
      --semantic-cache "$semantic" \
      --output "$semantic_query" \
      >"$RUN_ROOT/logs/${scene}.semantic_query.log" 2>&1
  fi

  if [[ ! -s "$cache" ]]; then
    wait_for_gpu
    field_sha256="$(sha256sum "$field" | awk '{print $1}')"
    CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/build_posefree_image_query_cache.py \
      --image "$crop" \
      --capability-cache "$capability" \
      --semantic-cache "$semantic_query" \
      --support-graph "$graph" \
      --field-checkpoint-sha256 "$field_sha256" \
      --output "$cache" \
      --radio-repo "$RADIO_REPO" \
      --radio-version c-radio_v4-h \
      --radio-checkpoint "$RADIO_CHECKPOINT" \
      --device cuda:0 \
      --prototype-count 4 \
      --semantic-weight 1.0 \
      --appearance-weight 1.0 \
      --iterations 12 \
      --residual 0.30 \
      --support-threshold 0.50 \
      >"$RUN_ROOT/logs/${query_id}.query.log" 2>&1
  fi

  if [[ ! -s "$ranking" || ! -s "$selection" ]]; then
    bash radio_gs/scripts/run_repo_python.sh \
      -m radio_gs.benchmarks.scannet_pfir.export_query_prediction \
      --query-cache "$cache" \
      --mesh-ply "$mesh" \
      --ranking-output "$ranking" \
      --selection-output "$selection" \
      --neighbors 3 \
      --maximum-distance-m 0.10 \
      --support-threshold 0.50 \
      >"$RUN_ROOT/logs/${query_id}.mesh_export.log" 2>&1
  fi
done

case "$RUN_EVALUATOR" in
  0|false|False|FALSE)
    echo "PFIR query shard complete; the aggregation invocation will evaluate the tracks."
    exit 0
    ;;
  1|true|True|TRUE)
    ;;
  *)
    echo "RUN_EVALUATOR must be 0/1 or true/false, got: $RUN_EVALUATOR" >&2
    exit 2
    ;;
esac

bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfir.evaluate_predictions \
  --benchmark-dir "$BENCHMARK_DIR" \
  --prediction-dir "$RANKING_ROOT" \
  --annotations-root "$ANNOTATIONS_ROOT" \
  --track ranking \
  --output "$RUN_ROOT/track_a_ranking.json" \
  >"$RUN_ROOT/logs/track_a.log" 2>&1

bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfir.evaluate_predictions \
  --benchmark-dir "$BENCHMARK_DIR" \
  --prediction-dir "$SELECTION_ROOT" \
  --annotations-root "$ANNOTATIONS_ROOT" \
  --track selection \
  --output "$RUN_ROOT/track_b_selection.json" \
  >"$RUN_ROOT/logs/track_b.log" 2>&1

case "$WRITE_TERMINAL" in
  1|true|True|TRUE)
    date -Iseconds >"$RUN_ROOT/pfir_evaluation.complete"
    ;;
  0|false|False|FALSE)
    echo "PFIR aggregation completed without writing a terminal marker."
    ;;
  *)
    echo "WRITE_TERMINAL must be 0/1 or true/false, got: $WRITE_TERMINAL" >&2
    exit 2
    ;;
esac
