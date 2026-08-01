#!/usr/bin/env bash

# Evaluate the ScanNet-frozen specificity-preserving scale rule on all four
# LERF scenes.  The primitive unaries are compiled before this queue starts;
# this script only performs the unchanged frozen rendering/evaluator pass.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:?set GPU to a physical GPU index}"
SOURCE_ROOT="${SOURCE_ROOT:-output/optimization_20260724/text_specificity_margin002}"
AFTER_MARKER="${AFTER_MARKER:-}"
PRIMITIVE_VALID_NORMALIZATION="${PRIMITIVE_VALID_NORMALIZATION:-0}"
PRIMITIVE_VALID_COVERAGE_POWER="${PRIMITIVE_VALID_COVERAGE_POWER:-0}"

if [[ ! "$PRIMITIVE_VALID_COVERAGE_POWER" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "PRIMITIVE_VALID_COVERAGE_POWER must be a non-negative decimal" >&2
  exit 2
fi

case "$PRIMITIVE_VALID_NORMALIZATION" in
  0)
    if [[ ! "$PRIMITIVE_VALID_COVERAGE_POWER" =~ ^0([.]0+)?$ ]]; then
      echo "nonzero PRIMITIVE_VALID_COVERAGE_POWER requires PRIMITIVE_VALID_NORMALIZATION=1" >&2
      exit 2
    fi
    VALID_NORMALIZATION_ARGS=()
    ;;
  1)
    VALID_NORMALIZATION_ARGS=(
      --primitive_valid_normalization
      --primitive_valid_coverage_power "$PRIMITIVE_VALID_COVERAGE_POWER"
    )
    ;;
  *)
    echo "PRIMITIVE_VALID_NORMALIZATION must be 0 or 1" >&2
    exit 2
    ;;
esac

if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT="$OUTPUT_ROOT"
elif [[ "$PRIMITIVE_VALID_NORMALIZATION" == 1 ]]; then
  COVERAGE_TAG="${PRIMITIVE_VALID_COVERAGE_POWER//./p}"
  OUTPUT_ROOT="${SOURCE_ROOT}_validnorm_beta${COVERAGE_TAG}"
else
  OUTPUT_ROOT="$SOURCE_ROOT"
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
    if (( available < 2 )); then sleep 20; fi
  done
}

validate_result() {
  local result="$1"
  local scene="$2"
  local unary="$3"
  EXPECTED_VALID_NORMALIZATION="$PRIMITIVE_VALID_NORMALIZATION" \
  EXPECTED_COVERAGE_POWER="$PRIMITIVE_VALID_COVERAGE_POWER" \
  bash radio_gs/scripts/run_repo_python.sh - "$result" "$scene" "$unary" <<'PY'
import json
import os
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
scene = sys.argv[2]
unary = Path(sys.argv[3])
payload = json.loads(report_path.read_text(encoding="utf-8"))
arguments = payload.get("args", {})
operator = payload.get("feature_observation_operator", {})
expected_normalization = os.environ["EXPECTED_VALID_NORMALIZATION"] == "1"
expected_power = float(os.environ["EXPECTED_COVERAGE_POWER"])

expected_arguments = {
    "rendered_only": "True",
    "render_readout": "primitive_unary",
    "scoring": "cosine",
    "threshold_mode": "fixed",
    "iou_threshold": "0.6",
    "heatmap_upsample": "4",
    "localization_mode": "polygon_argmax",
    "mask_refinement": "none",
    "feature_contribution_gamma": "1.0",
}
for key, expected in expected_arguments.items():
    if arguments.get(key) != expected:
        raise SystemExit(
            f"{report_path}: {key}={arguments.get(key)!r}, expected {expected!r}"
        )
if arguments.get("primitive_valid_normalization") != str(expected_normalization):
    raise SystemExit(f"{report_path}: valid-normalization mode does not match queue")
if float(arguments.get("primitive_valid_coverage_power", "nan")) != expected_power:
    raise SystemExit(f"{report_path}: coverage power does not match queue")
if Path(arguments.get("primitive_score_cache", "")).resolve() != unary.resolve():
    raise SystemExit(f"{report_path}: primitive unary cache does not match queue")

expected_formula = (
    "sum(w*v*s)/sum(w*v) * coverage**coverage_power"
    if expected_normalization
    else "sum(w*v*s)/sum(w)"
)
expected_operator = {
    "primitive_valid_normalization": expected_normalization,
    "semantic_score_formula": expected_formula,
    "semantic_coverage_power": (
        expected_power if expected_normalization else None
    ),
    "query_dependent": False,
    "changes_geometry_or_alpha": False,
}
for key, expected in expected_operator.items():
    if operator.get(key) != expected:
        raise SystemExit(
            f"{report_path}: operator {key}={operator.get(key)!r}, "
            f"expected {expected!r}"
        )

scenes = payload.get("scenes", {})
if set(scenes) != {scene}:
    raise SystemExit(f"{report_path}: expected only scene {scene!r}")
expected_samples = {
    "figurines": 56,
    "ramen": 71,
    "teatime": 59,
    "waldo_kitchen": 22,
}
aggregate = payload.get("aggregates", {}).get("rendered", {})
if int(aggregate.get("sample_count", -1)) != expected_samples[scene]:
    raise SystemExit(f"{report_path}: unexpected rendered sample count")
required_provenance = {
    "config", "checkpoint", "text_embedding_cache", "primitive_score_cache"
}
if not required_provenance.issubset(payload.get("provenance", {})):
    raise SystemExit(f"{report_path}: incomplete frozen-input provenance")
print(
    json.dumps(
        {
            "scene": scene,
            "sample_micro_miou": aggregate["sample_micro_miou"],
            "localization_accuracy": aggregate["localization_accuracy"],
            "sample_count": aggregate["sample_count"],
        }
    )
)
PY
}

if [[ -n "$AFTER_MARKER" ]]; then
  while [[ ! -s "$AFTER_MARKER" ]]; do sleep 30; done
fi

declare -A CONFIGS=(
  [figurines]="radio_gs/configs/generated/query_consistency/lerf_figurines_radio_verified_pose.yaml"
  [ramen]="radio_gs/configs/generated/query_consistency/lerf_ramen_radio_verified_pose.yaml"
  [teatime]="radio_gs/configs/generated/query_consistency/lerf_teatime_radio_verified_pose.yaml"
  [waldo_kitchen]="radio_gs/configs/generated/query_consistency/lerf_waldo_kitchen_radio_verified_pose.yaml"
)
declare -A CHECKPOINTS=(
  [figurines]="output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth"
  [ramen]="output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth"
  [teatime]="output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep/checkpoints/best.pth"
  [waldo_kitchen]="output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth"
)

mkdir -p "$OUTPUT_ROOT/logs"
REPORTS=()
for scene in figurines ramen teatime waldo_kitchen; do
  unary="$SOURCE_ROOT/${scene}_unary_specificity002.pt"
  result="$OUTPUT_ROOT/${scene}_eval_specificity002/lerf_ovs_results.json"
  if [[ -s "$result" ]]; then
    validate_result "$result" "$scene" "$unary"
    REPORTS+=("$result")
    continue
  fi
  for required in "$unary" "${CONFIGS[$scene]}" "${CHECKPOINTS[$scene]}"; do
    if [[ ! -s "$required" ]]; then
      echo "missing LERF specificity input: $required" >&2
      exit 2
    fi
  done
  wait_for_gpu
  CUDA_VISIBLE_DEVICES="$GPU" bash radio_gs/scripts/run_repo_python.sh \
    radio_gs/scripts/eval_lerf_grounding.py \
    --config "${CONFIGS[$scene]}" \
    --checkpoint "${CHECKPOINTS[$scene]}" \
    --rendered_only \
    --render_readout primitive_unary \
    --primitive_score_cache "$unary" \
    "${VALID_NORMALIZATION_ARGS[@]}" \
    --scene "$scene" \
    --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
    --output_dir "$OUTPUT_ROOT/${scene}_eval_specificity002" \
    --text_embedding_cache checkpoints/siglip2_lerf_text_embeddings_query_all_20260515.pt \
    --prompt_templates '{query}' \
    --iou_threshold 0.6 \
    --threshold_mode fixed \
    --scoring cosine \
    --heatmap_upsample 4 \
    --localization_mode polygon_argmax \
    --mask_refinement none \
    --gpu 0 \
    >"$OUTPUT_ROOT/logs/${scene}_specificity002.log" 2>&1
  validate_result "$result" "$scene" "$unary"
  REPORTS+=("$result")
done

bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/summarize_lerf_scene_reports.py \
  --reports "${REPORTS[@]}" \
  --expected-scenes figurines ramen teatime waldo_kitchen \
  --output "$OUTPUT_ROOT/lerf_specificity002_summary.json"
date -Iseconds >"$OUTPUT_ROOT/lerf_specificity002.complete"
