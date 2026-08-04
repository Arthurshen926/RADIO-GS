#!/usr/bin/env bash

# Freeze both strict-source-only selectors before the unchanged frozen target
# scorer opens either target mask.  Full8 is intentionally a separate launch
# that is permitted only after the registered sentinel gate is evaluated.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-1}"
SCENE_NAMES="${SCENE_NAMES:-horns_left fern}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/optimization_20260804/nvos_evidence_to_support_v2_sentinels}"
FIELD_ROOT="${FIELD_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/evaluation_closeout_20260716/canonical_mpr_v3_nvos8}"
EXACT_ROOT="${EXACT_ROOT:-$REPO_ROOT/output/optimization_20260803/nvos_exact_prompt_responsibility_dino_completion_v1}"
MANIFEST="${MANIFEST:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json}"
QUEUE_ROOT="${QUEUE_ROOT:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/gaussfm_jobs/nvos_strict_unseen_v1}"
REGISTRATION="$REPO_ROOT/paper/artifacts/nvos_evidence_to_support_v2_registration_20260804.json"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
THERMAL_GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"

[[ "$GPU" == "1" ]] || { echo "NVOS E2S-v2 sentinel queue is registered for physical GPU1" >&2; exit 2; }
mkdir -p "$RUN_ROOT"/{selectors,scores,reports,logs}

run_guarded() {
  local stage="$1"
  shift
  env \
    GPU="$GPU" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    GPU_MAX_POWER_LIMIT_W=300.5 \
    GPU_POLL_SECONDS=180 \
    GPU_START_MAX_TEMP_C=83 \
    GPU_MAX_TEMP_C=87 \
    GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS=2 \
    GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=3 \
    GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
    GPU_TELEMETRY_LOG="$RUN_ROOT/logs/${stage}.gpu${GPU}.telemetry.csv" \
    GPU_OWNER_AUDIT_LOG="$RUN_ROOT/logs/${stage}.gpu${GPU}.owner.csv" \
    bash "$THERMAL_GUARD" -- \
      env CUDA_VISIBLE_DEVICES="$GPU" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" "$@" \
      >"$RUN_ROOT/logs/${stage}.log" 2>&1
}

prompt_paths() {
  "$PYTHON" - "$MANIFEST" "$1" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
matches = [scene for scene in manifest["scenes"] if scene["scene_id"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit("expected exactly one manifest scene")
prompt = matches[0]["prompt"]
print(Path(prompt["positive_path"]).resolve())
print(Path(prompt["negative_path"]).resolve())
PY
}

artifact_hashes() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["output_sha256"])
print(payload["primitive_probability_sha256"])
PY
}

# Phase 1: construct and seal both target-blind selector receipts.
for scene in $SCENE_NAMES; do
  field_dir="$FIELD_ROOT/$scene"
  capability="$field_dir/official_dino_sam3_views.pt"
  graph="$field_dir/shared_support_graph_k16.pt"
  cache="$EXACT_ROOT/cache/$scene.pt"
  cache_report="$EXACT_ROOT/reports/$scene.export.json"
  selector="$RUN_ROOT/selectors/$scene.pt"
  for asset in "$capability" "$capability.json" "$graph" "$cache" "$cache_report" "$MANIFEST" "$REGISTRATION"; do
    [[ -s "$asset" ]] || { echo "$scene missing required asset: $asset" >&2; exit 3; }
  done
  mapfile -t prompts < <(prompt_paths "$scene")
  [[ "${#prompts[@]}" == "2" ]] || { echo "$scene prompt resolution failed" >&2; exit 3; }
  if [[ ! -s "$selector" ]]; then
    echo "[$(date --iso-8601=seconds)] $scene: build strict continuous E2S-v2 selector"
    run_guarded "${scene}.selector" \
      radio_gs/scripts/build_nvos_evidence_to_support_v2.py \
      --scene-id "$scene" \
      --experiment-registration "$REGISTRATION" \
      --cache "$cache" \
      --cache-report "$cache_report" \
      --positive-scribble "${prompts[0]}" \
      --negative-scribble "${prompts[1]}" \
      --capability-cache "$capability" \
      --support-graph "$graph" \
      --output "$selector" \
      --device cuda:0 \
      --feature-chunk-size 4096
  fi
  [[ -s "$selector.json" ]] || { echo "$scene selector receipt is missing" >&2; exit 4; }
done

for scene in $SCENE_NAMES; do
  [[ -s "$RUN_ROOT/selectors/$scene.pt" && -s "$RUN_ROOT/selectors/$scene.pt.json" ]] || {
    echo "$scene is missing its sealed pre-label selector" >&2
    exit 4
  }
done

# Phase 2: only now may the untouched evaluator render and score target masks.
for scene in $SCENE_NAMES; do
  cache_report="$EXACT_ROOT/reports/$scene.export.json"
  selector="$RUN_ROOT/selectors/$scene.pt"
  score="$RUN_ROOT/scores/$scene.pt"
  report="$RUN_ROOT/reports/$scene.json"
  mapfile -t hashes < <(artifact_hashes "$selector.json")
  [[ "${#hashes[@]}" == "2" ]] || { echo "$scene selector hashes are missing" >&2; exit 4; }
  if [[ ! -s "$report" ]]; then
    echo "[$(date --iso-8601=seconds)] $scene: frozen target readout"
    run_guarded "${scene}.score" \
      radio_gs/scripts/eval_nvos_evidence_to_support_v2.py \
      --manifest "$MANIFEST" \
      --queue-root "$QUEUE_ROOT" \
      --scene-id "$scene" \
      --cache-report "$cache_report" \
      --completion "$selector" \
      --expected-completion-sha256 "${hashes[0]}" \
      --expected-primitive-sha256 "${hashes[1]}" \
      --score-output "$score" \
      --report "$report" \
      --device cuda:0
  fi
done
