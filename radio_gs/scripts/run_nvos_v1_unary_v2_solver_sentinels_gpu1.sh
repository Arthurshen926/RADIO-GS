#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
GPU="${GPU:-1}"
SCENES="${SCENES:-horns_left fern}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/output/optimization_20260804/nvos_v1_unary_v2_solver_sentinels}"
FIELD_ROOT="${FIELD_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/evaluation_closeout_20260716/canonical_mpr_v3_nvos8}"
BASE_ROOT="${BASE_ROOT:-$REPO_ROOT/output/optimization_20260803/nvos_strict_hashed_query_conditioned_sentinels_v1}"
EXACT_ROOT="${EXACT_ROOT:-$REPO_ROOT/output/optimization_20260803/nvos_exact_prompt_responsibility_dino_completion_v1}"
MANIFEST="${MANIFEST:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json}"
QUEUE_ROOT="${QUEUE_ROOT:-/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/gaussfm_jobs/nvos_strict_unseen_v1}"
REGISTRATION="$REPO_ROOT/paper/artifacts/nvos_v1_unary_v2_continuous_solver_followup_registration_20260804.json"
PYTHON="${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}"
GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
[[ "$GPU" == 1 ]] || exit 2
mkdir -p "$RUN_ROOT"/{selectors,scores,reports,logs}

run_guarded() {
  local stage="$1"; shift
  env GPU="$GPU" CUDA_VISIBLE_DEVICES="$GPU" GPU_MAX_POWER_LIMIT_W=300.5 \
    GPU_POLL_SECONDS=180 GPU_START_MAX_TEMP_C=83 GPU_MAX_TEMP_C=87 \
    GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS=2 GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES=3 \
    GPU_OWNER_PID_NAMESPACE_MODE=exclusive-singleton-after-clear-v1 \
    GPU_TELEMETRY_LOG="$RUN_ROOT/logs/${stage}.gpu${GPU}.telemetry.csv" \
    GPU_OWNER_AUDIT_LOG="$RUN_ROOT/logs/${stage}.gpu${GPU}.owner.csv" \
    bash "$GUARD" -- env CUDA_VISIBLE_DEVICES="$GPU" \
      bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" "$@" \
      >"$RUN_ROOT/logs/${stage}.log" 2>&1
}

hashes() {
  "$PYTHON" - "$1" <<'PY'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
print(x["output_sha256"]); print(x["primitive_probability_sha256"])
PY
}

for scene in $SCENES; do
  selector="$RUN_ROOT/selectors/$scene.pt"
  if [[ ! -s "$selector" ]]; then
    run_guarded "$scene.selector" radio_gs/scripts/build_nvos_v1_unary_v2_continuous_solver.py \
      --scene-id "$scene" --experiment-registration "$REGISTRATION" \
      --base-selector "$BASE_ROOT/selectors/$scene.pt" \
      --cache-report "$EXACT_ROOT/reports/$scene.export.json" \
      --support-graph "$FIELD_ROOT/$scene/shared_support_graph_k16.pt" \
      --output "$selector" --device cuda:0
  fi
done
for scene in $SCENES; do
  [[ -s "$RUN_ROOT/selectors/$scene.pt.json" ]] || exit 4
done
for scene in $SCENES; do
  selector="$RUN_ROOT/selectors/$scene.pt"
  report="$RUN_ROOT/reports/$scene.json"
  mapfile -t values < <(hashes "$selector.json")
  if [[ ! -s "$report" ]]; then
    run_guarded "$scene.score" radio_gs/scripts/eval_nvos_v1_unary_v2_continuous_solver.py \
      --manifest "$MANIFEST" --queue-root "$QUEUE_ROOT" --scene-id "$scene" \
      --cache-report "$EXACT_ROOT/reports/$scene.export.json" \
      --completion "$selector" --expected-completion-sha256 "${values[0]}" \
      --expected-primitive-sha256 "${values[1]}" \
      --score-output "$RUN_ROOT/scores/$scene.pt" --report "$report" --device cuda:0
  fi
done
