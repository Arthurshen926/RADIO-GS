#!/usr/bin/env bash

# CPU-only end-to-end tests.  A fake nvidia-smi exercises NVML fail-closed,
# host/container PID fallback, six-sample thermal abort, process-group pause,
# cleanup, and output preservation without opening a real GPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_LAUNCHER="$REPO_ROOT/radio_gs/scripts/run_repo_python.sh"
WRAPPER_MODULE="radio_gs.scripts.run_with_fixed_cuda_duty_cycle"
SYNTHETIC_WORKER="$REPO_ROOT/tests/fixtures/fixed_cuda_duty_cycle_worker.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

FAKE_BIN="$TEST_ROOT/bin"
mkdir -p "$FAKE_BIN"

cat >"$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

arguments="$*"
if [[ "$arguments" == *"--query-gpu=pci.bus_id,uuid,temperature.gpu"* ]]; then
  count_file="$MOCK_STATE_DIR/telemetry_count"
  count=0
  if [[ -f "$count_file" ]]; then
    read -r count <"$count_file"
  fi
  count=$((count + 1))
  printf '%s\n' "$count" >"$count_file"
  if [[ "$MOCK_PLAN" == "nvml_fail" && "$count" -ge 2 ]]; then
    echo "synthetic NVML failure" >&2
    exit 9
  fi
  temperature=50
  if [[ "$MOCK_PLAN" == "six_hot" && "$count" -ge 2 ]]; then
    temperature=89
  fi
  printf '00000000:01:00.0, GPU-11111111-2222-3333-4444-555555555555, %s, 120.0, 300.0, 80, 1000, P2\n' "$temperature"
elif [[ "$arguments" == *"--query-compute-apps=gpu_uuid,pid"* ]]; then
  count_file="$MOCK_STATE_DIR/owner_count"
  count=0
  if [[ -f "$count_file" ]]; then
    read -r count <"$count_file"
  fi
  count=$((count + 1))
  printf '%s\n' "$count" >"$count_file"
  # The first query is the mandatory prelaunch-clear observation.  Subsequent
  # singleton host PIDs are deliberately invisible in this PID namespace.
  if [[ "$MOCK_PLAN" == "namespace" && "$count" -ge 2 ]]; then
    printf 'GPU-11111111-2222-3333-4444-555555555555, 999999\n'
  fi
else
  echo "unexpected fake nvidia-smi arguments: $arguments" >&2
  exit 97
fi
EOF
chmod +x "$FAKE_BIN/nvidia-smi"

run_scaled_case() {
  local case_name="$1"
  local plan="$2"
  local worker_count="$3"
  local run_seconds="$4"
  local cool_seconds="$5"
  local poll_seconds="$6"
  local case_root="$TEST_ROOT/$case_name"
  mkdir -p "$case_root/mock" "$case_root/worker"
  MOCK_STATE_DIR="$case_root/mock" \
  MOCK_PLAN="$plan" \
  CASE_ROOT="$case_root" \
  WRAPPER_MODULE="$WRAPPER_MODULE" \
  SYNTHETIC_WORKER="$SYNTHETIC_WORKER" \
  WORKER_COUNT="$worker_count" \
  TEST_RUN_SECONDS="$run_seconds" \
  TEST_COOL_SECONDS="$cool_seconds" \
  TEST_POLL_SECONDS="$poll_seconds" \
  FAKE_NVIDIA_SMI="$FAKE_BIN/nvidia-smi" \
    bash "$PYTHON_LAUNCHER" - <<'PY'
from dataclasses import replace
import importlib
import os
from pathlib import Path

module = importlib.import_module(os.environ["WRAPPER_MODULE"])
case_root = Path(os.environ["CASE_ROOT"])
policy = replace(
    module.FROZEN_POLICY,
    run_seconds=float(os.environ["TEST_RUN_SECONDS"]),
    cool_seconds=float(os.environ["TEST_COOL_SECONDS"]),
    nvml_poll_seconds=float(os.environ["TEST_POLL_SECONDS"]),
    nvml_query_timeout_seconds=0.01,
    terminate_grace_seconds=0.25,
)
raise SystemExit(
    module.run_guarded(
        [
            "bash",
            os.environ["SYNTHETIC_WORKER"],
            str(case_root / "output.txt"),
            os.environ["WORKER_COUNT"],
            "0.01",
            str(case_root / "worker"),
        ],
        physical_gpu=17,
        status_log=case_root / "status.jsonl",
        nvidia_smi=os.environ["FAKE_NVIDIA_SMI"],
        policy=policy,
    )
)
PY
}

# Production timing is immutable at the CLI boundary.  Tests shorten time only
# by calling the Python API with an injected policy and never exercise a GPU.
bash "$PYTHON_LAUNCHER" - <<'PY'
from radio_gs.scripts.run_with_fixed_cuda_duty_cycle import FROZEN_POLICY

assert FROZEN_POLICY.run_seconds == 45.0
assert FROZEN_POLICY.cool_seconds == 30.0
assert FROZEN_POLICY.nvml_poll_seconds == 30.0
assert FROZEN_POLICY.start_max_temperature_c == 65.0
assert FROZEN_POLICY.hard_abort_temperature_c == 88.0
assert FROZEN_POLICY.hard_abort_consecutive_samples == 6
PY

set +e
bash "$PYTHON_LAUNCHER" \
  "$REPO_ROOT/radio_gs/scripts/run_with_fixed_cuda_duty_cycle.py" \
  --gpu 0 --status-log "$TEST_ROOT/rejected.jsonl" --run-seconds 1 -- true \
  >"$TEST_ROOT/rejected.stdout" 2>"$TEST_ROOT/rejected.stderr"
rejected_status=$?
set -e
[[ "$rejected_status" -eq 2 ]]
grep -q 'unrecognized arguments: --run-seconds' "$TEST_ROOT/rejected.stderr"

# A stopped/resumed worker produces the exact deterministic byte sequence, and
# the invisible singleton host PID stays bound to this owner-free launch.
run_scaled_case success namespace 30 0.08 0.05 0.04
seq 1 30 >"$TEST_ROOT/success/expected.txt"
cmp "$TEST_ROOT/success/expected.txt" "$TEST_ROOT/success/output.txt"
test -f "$TEST_ROOT/success/worker/complete"
grep -q '"event": "duty_pause"' "$TEST_ROOT/success/status.jsonl"
grep -q '"event": "duty_resume"' "$TEST_ROOT/success/status.jsonl"
grep -q 'runtime_owner_audit_host_pid_singleton' \
  "$TEST_ROOT/success/status.jsonl"
grep -q '"event": "worker_complete"' "$TEST_ROOT/success/status.jsonl"

# The first failed runtime NVML query is fail-closed and EXIT cleanup terminates
# the complete worker group; no six-failure retry window is allowed.
set +e
run_scaled_case nvml_fail nvml_fail 500 0.08 0.05 0.04
nvml_status=$?
set -e
[[ "$nvml_status" -eq 86 ]]
grep -q '"event": "fail_closed_abort"' "$TEST_ROOT/nvml_fail/status.jsonl"
grep -q '"event": "worker_cleanup_complete"' \
  "$TEST_ROOT/nvml_fail/status.jsonl"
nvml_worker_pid="$(cat "$TEST_ROOT/nvml_fail/worker/worker.pid")"
if kill -0 "$nvml_worker_pid" 2>/dev/null; then
  echo "NVML-failure worker survived guard cleanup" >&2
  exit 1
fi

# Six consecutive >=88 C runtime samples hard-abort; the first hot sample also
# fail-safely holds the group instead of allowing it to run hot for 180 s.
set +e
run_scaled_case six_hot six_hot 500 0.05 0.04 0.025
hot_status=$?
set -e
[[ "$hot_status" -eq 86 ]]
grep -q '"event": "thermal_hold_1_of_6"' \
  "$TEST_ROOT/six_hot/status.jsonl"
grep -q '"event": "thermal_hard_abort_6_of_6"' \
  "$TEST_ROOT/six_hot/status.jsonl"
grep -q '"event": "worker_cleanup_resume_before_terminate"' \
  "$TEST_ROOT/six_hot/status.jsonl"
hot_worker_pid="$(cat "$TEST_ROOT/six_hot/worker/worker.pid")"
if kill -0 "$hot_worker_pid" 2>/dev/null; then
  echo "thermal-abort worker survived guard cleanup" >&2
  exit 1
fi

# An external TERM delivered while the worker is in its scheduled cool phase
# must reach the Python guard, which resumes the stopped PGID before terminating
# and reaping it.  This exercises the EXIT path rather than a telemetry abort.
set +e
run_scaled_case interrupted steady 500 0.05 0.20 0.04 &
interrupted_launcher_pid=$!
set -e
for ((wait_index=0; wait_index<200; wait_index++)); do
  if [[ -f "$TEST_ROOT/interrupted/status.jsonl" ]] \
      && grep -q '"event": "duty_pause"' \
        "$TEST_ROOT/interrupted/status.jsonl"; then
    break
  fi
  sleep 0.01
done
grep -q '"event": "duty_pause"' "$TEST_ROOT/interrupted/status.jsonl"
interrupted_guard_pid="$(
  sed -n 's/.*"guard_pid": \([0-9][0-9]*\).*/\1/p' \
    "$TEST_ROOT/interrupted/status.jsonl" | head -n 1
)"
[[ "$interrupted_guard_pid" =~ ^[0-9]+$ ]]
kill -TERM "$interrupted_guard_pid"
set +e
wait "$interrupted_launcher_pid"
interrupted_status=$?
set -e
[[ "$interrupted_status" -eq 143 ]]
grep -q '"event": "guard_interrupted"' \
  "$TEST_ROOT/interrupted/status.jsonl"
grep -q '"event": "worker_cleanup_resume_before_terminate"' \
  "$TEST_ROOT/interrupted/status.jsonl"
grep -q '"event": "worker_cleanup_complete"' \
  "$TEST_ROOT/interrupted/status.jsonl"
interrupted_worker_pid="$(cat "$TEST_ROOT/interrupted/worker/worker.pid")"
if kill -0 "$interrupted_worker_pid" 2>/dev/null; then
  echo "externally interrupted worker survived guard cleanup" >&2
  exit 1
fi

printf 'fixed CUDA duty-cycle CPU integration tests passed\n'
