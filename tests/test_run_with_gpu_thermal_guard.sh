#!/usr/bin/env bash

# CPU-only integration tests for bounded runtime telemetry retries.  The fake
# nvidia-smi and od binaries exercise the real guard without touching a GPU or
# PCIe configuration space.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$REPO_ROOT/radio_gs/scripts/run_with_gpu_thermal_guard.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

FAKE_BIN="$TEST_ROOT/bin"
mkdir -p "$FAKE_BIN"

cat >"$FAKE_BIN/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

arguments="$*"
if [[ "$arguments" == *"--query-gpu=pci.bus_id,uuid"* ]]; then
  printf '00000000:01:00.0, GPU-11111111-2222-3333-4444-555555555555\n'
elif [[ "$arguments" == *"--query-compute-apps=gpu_uuid,pid"* ]]; then
  :
elif [[ "$arguments" == *"--query-gpu=uuid"* ]]; then
  printf 'GPU-11111111-2222-3333-4444-555555555555\n'
elif [[ "$arguments" == *"--query-gpu=temperature.gpu,power.draw,power.limit"* ]]; then
  count_file="$MOCK_STATE_DIR/telemetry_count"
  count=0
  if [[ -f "$count_file" ]]; then
    read -r count <"$count_file"
  fi
  count=$((count + 1))
  printf '%s\n' "$count" >"$count_file"
  case "$MOCK_TELEMETRY_PLAN:$count" in
    recover:2|recover:4|threshold:2|threshold:3|pcie:2)
      exit 1
      ;;
  esac
  case "$MOCK_TELEMETRY_PLAN:$count" in
    overheat:2|overheat:3)
      printf '87, 250.0, 300.0, 99, 1000, P2\n'
      ;;
    *)
      printf '40, 50.0, 300.0, 10, 100, P2\n'
      ;;
  esac
else
  echo "unexpected fake nvidia-smi arguments: $arguments" >&2
  exit 97
fi
EOF
chmod +x "$FAKE_BIN/nvidia-smi"

cat >"$FAKE_BIN/od" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

count_file="$MOCK_STATE_DIR/pcie_count"
count=0
if [[ -f "$count_file" ]]; then
  read -r count <"$count_file"
fi
count=$((count + 1))
printf '%s\n' "$count" >"$count_file"
if [[ "${MOCK_PCIE_FAIL_AT:-0}" -gt 0 \
      && "$count" -ge "$MOCK_PCIE_FAIL_AT" ]]; then
  printf ' ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff\n'
else
  printf ' 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10\n'
fi
EOF
chmod +x "$FAKE_BIN/od"

run_guard() {
  local case_name="$1"
  local plan="$2"
  local maximum_failures="$3"
  local pcie_fail_at="$4"
  local child_seconds="$5"
  local maximum_overheat_polls="${6:-1}"
  local poll_seconds="${7:-1}"
  local case_root="$TEST_ROOT/$case_name"
  mkdir -p "$case_root/state"
  PATH="$FAKE_BIN:$PATH" \
    MOCK_STATE_DIR="$case_root/state" \
    MOCK_TELEMETRY_PLAN="$plan" \
    MOCK_PCIE_FAIL_AT="$pcie_fail_at" \
    GPU=0 \
    GPU_TELEMETRY_LOG="$case_root/telemetry.csv" \
    GPU_MAX_TEMP_C=86 \
    GPU_START_MAX_TEMP_C=80 \
    GPU_POLL_SECONDS="$poll_seconds" \
    GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES="$maximum_failures" \
    GPU_MAX_CONSECUTIVE_OVERHEAT_POLLS="$maximum_overheat_polls" \
    bash "$GUARD" -- bash -c "sleep $child_seconds" \
    >"$case_root/stdout.log" 2>"$case_root/stderr.log"
}

run_guard recover recover 2 0 4
[[ "$(grep -c ',telemetry_retry_1_of_2$' \
  "$TEST_ROOT/recover/telemetry.csv")" -eq 2 ]]
grep -q ',sample$' "$TEST_ROOT/recover/telemetry.csv"

# A poll interval greater than one must still produce periodic runtime samples.
# This specifically guards against using Bash's special ``_`` parameter as a
# counter: commands inside the loop rewrite ``_`` and previously made the wait
# loop effectively infinite for every production interval above one second.
run_guard periodic steady 2 0 8 1 2
periodic_count="$(cat "$TEST_ROOT/periodic/state/telemetry_count")"
[[ "$periodic_count" -ge 4 ]]
[[ "$(grep -c ',sample$' "$TEST_ROOT/periodic/telemetry.csv")" -ge 3 ]]

set +e
run_guard threshold threshold 2 0 20
threshold_status=$?
set -e
[[ "$threshold_status" -eq 86 ]]
grep -q ',telemetry_retry_1_of_2$' \
  "$TEST_ROOT/threshold/telemetry.csv"
grep -q ',telemetry_failed_2_of_2$' \
  "$TEST_ROOT/threshold/telemetry.csv"

set +e
run_guard pcie pcie 3 3 20
pcie_status=$?
set -e
[[ "$pcie_status" -eq 86 ]]
grep -q ',pcie_unresponsive_after_telemetry_failure$' \
  "$TEST_ROOT/pcie/telemetry.csv"
if grep -q 'telemetry_retry_' "$TEST_ROOT/pcie/telemetry.csv"; then
  echo "PCIe failure must not be retried as a transient telemetry failure" >&2
  exit 1
fi

set +e
run_guard overheat overheat 3 0 20 2
overheat_status=$?
set -e
[[ "$overheat_status" -eq 86 ]]
grep -q ',thermal_warning_1_of_2$' "$TEST_ROOT/overheat/telemetry.csv"
grep -q ',thermal_abort_2_of_2$' "$TEST_ROOT/overheat/telemetry.csv"

printf 'thermal guard telemetry retry tests passed\n'
