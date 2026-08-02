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
  printf '40, 50.0, 300.0, 10, 100, P2\n'
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
    GPU_POLL_SECONDS=1 \
    GPU_MAX_CONSECUTIVE_TELEMETRY_FAILURES="$maximum_failures" \
    bash "$GUARD" -- bash -c "sleep $child_seconds" \
    >"$case_root/stdout.log" 2>"$case_root/stderr.log"
}

run_guard recover recover 2 0 4
[[ "$(grep -c ',telemetry_retry_1_of_2$' \
  "$TEST_ROOT/recover/telemetry.csv")" -eq 2 ]]
grep -q ',sample$' "$TEST_ROOT/recover/telemetry.csv"

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

printf 'thermal guard telemetry retry tests passed\n'
