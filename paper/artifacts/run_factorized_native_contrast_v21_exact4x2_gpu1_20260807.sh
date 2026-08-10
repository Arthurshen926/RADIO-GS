#!/usr/bin/env bash
set -euo pipefail

# Physical GPU1 is exposed as process-local cuda:0.  V2.1 retains the V2
# 60-step objective and 5-step validation interval; only the frozen source
# promotion comparator changes.

REPO_ROOT="/root/RADIO-GS"
AUTHORITY="$REPO_ROOT/paper/artifacts/factorized_native_gauge_state_readout_exact4x2_contrast_v21_execution_authority_20260807.json"
AUTHORITY_SHA256="f24008d976067a5b0ee42eb2b9cf3e3276c11199c82b68754f5c000a5166c171"
OUTPUT="/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/factorized_native_gauge_state_exact4x2/contrast_v21_direction_only/model.pt"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=1
exec bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.scripts.train_factorized_native_gauge_state_readout_exact4x2_contrast_v21 \
  train \
  --execution-authority "$AUTHORITY" \
  --expected-execution-authority-sha256 "$AUTHORITY_SHA256" \
  --output "$OUTPUT" \
  --device cuda:0
