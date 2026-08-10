#!/usr/bin/env bash
set -euo pipefail

# Direct, one-shot GPU1 command for the frozen source-only contrast V2 arm.
# The trainer evaluates the full 2-scene validation cohort only every 5 steps
# (12 times over 60 steps) to reduce wall time and sustained thermal load.

REPO_ROOT="/root/RADIO-GS"
AUTHORITY="$REPO_ROOT/paper/artifacts/factorized_native_gauge_state_readout_exact4x2_contrast_v2_execution_authority_20260807.json"
AUTHORITY_SHA256="7d010800e9f64a0f45a3d9b881be0205dd56470b16c6cc872f739615bbedf607"
OUTPUT="/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/factorized_native_gauge_state_exact4x2/contrast_v2_direction_only/model.pt"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=1
exec bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.scripts.train_factorized_native_gauge_state_readout_exact4x2_contrast_v2 \
  train \
  --execution-authority "$AUTHORITY" \
  --expected-execution-authority-sha256 "$AUTHORITY_SHA256" \
  --output "$OUTPUT" \
  --device cuda:0
