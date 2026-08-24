#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/RADIO-GS}
export REPO
export VARIANT=query_native_categorical_coeff_h256p96_v4
export HIDDEN_DIM=256
export PAIR_HIDDEN_DIM=96
export STEPS=900
export GATE_STEPS=900

run_scene() {
  bash "$REPO/radio_gs/scripts/run_scannet_query_native_categorical_scene.sh" "$1" "$2"
}

(run_scene scene0000_00 0; run_scene scene0400_00 0) & p0=$!
(run_scene scene0062_00 1; run_scene scene0590_00 1) & p1=$!
run_scene scene0070_00 2 & p2=$!
run_scene scene0097_00 3 & p3=$!
run_scene scene0140_00 4 & p4=$!
run_scene scene0347_00 5 & p5=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do
  wait "$pid" || status=1
done
exit "$status"
