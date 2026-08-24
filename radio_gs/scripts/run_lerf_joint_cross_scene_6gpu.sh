#!/usr/bin/env bash
set -euo pipefail
PY=/root/miniconda3/envs/cybersim_agent/bin/python
ROOT=/root/RADIO-GS
SPECS=$ROOT/paper/artifacts/lerf_joint_cross_scene_specs_20260824.json
OUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/query_native_joint_lerf_v1
mkdir -p "$OUT"

run() {
  local gpu=$1 tag=$2 lr=$3 rank=$4 seed=$5
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 "$PY" -m \
    radio_gs.scripts.train_lerf_query_native_joint_cross_scene_decoder \
    --scene-specs "$SPECS" --output "$OUT/$tag.pt" --device cuda:0 \
    --steps 1800 --validation-interval 90 --learning-rate "$lr" \
    --scene-canonicalizer-rank "$rank" --seed "$seed" \
    >"$OUT/$tag.log" 2>&1
}

run 0 joint_lr1e3_r8_s24 0.001 8 20260824 & p0=$!
run 1 joint_lr1e3_r8_s25 0.001 8 20260825 & p1=$!
run 2 joint_lr5e4_r8_s24 0.0005 8 20260824 & p2=$!
run 3 joint_lr2e3_r8_s24 0.002 8 20260824 & p3=$!
run 4 joint_lr1e3_r4_s24 0.001 4 20260824 & p4=$!
run 5 joint_lr1e3_r16_s24 0.001 16 20260824 & p5=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do wait "$pid" || status=1; done
exit "$status"
