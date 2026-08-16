#!/usr/bin/env bash

set -euo pipefail

ROOT=/root/RADIO-GS
OUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260816/lerf3d_peak_retention_guard_full4_v1
SCENE=${SCENE:?set SCENE to figurines, ramen, teatime, or waldo_kitchen}
PHYSICAL_GPU=${PHYSICAL_GPU:?set PHYSICAL_GPU to an authorized free GPU}

case "$SCENE" in
  figurines)
    CONFIG="$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_figurines_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth
    ;;
  ramen)
    CONFIG="$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_ramen_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth
    ;;
  teatime)
    CONFIG="$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_teatime_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth
    ;;
  waldo_kitchen)
    CONFIG="$ROOT/radio_gs/configs/generated/frozen_eval_20260802/lerf_waldo_kitchen_radio_verified_pose.yaml"
    CHECKPOINT=/mnt/pool/sqy/results/RADIO-GS/output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth
    ;;
  *)
    echo "unsupported scene: $SCENE" >&2
    exit 2
    ;;
esac

RESULT="$OUT/$SCENE/$SCENE/lerf_direct_3d_selection_results.json"
if [[ -f "$RESULT" ]]; then
  sha256sum "$RESULT"
  exit 0
fi

SCENE_OUT="$OUT/$SCENE"
if [[ -d "$SCENE_OUT" ]] && [[ -n "$(find "$SCENE_OUT" -mindepth 1 -print -quit)" ]]; then
  echo "partial output must be audited before resume: $SCENE_OUT" >&2
  exit 3
fi
mkdir -p "$OUT/.locks" "$OUT/logs"
LOG="$OUT/logs/${SCENE}.log"
if [[ -e "$LOG" ]]; then
  echo "existing log must be audited before rerun: $LOG" >&2
  exit 3
fi
LOCK="$OUT/.locks/$SCENE"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "scene is already running: $SCENE ($LOCK)" >&2
  exit 4
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
mkdir -p "$SCENE_OUT"
CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
  bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
  "$ROOT/radio_gs/scripts/eval_lerf_direct_3d_selection.py" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --scene "$SCENE" \
  --protocol_preset vala_repo_3d \
  --vala_post_mask_refinement peak_component_retention_guard \
  --label_dir /mnt/pool/sqy/3d_understanding/lerf_ovs/label \
  --output_dir "$SCENE_OUT" \
  --summary_head_weights "$ROOT/checkpoints/siglip2_summary_head.pth" \
  --text_embedding_cache "$ROOT/checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt" \
  --canonical_embedding_cache "$ROOT/checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt" \
  --gpu 0 \
  >"$LOG" 2>&1

sha256sum "$RESULT"
