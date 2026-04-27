#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
MAX_UTIL="${MAX_UTIL:-40}"
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
LOG_ROOT="${LOG_ROOT:-output/radio_gs/pipeline_queue}"
MARKER_ROOT="$LOG_ROOT/markers"
AGG_EVAL_ROOT="$LOG_ROOT/lerf_eval_aggregate"

mkdir -p "$LOG_ROOT" "$MARKER_ROOT" "output/radio_gs/oracle_heads"
shopt -s nullglob
for stale_log in "$LOG_ROOT"/*.log; do
  if [[ "$(basename "$stale_log")" != "launcher.log" ]]; then
    rm -f "$stale_log"
  fi
done
shopt -u nullglob
rm -f "$MARKER_ROOT"/*.done
rm -rf "$AGG_EVAL_ROOT"
mkdir -p "$AGG_EVAL_ROOT"

queue_gpu_job() {
  local log_path="$1"
  shift
  nohup env \
    GPU_LIST="$GPU_LIST" \
    MIN_FREE_MIB="$MIN_FREE_MIB" \
    MAX_UTIL="$MAX_UTIL" \
    CHECK_INTERVAL="$CHECK_INTERVAL" \
    bash radio_gs/scripts/wait_and_run.sh "$log_path" "$@" \
    >/dev/null 2>&1 &
  echo $!
}

queue_file_job() {
  local target_file="$1"
  local log_path="$2"
  shift 2
  nohup env \
    GPU_LIST="$GPU_LIST" \
    MIN_FREE_MIB="$MIN_FREE_MIB" \
    MAX_UTIL="$MAX_UTIL" \
    CHECK_INTERVAL="$CHECK_INTERVAL" \
    bash radio_gs/scripts/wait_for_file_and_run.sh "$target_file" "$log_path" "$@" \
    >/dev/null 2>&1 &
  echo $!
}

queue_eval_copy_job() {
  local source_json="$1"
  local target_dir="$2"
  local log_path="$3"
  queue_file_job \
    "$source_json" \
    "$log_path" \
    bash radio_gs/scripts/run_and_mark_success.sh "$target_dir/.copied.done" \
      cp "$source_json" "$target_dir/lerf_ovs_results.json"
}

REPLICA_REBUILD_LOG="$LOG_ROOT/replica_rebuild.log"
LERF_REBUILD_LOG="$LOG_ROOT/lerf_rebuild.log"
REPLICA_REBUILD_MARKER="$MARKER_ROOT/replica_rebuild.done"
LERF_REBUILD_MARKER="$MARKER_ROOT/lerf_rebuild.done"

ROOM0_DEPTH_HEAD_MARKER="$MARKER_ROOT/room0_depth_head.done"
ROOM0_SEG_HEAD_MARKER="$MARKER_ROOT/room0_seg_head.done"
ROOM0_NOFDH_MARKER="$MARKER_ROOT/room0_nofdh.done"
ROOM0_PURE_MARKER="$MARKER_ROOT/room0_pure_frozen.done"
ROOM0_PURE_DEPTH_MARKER="$MARKER_ROOT/room0_pure_frozen_depth_only.done"
ROOM0_PURE_EVAL_MARKER="$MARKER_ROOT/room0_pure_frozen_eval.done"
ROOM0_PURE_DEPTH_EVAL_MARKER="$MARKER_ROOT/room0_pure_frozen_depth_only_eval.done"
ROOM0_FEATURES_READY="output/radio_features_1280d_reextract_20260407/room_0/Sequence_2/frame_manifest.json"

replica_pid=$(queue_gpu_job "$REPLICA_REBUILD_LOG" bash radio_gs/scripts/run_and_mark_success.sh "$REPLICA_REBUILD_MARKER" bash radio_gs/scripts/rebuild_3dgs_assets.sh replica)
lerf_pid=$(queue_gpu_job "$LERF_REBUILD_LOG" bash radio_gs/scripts/run_and_mark_success.sh "$LERF_REBUILD_MARKER" bash radio_gs/scripts/rebuild_3dgs_assets.sh lerf)

ROOM0_DEPTH_HEAD="output/radio_gs/oracle_heads/room_0_seq1_depth_head.pth"
ROOM0_SEG_HEAD="output/radio_gs/oracle_heads/room_0_seq1_seg_head.pth"
ROOM0_BASE_CKPT="output/radio_gs/room0_hybrid_v14_nofdh_240ep/checkpoints/best.pth"
ROOM0_PURE_OUT="output/radio_gs/room0_hybrid_v14_pure_frozen"
ROOM0_PURE_DEPTH_OUT="output/radio_gs/room0_hybrid_v14_pure_frozen_depth_only"

queue_file_job \
  "$REPLICA_REBUILD_MARKER" \
  "$LOG_ROOT/pretrain_room0_depth_head.log" \
  bash radio_gs/scripts/wait_and_run.sh "$LOG_ROOT/run_room0_depth_head.log" \
  bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_DEPTH_HEAD_MARKER" \
    python radio_gs/scripts/pretrain_oracle_head.py \
      --feature_dir output/radio_features_1280d_reextract_20260407/room_0/Sequence_1 \
      --depth_dir /mnt/pool/sqy/dataset/room_0/Sequence_1/depth \
      --val_feature_dir output/radio_features_1280d_reextract_20260407/room_0/Sequence_2 \
      --val_depth_dir /mnt/pool/sqy/dataset/room_0/Sequence_2/depth \
      --output_path output/radio_gs/oracle_heads/room_0_seq1_depth_head.pth \
      --epochs 500 --batch_size 8 --validate_every 10 --log_every 25 --feature_size 60,80 --gpu 0

queue_file_job \
  "$ROOM0_FEATURES_READY" \
  "$LOG_ROOT/pretrain_room0_depth_head_early.log" \
  bash radio_gs/scripts/wait_and_run.sh "$LOG_ROOT/run_room0_depth_head_early.log" \
  bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_DEPTH_HEAD_MARKER" \
    python radio_gs/scripts/pretrain_oracle_head.py \
      --feature_dir output/radio_features_1280d_reextract_20260407/room_0/Sequence_1 \
      --depth_dir /mnt/pool/sqy/dataset/room_0/Sequence_1/depth \
      --val_feature_dir output/radio_features_1280d_reextract_20260407/room_0/Sequence_2 \
      --val_depth_dir /mnt/pool/sqy/dataset/room_0/Sequence_2/depth \
      --output_path output/radio_gs/oracle_heads/room_0_seq1_depth_head.pth \
      --epochs 500 --batch_size 8 --validate_every 10 --log_every 25 --feature_size 60,80 --gpu 0

queue_file_job \
  "$REPLICA_REBUILD_MARKER" \
  "$LOG_ROOT/pretrain_room0_seg_head.log" \
  bash radio_gs/scripts/wait_and_run.sh "$LOG_ROOT/run_room0_seg_head.log" \
  bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_SEG_HEAD_MARKER" \
    python radio_gs/scripts/pretrain_oracle_seg_head.py \
      --feature_dir output/radio_features_1280d_reextract_20260407/room_0/Sequence_1 \
      --semantics_dir /mnt/pool/sqy/dataset/room_0/Sequence_1/semantic_class \
      --val_feature_dir output/radio_features_1280d_reextract_20260407/room_0/Sequence_2 \
      --val_semantics_dir /mnt/pool/sqy/dataset/room_0/Sequence_2/semantic_class \
      --output_path output/radio_gs/oracle_heads/room_0_seq1_seg_head.pth \
      --epochs 300 --batch_size 4 --validate_every 10 --log_every 25 --feature_size 60,80 --num_classes 101 --gpu 0

queue_file_job \
  "$ROOM0_FEATURES_READY" \
  "$LOG_ROOT/pretrain_room0_seg_head_early.log" \
  bash radio_gs/scripts/wait_and_run.sh "$LOG_ROOT/run_room0_seg_head_early.log" \
  bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_SEG_HEAD_MARKER" \
    python radio_gs/scripts/pretrain_oracle_seg_head.py \
      --feature_dir output/radio_features_1280d_reextract_20260407/room_0/Sequence_1 \
      --semantics_dir /mnt/pool/sqy/dataset/room_0/Sequence_1/semantic_class \
      --val_feature_dir output/radio_features_1280d_reextract_20260407/room_0/Sequence_2 \
      --val_semantics_dir /mnt/pool/sqy/dataset/room_0/Sequence_2/semantic_class \
      --output_path output/radio_gs/oracle_heads/room_0_seq1_seg_head.pth \
      --epochs 300 --batch_size 4 --validate_every 10 --log_every 25 --feature_size 60,80 --num_classes 101 --gpu 0

queue_file_job \
  "$REPLICA_REBUILD_MARKER" \
  "$LOG_ROOT/queue_room0_nofdh.log" \
  bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_NOFDH_MARKER" \
    bash radio_gs/scripts/wait_and_train.sh \
      radio_gs/configs/replica_hybrid_v14_room_0_nofdh_240ep.yaml \
      output/radio_gs/queue_room0_nofdh_240ep.log

queue_file_job \
  "$ROOM0_FEATURES_READY" \
  "$LOG_ROOT/queue_room0_nofdh_early.log" \
  bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_NOFDH_MARKER" \
    bash radio_gs/scripts/wait_and_train.sh \
      radio_gs/configs/replica_hybrid_v14_room_0_nofdh_240ep.yaml \
      output/radio_gs/queue_room0_nofdh_240ep.log

queue_file_job \
  "$ROOM0_NOFDH_MARKER" \
  "$LOG_ROOT/queue_room0_pure_frozen_after_base.log" \
  bash radio_gs/scripts/wait_for_file_and_run.sh \
    "$ROOM0_DEPTH_HEAD_MARKER" \
    "$LOG_ROOT/queue_room0_pure_frozen_after_heads.log" \
    bash radio_gs/scripts/wait_for_file_and_run.sh \
      "$ROOM0_SEG_HEAD_MARKER" \
      "$LOG_ROOT/queue_room0_pure_frozen_ready.log" \
      bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_PURE_MARKER" \
        bash radio_gs/scripts/wait_and_train.sh \
          radio_gs/configs/replica_hybrid_v14_room_0_pure_frozen.yaml \
          output/radio_gs/queue_room0_pure_frozen.log

queue_file_job \
  "$ROOM0_NOFDH_MARKER" \
  "$LOG_ROOT/queue_room0_pure_frozen_depth_only_after_base.log" \
  bash radio_gs/scripts/wait_for_file_and_run.sh \
    "$ROOM0_DEPTH_HEAD_MARKER" \
    "$LOG_ROOT/queue_room0_pure_frozen_depth_only_ready.log" \
    bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_PURE_DEPTH_MARKER" \
      bash radio_gs/scripts/wait_and_train.sh \
        radio_gs/configs/replica_hybrid_v14_room_0_pure_frozen_depth_only.yaml \
        output/radio_gs/queue_room0_pure_frozen_depth_only.log

declare -A LERF_NOFDH=(
  [figurines]="radio_gs/configs/lerf_hybrid_v14_figurines_nofdh_240ep.yaml"
  [ramen]="radio_gs/configs/lerf_hybrid_v14_ramen_nofdh_240ep.yaml"
  [teatime]="radio_gs/configs/lerf_hybrid_v14_teatime_nofdh_240ep.yaml"
  [waldo_kitchen]="radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_nofdh_240ep.yaml"
)

declare -A LERF_PURE=(
  [figurines]="radio_gs/configs/lerf_hybrid_v14_figurines_pure_frozen_depth_only.yaml"
  [ramen]="radio_gs/configs/lerf_hybrid_v14_ramen_pure_frozen_depth_only.yaml"
  [teatime]="radio_gs/configs/lerf_hybrid_v14_teatime_pure_frozen_depth_only.yaml"
  [waldo_kitchen]="radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_pure_frozen_depth_only.yaml"
)

declare -A LERF_FDH=(
  [figurines]="radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml"
  [ramen]="radio_gs/configs/lerf_hybrid_v14_ramen_fdh_ws240_240ep.yaml"
  [teatime]="radio_gs/configs/lerf_hybrid_v14_teatime_fdh_ws240_240ep.yaml"
  [waldo_kitchen]="radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep.yaml"
)

declare -A LERF_BASE_CKPT=(
  [figurines]="output/radio_gs/lerf_figurines_v14_nofdh_240ep/checkpoints/best.pth"
  [ramen]="output/radio_gs/lerf_ramen_v14_nofdh_240ep/checkpoints/best.pth"
  [teatime]="output/radio_gs/lerf_teatime_v14_nofdh_240ep/checkpoints/best.pth"
  [waldo_kitchen]="output/radio_gs/lerf_waldo_kitchen_v14_nofdh_240ep/checkpoints/best.pth"
)

declare -A LERF_PURE_OUT=(
  [figurines]="output/radio_gs/lerf_figurines_v14_pure_frozen_depth_only"
  [ramen]="output/radio_gs/lerf_ramen_v14_pure_frozen_depth_only"
  [teatime]="output/radio_gs/lerf_teatime_v14_pure_frozen_depth_only"
  [waldo_kitchen]="output/radio_gs/lerf_waldo_kitchen_v14_pure_frozen_depth_only"
)

declare -A LERF_FDH_OUT=(
  [figurines]="output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep"
  [ramen]="output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep"
  [teatime]="output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep"
  [waldo_kitchen]="output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep"
)

declare -A LERF_NOFDH_MARKER=()
declare -A LERF_PURE_MARKER=()
declare -A LERF_FDH_MARKER=()
declare -A LERF_PURE_EVAL_MARKER=()
declare -A LERF_FDH_EVAL_MARKER=()
declare -A LERF_FDH_COPY_MARKER=()

for scene in figurines ramen teatime waldo_kitchen; do
  LERF_NOFDH_MARKER[$scene]="$MARKER_ROOT/${scene}_nofdh.done"
  LERF_PURE_MARKER[$scene]="$MARKER_ROOT/${scene}_pure_frozen_depth_only.done"
  LERF_FDH_MARKER[$scene]="$MARKER_ROOT/${scene}_fdh_ws240_240ep.done"
  LERF_PURE_EVAL_MARKER[$scene]="$MARKER_ROOT/${scene}_pure_eval.done"
  LERF_FDH_EVAL_MARKER[$scene]="$MARKER_ROOT/${scene}_fdh_eval.done"
  LERF_FDH_COPY_MARKER[$scene]="$AGG_EVAL_ROOT/$scene/rendered/.copied.done"
done

for scene in figurines ramen teatime waldo_kitchen; do
  mkdir -p "$AGG_EVAL_ROOT/$scene/rendered"

  queue_file_job \
    "$LERF_REBUILD_MARKER" \
    "$LOG_ROOT/queue_${scene}_nofdh.log" \
    bash radio_gs/scripts/run_and_mark_success.sh "${LERF_NOFDH_MARKER[$scene]}" \
      bash radio_gs/scripts/wait_and_train.sh \
        "${LERF_NOFDH[$scene]}" \
        "output/radio_gs/queue_${scene}_nofdh_240ep.log"

  queue_file_job \
    "output/radio_features_lerf/${scene}/traj_w_c.txt" \
    "$LOG_ROOT/queue_${scene}_nofdh_early.log" \
    bash radio_gs/scripts/run_and_mark_success.sh "${LERF_NOFDH_MARKER[$scene]}" \
      bash radio_gs/scripts/wait_and_train.sh \
        "${LERF_NOFDH[$scene]}" \
        "output/radio_gs/queue_${scene}_nofdh_240ep.log"

  queue_file_job \
    "${LERF_NOFDH_MARKER[$scene]}" \
    "$LOG_ROOT/queue_${scene}_pure_after_base.log" \
    bash radio_gs/scripts/wait_for_file_and_run.sh \
      "$ROOM0_DEPTH_HEAD_MARKER" \
      "$LOG_ROOT/queue_${scene}_pure_after_depth_head.log" \
      bash radio_gs/scripts/run_and_mark_success.sh "${LERF_PURE_MARKER[$scene]}" \
        bash radio_gs/scripts/wait_and_train.sh \
          "${LERF_PURE[$scene]}" \
          "output/radio_gs/queue_${scene}_pure_frozen_depth_only.log"

  queue_file_job \
    "${LERF_NOFDH_MARKER[$scene]}" \
    "$LOG_ROOT/queue_${scene}_fdh_after_base.log" \
    bash radio_gs/scripts/wait_for_file_and_run.sh \
      "$ROOM0_DEPTH_HEAD_MARKER" \
      "$LOG_ROOT/queue_${scene}_fdh_after_depth_head.log" \
      bash radio_gs/scripts/run_and_mark_success.sh "${LERF_FDH_MARKER[$scene]}" \
        bash radio_gs/scripts/wait_and_train.sh \
          "${LERF_FDH[$scene]}" \
          "output/radio_gs/queue_${scene}_fdh_ws240_240ep.log"
done

queue_file_job \
  "$ROOM0_PURE_MARKER" \
  "$LOG_ROOT/autoeval_room0_pure.log" \
  bash radio_gs/scripts/wait_and_run.sh "$LOG_ROOT/run_autoeval_room0_pure.log" \
    bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_PURE_EVAL_MARKER" \
      bash radio_gs/scripts/auto_eval_room0_pure_frozen.sh none

queue_file_job \
  "$ROOM0_PURE_DEPTH_MARKER" \
  "$LOG_ROOT/autoeval_room0_pure_depth_only.log" \
  bash radio_gs/scripts/wait_and_run.sh "$LOG_ROOT/run_autoeval_room0_pure_depth_only.log" \
    bash radio_gs/scripts/run_and_mark_success.sh "$ROOM0_PURE_DEPTH_EVAL_MARKER" \
      env \
        CONFIG=radio_gs/configs/replica_hybrid_v14_room_0_pure_frozen_depth_only.yaml \
        OUTPUT_DIR="$ROOM0_PURE_DEPTH_OUT" \
        bash radio_gs/scripts/auto_eval_room0_pure_frozen.sh none

for scene in figurines ramen teatime waldo_kitchen; do
  queue_file_job \
    "${LERF_PURE_MARKER[$scene]}" \
    "$LOG_ROOT/autoeval_${scene}_pure.log" \
    bash radio_gs/scripts/wait_and_run.sh "$LOG_ROOT/run_autoeval_${scene}_pure.log" \
      bash radio_gs/scripts/run_and_mark_success.sh "${LERF_PURE_EVAL_MARKER[$scene]}" \
        bash radio_gs/scripts/auto_eval_lerf_after_train.sh none "${LERF_PURE[$scene]}" "${LERF_PURE_OUT[$scene]}"

  queue_file_job \
    "${LERF_FDH_MARKER[$scene]}" \
    "$LOG_ROOT/autoeval_${scene}_fdh.log" \
    bash radio_gs/scripts/wait_and_run.sh "$LOG_ROOT/run_autoeval_${scene}_fdh.log" \
      bash radio_gs/scripts/run_and_mark_success.sh "${LERF_FDH_EVAL_MARKER[$scene]}" \
        bash radio_gs/scripts/auto_eval_lerf_after_train.sh none "${LERF_FDH[$scene]}" "${LERF_FDH_OUT[$scene]}"

  queue_eval_copy_job \
    "${LERF_FDH_OUT[$scene]}/lerf_eval_best/best/lerf_ovs_results.json" \
    "$AGG_EVAL_ROOT/$scene/rendered" \
    "$LOG_ROOT/copy_${scene}_fdh_eval.log"
done

queue_file_job \
  "${LERF_FDH_COPY_MARKER[figurines]}" \
  "$LOG_ROOT/build_submission_tables_after_figurines.log" \
  bash radio_gs/scripts/wait_for_file_and_run.sh \
    "${LERF_FDH_COPY_MARKER[ramen]}" \
    "$LOG_ROOT/build_submission_tables_after_ramen.log" \
    bash radio_gs/scripts/wait_for_file_and_run.sh \
      "${LERF_FDH_COPY_MARKER[teatime]}" \
      "$LOG_ROOT/build_submission_tables_after_teatime.log" \
      bash radio_gs/scripts/wait_for_file_and_run.sh \
        "${LERF_FDH_COPY_MARKER[waldo_kitchen]}" \
        "$LOG_ROOT/build_submission_tables_ready.log" \
        python radio_gs/scripts/build_submission_tables.py --lerf_eval_dir output/radio_gs

cat <<EOF
Queued pipeline jobs.
- Replica rebuild waiter PID: $replica_pid
- LERF rebuild waiter PID: $lerf_pid
- Logs root: $LOG_ROOT
EOF
