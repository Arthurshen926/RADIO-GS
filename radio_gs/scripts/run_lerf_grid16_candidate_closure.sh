#!/usr/bin/env bash
set -euo pipefail

# Close one denser, query-independent source-SAM proposal experiment without
# reserving an otherwise useful GPU.  Each stage waits for enough free memory,
# commits its artifact atomically, and resumes from the first absent artifact.

SCENE=${1:-figurines}
ROOT=${ROOT:-/root/RADIO-GS}
SOURCE_ROOT=${SOURCE_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260820/lerf_multiscale_sam3_source32_grid16}
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260820/lerf_sam_siglip_object_posterior_grid16_v1}
FIELD_ROOT=${FIELD_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1}
PYTHON=${PYTHON:-/root/miniconda3/envs/cybersim_agent/bin/python}
GPU_SET=${GPU_SET:-4,5}
MINIMUM_FREE_MIB=${MINIMUM_FREE_MIB:-6000}
POLL_SECONDS=${POLL_SECONDS:-60}

case "$SCENE" in
  figurines)
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/exact_marginal_responsibility_authority.json
    ;;
  ramen)
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/ramen/fix4c_exact_marginal_v1/exact_marginal_responsibility_authority.json
    ;;
  teatime)
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/teatime/exact_marginal_target_v1/exact_marginal_responsibility_authority.json
    ;;
  waldo_kitchen)
    MPR=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/exact_marginal_responsibility_authority.json
    ;;
  *)
    echo "unsupported LERF scene: $SCENE" >&2
    exit 2
    ;;
esac

SCENE_ROOT="$SOURCE_ROOT/$SCENE"
MANIFEST="$SCENE_ROOT/manifest.json"
MEMBERSHIP="$SCENE_ROOT/gaussian_multiscale_memberships.pt"
TEACHER="$SCENE_ROOT/mask_aligned_siglip2_crop_summary_teacher.pt"
PRIMITIVE="$FIELD_ROOT/$SCENE/primitive_query_method_v1.pth"
SOURCE_AUTHORITY="$ROOT/paper/artifacts/lerf_${SCENE}_source32_exact_mpr_rgb_authority_20260817.json"
LOG_ROOT="$RUN_ROOT/logs"
LOG="$LOG_ROOT/${SCENE}_closure.log"
mkdir -p "$LOG_ROOT"

timestamp() {
  date -Is
}

best_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | tr -d ' ' \
    | awk -F, -v allowed="$GPU_SET" '
        BEGIN { split(allowed, ids, ","); for (i in ids) keep[ids[i]] = 1 }
        keep[$1] && $2 >= best_free { best = $1; best_free = $2 }
        END { if (best != "") print best "," best_free }
      '
}

run_with_available_gpu() {
  local label=$1
  shift
  while true; do
    local selection index free
    selection=$(best_gpu)
    if [[ -n "$selection" ]]; then
      IFS=, read -r index free <<<"$selection"
      if (( free >= MINIMUM_FREE_MIB )); then
        echo "[$(timestamp)] $label: trying GPU $index with ${free} MiB free" | tee -a "$LOG"
        if CUDA_VISIBLE_DEVICES="$index" "$@" >>"$LOG" 2>&1; then
          echo "[$(timestamp)] $label: complete" | tee -a "$LOG"
          return 0
        fi
        echo "[$(timestamp)] $label: failed; artifact remains uncommitted, retrying" | tee -a "$LOG"
      fi
    fi
    sleep "$POLL_SECONDS"
  done
}

while [[ ! -s "$MANIFEST" ]]; do
  echo "[$(timestamp)] waiting for sealed grid16 manifest" >>"$LOG"
  sleep "$POLL_SECONDS"
done

test -f "$MPR"
test -f "$PRIMITIVE"
test -f "$SOURCE_AUTHORITY"

if [[ ! -s "$MEMBERSHIP" ]]; then
  run_with_available_gpu exact_mpr_membership \
    "$PYTHON" -m radio_gs.scripts.build_lerf_multiscale_sam3_exact_mpr_memberships \
      --scene "$SCENE" \
      --responsibility-authority "$MPR" \
      --primitive-cache "$PRIMITIVE" \
      --source-authority "$SOURCE_AUTHORITY" \
      --mask-root "$SCENE_ROOT" \
      --min-membership 0.5 \
      --device cuda:0 \
      --output "$MEMBERSHIP"
fi

if [[ ! -s "$TEACHER" ]]; then
  run_with_available_gpu official_crop_summary_teacher \
    "$PYTHON" -m radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher \
      --scene "$SCENE" \
      --mask-root "$SCENE_ROOT" \
      --radio-checkpoint /root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar \
      --radio-checkpoint-sha256 bace44df72e750bc8555ea6979cc19d1a87e12ade89582edfe090513d5d6aab9 \
      --radio-repo /root/RADIO \
      --radio-version c-radio_v4-h \
      --context-expansion 1.5 \
      --crop-resolution 384 \
      --batch-size 2 \
      --device cuda:0 \
      --output "$TEACHER"
fi

SCORES="$RUN_ROOT/scores/$SCENE.pt"
while [[ ! -s "$RUN_ROOT/lerf2d/${SCENE}_eval/lerf_ovs_results.json" \
      || ! -s "$RUN_ROOT/lerf3d/$SCENE/$SCENE/lerf_direct_3d_selection_results.json" ]]; do
  selection=$(best_gpu)
  if [[ -n "$selection" ]]; then
    IFS=, read -r index free <<<"$selection"
    if (( free >= MINIMUM_FREE_MIB )); then
      echo "[$(timestamp)] typed_readout: trying GPU $index with ${free} MiB free" | tee -a "$LOG"
      if RUN_ROOT="$RUN_ROOT" SOURCE_ROOT="$SOURCE_ROOT" FIELD_ROOT="$FIELD_ROOT" \
        SCORE_DEVICE="cuda:$index" EVAL_GPU="$index" \
        bash "$ROOT/radio_gs/scripts/run_lerf_sam_siglip_object_posterior_v3_scene.sh" \
          "$SCENE" >>"$LOG" 2>&1; then
        break
      fi
      echo "[$(timestamp)] typed_readout: failed; retrying from sealed artifacts" | tee -a "$LOG"
    fi
  fi
  sleep "$POLL_SECONDS"
done

test -s "$SCORES"
chmod -R o+rwX "$SCENE_ROOT" "$RUN_ROOT"
echo "[$(timestamp)] grid16 candidate closure complete" | tee -a "$LOG"
