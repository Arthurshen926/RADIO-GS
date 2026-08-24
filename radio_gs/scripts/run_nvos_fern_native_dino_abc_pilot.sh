#!/usr/bin/env bash
set -euo pipefail

# Independent NVOS confirmation for the same-capacity native-DINO A/B/C
# source-heldout sentinel.  This runner never opens the registered prompt,
# target RGB, benchmark mask, or target metric.

ROOT=${ROOT:-/root/RADIO-GS}
GPU=${GPU:-1}
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/nvos_fern}
CORE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/nvos/fern
SOURCE_MANIFEST="$CORE/source_only_features/frame_manifest.json"
MPR="$CORE/exact_marginal_responsibility_heldout4.json"
FIELD="$CORE/generic_text_response_w005_s0_64.pth"
DINO="$RUN_ROOT/native_dinov2_exact_mpr_trainval.pt"
RESUME="$RUN_ROOT/native_dinov2_frames"
ABC="$RUN_ROOT/native_dinov2_abc_matched_v2"
LOG="$RUN_ROOT/native_dino_abc_closure.log"

mkdir -p "$RUN_ROOT"

if [[ ! -s "$DINO" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.build_native_dinov2_exact_mpr_teacher \
      --source-frame-manifest "$SOURCE_MANIFEST" \
      --exact-mpr-authority "$MPR" \
      --checkpoint /root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth \
      --validation-stride 5 \
      --validation-offset 0 \
      --channel-chunk-size 96 \
      --row-chunk-size 8192 \
      --device cuda:0 \
      --resume-root "$RESUME" \
      --output "$DINO" >>"$LOG" 2>&1
fi

if [[ ! -s "$ABC/abc_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$ROOT/radio_gs/scripts/run_repo_python.sh" \
      "$ROOT/radio_gs/scripts/train_native_multiteacher_abc_pilot.py" \
      --base-field "$FIELD" \
      --native-teacher "$DINO" \
      --output-dir "$ABC" \
      --device cuda:0 \
      --steps 800 \
      --batch-size 1024 \
      --radio-weight 1.0 \
      --seed 0 >>"$LOG" 2>&1
fi

chmod -R o+rwX "$RUN_ROOT"
