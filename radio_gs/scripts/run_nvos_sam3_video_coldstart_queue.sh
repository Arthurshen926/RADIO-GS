#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU SCENE..." >&2
  exit 2
fi
GPU=$1
shift
REPO=${REPO:-/root/RADIO-GS}
MANIFEST=/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json
CHECKPOINT=$REPO/checkpoints/sam3_modelscope/sam3.pt
OUTPUT=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/nvos_sam3_video_prompt_proposal_coldstart_v4
MANIFEST_SHA=bafc48ce30a0a637f5ea4d81a196ea240f80c153c41a3e257b6a2fd45fa3f2ea
CHECKPOINT_SHA=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
for scene in "$@"; do
  target=$OUTPUT/$scene
  if [[ -s "$target/prediction_receipt.json" ]]; then
    continue
  fi
  mkdir -p "$target"
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    /root/miniconda3/envs/sam3/bin/python \
      -m radio_gs.scripts.predict_nvos_sam3_video_from_official_scribble \
      --scene "$scene" --manifest "$MANIFEST" \
      --expected-manifest-sha256 "$MANIFEST_SHA" \
      --checkpoint "$CHECKPOINT" --expected-checkpoint-sha256 "$CHECKPOINT_SHA" \
      --output-dir "$target" --prompt-proposal-seed --device cuda:0 \
      >"$target/run.log" 2>&1
  chmod -R o+rwX "$target"
done
