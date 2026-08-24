#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/envs/cybersim_agent/bin/python
ROOT=/root/RADIO-GS
BASE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/query_native_gaussian_memory
SCENES=scene0000_00,scene0062_00,scene0070_00,scene0097_00,scene0140_00,scene0347_00,scene0400_00,scene0590_00
TEXT=/root/RADIO-GS/checkpoints/siglip2_scannet_og_text_embeddings_ens5_split19.pt,/root/RADIO-GS/checkpoints/siglip2_scannet_og_text_embeddings_ens5_split15.pt,/root/RADIO-GS/checkpoints/siglip2_scannet_og_text_embeddings_ens5_split10.pt
TEXT_SHA=098b67081c33bc9c6c8d021217189e7cf1fd33aa5603d4680754dd3fc2bf8d89,3811b3bc98c6931dd47db8f9eb699927b1351a02f4507980da5d627908645ceb,18999df2a6d0ed8e272c9d15cde9b4a6de3e3f6880b79cc4848bcf0d6a88eead

run_arm() {
  local gpu=$1 tag=$2 rank=$3 seed=$4 factorized=${5:-no}
  local output="$BASE/$tag"
  if [[ -s "$output/source_gate.json" && -s "$output/shared_decoder.pt" ]]; then
    echo "[$tag] complete; skipping immutable result"
    return 0
  fi
  if [[ -e "$output/source_gate.json" || -e "$output/shared_decoder.pt" ]]; then
    echo "[$tag] incomplete output pair; refusing to overwrite" >&2
    return 1
  fi
  cd "$ROOT"
  local -a command=(
    "$PYTHON" -m radio_gs.scripts.train_scannet_query_native_shared_decoder \
      --scenes "$SCENES" \
      --primitive-text-banks "$TEXT" \
      --expected-primitive-text-sha256 "$TEXT_SHA" \
      --output-dir "$output" --device cuda:0 \
      --memory-representation coefficients \
      --hidden-dim 192 --pair-hidden-dim 48 \
      --scene-canonicalizer-rank "$rank" \
      --query-holdout-modulus 4 --query-holdout-residue 0 \
      --steps 2400 --gate-steps 1600 --seed "$seed"
  )
  if [[ "$factorized" == yes ]]; then
    command+=(--factorized-identity-competition)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "${command[@]}" >"$BASE/$tag.log" 2>&1
}

run_arm 0 shared_coeff_queryholdout4_seed24_v2 0 20260824 & p0=$!
run_arm 1 shared_coeff_canon8_queryholdout4_seed24_v2 8 20260824 & p1=$!
status=0
for pid in "$p0" "$p1"; do
  wait "$pid" || status=1
done
if [[ "$status" == 0 ]]; then
  # Loading all eight fields dominates host-memory pressure.  Stability seeds
  # run after the two primary arms rather than duplicating four full cohorts.
  run_arm 2 shared_coeff_canon8_queryholdout4_seed25_v2 8 20260825 || status=1
  run_arm 3 shared_coeff_canon8_queryholdout4_seed26_v2 8 20260826 || status=1
  run_arm 4 shared_coeff_canon8_factorized_queryholdout4_seed24_v3 8 20260824 yes || status=1
fi
exit "$status"
