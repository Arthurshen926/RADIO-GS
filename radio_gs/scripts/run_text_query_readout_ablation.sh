#!/usr/bin/env bash
set -euo pipefail

# Two predeclared, label-free readouts on one frozen canonical field.
# This is a diagnostic of score-scale compatibility, not a test-set sweep.

physical_gpu="${1:?physical GPU index is required}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python="/root/miniconda3/envs/cybersim_agent/bin/python"
script="radio_gs/scripts/eval_lerf_direct_3d_selection.py"
config="radio_gs/configs/lerf_hybrid_v14_ramen_fdh_ws240_240ep.yaml"
checkpoint="output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth"
labels="/mnt/pool/sqy/3d_understanding/lerf_ovs/label"
text_cache="checkpoints/frozen_protocol/siglip2_lerf_all_queries_raw.pt"
negative_cache="checkpoints/frozen_protocol/siglip2_lerf_negatives_raw.pt"
summary_head="checkpoints/siglip2_summary_head.pth"
root="output/optimization_20260725/text_readout"

common=(
  "${python}" "${script}"
  --scene ramen
  --config "${config}"
  --checkpoint "${checkpoint}"
  --label_dir "${labels}"
  --summary_head_weights "${summary_head}"
  --text_embedding_cache "${text_cache}"
  --canonical_embedding_cache "${negative_cache}"
  --prompt_templates "{query}"
  --text_encoder siglip2
  --score_source direct
  --direct_readout_mode gaussian
  --direct_readout_k 8
  --compact_feature_key features
  --selection_refinement none
  --gpu 0
)

repo_output="${root}/vala_repo_3d"
repo_result="${repo_output}/ramen/lerf_direct_3d_selection_results.json"
if [[ ! -s "${repo_result}" ]]; then
  mkdir -p "${repo_output}"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" "${common[@]}" \
    --output_dir "${repo_output}" \
    --protocol_preset vala_repo_3d \
    > "${repo_output}/run.log" 2>&1
fi

adaptive_output="${root}/mean_std_1"
adaptive_result="${adaptive_output}/ramen/lerf_direct_3d_selection_results.json"
if [[ ! -s "${adaptive_result}" ]]; then
  mkdir -p "${adaptive_output}"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" "${common[@]}" \
    --output_dir "${adaptive_output}" \
    --protocol_preset none \
    --selection_mode mean_std \
    --mean_std 1.0 \
    --scoring relevancy \
    --score_postprocess none \
    > "${adaptive_output}/run.log" 2>&1
fi

# Reuse the same released GPU for the frozen-unary PFIR readout immediately
# after the text diagnostic.  This avoids a second waiter racing the first
# job onto the same physical device.
bash radio_gs/scripts/run_pfir_support_readout_ablation.sh "${physical_gpu}"
