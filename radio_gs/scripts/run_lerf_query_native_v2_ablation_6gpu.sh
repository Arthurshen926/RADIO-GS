#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/envs/cybersim_agent/bin/python
ROOT=/root/RADIO-GS
BASE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/query_native_gaussian_memory_v2/figurines32
FIELD=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/figurines/generic_text_response_w005_s0_64_lineage.pth
UNIVERSAL=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260816/universal_field_v1/figurines/universal_field_v1_shared_reliability.pth
MEMBERSHIP=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/figurines32/native_sam3_multiscale_memberships.pt
LANGUAGE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/figurines32/native_siglip2_sam_crop_teacher.pt
APPEARANCE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/native_multiteacher_v1/figurines32/native_dinov2_sam_proposal_teacher.pt
QUERY=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/figurines/primitive_query_method_v1.pth

mkdir -p "$BASE"

run_arm() {
  local gpu=$1 tag=$2 representation=$3 topk=$4 negative_max=$5 appearance=$6
  local output="$BASE/$tag.pt"
  if [[ -s "$output" && -s "$output.json" ]]; then
    echo "[$tag] complete; skipping immutable result"
    return 0
  fi
  if [[ -e "$output" || -e "$output.json" ]]; then
    echo "[$tag] incomplete output pair; refusing to overwrite: $output" >&2
    return 1
  fi
  local -a command=(
    "$PYTHON" -m radio_gs.scripts.train_evaluate_query_native_membership_decoder
    --scene figurines
    --field "$FIELD" --expected-field-sha256 9beeb9db4f91055ee17eaee4b85c60f790417fb9cc109772fea853b1c5b86e8b
    --universal-field "$UNIVERSAL" --expected-universal-field-sha256 e52bcced0d09a34a8ca1cd65e361a866916d0eec75f70c7fe2be19edf8615a8d
    --membership "$MEMBERSHIP" --expected-membership-sha256 e4fa35abc76c73b5e1e80daf761d63659c0d2cae2a9b7b371417f61d8836bee1
    --language-teacher "$LANGUAGE" --expected-language-teacher-sha256 b25ea55d6f55c32cda33effb9e611d1ce64e9514f7c6aadbcb37dad668402cbb
    --query-cache "$QUERY" --expected-query-cache-sha256 acc0b8b4cbf429d92e2f9df05865898066349fb79bcbe0bd3933ae1e504f1e18
    --output "$output" --device "cuda:$gpu"
    --memory-representation "$representation" --topk-anchors "$topk"
    --negative-semantic-max "$negative_max" --negative-max-support-iou 0.05
    --steps 800 --validation-interval 40 --validation-proposals 8
  )
  if [[ "$appearance" == yes ]]; then
    command+=(
      --appearance-teacher "$APPEARANCE"
      --expected-appearance-teacher-sha256 d86835e1a4a9fa2cdb41a2862c99020815d54b5fb80fbb4ddae0cad439fa4a7b
    )
  fi
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "${command[@]/cuda:$gpu/cuda:0}" >"$BASE/$tag.log" 2>&1
}

run_arm 0 local_k6_neg065 local_codes 6 0.65 no & p0=$!
run_arm 1 coefficient_k6_neg065 coefficients 6 0.65 no & p1=$!
run_arm 2 coefficient_k4_neg065 coefficients 4 0.65 no & p2=$!
run_arm 3 coefficient_k8_neg065 coefficients 8 0.65 no & p3=$!
run_arm 4 coefficient_k6_neg065_dino coefficients 6 0.65 yes & p4=$!
run_arm 5 radio_projected_k6_neg065 radio_projected 6 0.65 no & p5=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do
  wait "$pid" || status=1
done
exit "$status"
