#!/usr/bin/env bash

set -euo pipefail

ROOT=/root/RADIO-GS
SCENE=${SCENE:?set SCENE}
PHYSICAL_GPU=${PHYSICAL_GPU:?set PHYSICAL_GPU}
OUT_ROOT=${OUT_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260817/lerf_sam3_exact_mpr_memberships_v1}
SAM_ROOT="$ROOT/output/radio_gs/foundation_cache_sam3_modelscope_mapped_trainviews"

case "$SCENE" in
  figurines)
    RESPONSIBILITY=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260805/canonical_factorized_radio_v1/figurines/exact_marginal_v3/exact_marginal_responsibility_authority.json
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/figurines/primitive_query_method_v1.pth
    PRIMITIVE_SHA=acc0b8b4cbf429d92e2f9df05865898066349fb79bcbe0bd3933ae1e504f1e18
    ;;
  ramen)
    RESPONSIBILITY=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/ramen/fix4c_exact_marginal_v1/exact_marginal_responsibility_authority.json
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/ramen/primitive_query_method_v1.pth
    PRIMITIVE_SHA=893fda2a90142f71ee8175e666f12353a93e08a8125d8d5bdaf26d3a95dc54b5
    ;;
  teatime)
    RESPONSIBILITY=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/canonical_factorized_radio_v1/teatime/exact_marginal_target_v1/exact_marginal_responsibility_authority.json
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/teatime/primitive_query_method_v1.pth
    PRIMITIVE_SHA=3938c13cd5f2c78cc2522aeff26cb0f77ba08cbeb519288b4b564dffd629b96b
    ;;
  waldo_kitchen)
    RESPONSIBILITY=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/exact_marginal_responsibility_authority.json
    PRIMITIVE=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/waldo_kitchen/primitive_query_method_v1.pth
    PRIMITIVE_SHA=01ffe08e54466dc0da720bcc2e25029ae2b085e24e78f8ac5ad9ced28085159f
    ;;
  *)
    echo "unsupported scene: $SCENE" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_ROOT/logs"
OUTPUT="$OUT_ROOT/$SCENE.pt"
LOG="$OUT_ROOT/logs/$SCENE.log"
if [[ -e "$OUTPUT" || -e "$OUTPUT.json" || -e "$LOG" ]]; then
  echo "refusing to overwrite an existing output or log for $SCENE" >&2
  exit 3
fi

cd "$ROOT"
CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" \
  bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.scripts.build_lerf_sam3_exact_mpr_memberships \
  --scene "$SCENE" \
  --responsibility-authority "$RESPONSIBILITY" \
  --sam3-cache-root "$SAM_ROOT" \
  --primitive-cache "$PRIMITIVE" \
  --expected-primitive-cache-sha256 "$PRIMITIVE_SHA" \
  --min-membership 0.50 \
  --device cuda:0 \
  --output "$OUTPUT" \
  >"$LOG" 2>&1
sha256sum "$OUTPUT" "$OUTPUT.json"
