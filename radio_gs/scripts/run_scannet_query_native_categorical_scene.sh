#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SCENE GPU" >&2
  exit 2
fi
SCENE=$1
GPU=$2
REPO=${REPO:-/root/RADIO-GS}
ROOT=/mnt/pool/sqy/results/RADIO-GS/output
CORE=$ROOT/optimization_20260815/core_method_v1/$SCENE
NATIVE=$ROOT/optimization_20260824/native_multiteacher_v1/$SCENE/native_sam_siglip
VARIANT=${VARIANT:-query_native_categorical_v1}
RUN=$ROOT/optimization_20260824/query_native_gaussian_memory/$SCENE/$VARIANT
FIELD=$CORE/generic_text_response_w005_s0_64.pth
BASE=$ROOT/optimization_20260823/scannet_semantic_ladder/restored_direct_capability/$SCENE/primitive_query_restored_direct_capability.pt
MEMBERSHIP=$NATIVE/native_sam3_multiscale_memberships.pt
TEACHER=$NATIVE/native_siglip2_sam_crop_teacher.pt
UNIVERSAL=$ROOT/optimization_20260816/universal_field_v1/$SCENE/universal_field_v1.pth
CACHE=$RUN/source_distilled_scores.pt
RESULT=$RUN/eval/scannet_vala_gaussian_protocol_results.json

case "$SCENE" in
  scene0070_00) CHECKPOINT=$ROOT/radio_gs/scannet_og_scene0070_00_v14_b1/checkpoints/best.pth ;;
  scene0400_00) CHECKPOINT=$ROOT/radio_gs/scannet_og_scene0400_00_v67_dino_cv001_b2_s32768_ft20/checkpoints/best.pth ;;
  *) CHECKPOINT=$ROOT/radio_gs/scannet_og_${SCENE}_v14/checkpoints/best.pth ;;
esac
CONFIG=$REPO/radio_gs/configs/generated/frozen_eval_20260802/scannet_${SCENE%_00}_canonical_mpr_v3_paper8.yaml
LABEL=/mnt/pool/sqy/3d_understanding/scannet_og/$SCENE/${SCENE}_vh_clean_2.labels.ply
P19=$REPO/checkpoints/siglip2_scannet_og_text_embeddings_ens5_split19.pt
P15=$REPO/checkpoints/siglip2_scannet_og_text_embeddings_ens5_split15.pt
P10=$REPO/checkpoints/siglip2_scannet_og_text_embeddings_ens5_split10.pt
R19=$REPO/checkpoints/siglip2_scannet_og_text_embeddings_exact_split19.pt
R15=$REPO/checkpoints/siglip2_scannet_og_text_embeddings_exact_split15.pt
R10=$REPO/checkpoints/siglip2_scannet_og_text_embeddings_exact_split10.pt
for value in "$FIELD" "$BASE" "$MEMBERSHIP" "$TEACHER" "$UNIVERSAL" "$CHECKPOINT" "$CONFIG" "$LABEL" "$P19" "$P15" "$P10" "$R19" "$R15" "$R10"; do
  [[ -s "$value" ]] || { echo "incomplete input: $value" >&2; exit 1; }
done
mkdir -p "$RUN"
exec 9>"$RUN/.materialize.lock"
flock 9
sha() { sha256sum "$1" | awk '{print $1}'; }
if [[ ! -s "$CACHE" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    bash "$REPO/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.train_scannet_query_native_categorical_decoder \
      --scene "$SCENE" --field "$FIELD" --expected-field-sha256 "$(sha "$FIELD")" \
      --universal-field "$UNIVERSAL" --expected-universal-field-sha256 "$(sha "$UNIVERSAL")" \
      --membership "$MEMBERSHIP" --expected-membership-sha256 "$(sha "$MEMBERSHIP")" \
      --proposal-teacher "$TEACHER" --expected-proposal-teacher-sha256 "$(sha "$TEACHER")" \
      --baseline-query-cache "$BASE" --expected-baseline-query-cache-sha256 "$(sha "$BASE")" \
      --primitive-text-banks "$P19,$P15,$P10" \
      --expected-primitive-text-sha256 "$(sha "$P19"),$(sha "$P15"),$(sha "$P10")" \
      --region-text-banks "$R19,$R15,$R10" \
      --expected-region-text-sha256 "$(sha "$R19"),$(sha "$R15"),$(sha "$R10")" \
      --output-model "$RUN/decoder.pt" --output-score-cache "$CACHE" \
      --device cuda:0 --steps "${STEPS:-900}" --gate-steps "${GATE_STEPS:-900}" \
      --hidden-dim "${HIDDEN_DIM:-192}" --pair-hidden-dim "${PAIR_HIDDEN_DIM:-48}" \
      --split-training-sequence "${SPLIT_TRAINING_SEQUENCE:-19,15,10}" \
      --query-holdout-modulus "${QUERY_HOLDOUT_MODULUS:-0}" \
      --query-holdout-residue "${QUERY_HOLDOUT_RESIDUE:-0}"
fi
if [[ ! -s "$RESULT" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU bash "$REPO/radio_gs/scripts/run_repo_python.sh" \
    "$REPO/radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py" \
    --scene_list "$SCENE" --prepared_root /mnt/pool/sqy/3d_understanding/scannet_og \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --label_ply "$LABEL" \
    --output_dir "$RUN/eval" --class_splits 19,15,10 \
    --feature_chunk_size 8192 --pseudo_chunk_size 512 \
    --pseudo_gt_cache_dir "$CORE/scannet_vala_method_v1/pseudo_gt" \
    --radius_factor 5.0 --candidate_k 1000 --fallback_k 1 \
    --row_opacity_threshold 0.1 --compact_feature_key features \
    --prompt_templates '{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}' \
    --text_embedding_cache "$REPO/checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt" \
    --text_encoder siglip2 --class_aliases none \
    --projection_weights "$REPO/checkpoints/siglip2_feat_projection.pth" \
    --summary_head_weights "$REPO/checkpoints/siglip2_summary_head.pth" \
    --external_query_score_cache "$CACHE" \
    --expected_external_query_score_cache_sha256 "$(sha "$CACHE")" --device cuda
fi
chmod -R o+rwX "$RUN"
