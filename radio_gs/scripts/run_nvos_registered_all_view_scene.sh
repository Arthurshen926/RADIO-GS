#!/usr/bin/env bash
set -euo pipefail

# Native registered-all-view NVOS development compiler for one scene.  The
# plan is source/target-mask blind, official SAM3 owns region extent, and
# native SigLIP2 selects a fixed top-k mapping-view identity cohort.

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SCENE GPU" >&2
  exit 2
fi

SCENE=$1
GPU=$2
REPO_ROOT=${REPO_ROOT:-/root/RADIO-GS}
RUN_ROOT=${RUN_ROOT:-/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260824/nvos_registered_all_view_v1}
SCENE_ROOT=$RUN_ROOT/$SCENE
PLAN=$SCENE_ROOT/plan/candidate_plan.json
INVENTORY=$SCENE_ROOT/sam3_inventory/inventory.json
WEIGHTED=$SCENE_ROOT/native_siglip2_weighted_inventory.json
MARGINAL_ROOT=$SCENE_ROOT/marginal_siglip2_weighted_neutral_v2
MARGINAL=$MARGINAL_ROOT/primitive_posterior.pt
PREDICTION_ROOT=$SCENE_ROOT/prediction_siglip2_weighted_neutral_v2
PREDICTION_RECEIPT=$PREDICTION_ROOT/prediction_receipt.json
RESULT=$SCENE_ROOT/result_siglip2_weighted_neutral_v2.json
PROMPT_MANIFEST=/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1/nvos/method_v1_readout/full8_20260816/signed_field_prompt/prediction_manifest.json
DATASET_MANIFEST=/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/manifests/nvos_strict_unseen_v1.json
SIGLIP_ROOT=/root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots/a713301b217d38485fb2204c808367d10bc3cc40
SAM_CHECKPOINT=$REPO_ROOT/checkpoints/sam3_modelscope/sam3.pt
SAM_SHA=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e

mkdir -p "$SCENE_ROOT"

if [[ ! -s "$PLAN" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.build_nvos_synchronous_multiview_candidate_plan \
      --scene-id "$SCENE" \
      --signed-field-prompt-manifest "$PROMPT_MANIFEST" \
      --output-dir "$SCENE_ROOT/plan" --candidates 10 --points-per-sign 3 \
      --all-registered-views --device cuda:0
fi

if [[ ! -s "$INVENTORY" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU \
    bash "$REPO_ROOT/radio_gs/scripts/run_official_sam3_python.sh" \
      -m radio_gs.scripts.run_nvos_multiscene_box_inventory \
      --scene "$SCENE" --plan-root "$RUN_ROOT" --output-root "$RUN_ROOT" \
      --checkpoint "$SAM_CHECKPOINT" \
      --expected-checkpoint-sha256 "$SAM_SHA" --device cuda:0
fi

if [[ ! -s "$WEIGHTED" ]]; then
  PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
  INVENTORY_SHA=$(sha256sum "$INVENTORY" | awk '{print $1}')
  CUDA_VISIBLE_DEVICES=$GPU \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.build_nvos_native_siglip_view_reliability \
      --inventory "$INVENTORY" --expected-inventory-sha256 "$INVENTORY_SHA" \
      --plan "$PLAN" --expected-plan-sha256 "$PLAN_SHA" \
      --native-siglip2-model "$SIGLIP_ROOT" --mapping-top-k 4 \
      --appearance-temperature 16 --batch-size 2 --device cuda:0 \
      --output "$WEIGHTED"
fi

if [[ ! -s "$MARGINAL" ]]; then
  WEIGHTED_SHA=$(sha256sum "$WEIGHTED" | awk '{print $1}')
  CUDA_VISIBLE_DEVICES=$GPU \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.materialize_nvos_synchronous_candidate_marginal \
      --inventory "$WEIGHTED" --expected-inventory-sha256 "$WEIGHTED_SHA" \
      --output-dir "$MARGINAL_ROOT" --expected-candidates 1 \
      --view-fusion positive_unknown_noisy_or --device cuda:0
fi

if [[ ! -s "$PREDICTION_RECEIPT" ]]; then
  PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
  MARGINAL_SHA=$(sha256sum "$MARGINAL" | awk '{print $1}')
  CUDA_VISIBLE_DEVICES=$GPU \
    bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
      -m radio_gs.scripts.render_nvos_synchronous_candidate_marginal \
      --plan "$PLAN" --expected-plan-sha256 "$PLAN_SHA" \
      --marginal "$MARGINAL" --expected-marginal-sha256 "$MARGINAL_SHA" \
      --output-dir "$PREDICTION_ROOT" --device cuda:0
fi

if [[ ! -s "$RESULT" ]]; then
  bash "$REPO_ROOT/radio_gs/scripts/run_repo_python.sh" \
    -m radio_gs.scripts.score_nvos_synchronous_candidate_batch \
    --manifest "$DATASET_MANIFEST" --receipt "$PREDICTION_RECEIPT" \
    --output "$RESULT"
fi

chmod -R o+rwX "$SCENE_ROOT"
