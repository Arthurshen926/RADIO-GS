#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/RADIO-GS"
cd "${REPO_ROOT}"

SCENES=(figurines ramen teatime waldo_kitchen)

config_for_scene() {
  case "$1" in
    figurines) echo "radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml" ;;
    ramen) echo "radio_gs/configs/lerf_hybrid_v14_ramen_fdh_ws240_240ep.yaml" ;;
    teatime) echo "radio_gs/configs/lerf_hybrid_v14_teatime_fdh_ws240_240ep.yaml" ;;
    waldo_kitchen) echo "radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep.yaml" ;;
    *) return 1 ;;
  esac
}

checkpoint_for_scene() {
  case "$1" in
    figurines) echo "output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/best.pth" ;;
    ramen) echo "output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/best.pth" ;;
    teatime) echo "output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep/checkpoints/best.pth" ;;
    waldo_kitchen) echo "output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/checkpoints/best.pth" ;;
    *) return 1 ;;
  esac
}

run_sweep() {
  local physical_gpu="$1"
  local weight="$2"
  local radius="$3"
  local outdir="$4"
  mkdir -p "${outdir}/logs"
  for scene in "${SCENES[@]}"; do
    local cfg ckpt log
    cfg="$(config_for_scene "${scene}")"
    ckpt="$(checkpoint_for_scene "${scene}")"
    log="${outdir}/logs/${scene}.log"
    mkdir -p "${outdir}/${scene}"
    if [[ -f "${outdir}/${scene}/lerf_sam_dino_task_results.json" ]]; then
      echo "[dino-transport] skip existing scene=${scene} out=${outdir}/${scene}"
      continue
    fi
    echo "[dino-transport] gpu=${physical_gpu} weight=${weight} radius=${radius} scene=${scene}"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" bash radio_gs/scripts/run_repo_python.sh \
      radio_gs/scripts/eval_lerf_sam_dino_tasks.py \
      --config "${cfg}" \
      --checkpoint "${ckpt}" \
      --scene "${scene}" \
      --output_dir "${outdir}/${scene}" \
      --max_visuals 0 \
      --dino_background_contrast 1.1 \
      --dino_foreground_pool topk_mean \
      --dino_area_scale 2.0 \
      --dino_component_cleanup peak \
      --dino_match_mutual \
      --dino_transport_match_weight "${weight}" \
      --dino_transport_match_radius "${radius}" \
      --dino_feature_boundary_refinement \
      --gpu 0 \
      > "${log}" 2>&1
  done
  bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/aggregate_lerf_sam_dino_tasks.py \
    "${outdir}"/{figurines,ramen,teatime,waldo_kitchen}/lerf_sam_dino_task_results.json \
    --output_dir "${outdir}" \
    --title "LERF SAM3/DINOv3 DINO Transport Review1 Sweep" \
    --note "Protocol: formal v9 DINO readout plus mutual-cycle transported match evidence and feature-boundary refinement. GT masks are used for prompts/support masks and final metrics only." \
    > "${outdir}/logs/aggregate.log" 2>&1
}

case "${1:-}" in
  w025)
    run_sweep "${2:-2}" 0.25 1 "output/lerf_sam_dino_tasks/formal_v11_dino_transport_w025_r1_boundary_20260527"
    ;;
  w050)
    run_sweep "${2:-2}" 0.5 1 "output/lerf_sam_dino_tasks/formal_v11_dino_transport_w050_r1_boundary_20260527"
    ;;
  w100)
    run_sweep "${2:-4}" 1.0 1 "output/lerf_sam_dino_tasks/formal_v11_dino_transport_w100_r1_boundary_20260527"
    ;;
  *)
    echo "Usage: $0 {w050|w100} [physical_gpu]" >&2
    exit 2
    ;;
esac
