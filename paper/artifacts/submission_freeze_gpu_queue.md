# Submission Freeze GPU Queue

Date: 2026-05-02

## Running Jobs

| GPU | PID | Purpose | Profile Dir | Launch Log |
|---:|---:|---|---|---|
| 4 | 2904231 | LERF Figurines formal overlay/profile at frozen T50 | `output/radio_gs/profiles/freeze_lerf_figurines_overlay_20260502` | `output/radio_gs/reports/freeze_lerf_figurines_overlay_20260502.launch.log` |
| 5 | 2904229 | ScanNet v67 10-scene fair direct point re-eval/profile | `output/radio_gs/profiles/freeze_scannet_v67_all_eval_20260502` | `output/radio_gs/reports/freeze_scannet_v67_all_eval_20260502.launch.log` |
| 4 | 2905383 | LERF Waldo Kitchen formal overlay/profile at frozen T25 | `output/radio_gs/profiles/freeze_lerf_waldo_overlay_20260502` | `output/radio_gs/reports/freeze_lerf_waldo_overlay_20260502.launch.log` |
| 4 | 2906850 | LERF Ramen formal overlay/profile at frozen T40 | `output/radio_gs/profiles/freeze_lerf_ramen_overlay_20260502` | `output/radio_gs/reports/freeze_lerf_ramen_overlay_20260502.launch.log` |
| 5 | 2906856 | LERF Teatime formal overlay/profile at frozen T25 | `output/radio_gs/profiles/freeze_lerf_teatime_overlay_20260502` | `output/radio_gs/reports/freeze_lerf_teatime_overlay_20260502.launch.log` |

## Commands

```bash
SELECTED_GPU=4 bash radio_gs/scripts/profile_command.sh \
  --gpu 4 \
  --output_dir output/radio_gs/profiles/freeze_lerf_figurines_overlay_20260502 \
  -- bash radio_gs/scripts/run_repo_python.sh -m radio_gs.scripts.eval_lerf_grounding \
  --config radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml \
  --checkpoint output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/latest.pth \
  --scene figurines \
  --output_dir output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502 \
  --text_embedding_cache checkpoints/siglip2_lerf_text_embeddings.pt \
  --scoring softmax_scene \
  --relevancy_temp 50 \
  --heatmap_upsample 4 \
  --save_vis \
  --save_overlay_vis \
  --save_per_query_vis \
  --gpu 4
```

```bash
SELECTED_GPU=5 bash radio_gs/scripts/profile_command.sh \
  --gpu 5 \
  --output_dir output/radio_gs/profiles/freeze_scannet_v67_all_eval_20260502 \
  -- bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py \
  --scene all \
  --prepared_root dataset/scannet_og \
  --config radio_gs/configs/generated/scannet_og/scannet_og_hybrid_v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63_{scene}.yaml \
  --checkpoint output/radio_gs/scannet_og_{scene}_v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63/checkpoints/best.pth \
  --output_dir output/scannet_pointcloud_eval/freeze_v67_all_eval_20260502 \
  --class_splits 19,15,10 \
  --query_mode gaussian_index \
  --gaussian_index_position_mode label_point \
  --opacity_filter_mode label_index \
  --opacity_threshold 0.1 \
  --save_logits_npz \
  --save_feature_rgb_ply \
  --text_embedding_cache checkpoints/siglip2_scannet_og_text_embeddings_ens5.pt \
  --use_summary_head \
  --device cuda:5
```

```bash
SELECTED_GPU=4 bash radio_gs/scripts/profile_command.sh \
  --gpu 4 \
  --output_dir output/radio_gs/profiles/freeze_lerf_waldo_overlay_20260502 \
  -- bash radio_gs/scripts/run_repo_python.sh -m radio_gs.scripts.eval_lerf_grounding \
  --config radio_gs/configs/generated/seeds/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed7.yaml \
  --checkpoint output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth \
  --scene waldo_kitchen \
  --output_dir output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502 \
  --text_embedding_cache checkpoints/siglip2_lerf_text_embeddings.pt \
  --scoring softmax_scene \
  --relevancy_temp 25 \
  --heatmap_upsample 4 \
  --save_vis \
  --save_overlay_vis \
  --save_per_query_vis \
  --gpu 4
```

```bash
SELECTED_GPU=4 bash radio_gs/scripts/profile_command.sh \
  --gpu 4 \
  --output_dir output/radio_gs/profiles/freeze_lerf_ramen_overlay_20260502 \
  -- bash radio_gs/scripts/run_repo_python.sh -m radio_gs.scripts.eval_lerf_grounding \
  --config radio_gs/configs/generated/seeds/lerf_hybrid_v14_ramen_fdh_ws240_240ep_seed7.yaml \
  --checkpoint output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth \
  --scene ramen \
  --output_dir output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502 \
  --text_embedding_cache checkpoints/siglip2_lerf_text_embeddings.pt \
  --scoring softmax_scene \
  --relevancy_temp 40 \
  --heatmap_upsample 4 \
  --save_vis \
  --save_overlay_vis \
  --save_per_query_vis \
  --gpu 4
```

```bash
SELECTED_GPU=5 bash radio_gs/scripts/profile_command.sh \
  --gpu 5 \
  --output_dir output/radio_gs/profiles/freeze_lerf_teatime_overlay_20260502 \
  -- bash radio_gs/scripts/run_repo_python.sh -m radio_gs.scripts.eval_lerf_grounding \
  --config radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed7.yaml \
  --checkpoint output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth \
  --scene teatime \
  --output_dir output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502 \
  --text_embedding_cache checkpoints/siglip2_lerf_text_embeddings.pt \
  --scoring softmax_scene \
  --relevancy_temp 25 \
  --heatmap_upsample 4 \
  --save_vis \
  --save_overlay_vis \
  --save_per_query_vis \
  --gpu 5
```
