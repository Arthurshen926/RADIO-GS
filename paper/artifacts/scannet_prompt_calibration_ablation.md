# ScanNet Prompt and Calibration Ablation

This report records the Expert (5) ScanNet follow-up. All rows use the same
10-scene v67 teacher-balanced direct point-query protocol as the main ScanNet
table. The original conservative rows use `gaussian_index` queries,
`label_point` positions, `label_index` opacity filtering, and the fixed prompt template bank
`{query}|a photo of a {query}|a 3d scan of a {query}|a point cloud of a {query}|an indoor scene containing a {query}`.
The 2026-05-14 contextual rows use the same checkpoints and text bank but query
label vertices with `query_mode=knn`, `k=8`, `candidate_k=32`, default point
opacity filtering, and scene-mean calibration.

| Variant | split19 mIoU / mAcc | split15 mIoU / mAcc | split10 mIoU / mAcc | Conclusion |
|---|---:|---:|---:|---|
| v67 baseline | 0.3538 / 0.6076 | 0.3573 / 0.6203 | 0.4293 / 0.7051 | Main conservative row |
| scene-mean calibration, alpha=0.5 | 0.3575 / 0.6101 | 0.3604 / 0.6227 | 0.4353 / 0.7074 | Positive label-free ablation |
| contextual kNN + scene-mean alpha=0.5 | 0.3637 / 0.6033 | 0.3708 / 0.6224 | 0.4512 / 0.7079 | Strongest balanced support row |
| contextual kNN + scene-mean alpha=0.75 | 0.3620 / 0.5994 | 0.3692 / 0.6187 | 0.4534 / 0.7078 | Higher split10 mIoU but weaker balance |
| ScanNet aliases | 0.3592 / 0.6191 | 0.3561 / 0.6192 | 0.4234 / 0.7002 | Mixed; not promoted |
| aliases + scene-mean alpha=0.5 | 0.3617 / 0.6180 | 0.3554 / 0.6174 | 0.4295 / 0.7026 | Mixed; not promoted |
| scene-mean calibration, alpha=1.0 | 0.3528 / 0.5834 | 0.3541 / 0.5935 | 0.4386 / 0.7048 | Hurts 19/15 and mAcc |

Recommended paper use: keep the Gaussian-index baseline as the conservative
protocol anchor, and report contextual kNN + scene-mean alpha=0.5 as the
stronger balanced direct point-query support row. Keep alpha=0.75,
aliases/stronger calibration, and label-informed diagnostics as appendix or
negative/mixed evidence.

Artifacts:

- `output/scannet_pointcloud_eval/v67_scene_mean_a05_gidx_labelpoint_20260513/scannet_pointcloud_radio_gs_results.json`
- `output/scannet_pointcloud_eval/v67_alias_scannet_gidx_labelpoint_20260513/scannet_pointcloud_radio_gs_results.json`
- `output/scannet_pointcloud_eval/v67_alias_scene_mean_a05_gidx_labelpoint_20260513/scannet_pointcloud_radio_gs_results.json`
- `output/scannet_pointcloud_eval/v67_scene_mean_a10_gidx_labelpoint_20260513/scannet_pointcloud_radio_gs_results.json`
- `output/scannet_pointcloud_eval/v67_knn8_cand32_scene_mean_a05_20260514/scannet_pointcloud_radio_gs_results.json`
- `output/scannet_pointcloud_eval/v67_knn8_cand32_scene_mean_a075_20260514/scannet_pointcloud_radio_gs_results.json`
