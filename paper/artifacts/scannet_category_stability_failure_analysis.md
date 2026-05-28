# ScanNet VALA8 Category Stability / Failure Analysis

Source: `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json`

This table supports the paper appendix row for the promoted ScanNet DINO-CV
contextual kNN readout. Statistics are per-class IoU across scenes where the
class appears.

| Split | Weakest class | Scenes | Mean IoU | IoU range | Most unstable class | Std IoU | IoU range | Reading |
|---|---|---:|---:|---:|---|---:|---:|---|
| 19 | picture | 3 | 0.0177 | 0.0000-0.0528 | refrigerator | 0.2969 | 0.0041-0.8273 | small/decorative regions and scene-dependent appliances |
| 15 | table | 4 | 0.1262 | 0.0000-0.3738 | door | 0.2378 | 0.0148-0.7864 | broad cluttered support surfaces and geometry-dependent openings |
| 10 | toilet | 2 | 0.2158 | 0.1138-0.3179 | window | 0.2443 | 0.1798-0.8674 | sparse class support and strong view/lighting variation |

Conclusion: the promoted ScanNet row is stable enough as cross-domain
direct-query evidence, but category-macro reporting remains important because
the aggregate gains can hide weak or sparse classes such as picture/table/toilet.
