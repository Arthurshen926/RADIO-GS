# ScanNet VALA8 Category Stability / Failure Analysis

Source: `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_results.json`

This table supports the paper appendix row for the promoted ScanNet DINO-CV
contextual kNN readout. Statistics are per-class IoU across scenes where the
class appears.

| Split | Weakest class | Scenes | Mean IoU | IoU range | Most unstable class | Std IoU | IoU range | Reading |
|---|---|---:|---:|---:|---|---:|---:|---|
| 19 | picture | 3 | 0.0247 | 0.0000-0.0740 | refrigerator | 0.2628 | 0.0170-0.7366 | small/decorative regions and scene-dependent appliances |
| 15 | table | 4 | 0.1223 | 0.0000-0.3560 | door | 0.2393 | 0.0141-0.7818 | broad cluttered support surfaces and geometry-dependent openings |
| 10 | toilet | 2 | 0.2203 | 0.1070-0.3337 | window | 0.2558 | 0.1445-0.8609 | sparse class support and strong view/lighting variation |

Conclusion: the promoted ScanNet row is stable enough as cross-domain
direct-query evidence, but category-macro reporting remains important because
the aggregate gains can hide weak or sparse classes such as picture/table/toilet.
