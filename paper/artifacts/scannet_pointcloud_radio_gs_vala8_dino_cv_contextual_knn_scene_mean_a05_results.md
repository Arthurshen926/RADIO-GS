# RADIO-GS VALA8 DINO-CV contextual kNN scene-mean alpha0.5 (Previous Diagnostic)

This artifact is retained for provenance only. The current paper-facing ScanNet
row is `scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn_scene_mean_a045_results`.

Protocol: VALA-aligned ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3704 | 0.6017 |
| 15 classes | 0.3771 | 0.6198 |
| 10 classes | 0.4585 | 0.7032 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3049/0.5838 | 0.2858/0.5738 | 0.3404/0.6789 |
| scene0062_00 | 0.3988/0.7315 | 0.3988/0.7315 | 0.5530/0.8315 |
| scene0070_00 | 0.2168/0.3117 | 0.2346/0.3411 | 0.3584/0.5222 |
| scene0097_00 | 0.4605/0.7597 | 0.4360/0.7462 | 0.4903/0.7093 |
| scene0140_00 | 0.3466/0.5370 | 0.3978/0.6137 | 0.4204/0.6718 |
| scene0347_00 | 0.5773/0.7344 | 0.5558/0.7245 | 0.6827/0.7795 |
| scene0400_00 | 0.3914/0.7126 | 0.3914/0.7126 | 0.4290/0.7797 |
| scene0590_00 | 0.2670/0.4428 | 0.3167/0.5154 | 0.3939/0.6530 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1500 | picture 0.0251 | refrigerator 0.2613 |
| 15 classes | 0.1494 | table 0.1240 | door 0.2401 |
| 10 classes | 0.1542 | toilet 0.2216 | window 0.2545 |
