# LERF Direct 3D Silhouette Threshold Sweep

LERF direct 3D selection silhouette threshold sweep. Same cached VPR primitive scores, meanstd2p5 selector, selection floor=0.005 cap=0.018, voxel_max aggregation, and GT-free RGB snap; only rendered selected-primitive silhouette threshold changes.

| Silhouette | Macro mIoU | Delta mIoU vs 0.70 | Macro Acc@0.25 | Macro Acc@0.50 | Weighted mIoU |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 0.4548 | +0.0049 | 0.7040 | 0.4544 | 0.4932 |
| 0.60 | 0.4554 | +0.0055 | 0.7014 | 0.4663 | 0.4932 |
| 0.65 | 0.4539 | +0.0040 | 0.7085 | 0.4506 | 0.4906 |
| 0.70 | 0.4499 | +0.0000 | 0.6939 | 0.4663 | 0.4858 |
| 0.75 | 0.4349 | -0.0150 | 0.6942 | 0.4398 | 0.4718 |

Per-scene mIoU:

| Silhouette | Figurines | Ramen | Teatime | Waldo |
|---:|---:|---:|---:|---:|
| 0.55 | 0.5516 | 0.4707 | 0.5607 | 0.2361 |
| 0.60 | 0.5484 | 0.4706 | 0.5621 | 0.2406 |
| 0.65 | 0.5457 | 0.4698 | 0.5546 | 0.2455 |
| 0.70 | 0.5347 | 0.4717 | 0.5450 | 0.2481 |
| 0.75 | 0.5157 | 0.4733 | 0.5172 | 0.2336 |

Best macro silhouette: 0.60 (macro mIoU 0.4554, Acc@0.25 0.7014).
