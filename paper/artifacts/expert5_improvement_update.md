# Expert Suggestion (5) Improvement Update

This pass followed `ChatGPT-项目总结与优化 (5).md` and focused on the three
highest-return directions: DINOv3 propagation mIoU, LERF direct-3D Waldo
diagnostics, and ScanNet prompt/calibration support. All promoted settings are
fixed global rules and do not use per-query GT threshold tuning.

## Effective Updates

| Area | Previous paper-facing result | New result | Conclusion |
|---|---:|---:|---|
| DINOv3 rendered mask propagation | 0.7730 LocAcc / 0.4289 mIoU | 0.7730 LocAcc / 0.4456 mIoU | Fixed 2.0x area scaling improves mIoU without hurting LocAcc |
| DINOv3 same-readout teacher | 0.7801 LocAcc / 0.4512 mIoU | 0.7801 LocAcc / 0.4806 mIoU | Teacher also improves, so the claim remains gap reduction |
| ScanNet scene-mean calibration, alpha=0.5 | 0.3538/0.3573/0.4293 mIoU | 0.3575/0.3604/0.4353 mIoU | Positive label-free support row |
| ScanNet contextual kNN + scene-mean alpha=0.5 | 0.3538/0.3573/0.4293 mIoU | 0.3637/0.3708/0.4512 mIoU | Strongest balanced direct point-query support row |
| LERF direct 3D VPR view budget | 0.4133 mIoU / 0.6741 Acc@0.25 | 0.4185 mIoU / 0.6899 Acc@0.25 | 128 all-pose VPR views improve both fixed paper metrics under the 2% cap |
| LERF direct 3D global cap | 0.4185 mIoU / 0.6899 Acc@0.25 | 0.4227 mIoU / 0.6906 Acc@0.25 | 1.8% cap is promoted as the paper-facing row; 1.75% is a near-tie diagnostic at 0.4226 / 0.6864 and 1.5% gives 0.4184 / 0.7013 |

The DINO update uses `output/lerf_sam_dino_tasks/formal_v6_dino_topk_area200_peak/`.
The earlier formal_v4 rendered DINO mask propagation was 0.7376 LocAcc /
0.3684 mIoU, so the robust readout now gives a large absolute mIoU gain while
keeping the correct same-protocol caveat.

## Negative or Mixed Updates

| Area | Variant | Result | Decision |
|---|---|---:|---|
| DINOv3 readout | score-sum component cleanup | 0.7730 / 0.4204 | Worse than peak cleanup; do not promote |
| DINOv3 readout | largest-component cleanup | 0.7730 / 0.4144 | Worse than peak cleanup; do not promote |
| LERF direct 3D Waldo | top-score component keep-1 | 0.1757 mIoU | Worse than current 0.2217; negative ablation |
| LERF direct 3D Waldo | top-score component keep-2 | 0.1827 mIoU | Worse than current 0.2217; negative ablation |
| LERF direct 3D Waldo | voxel_mean | 0.0931 mIoU at meanstd2p5 | Strong regression; keep voxel_max |
| LERF direct 3D Waldo | voxel_max_dilate | 0.1286 mIoU at meanstd2p5 | Regression; keep voxel_max |
| LERF direct 3D | prompt ensemble | 0.4079 macro mIoU at meanstd2p5 | Worse than current 0.4227; do not promote |
| LERF direct 3D | registration confidence blend=0.25 | 0.4065 macro mIoU at meanstd2p5 | Slight Waldo gain but macro regression; do not promote |
| LERF direct 3D | registered-view fallback=low | 0.2692 mIoU / 0.4229 Acc@0.25 | Strong recall regression; keep direct fallback for unregistered primitives |
| LERF direct 3D | 160 all-pose VPR views | 0.4165 mIoU / 0.6613 Acc@0.25 | mIoU improves over 96-view row but Acc drops; 128 views is better balanced |
| ScanNet prompts | ScanNet aliases | 0.3592/0.3561/0.4234 mIoU | Helps split19 but hurts split15/10; not promoted |
| ScanNet calibration | scene-mean alpha=1.0 | 0.3528/0.3541/0.4386 mIoU | Helps split10 mIoU but hurts 19/15 and mAcc; not promoted |
| ScanNet contextual kNN | scene-mean alpha=0.75 | 0.3620/0.3692/0.4534 mIoU | Higher split10 mIoU but weaker 19/15 and mAcc than alpha=0.5 |

## Paper-Facing Recommendation

Keep the main method framing unchanged: GaussFM learns a compact teacher-feature
field with rendered-view and direct-3D readouts. Promote the formal_v6 DINO
robust readout as the latest downstream task result, keep 128-view VPR +
voxel-max + meanstd2p5 floor/cap as the direct-3D mainline, and report ScanNet contextual
kNN + scene-mean alpha=0.5 as the stronger balanced point-query support row
while retaining Gaussian-index results as the conservative protocol anchor.

Remaining non-local gaps are unchanged: strict same-evaluator external baseline
reproduction, true rasterizer-level contribution assignment, and a real frozen
SAM3 decoder/mask-logit distillation path.
