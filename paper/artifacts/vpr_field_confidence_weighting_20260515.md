# VPR-to-Field Confidence Weighting Audit

Date: 2026-05-15

## Method Change

Added GT-free registration-confidence weighting to
`train_scannet_point_summary_adapter.py`. For VPR feature caches with
`view_counts`, each valid Gaussian receives a normalized `log1p(view_counts)`
sample weight. The weight is applied consistently to summary reconstruction,
text KL distillation, pseudo-label CE, and decoder-anchor terms.

The promoted setting uses:

- `--teacher_sample_weight_mode log`
- `--teacher_sample_weight_min 0.0`
- point-summary adapter blend `alpha=0.5` at evaluation
- same score-threshold sweep, 0.5% selection floor, 1.8% cap, silhouette 0.60,
  and GT-free RGB GrabCut boundary snap as the previous VPR-to-field row

## Results

| Variant | Figurines | Ramen | Teatime | Waldo | Macro mIoU | Macro Acc@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| VPR-to-field previous | 0.4877 | 0.4381 | 0.5103 | 0.2115 | 0.4119 | 0.5876 |
| VPR-to-field + log confidence | 0.4037 | 0.5975 | 0.5196 | 0.2245 | 0.4363 | 0.6191 |
| Registered VPR readout | 0.5307 | 0.5796 | 0.5659 | 0.2433 | 0.4799 | 0.6760 |

The confidence-weighted field improves the direct-field macro mIoU by +0.0244
and Acc@0.25 by +0.0315 over the previous VPR-to-field row. The strongest gain
is on Ramen; Figurines remains a failure case for this transfer objective.

## Negative Follow-Ups

- Lowering point-summary adapter blend to `alpha=0.3` reduced both tested scenes
  (Figurines 0.3416, Ramen 0.5067), so it is not promoted.
- Raising the log-weight floor to `2.0` did not recover Figurines
  (0.4039) and reduced Ramen (0.5511), so it is not promoted.

## Conclusion

Promote `log` confidence-weighted VPR-to-field as the current strongest
direct-field transfer result. Keep streamed registered VPR as the main
direct-3D readout because it remains stronger and more stable across scenes.
