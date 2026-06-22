# LERF Rendered Grounding Adaptive Threshold Diagnostic

GT-free rendered grounding adaptive threshold diagnostic: threshold_mode=mean_std, k=1.0, clamp=[0.50,0.70]. Same paper-facing checkpoints, temperatures, text embeddings, scoring and heatmap upsample as main table.

| Scene | LocAcc | mIoU | Frame-wise RADIO mIoU |
|---|---:|---:|---:|
| figurines | 0.8214 | 0.4309 | 0.4308 |
| ramen | 0.9014 | 0.5863 | 0.5883 |
| teatime | 0.8983 | 0.5565 | 0.5255 |
| waldo_kitchen | 0.8636 | 0.4019 | 0.3925 |

Macro LocAcc: `0.8712`
Macro mIoU: `0.4939`

Negative diagnostic: adaptive mean+std does not beat the fixed threshold-0.60 calibrated main row (0.5243 macro mIoU). Keep fixed threshold-0.60 as paper-facing readout.
