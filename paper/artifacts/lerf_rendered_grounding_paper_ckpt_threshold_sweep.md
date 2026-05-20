# Paper Checkpoint Rendered Grounding Threshold Sweep

Paper-facing LERF rendered grounding checkpoint threshold sweep. Same scene-specific paper checkpoints, SigLIP2 text embeddings, temperatures, softmax_scene scoring, and heatmap upsample; only peak-relative mask threshold changes.

| Threshold | Macro LocAcc | Macro mIoU | Delta Macro mIoU | Weighted mIoU | Delta Weighted mIoU |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.8712 | 0.4941 | +0.0000 | 0.5151 | +0.0000 |
| 0.55 | 0.8712 | 0.5151 | +0.0210 | 0.5333 | +0.0182 |
| 0.60 | 0.8712 | 0.5243 | +0.0303 | 0.5397 | +0.0246 |
| 0.65 | 0.8712 | 0.5253 | +0.0313 | 0.5386 | +0.0234 |
| 0.70 | 0.8712 | 0.5063 | +0.0122 | 0.5182 | +0.0031 |
| 0.75 | 0.8712 | 0.4710 | -0.0230 | 0.4846 | -0.0306 |

Per-scene rendered mIoU:

| Threshold | Figurines | Ramen | Teatime | Waldo |
|---:|---:|---:|---:|---:|
| 0.50 | 0.4308 | 0.5862 | 0.5486 | 0.4106 |
| 0.55 | 0.4302 | 0.6113 | 0.5682 | 0.4505 |
| 0.60 | 0.4244 | 0.6201 | 0.5760 | 0.4769 |
| 0.65 | 0.4148 | 0.6187 | 0.5778 | 0.4901 |
| 0.70 | 0.3954 | 0.5963 | 0.5561 | 0.4775 |
| 0.75 | 0.3669 | 0.5563 | 0.5298 | 0.4311 |

Best macro threshold: 0.65 (macro mIoU 0.5253).
Safer non-over-shrinking threshold <=0.65: 0.65 (macro mIoU 0.5253).
