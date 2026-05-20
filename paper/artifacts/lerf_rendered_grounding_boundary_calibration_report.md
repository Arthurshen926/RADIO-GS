# LERF Rendered Grounding Boundary Calibration Report

LERF rendered-view grounding readout calibration. All variants use the same checkpoints, temperatures, SigLIP2 text cache, softmax_scene scoring, and heatmap upsample. Threshold variants change only the peak-relative binary mask threshold and do not use GT masks. RGB-snap uses GT-free GrabCut initialized from predicted masks/RGB.

| Variant | Macro LocAcc | Macro mIoU | Delta Macro mIoU | Weighted LocAcc | Weighted mIoU | Delta Weighted mIoU |
|---|---:|---:|---:|---:|---:|---:|
| thr0.5_base | 0.8327 | 0.4573 | +0.0000 | 0.8462 | 0.4786 | +0.0000 |
| thr0.6 | 0.8327 | 0.4994 | +0.0422 | 0.8462 | 0.5183 | +0.0397 |
| thr0.7 | 0.8327 | 0.5049 | +0.0476 | 0.8462 | 0.5187 | +0.0401 |
| rgb_snap | 0.8327 | 0.4570 | -0.0002 | 0.8462 | 0.4788 | +0.0002 |

Per-scene rendered mIoU:

| Scene | Base 0.5 | Thr 0.6 | Thr 0.7 | RGB snap |
|---|---:|---:|---:|---:|
| figurines | 0.4312 | 0.4264 | 0.3957 | 0.4683 |
| ramen | 0.5935 | 0.6233 | 0.5972 | 0.5789 |
| teatime | 0.4206 | 0.5086 | 0.5605 | 0.4079 |
| waldo_kitchen | 0.3837 | 0.4395 | 0.4661 | 0.3730 |

Conclusion: global threshold 0.6 is the most stable calibrated 2D rendered-mask readout here: it improves macro mIoU by +0.0422 and weighted mIoU by +0.0397 over the existing 0.5 readout without changing LocAcc. Threshold 0.7 further sharpens some scenes but over-shrinks figurines/ramen, so 0.6 is the safer global choice. RGB snap is useful for direct 3D selection and some qualitative boundaries, but should be reported separately for rendered grounding unless combined with a conservative rule.
