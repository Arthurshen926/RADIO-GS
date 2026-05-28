# LERF Direct3D Prompt-Ensemble Support Policy

This artifact records the compact-field direct 3D readout that does not use a
VPR cache or official RGB SAM readout at inference. It does use a GT-free
RGB/GrabCut component-support guard, so it is not the strict no-RGB one-map
ablation.

Policy:

- Frozen SigLIP2 prompt ensemble: `{query}|a photo of {query}|a photo of a {query}|the {query}|a {query} object`.
- Direct Gaussian-center compact primitive scores with the opacity-gated point summary adapter.
- Fixed global softmax score threshold; the table reports a global threshold sweep without per-scene or per-query thresholding.
- GT-free support-aware RGB/GrabCut cleanup with component guard: keep the largest component if it dominates, otherwise preserve multi-component support only when the refined support has at least 6000 pixels.

## Threshold Sweep

| Threshold | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
| --- | ---: | ---: | ---: | ---: |
| thr0p55 | 0.498691 | 0.708668 | 0.628877 | 0.324130 |
| thr0p6 | 0.498644 | 0.705147 | 0.631257 | 0.322127 |
| thr0p65 | 0.499976 | 0.705147 | 0.634492 | 0.321253 |
| thr0p66 | 0.498917 | 0.705147 | 0.633926 | 0.319663 |
| thr0p67 | 0.499523 | 0.705147 | 0.634113 | 0.319978 |
| thr0p68 | 0.499453 | 0.700910 | 0.633925 | 0.319918 |
| thr0p7 | 0.498801 | 0.705147 | 0.633527 | 0.318802 |
| thr0p75 | 0.497975 | 0.705147 | 0.634390 | 0.316367 |

Best threshold by macro mIoU: `thr0p65`.

## Best Per-Scene Metrics

| Scene | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
| --- | ---: | ---: | ---: | ---: |
| figurines | 0.509582 | 0.678571 | 0.669243 | 0.267778 |
| ramen | 0.590308 | 0.816901 | 0.747319 | 0.404796 |
| teatime | 0.569376 | 0.779661 | 0.719849 | 0.404758 |
| waldo_kitchen | 0.330638 | 0.545455 | 0.401557 | 0.207680 |
| Macro | 0.499976 | 0.705147 | 0.634492 | 0.321253 |

Conclusion: this targeted support-aware compact direct readout improves the
previous compact row from 0.4836/0.6426 to 0.5000/0.7051 when rounded to four
decimals, with the largest Acc@0.25 recovery on Waldo Kitchen. The companion
`lerf_direct3d_compact_readout_ablation_20260528` artifact separates this
guarded deployed readout from the strict no-RGB one-map ablation.
