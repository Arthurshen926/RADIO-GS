# LERF-OVS Direct 3D Object Selection

Protocol: OpenGaussian-style direct 3D primitive selection. Query scores are computed at Gaussian centers from pre-refiner RADIO-GS features; selected primitives are rendered only to compare with LERF-OVS object masks.

Input root: `output/radio_gs/lerf_direct_3d_selection_registered_p0_softmax24_20260512`
Fixed-protocol candidate selected by global macro mIoU among the sweep: `top0p02`. This is a diagnostic sweep result and should be reported separately from rendered-view grounding.

## Fixed-Ratio Sweep

| Selection | Figurines mIoU | Ramen mIoU | Teatime mIoU | Waldo mIoU | Macro mIoU | Macro Acc@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| top0p005 | 0.3475 | 0.2766 | 0.1888 | 0.0831 | 0.2240 | 0.3531 |
| top0p01 | 0.3606 | 0.3962 | 0.3214 | 0.1184 | 0.2991 | 0.5188 |
| top0p02 | 0.3246 | 0.4561 | 0.4466 | 0.1413 | 0.3421 | 0.5547 |
| top0p03 | 0.3058 | 0.4093 | 0.4796 | 0.1515 | 0.3365 | 0.5381 |
| top0p05 | 0.2758 | 0.3254 | 0.4675 | 0.1450 | 0.3034 | 0.4722 |
| top0p1 | 0.2294 | 0.2333 | 0.3511 | 0.1483 | 0.2405 | 0.3519 |

## Paper-Facing Direct-Selection Context

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| RADIO-GS | SigLIP2 | fixed top0p02 mIoU | 0.3246 | 0.4561 | 0.4466 | 0.1413 | 0.3421 |
| RADIO-GS | SigLIP2 | diagnostic best-by-scene mIoU | 0.3606 | 0.4561 | 0.4796 | 0.1515 | 0.3619 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| RADIO-GS | SigLIP2 | fixed top0p02 Acc@0.25 | 0.5357 | 0.6761 | 0.7797 | 0.2273 | 0.5547 |
| RADIO-GS | SigLIP2 | diagnostic best-by-scene Acc@0.25 | 0.6607 | 0.6761 | 0.8305 | 0.2727 | 0.6100 |

Interpretation: direct Gaussian-center text selection is implemented and aligned with the query-select-render-evaluate protocol, but current pre-refiner Gaussian features are substantially weaker than rendered-view grounding for LERF object selection. This should be framed as a remaining direct-3D selection gap unless new direct 3D supervision or instance aggregation is added.

