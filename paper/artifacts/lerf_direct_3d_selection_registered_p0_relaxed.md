# LERF-OVS Direct 3D Object Selection

Protocol: OpenGaussian-style direct 3D primitive selection. Query scores are computed at Gaussian centers from pre-refiner RADIO-GS features; selected primitives are rendered only to compare with LERF-OVS object masks.

Input root: `output/radio_gs/lerf_direct_3d_selection_registered_p0_relaxed_20260512`
Fixed-protocol candidate selected by global macro mIoU among the sweep: `top0p02`. This is a diagnostic sweep result and should be reported separately from rendered-view grounding.

## Fixed-Ratio Sweep

| Selection | Figurines mIoU | Ramen mIoU | Teatime mIoU | Waldo mIoU | Macro mIoU | Macro Acc@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| top0p005 | 0.3599 | 0.1993 | 0.2137 | 0.0557 | 0.2071 | 0.3318 |
| top0p01 | 0.3695 | 0.3300 | 0.3528 | 0.0868 | 0.2848 | 0.5098 |
| top0p02 | 0.2962 | 0.4375 | 0.3997 | 0.1149 | 0.3121 | 0.5474 |
| top0p03 | 0.2563 | 0.4232 | 0.4403 | 0.1210 | 0.3102 | 0.5204 |
| top0p05 | 0.2322 | 0.3901 | 0.4304 | 0.1058 | 0.2896 | 0.4737 |
| top0p1 | 0.2052 | 0.3120 | 0.3059 | 0.1481 | 0.2428 | 0.3805 |

## Paper-Facing Direct-Selection Context

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| RADIO-GS | SigLIP2 | fixed top0p02 mIoU | 0.2962 | 0.4375 | 0.3997 | 0.1149 | 0.3121 |
| RADIO-GS | SigLIP2 | diagnostic best-by-scene mIoU | 0.3695 | 0.4375 | 0.4403 | 0.1481 | 0.3488 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| RADIO-GS | SigLIP2 | fixed top0p02 Acc@0.25 | 0.5714 | 0.6620 | 0.7288 | 0.2273 | 0.5474 |
| RADIO-GS | SigLIP2 | diagnostic best-by-scene Acc@0.25 | 0.7321 | 0.6620 | 0.7966 | 0.2273 | 0.6045 |

Interpretation: direct Gaussian-center text selection is implemented and aligned with the query-select-render-evaluate protocol, but current pre-refiner Gaussian features are substantially weaker than rendered-view grounding for LERF object selection. This should be framed as a remaining direct-3D selection gap unless new direct 3D supervision or instance aggregation is added.

