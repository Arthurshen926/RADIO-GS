# LERF Direct 3D Query-Level Audit

- Result root: `output/radio_gs/lerf_direct_3d_selection_threshold_grabcut_20260515`
- Selection tag: `thr0p25`
- Purpose: expose query-level uncertainty, zero-prediction failures, and over-selection failures for the VPR direct 3D readout.

## Per-Scene Summary

| Scene | Queries | mIoU | 95% bootstrap CI | Acc@0.25 | Zero-pred rate | Mean overselect |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 56 | 0.5309 | [0.4529, 0.6073] | 0.7857 | 0.0714 | 1.38 |
| Ramen | 71 | 0.5805 | [0.5077, 0.6487] | 0.7465 | 0.0282 | 1.31 |
| Teatime | 59 | 0.5662 | [0.4882, 0.6432] | 0.7627 | 0.0169 | 0.78 |
| Waldo Kitchen | 22 | 0.2429 | [0.1487, 0.3403] | 0.4091 | 0.1818 | 0.52 |
| Macro/query pool | 208 | 0.5274 | [0.4869, 0.5704] | 0.7260 | 0.0529 | 1.10 |

## Worst Queries

| Scene | Frame | Category | IoU | Pred px | GT px | Overselect |
|---|---|---|---:|---:|---:|---:|
| Waldo Kitchen | frame_00066 | pot | 0.0000 | 0 | 19457 | 0.00 |
| Ramen | frame_00128 | hand | 0.0000 | 0 | 9381 | 0.00 |
| Waldo Kitchen | frame_00053 | pour-over vessel | 0.0000 | 0 | 7705 | 0.00 |
| Teatime | frame_00129 | dall-e brand | 0.0000 | 0 | 7442 | 0.00 |
| Waldo Kitchen | frame_00154 | knife | 0.0000 | 0 | 4580 | 0.00 |
| Waldo Kitchen | frame_00066 | plastic ladle | 0.0000 | 0 | 3558 | 0.00 |
| Ramen | frame_00128 | napkin | 0.0000 | 0 | 2891 | 0.00 |
| Figurines | frame_00152 | pirate hat | 0.0000 | 0 | 2080 | 0.00 |
| Figurines | frame_00195 | pirate hat | 0.0000 | 0 | 1873 | 0.00 |
| Figurines | frame_00041 | pirate hat | 0.0000 | 0 | 1835 | 0.00 |
| Figurines | frame_00105 | pirate hat | 0.0000 | 0 | 899 | 0.00 |
| Waldo Kitchen | frame_00053 | yellow desk | 0.0000 | 1959 | 66361 | 0.03 |
| Teatime | frame_00129 | yellow pouf | 0.0000 | 7 | 23108 | 0.00 |
| Figurines | frame_00195 | bag | 0.0000 | 645 | 6112 | 0.11 |
| Ramen | frame_00006 | napkin | 0.0000 | 5276 | 5538 | 0.95 |
| Ramen | frame_00065 | napkin | 0.0000 | 7837 | 5392 | 1.45 |

Interpretation: zero-prediction rows usually indicate that selected primitives are not visible in the annotated view after rendering, while high overselect ratios indicate clutter/background leakage. These are the two dominant failure modes to discuss for Waldo Kitchen.
