# LERF Nearest-View Cache Baseline

Protocol: unwarped nearest-view cached teacher features. For each annotated target frame, the baseline substitutes the closest cached RADIO frame by camera-center distance, excluding the target frame itself, then runs the same LERF text scoring and thresholded-mask evaluator.

| Scene | LocAcc | mIoU | N | Mean nearest distance |
|---|---:|---:|---:|---:|
| Figurines | 0.1786 | 0.0755 | 56 | 0.2831 |
| Ramen | 0.2817 | 0.1740 | 71 | 0.5443 |
| Teatime | 0.3559 | 0.2114 | 59 | 0.4266 |
| Waldo Kitchen | 0.2727 | 0.1570 | 22 | 0.5787 |

## Aggregate

| Aggregate | LocAcc | mIoU |
|---|---:|---:|
| Macro | 0.2722 | 0.1545 |
| Query-weighted | 0.2740 | 0.1563 |

## Interpretation

- This is a cache-only baseline, not a 3D scene representation.
- The source feature map is not warped into the target camera, so the result measures how far a simple nearest-view cache can go without RADIO-GS reconstruction.
- It should be reported separately from the same-frame RADIO teacher row and the rendered 3D feature-field row.
