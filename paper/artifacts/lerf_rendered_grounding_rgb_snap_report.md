# LERF Rendered Grounding RGB Boundary Snap Report

LERF rendered-view grounding with optional GT-free RGB GrabCut boundary snap; same checkpoint, same best-by-LocAcc temperature, same text scoring as existing runs.

| Scene | Temp | N | Base LocAcc | RGB-snap LocAcc | Base mIoU | RGB-snap mIoU | Delta mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| figurines | 50 | 56 | 0.8036 | 0.8036 | 0.4312 | 0.4683 | +0.0371 |
| ramen | 40 | 71 | 0.8732 | 0.8732 | 0.5935 | 0.5789 | -0.0147 |
| teatime | 15 | 59 | 0.8814 | 0.8814 | 0.4206 | 0.4079 | -0.0127 |
| waldo_kitchen | 25 | 22 | 0.7727 | 0.7727 | 0.3837 | 0.3730 | -0.0106 |
| **Macro** | - | - | 0.8327 | 0.8327 | 0.4573 | 0.4570 | -0.0002 |
| **Weighted** | - | 208 | 0.8462 | 0.8462 | 0.4786 | 0.4788 | +0.0002 |

Conclusion: RGB boundary snap improves figurines rendered-view mIoU and leaves LocAcc unchanged, but this global setting is mixed across scenes. It should be treated as an optional boundary-aware rendered-mask readout/qualitative refinement unless a GT-free conservative rule or globally tuned setting improves macro metrics.
