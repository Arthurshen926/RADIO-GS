# LERF Rendered Grounding Peak-Component Readout

LERF rendered-view peak-connected-component mask readout. Heatmap scoring/checkpoints/temperatures match the paper LERF rendered protocol; binary masks use fixed peak-relative threshold 0.60, then keep only the connected component containing the query heatmap peak. No GT masks are used by the readout.

| Scene | LocAcc | mIoU | Teacher LocAcc | Teacher mIoU | N | Source |
|---|---:|---:|---:|---:|---:|---|
| figurines | 0.8214 | 0.5134 | 0.7500 | 0.4031 | 56 | `output/radio_gs/lerf2d_peak_component_probe_20260524/figurines/lerf_ovs_results.json` |
| ramen | 0.9014 | 0.6249 | 0.9014 | 0.5241 | 71 | `output/radio_gs/lerf2d_peak_component_probe_20260524/ramen/lerf_ovs_results.json` |
| teatime | 0.8983 | 0.6177 | 0.8475 | 0.5429 | 59 | `output/radio_gs/lerf2d_peak_component_probe_20260524/teatime/lerf_ovs_results.json` |
| waldo_kitchen | 0.8182 | 0.5268 | 0.7273 | 0.3688 | 22 | `output/radio_gs/lerf2d_peak_component_probe_20260524/waldo_kitchen/lerf_ovs_results.json` |
| **Macro** | **0.8598** | **0.5707** | **0.8065** | **0.4597** | - | - |
| **Weighted** | **0.8702** | **0.5825** | **0.8269** | **0.4804** | 208 | - |

## Delta vs threshold-0.60 historical paper row

- Historical threshold row: LocAcc `0.8712`, mIoU `0.5243`.
- Peak-component readout: LocAcc `0.8598`, mIoU `0.5707`.
- Macro mIoU delta: `+0.0464`; LocAcc delta under current rerun artifacts: `-0.0114`.
