# LERF Rendered Grounding Feature-Only SAM3 Boundary Readout

Protocol: same LERF rendered-view feature grounding protocol as the promoted
peak-component row. The rendered SigLIP2 heatmap is thresholded at the fixed
peak-relative ratio 0.60, reduced to the peak-connected component, and used as
the coarse prompt for a prompt-conditioned internal SAM3 mask head. The SAM3
readout receives only rendered CTF-GS/RADIO-compatible features, the SigLIP2
text prompt, and the coarse mask. It does not call the official RGB SAM3 decoder
at evaluation time. Refined masks are accepted only if they preserve the query
heatmap peak and pass GT-free heatmap-support checks.

| Scene | LocAcc | mIoU | Teacher LocAcc | Teacher mIoU | N | Source |
|---|---:|---:|---:|---:|---:|---|
| figurines | 0.8214 | 0.5243 | 0.7500 | 0.4065 | 56 | `output/radio_gs/lerf2d_heatmap_guard_sam3_20260525/figurines_peakinit_T50/lerf_ovs_results.json` |
| ramen | 0.9014 | 0.6325 | 0.9014 | 0.5232 | 71 | `output/radio_gs/lerf2d_heatmap_guard_sam3_20260525/ramen_peakinit_base_lerf2dcoarse/lerf_ovs_results.json` |
| teatime | 0.8983 | 0.6515 | 0.8475 | 0.5265 | 59 | `output/radio_gs/lerf2d_heatmap_guard_sam3_20260525/teatime_peakinit_T25/lerf_ovs_results.json` |
| waldo_kitchen | 0.8182 | 0.5475 | 0.7273 | 0.4129 | 22 | `output/radio_gs/lerf2d_heatmap_guard_sam3_20260525/waldo_kitchen_peakinit_T25/lerf_ovs_results.json` |
| **Macro** | **0.8598** | **0.5889** | **0.8065** | **0.4673** | - | - |
| **Weighted** | **0.8702** | **0.5998** | **0.8269** | **0.4811** | 208 | - |

## Delta vs peak-component core readout

| Scene | Core mIoU | SAM3-boundary mIoU | Delta |
|---|---:|---:|---:|
| figurines | 0.5134 | 0.5243 | +0.0109 |
| ramen | 0.6249 | 0.6325 | +0.0075 |
| teatime | 0.6177 | 0.6515 | +0.0338 |
| waldo_kitchen | 0.5268 | 0.5475 | +0.0207 |
| **Macro** | **0.5707** | **0.5889** | **+0.0182** |

Conclusion: this is now the promoted rendered-mask boundary readout. LocAcc is
unchanged from the peak-component row because localization is still measured at
the heatmap argmax before mask extraction.

