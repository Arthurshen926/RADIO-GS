# LERF 2D and 3D OVS Qualitative

Figure: `paper/figures/lerf_2d3d_ovs_qualitative.png`

Layout: each query has GT, Prior, and Ours rows with separate 2D rendered-view OVS and 3D direct-selection OVS panels. The Prior row uses LangSplatV2 for 2D OVS and Dr. Splat for 3D OVS.

| Scene | Frame | Query | 2D Prior | 3D Prior | Fallback? | Prior 2D IoU | Prior 3D IoU | Ours 3D IoU |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| figurines | 00105 | `old camera` | LangSplatV2 (repro.) | Dr. Splat (repro.) | no | 0.5951 | 0.1843 | 0.9649 |
| figurines | 00105 | `green apple` | LangSplatV2 (repro.) | Dr. Splat (repro.) | no | 0.9384 | 0.7201 | 0.9690 |
| figurines | 00105 | `pumpkin` | LangSplatV2 (repro.) | Dr. Splat (repro.) | no | 0.7358 | 0.1740 | 0.9786 |
| teatime | 00140 | `tea in a glass` | LangSplatV2 (repro.) | Dr. Splat (repro.) | no | 0.9585 | 0.2474 | 0.7435 |
| teatime | 00140 | `apple` | LangSplatV2 (repro.) | Dr. Splat (repro.) | no | 0.9607 | 0.1086 | 0.9509 |
| teatime | 00140 | `bag of cookies` | LangSplatV2 (repro.) | Dr. Splat (repro.) | no | 0.8856 | 0.3696 | 0.9743 |

Note: 2D prior panels show the locally reproduced LangSplatV2 heatmap/RGB visualizations to match the rendered-view style of the Ours 2D panels; IoU is still computed from the corresponding chosen masks. 3D prior panels show Dr. Splat direct-selection masks rendered alone on a blank canvas, matching the 3D OVS display style of our selected primitive masks.
