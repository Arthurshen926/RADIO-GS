# LERF 2D/3D OVS Qualitative Figure

- Figure: `paper/figures/lerf_2d3d_ovs_qualitative.png`
- 2D reproduced prior: `LangSplat` rendered-view mask.
- 3D reproduced prior: `Dr. Splat` direct-selection render and silhouette.
- Ours 3D source root: `output/radio_gs/lerf_direct3d_prompt_ensemble_policy_masks_20260528`

| Scene | Frame | Query | Prior 2D IoU | Prior 3D IoU | Ours 3D IoU |
| --- | --- | --- | ---: | ---: | ---: |
| figurines | 00105 | `old camera` | 0.4207 | 0.1843 | 0.9649 |
| figurines | 00105 | `green apple` | 0.6742 | 0.7201 | 0.9690 |
| figurines | 00105 | `pumpkin` | 0.5966 | 0.1740 | 0.9786 |
| teatime | 00140 | `tea in a glass` | 0.6358 | 0.2474 | 0.7435 |
| teatime | 00140 | `apple` | 0.8086 | 0.1086 | 0.9509 |
| teatime | 00140 | `bag of cookies` | 0.3255 | 0.3696 | 0.9743 |

Protocol note: 2D panels visualize rendered-view OVS masks/heatmaps on RGB, while 3D panels visualize primitives selected by direct Gaussian-level query and rendered/cut out on a white background. The CTF-GS 3D panels use the compact direct-field mask source only; no VPR cache or official RGB SAM3 decoder is called when producing these panels.
