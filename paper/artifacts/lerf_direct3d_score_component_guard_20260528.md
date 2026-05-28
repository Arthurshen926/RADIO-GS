# LERF Direct3D Score-Component Guard

This artifact records a GT-free support-policy upgrade for the compact-field
direct 3D readout. After compact primitive selection and RGB/GrabCut boundary
snapping, connected components are ranked by rendered compact score-heatmap mass
rather than only by image area. The policy does not read a VPR cache and does
not call the official RGB SAM decoder.

Policy:

- Frozen SigLIP2 prompt ensemble: `{query}|a photo of {query}|a photo of a {query}|the {query}|a {query} object`.
- Direct Gaussian-center compact primitive scores with the opacity-gated point summary adapter.
- Fixed global score threshold.
- RGB/GrabCut boundary snapping followed by score-heatmap component filtering.
- Component filtering uses `min_mass_fraction=0.50`, `max_components=2`, and a 6000-pixel small-support floor.

## Comparison

| Row | Threshold | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| Area component guard, Acc-best | thr0p65 | 0.499976 | 0.705147 | 0.634492 | 0.321253 |
| Score-component guard, overlap-balanced | thr0p55 | 0.501374 | 0.704431 | 0.630533 | 0.322466 |
| Score-component guard, boundary-favored | thr0p65 | 0.501154 | 0.697389 | 0.635800 | 0.319379 |

## Promoted Per-Scene Metrics

The promoted overlap-balanced row uses the global threshold `thr0p55`.

| Scene | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
| --- | ---: | ---: | ---: | ---: |
| figurines | 0.510437 | 0.678571 | 0.668857 | 0.270293 |
| ramen | 0.599875 | 0.830986 | 0.755701 | 0.407219 |
| teatime | 0.563975 | 0.762712 | 0.714854 | 0.400064 |
| waldo_kitchen | 0.331208 | 0.545455 | 0.382722 | 0.212287 |
| Macro | 0.501374 | 0.704431 | 0.630533 | 0.322466 |

Conclusion: score-component support filtering gives the first compact direct-3D
row with exact macro mIoU above 0.50 while preserving the Acc@0.25 gain over the
registered VPR row. The previous area component guard remains the Acc-best
compact row by a small margin.
