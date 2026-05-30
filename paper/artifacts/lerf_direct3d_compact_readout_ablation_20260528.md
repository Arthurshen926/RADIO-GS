# LERF Direct3D Compact Readout Ablation

This artifact separates the strict compact-field readout from the current
deployed guarded readout for LERF-OVS direct 3D object selection.

Definitions:

- **Pure one-map**: compact Gaussian-center primitive scores only; no VPR cache,
  no official RGB SAM decoder, and no RGB postprocess. The stored score cache is
  an evaluation-speed cache of compact-field primitive scores, not an external
  VPR cache.
- **Guarded compact readout**: the same compact primitive scores plus a frozen
  SigLIP2 prompt ensemble and a GT-free RGB/GrabCut component-support guard.
  It does not use a VPR cache or official RGB SAM decoder, but it is not a
  strict no-RGB-postprocess row.

## Main Ablation

| Row | VPR cache | Official RGB SAM | RGB postprocess | Prompt ensemble | Threshold | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Compact single prompt, pure one-map | no | no | no | no | thr0p70 | 0.448897 | 0.672370 | 0.612399 | 0.266532 |
| Compact direct, previous promoted baseline | no | no | yes | no | fixed | 0.4836 | 0.6426 | 0.5983 | - |
| Compact prompt ensemble, pure one-map | no | no | no | yes | thr0p70 | 0.457028 | 0.685082 | 0.616570 | 0.277138 |
| Compact prompt ensemble + RGB component guard | no | no | yes | yes | thr0p65 | 0.499976 | 0.705147 | 0.634492 | 0.321253 |
| Compact prompt ensemble + RGB/score-component guard | no | no | yes | yes | thr0p55 | 0.501374 | 0.704431 | 0.630533 | 0.322466 |

## Single-Prompt Pure One-Map Per-Scene Metrics

This row was rerun on 2026-05-30 from the single-prompt compact-score caches
using the cache-matched checkpoint provenance and official-frame setting. It
uses no VPR cache, no official SAM decoder, no RGB postprocess, and no prompt
ensemble. The best global threshold in the evaluated sweep is `thr0p70`.

| Scene | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
| --- | ---: | ---: | ---: | ---: |
| figurines | 0.418006 | 0.660714 | 0.608767 | 0.164607 |
| ramen | 0.600002 | 0.816901 | 0.756113 | 0.398472 |
| teatime | 0.519715 | 0.711864 | 0.718381 | 0.321810 |
| waldo_kitchen | 0.257866 | 0.500000 | 0.366335 | 0.181238 |
| Macro | 0.448897 | 0.672370 | 0.612399 | 0.266532 |

## Pure One-Map Per-Scene Metrics

The pure one-map row uses the global threshold `thr0p70`, selected by macro mIoU
over the same global threshold sweep.

| Scene | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
| --- | ---: | ---: | ---: | ---: |
| figurines | 0.421347 | 0.660714 | 0.622372 | 0.165981 |
| ramen | 0.579621 | 0.816901 | 0.754534 | 0.384837 |
| teatime | 0.528667 | 0.762712 | 0.701729 | 0.375980 |
| waldo_kitchen | 0.298477 | 0.500000 | 0.387644 | 0.181754 |
| Macro | 0.457028 | 0.685082 | 0.616570 | 0.277138 |

## Guarded Compact Per-Scene Metrics

The guarded row uses the global threshold `thr0p65`.

| Scene | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
| --- | ---: | ---: | ---: | ---: |
| figurines | 0.509582 | 0.678571 | 0.669243 | 0.267778 |
| ramen | 0.590308 | 0.816901 | 0.747319 | 0.404796 |
| teatime | 0.569376 | 0.779661 | 0.719849 | 0.404758 |
| waldo_kitchen | 0.330638 | 0.545455 | 0.401557 | 0.207680 |
| Macro | 0.499976 | 0.705147 | 0.634492 | 0.321253 |

Conclusion: the latest overlap-balanced direct-3D row is no-VPR-cache and
no-official-SAM, but it is not the strict no-RGB one-map row. The strict
single-prompt pure one-map row reaches 0.4489 mIoU / 0.6724 Acc@0.25; adding
the frozen SigLIP2 prompt ensemble improves it to 0.4570 / 0.6851. The deployed
score-component guarded compact readout reaches 0.5014 mIoU / 0.7044 Acc@0.25,
so the largest Direct3D readout-policy gain comes from support recovery rather
than prompt wording alone.
