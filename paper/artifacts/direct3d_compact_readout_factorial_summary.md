# Direct3D Compact Readout Factorial Summary

This summary is generated from `paper/artifacts/lerf_direct3d_compact_readout_ablation_20260528.json`.
It separates compact-map evidence from RGB/component support policies and official SAM3 diagnostics.

| Row | VPR cache | Official SAM3 | RGB postprocess | Prompt ensemble | Threshold | mIoU | Acc@0.25 | Boundary-F | Trimap IoU | Note |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Compact single prompt, pure one-map | no | no | no | no | thr0p70 | 0.448897 | 0.672370 | 0.612399 | 0.266532 | strict no-prompt/no-RGB compact-score readout |
| Compact direct, previous promoted baseline | no | no | yes | no | fixed | 0.483600 | 0.642600 | 0.598300 | - | older guarded baseline; not strict no-RGB |
| Compact prompt ensemble, pure one-map | no | no | no | yes | thr0p70 | 0.457028 | 0.685082 | 0.616570 | 0.277138 | strict no-RGB compact-score readout |
| Compact prompt ensemble + RGB component guard | no | no | yes | yes | thr0p65 | 0.499976 | 0.705147 | 0.634492 | 0.321253 | Acc-best compact guarded row |
| Compact prompt ensemble + RGB/score-component guard | no | no | yes | yes | thr0p55 | 0.501374 | 0.704431 | 0.630533 | 0.322466 | overlap-balanced promoted compact row |

## Main Delta

- Prompt ensemble over strict single-prompt pure one-map: mIoU 0.448897->0.457028 (+0.008131), Acc@0.25 0.672370->0.685082 (+0.012712), Boundary-F 0.612399->0.616570 (+0.004171).
- Guarded compact support over prompt-ensemble pure one-map: mIoU 0.457028->0.501374 (+0.044346), Acc@0.25 0.685082->0.704431 (+0.019349), Boundary-F 0.616570->0.630533 (+0.013964).
- Remaining optional cell: no-prompt plus RGB/score-component guard, which would isolate support-policy effects without prompt ensembling.
