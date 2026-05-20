# LERF SAM3 Box Global Threshold Sweep

Fixed global threshold is the strict paper-facing readout. Best fixed macro threshold and scene-locked best are diagnostic/post-hoc upper-bound readouts unless selected on held-out validation scenes.

- Strict fixed global threshold: `thr0p25`

| Run | Scenes | Fixed global threshold | Fixed mIoU | Fixed Acc@0.25 | Fixed Acc@0.50 | Boundary-F | Trimap IoU | Best fixed tag | Best fixed mIoU | Scene-locked mIoU | Source root |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| pad16 | 4 | `thr0p25` | 0.5705 | 0.6835 | 0.6081 | 0.6681 | 0.3958 | `thr0p18` | 0.5863 | 0.5972 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5` |
| pad0 | 4 | `thr0p25` | 0.5421 | 0.6816 | 0.5630 | 0.6598 | 0.3601 | `thr0p18` | 0.5642 | 0.5795 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_global_selector_20260516_210200_pad0_gpu5b` |
| pad8 | 4 | `thr0p25` | 0.5401 | 0.6629 | 0.5606 | 0.6532 | 0.3700 | `thr0p15` | 0.5619 | 0.5724 | `output/radio_gs/lerf_direct3d_sam3_box_pad8_global_selector_20260516_210500_pad8_gpu0` |
| pad4 | 4 | `thr0p25` | 0.5471 | 0.6950 | 0.5604 | 0.6655 | 0.3746 | `thr0p15` | 0.5641 | 0.5748 | `output/radio_gs/lerf_direct3d_sam3_box_pad4_global_selector_20260516_212900_pad4_gpu5` |

## pad16

- Source root: `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5`
- Missing scenes: `none`
- Fixed global threshold `thr0p25`: macro mIoU `0.5705`, Acc@0.50 `0.6081`, Boundary-F `0.6681`, Trimap IoU `0.3958`.
- Diagnostic best fixed macro threshold `thr0p18`: macro mIoU `0.5863`.
- Diagnostic scene-locked best: macro mIoU `0.5972`.

### Fixed Global Threshold Scene Rows

| Scene | Selection | mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Trimap IoU | N | Source JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|
| figurines | `thr0p25` | 0.6136 | 0.6964 | 0.6607 | 0.7116 | 0.3986 | 56 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/figurines/lerf_direct_3d_selection_results.json` |
| ramen | `thr0p25` | 0.6409 | 0.7465 | 0.6901 | 0.7713 | 0.4400 | 71 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/ramen/lerf_direct_3d_selection_results.json` |
| teatime | `thr0p25` | 0.6130 | 0.7458 | 0.6271 | 0.7255 | 0.4626 | 59 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/teatime/lerf_direct_3d_selection_results.json` |
| waldo_kitchen | `thr0p25` | 0.4142 | 0.5455 | 0.4545 | 0.4638 | 0.2820 | 22 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/waldo_kitchen/lerf_direct_3d_selection_results.json` |

## pad0

- Source root: `output/radio_gs/lerf_direct3d_sam3_box_pad0_global_selector_20260516_210200_pad0_gpu5b`
- Missing scenes: `none`
- Fixed global threshold `thr0p25`: macro mIoU `0.5421`, Acc@0.50 `0.5630`, Boundary-F `0.6598`, Trimap IoU `0.3601`.
- Diagnostic best fixed macro threshold `thr0p18`: macro mIoU `0.5642`.
- Diagnostic scene-locked best: macro mIoU `0.5795`.

### Fixed Global Threshold Scene Rows

| Scene | Selection | mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Trimap IoU | N | Source JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|
| figurines | `thr0p25` | 0.5614 | 0.6607 | 0.6250 | 0.6652 | 0.3284 | 56 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_global_selector_20260516_210200_pad0_gpu5b/figurines/lerf_direct_3d_selection_results.json` |
| ramen | `thr0p25` | 0.6736 | 0.7746 | 0.7324 | 0.7881 | 0.4339 | 71 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_global_selector_20260516_210200_pad0_gpu5b/ramen/lerf_direct_3d_selection_results.json` |
| teatime | `thr0p25` | 0.5761 | 0.7458 | 0.5763 | 0.7211 | 0.4500 | 59 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_global_selector_20260516_210200_pad0_gpu5b/teatime/lerf_direct_3d_selection_results.json` |
| waldo_kitchen | `thr0p25` | 0.3574 | 0.5455 | 0.3182 | 0.4649 | 0.2283 | 22 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_global_selector_20260516_210200_pad0_gpu5b/waldo_kitchen/lerf_direct_3d_selection_results.json` |

## pad8

- Source root: `output/radio_gs/lerf_direct3d_sam3_box_pad8_global_selector_20260516_210500_pad8_gpu0`
- Missing scenes: `none`
- Fixed global threshold `thr0p25`: macro mIoU `0.5401`, Acc@0.50 `0.5606`, Boundary-F `0.6532`, Trimap IoU `0.3700`.
- Diagnostic best fixed macro threshold `thr0p15`: macro mIoU `0.5619`.
- Diagnostic scene-locked best: macro mIoU `0.5724`.

### Fixed Global Threshold Scene Rows

| Scene | Selection | mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Trimap IoU | N | Source JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|
| figurines | `thr0p25` | 0.5967 | 0.6964 | 0.6607 | 0.6920 | 0.3766 | 56 | `output/radio_gs/lerf_direct3d_sam3_box_pad8_global_selector_20260516_210500_pad8_gpu0/figurines/lerf_direct_3d_selection_results.json` |
| ramen | `thr0p25` | 0.6584 | 0.7887 | 0.7042 | 0.7749 | 0.4306 | 71 | `output/radio_gs/lerf_direct3d_sam3_box_pad8_global_selector_20260516_210500_pad8_gpu0/ramen/lerf_direct_3d_selection_results.json` |
| teatime | `thr0p25` | 0.5629 | 0.7119 | 0.5593 | 0.7045 | 0.4263 | 59 | `output/radio_gs/lerf_direct3d_sam3_box_pad8_global_selector_20260516_210500_pad8_gpu0/teatime/lerf_direct_3d_selection_results.json` |
| waldo_kitchen | `thr0p25` | 0.3423 | 0.4545 | 0.3182 | 0.4414 | 0.2466 | 22 | `output/radio_gs/lerf_direct3d_sam3_box_pad8_global_selector_20260516_210500_pad8_gpu0/waldo_kitchen/lerf_direct_3d_selection_results.json` |

## pad4

- Source root: `output/radio_gs/lerf_direct3d_sam3_box_pad4_global_selector_20260516_212900_pad4_gpu5`
- Missing scenes: `none`
- Fixed global threshold `thr0p25`: macro mIoU `0.5471`, Acc@0.50 `0.5604`, Boundary-F `0.6655`, Trimap IoU `0.3746`.
- Diagnostic best fixed macro threshold `thr0p15`: macro mIoU `0.5641`.
- Diagnostic scene-locked best: macro mIoU `0.5748`.

### Fixed Global Threshold Scene Rows

| Scene | Selection | mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Trimap IoU | N | Source JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|
| figurines | `thr0p25` | 0.5931 | 0.7143 | 0.6429 | 0.7010 | 0.3731 | 56 | `output/radio_gs/lerf_direct3d_sam3_box_pad4_global_selector_20260516_212900_pad4_gpu5/figurines/lerf_direct_3d_selection_results.json` |
| ramen | `thr0p25` | 0.6616 | 0.7746 | 0.7042 | 0.7705 | 0.4323 | 71 | `output/radio_gs/lerf_direct3d_sam3_box_pad4_global_selector_20260516_212900_pad4_gpu5/ramen/lerf_direct_3d_selection_results.json` |
| teatime | `thr0p25` | 0.5607 | 0.7458 | 0.5763 | 0.7094 | 0.4288 | 59 | `output/radio_gs/lerf_direct3d_sam3_box_pad4_global_selector_20260516_212900_pad4_gpu5/teatime/lerf_direct_3d_selection_results.json` |
| waldo_kitchen | `thr0p25` | 0.3731 | 0.5455 | 0.3182 | 0.4809 | 0.2641 | 22 | `output/radio_gs/lerf_direct3d_sam3_box_pad4_global_selector_20260516_212900_pad4_gpu5/waldo_kitchen/lerf_direct_3d_selection_results.json` |

## Warnings

- best_fixed_macro_threshold and scene_locked_best are diagnostic/post-hoc readouts; fixed_global_threshold is the strict global-threshold protocol.
