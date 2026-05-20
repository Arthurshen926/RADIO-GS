# LERF SAM3 Box Global Threshold Sweep

Fixed global threshold is the strict paper-facing readout. Best fixed macro threshold and scene-locked best are diagnostic/post-hoc upper-bound readouts unless selected on held-out validation scenes.

- Strict fixed global threshold: `thr0p25`

| Run | Scenes | Fixed global threshold | Fixed mIoU | Fixed Acc@0.25 | Fixed Acc@0.50 | Boundary-F | Trimap IoU | Best fixed tag | Best fixed mIoU | Scene-locked mIoU | Source root |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| pad16_geometry | 4 | `thr0p25` | 0.5705 | 0.6835 | 0.6081 | 0.6681 | 0.3958 | `thr0p18` | 0.5863 | 0.5972 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260517_geometry_pad16_full` |

## pad16_geometry

- Source root: `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260517_geometry_pad16_full`
- Missing scenes: `none`
- Fixed global threshold `thr0p25`: macro mIoU `0.5705`, Acc@0.50 `0.6081`, Boundary-F `0.6681`, Trimap IoU `0.3958`.
- Diagnostic best fixed macro threshold `thr0p18`: macro mIoU `0.5863`.
- Diagnostic scene-locked best: macro mIoU `0.5972`.

### Fixed Global Threshold Scene Rows

| Scene | Selection | mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Trimap IoU | N | Source JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|
| figurines | `thr0p25` | 0.6136 | 0.6964 | 0.6607 | 0.7116 | 0.3986 | 56 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260517_geometry_pad16_full/figurines/lerf_direct_3d_selection_results.json` |
| ramen | `thr0p25` | 0.6409 | 0.7465 | 0.6901 | 0.7713 | 0.4400 | 71 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260517_geometry_pad16_full/ramen/lerf_direct_3d_selection_results.json` |
| teatime | `thr0p25` | 0.6130 | 0.7458 | 0.6271 | 0.7255 | 0.4626 | 59 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260517_geometry_pad16_full/teatime/lerf_direct_3d_selection_results.json` |
| waldo_kitchen | `thr0p25` | 0.4142 | 0.5455 | 0.4545 | 0.4638 | 0.2820 | 22 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260517_geometry_pad16_full/waldo_kitchen/lerf_direct_3d_selection_results.json` |

## Warnings

- best_fixed_macro_threshold and scene_locked_best are diagnostic/post-hoc readouts; fixed_global_threshold is the strict global-threshold protocol.
