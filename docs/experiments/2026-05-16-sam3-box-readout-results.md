# Official SAM3 Box Readout Results, 2026-05-16

## What Changed

The direct 3D object-selection evaluator now supports a frozen official SAM3
box-prompt readout refinement:

1. Compact direct-field Gaussian primitives are scored by the SigLIP2 text head.
2. Selected 3D primitives are rendered into a binary prediction mask.
3. The rendered prediction is converted into a normalized SAM3 box prompt.
4. Official SAM3 candidate masks are selected by overlap with the rendered
   prediction; ground-truth masks are not used for candidate selection.

This is a readout refinement, not a replacement for the learned feature field.
The 3D selection signal still comes from RADIO-GS.

## Fixed-Protocol Results

| Readout | Fig. mIoU | Ramen mIoU | Tea. mIoU | Waldo mIoU | Macro mIoU | Macro Acc@0.25 | Boundary-F | Trimap IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct field + RGB/GrabCut | 0.5203 | 0.5861 | 0.5723 | 0.2736 | 0.4881 | 0.6788 | 0.6226 | 0.3360 |
| direct field + official SAM3 box, pad0 | 0.5924 | 0.6830 | 0.6556 | 0.3949 | 0.5815 | 0.7150 | 0.6731 | 0.3777 |
| direct field + official SAM3 box, pad16 | 0.6422 | 0.6529 | 0.6165 | 0.4110 | 0.5807 | 0.6967 | 0.6814 | 0.3897 |

The pad0 row is the strongest exported SAM3-box boundary-readout diagnostic and
pad16 is the boundary-analysis row. The regenerated freeze manifest records
these rows with `selector_policy=best_by_miou` because each scene uses the
exported best threshold (`thr0p11`, `thr0p16`, `thr0p38`, and `thr0p3` for
pad0). Treat the SAM3-box rows as boundary-readout diagnostics until a
validation-selected or global threshold rule is added. The strict direct-3D
primitive row remains the fixed-threshold VPR + RGB-snap result.

## Global-Threshold Padding Sweep

A later GPU0/GPU5 sweep reran the official SAM3 box readout with a shared
global threshold grid for all four LERF scenes and four box paddings. The
generated report is
`output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260516.md`;
the machine-readable manifest is
`output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260516.json`.

The strict paper-facing SAM3-box readout now uses the fixed global selector
`thr0p25`. Best fixed macro threshold and scene-locked best are retained only
as diagnostics/post-hoc upper bounds unless selected on held-out validation
scenes.

| Padding | Fixed selector | Macro mIoU | Macro Acc@0.25 | Macro Acc@0.50 | Boundary-F | Trimap IoU | Best fixed selector (diag.) | Best fixed mIoU (diag.) | Scene-locked mIoU (diag.) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| pad16 | `thr0p25` | 0.5705 | 0.6835 | 0.6081 | 0.6681 | 0.3958 | `thr0p18` | 0.5863 | 0.5972 |
| pad0 | `thr0p25` | 0.5421 | 0.6816 | 0.5630 | 0.6598 | 0.3601 | `thr0p18` | 0.5642 | 0.5795 |
| pad8 | `thr0p25` | 0.5401 | 0.6629 | 0.5606 | 0.6532 | 0.3700 | `thr0p15` | 0.5619 | 0.5724 |
| pad4 | `thr0p25` | 0.5471 | 0.6950 | 0.5604 | 0.6655 | 0.3746 | `thr0p15` | 0.5641 | 0.5748 |

Under the strict fixed-global protocol, pad16 is the strongest tested SAM3-box
padding by macro mIoU and Acc@0.50. The best fixed macro threshold for pad16 is
`thr0p18` at 0.5863 mIoU, and the scene-locked diagnostic upper bound reaches
0.5972 mIoU / 0.7009 Acc@0.25, but both are post-hoc readouts.

The measured boundary-error audit for the strict pad16 row is
`output/radio_gs/reports/boundary_error_readout_report.md`. It uses 208
per-query records and finds IoU vs Boundary-F Pearson r=0.9148, while
under-selection and over-selection have high mean boundary errors (0.7042 and
0.6919). The Direct3D evaluator now has `--save_geometry_maps` instrumentation,
and the geometry-map rerun
`output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260517_geometry.md`
keeps the strict pad16 metrics unchanged while populating
`output/radio_gs/reports/alpha_depth_boundary_alignment_report.md`. The report
now has 208/208 alpha/depth geometry records and overlays. The correlations are
weak, so this should be presented as boundary-mechanism context rather than a
causal occlusion analysis.

## Output Locations

Fixed pad0 sweep:

- `output/radio_gs/lerf_direct3d_sam3_box_pad0_sweep_gpu5_20260516_180129`
- `output/radio_gs/lerf_direct3d_waldo_direct_sam3_box_padding_sweep_gpu4_20260516_175619/pad_0`

Fixed pad16 sweep:

- `output/radio_gs/lerf_direct3d_sam3_box_pad16_sweep_gpu4_20260516_180649`
- `output/radio_gs/lerf_direct3d_waldo_direct_sam3_box_padding_sweep_gpu4_20260516_175619/pad_16`

Final mask exports:

- `output/radio_gs/lerf_direct3d_sam3_box_pad0_best_masks_20260516`
- `output/radio_gs/lerf_direct3d_sam3_box_pad16_best_masks_20260516`

Global-threshold sweep roots:

- `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5`
- `output/radio_gs/lerf_direct3d_sam3_box_pad0_global_selector_20260516_210200_pad0_gpu5b`
- `output/radio_gs/lerf_direct3d_sam3_box_pad8_global_selector_20260516_210500_pad8_gpu0`
- `output/radio_gs/lerf_direct3d_sam3_box_pad4_global_selector_20260516_212900_pad4_gpu5`

Qualitative figures:

- `paper/figures/lerf_vpr_direct_3d_qualitative.png` is regenerated from the
  fixed-pad0 SAM3-box readout masks.
- `paper/figures/lerf_sam3_box_direct_3d_qualitative_pad16.png` is the
  fixed-pad16 boundary diagnostic.

## Verification

- `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_lerf_direct_3d_selection.py tests/test_build_sam3_foundation_cache.py`
  passed: 57 tests.
- `bash radio_gs/scripts/run_repo_python.sh -m py_compile radio_gs/scripts/eval_lerf_direct_3d_selection.py radio_gs/scripts/build_sam3_foundation_cache.py`
  passed.
- `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_summarize_direct3d_threshold_sweeps.py`
  passed: 2 tests.
