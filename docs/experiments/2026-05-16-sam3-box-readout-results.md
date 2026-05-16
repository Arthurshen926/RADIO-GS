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

The paper-safe main row is fixed pad0 because it has the best fixed macro
mIoU/Acc@0.25. Fixed pad16 is the boundary-analysis row. A best-by-scene
padding diagnostic reaches 0.5980 macro mIoU and 0.7150 Acc@0.25, but it should
remain appendix-only unless a validation-selected padding rule is added.

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
