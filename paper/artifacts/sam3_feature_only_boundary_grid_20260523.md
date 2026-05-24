# Feature-Only SAM3 Boundary Grid Audit

Date: 2026-05-23

This audit evaluates the feature-only SAM3 boundary branches for LERF direct 3D
object selection. Unlike the official SAM3 RGB readout, these rows do not call
the frozen SAM3 image encoder/decoder on the evaluation RGB image. They refine
the feature-derived rendered primitive mask through the existing
`sam3_adaptor_grabcut` path and a GT-free geometry boundary gate.

## Protocol

Common setting:

```bash
--score_source direct
--scoring softmax_scene
--softmax_temperature 50
--selection_mode score_threshold
--threshold_sweep 0.35
--selection_min_ratio 0.005
--selection_max_ratio 0.05
--use_point_summary_adapter
--strict_direct_head_consistency
--point_summary_adapter_blend_alpha 1.0
--point_summary_adapter_valid_mask_mode opacity
--direct_primitive_confidence_mode none
--silhouette_threshold 0.6
--mask_refinement sam3_adaptor_grabcut
--mask_refinement_iters 2
--mask_refinement_dilate 6
--mask_refinement_erode 2
--sam3_refinement_geometry_gate
```

The grid varied the GT-free geometry gate:

| Run | Min area | Max area | Min boundary gain |
|---|---:|---:|---:|
| `four_scene` | default | default | default |
| `a50_150_gn005` | 0.50 | 1.50 | -0.005 |
| `a70_130_gn005` | 0.70 | 1.30 | -0.005 |
| `a90_110_gn005` | 0.90 | 1.10 | -0.005 |
| `a70_130_gp005` | 0.70 | 1.30 | 0.005 |

## Four-Scene Means

| Run | mIoU | Initial mIoU | Delta | Boundary-F | Initial B-F | Delta | Trimap IoU | Initial Trimap | Delta | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `four_scene` | 0.4420 | 0.4430 | -0.0010 | 0.5881 | 0.5880 | +0.0001 | 0.2810 | 0.2821 | -0.0010 | 0.6574 | 0.4858 |
| `a50_150_gn005` | 0.4421 | 0.4431 | -0.0010 | 0.5876 | 0.5887 | -0.0011 | 0.2819 | 0.2837 | -0.0018 | 0.6574 | 0.4814 |
| `a70_130_gn005` | 0.4421 | 0.4431 | -0.0010 | 0.5888 | 0.5887 | +0.0001 | 0.2827 | 0.2837 | -0.0010 | 0.6574 | 0.4814 |
| `a90_110_gn005` | 0.4431 | 0.4431 | +0.0000 | 0.5905 | 0.5887 | +0.0018 | 0.2846 | 0.2837 | +0.0009 | 0.6616 | 0.4814 |
| `a70_130_gp005` | 0.4422 | 0.4431 | -0.0009 | 0.5884 | 0.5887 | -0.0003 | 0.2827 | 0.2837 | -0.0010 | 0.6574 | 0.4814 |

## Interpretation

The best feature-only setting is `a90_110_gn005`. It is effectively no-harm on
mIoU and gives small positive boundary metrics:

- mIoU: +0.0000 absolute over the initial rendered primitive mask.
- Boundary-F: +0.0018 absolute.
- Trimap IoU: +0.0009 absolute.
- Acc@0.25: +0.0042 absolute.

This is not comparable to the official SAM3 RGB boundary readout, which improves
the same deployed compact direct-field masks from 0.4430 to 0.5474 macro mIoU.
Therefore the feature-only branch should not be promoted as a paper main row.
It is currently a diagnostic branch showing that the adaptor/GrabCut route is
stable under a conservative geometry gate, but it does not yet provide a strong
boundary-refinement claim.

For the main paper, the safe claim remains:

> The compact direct field provides the 3D evidence. Frozen official SAM3 can be
> used as an image-assisted boundary readout when sharp masks are required.

The stronger claim that existing CTF-GS/RADIO features can directly emulate
official SAM3 mask decoding is not supported by this grid.

