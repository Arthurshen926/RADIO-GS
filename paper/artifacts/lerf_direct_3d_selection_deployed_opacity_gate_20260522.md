# LERF Direct 3D Deployed Compact Field Audit

Date: 2026-05-22

This audit follows the OpenGaussian-style LERF-OVS direct 3D query-select-render protocol, but uses the deployed compact direct field instead of rendered-view VPR features at inference time. VPR is used only as a label-free multiview distillation teacher during point-summary-adapter training. At evaluation time, the point adapter is gated by a GT-free per-Gaussian opacity mask and no VPR feature cache or VPR valid mask is read.

## Method Change

The direct Gaussian text-score path now supports `--point_summary_adapter_valid_mask_mode opacity`. Low-opacity primitives fall back to the decoded compact RADIO-GS feature instead of using the point adapter on primitives that were not covered by the VPR distillation teacher. A separate `--direct_primitive_confidence_mode` was added for score calibration, but the final promoted setting keeps it disabled because score scaling reduced recall on several scenes.

Promoted deployed compact-field setting:

```bash
--score_source direct
--scoring softmax_scene
--use_point_summary_adapter
--point_summary_adapter_blend_alpha 1.0
--point_summary_adapter_valid_mask_mode opacity
--direct_primitive_confidence_mode none
--direct_primitive_opacity_threshold 0.02
--mask_refinement rgb_grabcut_largest_component
```

## Four-Scene Result

| Scene | Fixed tag | mIoU | Acc@0.25 | Acc@0.50 | Boundary F | Trimap IoU | Result JSON |
|---|---|---:|---:|---:|---:|---:|---|
| Figurines | `thr0p35` | 0.5147 | 0.6607 | 0.6071 | 0.6475 | 0.2659 | `output/radio_gs/lerf_direct3d_20260522_fig_deployed_opacity_gate_only/figurines/lerf_direct_3d_selection_results.json` |
| Ramen | `thr0p35` | 0.5726 | 0.7887 | 0.6901 | 0.7258 | 0.3809 | `output/radio_gs/lerf_direct3d_20260522_ramen_deployed_opacity_gate_only/ramen/lerf_direct_3d_selection_results.json` |
| Teatime | `thr0p35` | 0.5482 | 0.7119 | 0.6271 | 0.6910 | 0.3933 | `output/radio_gs/lerf_direct3d_20260522_teatime_deployed_opacity_gate_only/teatime/lerf_direct_3d_selection_results.json` |
| Waldo Kitchen | `thr0p35` | 0.2988 | 0.4091 | 0.3182 | 0.3288 | 0.2013 | `output/radio_gs/lerf_direct3d_20260522_waldo_deployed_opacity_gate_only/waldo_kitchen/lerf_direct_3d_selection_results.json` |
| Mean | `thr0p35` | 0.4836 | 0.6426 | 0.5606 | 0.5983 | 0.3103 | - |

Best-by-scene diagnostic macro mIoU is 0.4880:

| Scene | Best tag | mIoU | Acc@0.25 | Boundary F | Trimap IoU |
|---|---|---:|---:|---:|---:|
| Figurines | `thr0p15` | 0.5201 | 0.6607 | 0.6590 | 0.2826 |
| Ramen | `thr0p3` | 0.5732 | 0.7887 | 0.7251 | 0.3819 |
| Teatime | `thr0p55` | 0.5600 | 0.7119 | 0.7106 | 0.3973 |
| Waldo Kitchen | `thr0p35` | 0.2988 | 0.4091 | 0.3288 | 0.2013 |
| Mean | - | 0.4880 | 0.6426 | 0.6059 | 0.3158 |

## Interpretation

The deployed compact direct field now exceeds the previous registered-view VPR fixed row on macro mIoU: 0.4836 vs. 0.4801, while preserving the paper claim that a single compact foundation-feature Gaussian map supports both rendered-view query and direct primitive query. The gain is most visible on Waldo Kitchen, where the opacity-gated compact readout reaches 0.2988 mIoU, compared with the previous compact-field row around 0.2560 and the registered VPR table row around 0.2429.

The result also clarifies a failed branch: using no adapter support mask at inference collapses to roughly 0.05 mIoU on Figurines/Waldo because uncovered primitives are scored by an adapter trained only on VPR-visible primitives. The opacity gate fixes this without reading VPR cache during evaluation.
