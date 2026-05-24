# SAM3 Feature-Prompt Boundary Readout Audit

Date: 2026-05-22

This audit implements the expert-recommended SAM3 boundary experiment for LERF
direct 3D object selection. The coarse evidence is produced by the existing
compact direct field: text scores select 3D Gaussians, the selected primitives
are rendered into a coarse binary silhouette, and that feature-derived mask is
converted into a normalized SAM3 box prompt. Frozen official SAM3 then refines
the boundary on the evaluation RGB image. SAM3 candidate masks are selected by
overlap with the initial rendered mask only; ground-truth masks are used only
afterward for metrics.

## Protocol Fixes

- Added paired pre/post metrics in `eval_lerf_direct_3d_selection.py`:
  `initial_miou`, `delta_miou`, `initial_boundary_f`, `delta_boundary_f`,
  `initial_trimap_iou`, and `delta_trimap_iou`.
- Added SAM3 behavior audit fields: `sam3_attempt_count`,
  `sam3_skip_count`, `sam3_accept_count`, `sam3_accept_rate`,
  `sam3_fallback_reasons`, and per-query prompt/candidate records.
- Added protocol provenance hashes for the RADIO-GS config/checkpoint, score
  cache, text cache, summary head, official SAM3 checkpoint, and repo commit.
- Fixed a SAM3 logits fallback threshold from `>0.5` to `>0.0`.
- Wrapped official SAM3 image/prompt inference in `torch.no_grad()`.
- Restored `checkpoints/siglip2_lerf_text_embeddings.pt` to the full 63-query
  LERF cache after detecting that parallel scene jobs can overwrite a shared
  single-scene text cache. Because score-cache metadata stores the text-cache
  path, current cache-backed runs must keep the original path.

## Main Audited Readout

Setting:

```bash
--score_source direct
--scoring softmax_scene
--softmax_temperature 50
--selection_mode score_threshold
--threshold_sweep 0.35
--selection_min_ratio 0.005
--use_point_summary_adapter
--point_summary_adapter_blend_alpha 1.0
--point_summary_adapter_valid_mask_mode opacity
--direct_primitive_confidence_mode none
--silhouette_threshold 0.6
--mask_refinement sam3_box
--sam3_box_padding 8
--sam3_min_initial_iou 0.05
```

| Scene | Raw mIoU | SAM3 mIoU | Delta | Raw B-F | SAM3 B-F | Raw Trimap | SAM3 Trimap | Acc@0.25 | Acc@0.50 | Accept / Attempt | JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Figurines | 0.4202 | 0.5626 | +0.1424 | 0.5953 | 0.6581 | 0.1750 | 0.3629 | 0.6786 | 0.6071 | 46 / 56 | `output/radio_gs/lerf_direct3d_20260522_fig_deployed_opacity_gate_sam3_box/figurines/lerf_direct_3d_selection_results.json` |
| Ramen | 0.5698 | 0.6630 | +0.0933 | 0.7514 | 0.7555 | 0.3902 | 0.4559 | 0.8028 | 0.6901 | 70 / 71 | `output/radio_gs/lerf_direct3d_20260522_ramen_deployed_opacity_gate_sam3_box/ramen/lerf_direct_3d_selection_results.json` |
| Teatime | 0.5151 | 0.5622 | +0.0470 | 0.6896 | 0.7220 | 0.3709 | 0.4063 | 0.6780 | 0.6271 | 53 / 59 | `output/radio_gs/lerf_direct3d_20260522_teatime_deployed_opacity_gate_sam3_box/teatime/lerf_direct_3d_selection_results.json` |
| Waldo Kitchen | 0.2669 | 0.4019 | +0.1350 | 0.3156 | 0.4420 | 0.1922 | 0.2445 | 0.5000 | 0.5000 | 17 / 22 | `output/radio_gs/lerf_direct3d_20260522_waldo_deployed_opacity_gate_sam3_box/waldo_kitchen/lerf_direct_3d_selection_results.json` |
| Mean | 0.4430 | 0.5474 | +0.1044 | 0.5880 | 0.6444 | 0.2821 | 0.3674 | 0.6648 | 0.6061 | 186 / 208 | - |

All four scenes have `sam3_skip_count=0`; rejected cases are empty initial
masks, not missing SAM3 state.

## Comparison to RGB Cleanup

The previous deployed compact-field row used GT-free
`rgb_grabcut_largest_component` boundary cleanup, not raw masks. Against that
stronger cleanup baseline, the audited official SAM3 readout still improves the
strict fixed-threshold result:

| Readout | Selector | Macro mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Trimap IoU |
|---|---|---:|---:|---:|---:|---:|
| Raw rendered silhouette | fixed `thr0p35` | 0.4430 | 0.6616 | 0.4816 | 0.5880 | 0.2821 |
| RGB cleanup | fixed `thr0p35` | 0.4836 | 0.6426 | 0.5606 | 0.5983 | 0.3103 |
| Official SAM3 box readout | fixed `thr0p35`, pad8 | 0.5474 | 0.6648 | 0.6061 | 0.6444 | 0.3674 |

## Secondary Check

A fixed pad16 / `thr0p25` rerun was also tested with the same deployed score
caches and new audit fields. It improves raw masks by a similar amount but is
weaker than the pad8 strict row on macro mIoU:

| Readout | Selector | Macro Raw mIoU | Macro SAM3 mIoU | Delta mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
|---|---|---:|---:|---:|---:|---:|---:|
| Official SAM3 box, pad16 | fixed `thr0p25` | 0.4352 | 0.5399 | +0.1047 | 0.6706 | 0.6339 | 0.3813 |

The older frozen pad16 global-threshold submission row remains a stronger
historical diagnostic for a different score source, but the table above is the
current paired audit for the deployed compact direct field.

## Interpretation

This experiment supports a conservative paper claim:

> The compact foundation-feature Gaussian field provides the coarse 3D evidence
> for direct object selection, and a frozen official SAM3 box-prompt boundary
> readout can turn that feature-derived coarse mask into objectively sharper
> masks without using ground truth for candidate selection.

It should not be described as "RADIO features directly emulate the official
SAM3 image-encoder state." The official SAM3 decoder-state bridge remains a
separate high-risk branch; the promoted result here is a feature-prompted,
image-assisted frozen SAM3 boundary readout with paired objective metrics.
