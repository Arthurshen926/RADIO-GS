# Prompt-Conditioned Internal SAM3 Boundary Readout: Training-View Pseudo Masks

Date: 2026-05-23

This run upgrades the earlier Figurines-only diagnostic into a training-view
pseudo-mask experiment. Official SAM3 masks are generated on unlabelled
training views and used only as pseudo supervision. Evaluation does not call
the official SAM3 RGB readout; it uses rendered CTF-GS features, SigLIP2 text
prompts, and the method's coarse mask.

## Implementation Notes

- Added training-view direct-3D coarse-mask rendering:
  `radio_gs/scripts/render_lerf_direct3d_coarse_masks.py`.
- Updated `train_prompt_conditioned_sam3_mask_head.py` to train on unlabelled
  cache frames and all scene queries, not only labelled/eval frames.
- Added cached-source-feature training to avoid repeated CPU SAM3-mask resize
  and repeated feature rendering inside every epoch.
- Fixed `foundation_cache` parsing for official SAM3 frames with zero proposals.
- Added shared feature-only prompt-mask refinement utilities in
  `radio_gs/models/prompt_conditioned_mask_refinement.py`.
- Added `sam3_prompt_mask_head` support to LERF rendered-view grounding, with
  default application only to rendered/student features.

## Direct 3D Selection Result

Protocol: LERF-OVS official frames, direct compact field scores,
`score_threshold=0.35`, `silhouette_threshold=0.7`, geometry/area gate with
refined-mask area ratio cap `1.6`. The gate is global and GT-free.

| Scene | n | Initial mIoU | Refined mIoU | Delta | Initial Acc@0.50 | Refined Acc@0.50 | Initial Boundary-F | Refined Boundary-F | Initial Trimap | Refined Trimap | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Figurines | 56 | 0.3746 | 0.4010 | +0.0264 | 0.4107 | 0.4464 | 0.6109 | 0.6204 | 0.1367 | 0.1808 | 21/56 |
| Ramen | 71 | 0.5022 | 0.5049 | +0.0028 | 0.5352 | 0.5352 | 0.6441 | 0.6449 | 0.3390 | 0.3375 | 8/71 |
| Teatime | 59 | 0.5289 | 0.5336 | +0.0047 | 0.6102 | 0.6271 | 0.6955 | 0.6968 | 0.3732 | 0.3702 | 14/59 |
| Waldo Kitchen | 22 | 0.2471 | 0.2493 | +0.0022 | 0.2273 | 0.2727 | 0.2789 | 0.2797 | 0.1652 | 0.1673 | 1/22 |
| Weighted mean | 208 | 0.4484 | 0.4580 | +0.0096 | 0.4904 | 0.5096 | 0.6111 | 0.6144 | 0.2758 | 0.2866 | 44/208 |

Artifacts:

- Training-view coarse masks:
  `output/radio_gs/prompt_sam3_trainview_coarse_20260523/`
- Official SAM3 training-view caches:
  `output/radio_gs/foundation_cache_sam3_modelscope_mapped_trainviews/`
- Trained heads:
  `output/radio_gs/prompt_sam3_mask_head_20260523/*_trainviews_directcoarse_e60_cache/`
- Direct-3D eval:
  `output/radio_gs/lerf_direct3d_prompt_sam3_trainviews_gate16_20260523/`

## Rendered-View Grounding Diagnostic

The same head was also connected to LERF rendered-view grounding. It should not
be promoted as a main result yet: the prompt head was trained with direct-3D
coarse masks, whose distribution differs from heatmap-threshold masks.

Rendered/student-only application gives a small weighted mIoU increase:
`0.0281 -> 0.0307` (`+0.0027`, 187/208 accepted). Teacher features are left
unmodified by default because applying this rendered/direct-coarse head to
frame-wise teacher features degraded teacher mIoU in a diagnostic run.

Artifacts:

- `output/radio_gs/lerf2d_prompt_sam3_trainviews_renderedonly_gate16_20260523/`

## Conclusion

This is now positive method-level evidence for an internal feature-only SAM3
boundary readout on the direct-3D object-selection task. It is no longer merely
an eval-frame diagnostic. The strongest paper-safe claim is:

> A prompt-conditioned internal SAM3 boundary readout, distilled from official
> SAM3 training-view pseudo masks, improves direct 3D object-selection mask
> quality and boundary metrics without calling official SAM3 at evaluation time.

The module should be reported for direct-3D selection. Rendered-view 2D usage
needs a separately trained 2D-heatmap-coarse variant before it can become a main
claim.

## 2026-05-23 Rendered-View Follow-up

The rendered-view variant has now been implemented with heatmap-threshold
training-view coarse masks and a support-constrained acceptance gate. See
`paper/artifacts/sam3_prompt_mask_head_lerf2d_support_readout_20260523.md`.
The latest stable gate gives positive weighted sample-mIoU evidence on active
checkpoints and removes the previous Ramen negative delta, but it is currently a
boundary-readout ablation rather than an unconditional replacement for the LERF
category-macro main table.

## 2026-05-24 Direct-3D Support-Gate Diagnostic

The rendered-view support gate was also ported to the direct-3D prompt-mask-head
readout. This did not improve the direct-3D result:

| Variant | Weighted initial mIoU | Weighted refined mIoU | Delta | Acc@0.25 |
|---|---:|---:|---:|---:|
| original direct-coarse gate | 0.4484 | 0.4580 | +0.0096 | 0.6683 |
| support gate, logit -1 / area 0.50-1.20 / support 12 | 0.4484 | 0.4541 | +0.0057 | 0.6683 |
| support gate, logit 0 / area 0.70-1.30 / support 12 | 0.4484 | 0.4525 | +0.0040 | 0.6635 |

Artifacts:

- `output/radio_gs/lerf_direct3d_prompt_sam3_support_sweep_20260524/`

Conclusion: support clipping is useful for rendered 2D heatmap masks, where the
coarse prompt is already a dense query support. It is too restrictive for
direct-3D selected-primitive projections, where missing/fragmented primitive
coverage needs more freedom. Keep the original direct-coarse gate for direct 3D
and use the support gate only for rendered-view boundary readout.
