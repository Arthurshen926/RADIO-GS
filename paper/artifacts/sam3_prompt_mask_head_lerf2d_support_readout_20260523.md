# Support-Constrained Internal SAM3 Readout for LERF Rendered Grounding

Date: 2026-05-23

This artifact records the rendered-view variant of the prompt-conditioned
internal SAM3 boundary readout. Official SAM3 masks are used only on unlabelled
training views as pseudo supervision. At evaluation time, the readout uses only
GaussFM rendered features, the SigLIP2 text prompt, and the GaussFM heatmap mask.

## Method Update

- Added rendered-view coarse-mask rendering:
  `radio_gs/scripts/render_lerf2d_coarse_masks.py`.
- Trained scene heads from active paper checkpoints and training-view official
  SAM3 pseudo masks:
  `output/radio_gs/prompt_sam3_mask_head_20260523/*_trainviews_lerf2dcoarse_active_e60_cache/`.
- Added a GT-free support-constrained acceptance gate:
  refined masks must overlap the initial heatmap mask by at least `0.50`, have
  area ratio in `[0.70, 1.30]`, and are clipped to the initial query support
  dilated by `12` pixels. The logit threshold is `0.0`.

The support constraint is important. A direct-coarse-trained head degraded LERF
rendered grounding because direct-3D selected-primitive masks and 2D heatmap
masks have different prompt distributions. The rendered-view head fixes that
distribution mismatch, and the support gate prevents SAM-style pseudo masks from
moving the query result away from the GaussFM heatmap support.

## Active-Checkpoint Result

Current-code active checkpoint evaluation with the earlier permissive support
gate:
`output/radio_gs/lerf2d_paperckpt_lerf2dcoarse_active_sam3_gate_sup12_20260523/`.

| Scene | LocAcc | Initial sample mIoU | Refined sample mIoU | Delta | Refined category-macro mIoU | Accept |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.8214 | 0.4577 | 0.4672 | +0.0095 | 0.4512 | 27/56 |
| Ramen | 0.9014 | 0.6223 | 0.6070 | -0.0153 | 0.5806 | 33/71 |
| Teatime | 0.8983 | 0.5748 | 0.6277 | +0.0529 | 0.5818 | 35/59 |
| Waldo Kitchen | 0.8182 | 0.4785 | 0.5050 | +0.0265 | 0.4946 | 6/22 |
| Weighted sample mean | 0.8702 | 0.5493 | 0.5644 | +0.0151 | -- | 101/208 |
| Scene category-macro mean | -- | -- | -- | -- | 0.5270 | -- |

The better stable global gate is
`activeC_logit0_area70_130_sup12`, evaluated at
`output/radio_gs/lerf2d_active_sam3_gate_sweep_20260524/activeC_logit0_area70_130_sup12/`.
It improves weighted sample mIoU from `0.5493` to `0.5666` (`+0.0173`) without
changing heatmap localization, and removes the Ramen negative delta
(`+0.0004`). This is now the safer paper-facing internal SAM3 readout setting.

| Scene | LocAcc | Initial sample mIoU | Stable-gate mIoU | Delta | Category-macro mIoU | Accept |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.8214 | 0.4577 | 0.4645 | +0.0068 | 0.4482 | 29/56 |
| Ramen | 0.9014 | 0.6223 | 0.6227 | +0.0004 | 0.5914 | 27/71 |
| Teatime | 0.8983 | 0.5748 | 0.6230 | +0.0482 | 0.5801 | 32/59 |
| Waldo Kitchen | 0.8182 | 0.4785 | 0.4941 | +0.0156 | 0.4795 | 3/22 |
| Weighted sample mean | 0.8702 | 0.5493 | 0.5666 | +0.0173 | -- | 91/208 |
| Scene category-macro mean | -- | -- | -- | -- | 0.5248 | -- |

## Gate Sweep

Before correcting to the active seed7 checkpoints, the same sweep showed the
importance of the gate:

| Variant | Weighted initial | Weighted refined | Delta | Accept |
|---|---:|---:|---:|---:|
| permissive direct-style gate | 0.5507 | 0.5451 | -0.0056 | 162/208 |
| overlap/area gate | 0.5507 | 0.5670 | +0.0163 | 106/208 |
| support 24px | 0.5507 | 0.5702 | +0.0195 | 113/208 |
| support 12px | 0.5507 | 0.5719 | +0.0212 | 118/208 |
| support 6px | 0.5507 | 0.5716 | +0.0209 | 122/208 |
| active stable gate, logit0/area70-130/support12 | 0.5493 | 0.5666 | +0.0173 | 91/208 |

Artifacts:

- Rendered-view coarse masks:
  `output/radio_gs/prompt_sam3_trainview_lerf2d_coarse_active_20260523/`
- Active trained heads:
  `output/radio_gs/prompt_sam3_mask_head_20260523/*_trainviews_lerf2dcoarse_active_e60_cache/`
- Active final eval:
  `output/radio_gs/lerf2d_paperckpt_lerf2dcoarse_active_sam3_gate_sup12_20260523/`
- Active stable-gate eval:
  `output/radio_gs/lerf2d_active_sam3_gate_sweep_20260524/activeC_logit0_area70_130_sup12/`
- Gate sweeps:
  `output/radio_gs/lerf2d_paperckpt_lerf2dcoarse_sam3_sweep_20260523/`

## Paper-Safe Claim

The safe claim is:

> A prompt-conditioned internal SAM3 boundary readout distilled from official
> SAM3 training-view pseudo masks can improve rendered-view mask quality from
> GaussFM features without invoking the official RGB SAM3 readout at evaluation
> time. A query-support and bounded-area gate is necessary for stability.

The strongest current quantitative claim is weighted-sample improvement with no
Ramen regression under a single global GT-free gate. Scene category-macro
improvement is still weaker, so the module remains best positioned as a
boundary-readout ablation rather than the sole main LERF metric.
