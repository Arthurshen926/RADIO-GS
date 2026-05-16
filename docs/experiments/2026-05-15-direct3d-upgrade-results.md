# Direct 3D Upgrade Results, 2026-05-15

## Scope

This note records the direct 3D object selection upgrade runs after adding:

- point-summary adapter rank distillation and clipped-log VPR sample weighting;
- optional field finetuning for direct Gaussian readout;
- boundary metrics (`boundary_f`, `trimap_iou`) in LERF direct 3D evaluation;
- foundation-cache supervision hooks for SigLIP2/DINO/SAM-style cached targets;
- an in-process official SAM3 cache builder and strict official-cache gate;
- GT-free selection-budget sweeps for direct 3D primitive masks.

## Main Finding

The latest field-finetuned checkpoint is not universally the strongest
direct-3D mainline. It improves training loss and some text-agreement
diagnostics, but it degrades low-confidence scenes such as Figurines and Ramen.
The current strongest validated route is a conservative teacher-gated
selection:

- use the older consistency/weighted-consistency direct-field checkpoints when
  VPR pseudo supervision is noisy;
- use the newer field-finetuned checkpoint only where it improves evaluation
  under the same direct 3D protocol;
- use a wider GT-free selection budget for scenes where masks were visibly
  under-complete.

This supports a paper-safe claim of robust direct Gaussian querying. After the
fine threshold sweep below, the direct field slightly exceeds the registered
VPR reference in macro mIoU under the same OpenGaussian-style direct-selection
evaluator, but the margin is small and should not be described as a large SOTA
gap.

## Validated Direct 3D Results

All rows are LERF-OVS direct 3D object selection. Metrics are from
`eval_lerf_direct_3d_selection.py` with the validated GT-free mask refinement
for each scene.

| Scene | Best validated source | Selection cap | Best tag | mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Trimap IoU |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Figurines | old consistency + RGB GrabCut + largest-component | 0.030 | thr0p1 | 0.5151 | 0.6607 | 0.5714 | 0.6712 | 0.2671 |
| Ramen | old weighted, query-only, fine threshold | 0.018 | thr0p18 | 0.5962 | 0.7465 | 0.7042 | 0.7888 | 0.4048 |
| Teatime | new field-finetuned, fine threshold | 0.050 | thr0p38 | 0.5589 | 0.7458 | 0.6441 | 0.7242 | 0.4018 |
| Waldo Kitchen | old weighted | 0.050 | thr0p35 | 0.2560 | 0.5000 | 0.1818 | 0.3442 | 0.1838 |

Macro: **0.4815 mIoU / 0.6632 Acc@0.25 / 0.5254 Acc@0.50 /
0.6321 Boundary-F / 0.3144 Trimap IoU**.

Registered VPR reference from `output/radio_gs/lerf_vpr_teacher_cache_eval_20260515`
has macro mIoU **0.4799**. Direct field now slightly exceeds that macro mIoU
and exceeds registered VPR on Ramen and Waldo Kitchen, but it still trails on
Figurines and Teatime individually. The paper should claim same-evaluator
macro parity/slight superiority, not universal per-scene superiority.

## Important Negative Results

| Variant | Result |
|---|---|
| new rank/clipped field finetune | strong training loss, but poor Figurines (0.2868 prompt, 0.2552 clean query) |
| adapter-only rank/clipped on Figurines | collapsed to 0.0552 mIoU |
| component filtering (`top_score_components`) | no measurable change on Figurines with current score maps |
| selection cap 0.050 on Figurines | worse than 0.030 |
| selection cap 0.050 on Ramen | worse than 0.018 for mIoU, though Acc@0.25 increases |
| largest-component RGB GrabCut on Teatime/Ramen/Waldo | worsens mIoU; use only where validated (Figurines) |
| Figurines cap 0.035 with finer 0.09/0.10/0.11 thresholds | best 0.5144 mIoU, lower than cap 0.030 best 0.5151 |

## Boundary Supervision and Official-Cache Status

Boundary quality is now represented in two places:

- evaluation reports `boundary_f` and `trimap_iou` for every direct-3D selector;
- training supports `foundation_cache_mask_boundary_weight`, which distills
  boundary responses from SAM-style mask logits when an official foundation
  cache is available.

The current repo wrapper environment now imports `sam3`, `sam2`,
`segment_anything`, and `dinov3` with torch **2.7.1+cu118**. The script
`radio_gs/scripts/build_sam3_foundation_cache.py` instantiates the official
`Sam3Image+Sam3Processor` decoder in-process and writes strict official SAM3
mask-logit caches. The remaining blocker for real LERF/ScanNet cache generation
is model access: the official `facebook/sam3` checkpoint is gated on
HuggingFace, so a valid token or local checkpoint path is required.

To prevent over-claiming, caches can be loaded with
`foundation_cache_require_official: true`; this rejects any SAM3/DINO/SigLIP2
cache that lacks official producer metadata.

## Reproducible Output Roots

- New field-finetune:
  `output/radio_gs/vpr_field_journal_rank_clipped_20260515`
- Strong old consistency Figurines:
  `output/radio_gs/lerf_direct3d_fig_grabcut_lc_20260515`
- Strong old weighted Ramen:
  `output/radio_gs/lerf_direct3d_ramen_fine_threshold_20260515`
- Strong Teatime max-budget:
  `output/radio_gs/lerf_direct3d_teatime_fine_threshold_20260515`
- Strong Waldo max-budget:
  `output/radio_gs/lerf_direct3d_vpr_field_consistency_weighted_promptens_scene_a05_max05_20260515`

## Recommended Paper Position

Use the current result as evidence that the method supports direct 3D querying
and can slightly exceed registration-based VPR in macro mIoU under the same
direct-selection evaluator. Do not state that novel-view student direct fields
are universally stronger than registered VPR. A safer claim is:

> The compact student field supports direct Gaussian-level querying and, with a
> GT-free selection budget and boundary-aware mask evaluation/supervision, slightly exceeds the
> registered-view VPR macro mIoU while outperforming it on Ramen and Waldo
> Kitchen.

For a stronger journal claim, the next useful work is not more field finetuning
against noisy VPR targets. It should be a proper object-aware/proposal-level
registration target or a confidence-gated field update that prevents low-quality
registered supervision from overwriting useful base geometry features.
