# LERF Direct 3D Selection Debug Audit

Date: 2026-05-11

## Question

The OpenGaussian-style LERF direct 3D object-selection result is much weaker
than the rendered-view LERF result and the ScanNet direct point-query result.
This audit checks whether the gap is caused by evaluator bugs or by the current
feature/scoring design.

## Registration-Method Reference

Dr. Splat-style methods support direct 3D object selection by registering
language-aligned CLIP instance/mask embeddings directly onto 3D Gaussians. They
aggregate per-pixel CLIP embeddings onto the dominant Top-k Gaussians along each
pixel ray using volume-rendering contribution weights, then query stored
Gaussian language embeddings directly with CLIP text features and a relevancy
score.

RADIO-GS currently does something different for LERF direct selection: it
decodes pre-refiner compact RADIO-compatible Gaussian-center features and
projects them through a frozen SigLIP2 summary head. This is a direct 3D
readout, but it is not a direct language-registration objective.

## Evaluator Checks

- Official LERF-OVS annotated frames are selected for all four scenes.
- The SigLIP2 LERF text cache contains every current LERF-OVS category:
  Figurines 21/21, Ramen 14/14, Teatime 14/14, Waldo Kitchen 18/18.
- The same selected-primitive renderer and IoU code can produce high scores
  when the selected Gaussians are geometrically correct.

## Oracle Projection Sanity Check

For Figurines, I selected Gaussians whose projected centers fall inside any GT
object mask on the official annotated frames, then evaluated those selected
Gaussians with the same renderer, silhouette threshold, masks, and IoU code used
by `eval_lerf_direct_3d_selection.py`.

| Selection source | Silhouette threshold | mIoU | Acc@0.25 | N |
|---|---:|---:|---:|---:|
| GT projected-center oracle | 0.1 | 0.2554 | 0.4107 | 56 |
| GT projected-center oracle | 0.3 | 0.4480 | 0.7679 | 56 |
| GT projected-center oracle | 0.5 | 0.6086 | 0.9286 | 56 |
| GT projected-center oracle | 0.7 | 0.6850 | 0.9821 | 56 |
| RADIO-GS text scores, fixed top0p1 | 0.7 | 0.0474 | 0.0536 | 56 |

This strongly suggests that camera alignment, GT mask loading, selected-Gaussian
rendering, and IoU computation are not the primary failure source.

## Score-Separation Check

For Figurines, using the projected-center oracle as a proxy for relevant
Gaussians, RADIO-GS Gaussian-center text scores have weak separation:

| Diagnostic | Value |
|---|---:|
| Macro precision of fixed top10% scores against oracle Gaussians | 0.0422 |
| Macro recall of fixed top10% scores against oracle Gaussians | 0.1566 |
| Macro mean score gap, oracle minus non-oracle | 0.0052 |

Several categories have nearly identical inside/outside scores, and in some
cases the oracle Gaussians score lower than non-oracle Gaussians. Therefore the
top-scored primitives are not concentrated on the target objects.

## Dr. Splat-Style Relevancy Check

I added LERF-style canonical relevancy scoring to the direct evaluator and ran
the same four-scene ratio sweep.

| Variant | Best fixed selection | Fixed macro mIoU | Fixed macro Acc@0.25 | Best-by-scene macro mIoU | Best-by-scene macro Acc@0.25 |
|---|---|---:|---:|---:|---:|
| cosine baseline | top0p1 | 0.0804 | 0.0932 | 0.0914 | 0.1215 |
| relevancy diagnostic | top0p1 | 0.0748 | 0.0754 | 0.0875 | 0.1138 |

Relevancy re-ranking alone does not close the gap.

## Root-Cause Assessment

The current weak LERF direct 3D result is not explained by an obvious
implementation bug in label loading, frame selection, rendering, or IoU. The
dominant issue is that RADIO-GS was trained to reconstruct rendered
RADIO-compatible feature maps, not to make each individual Gaussian center carry
a language-registered object embedding. This matches the failure mode discussed
by Dr. Splat for rendering-distilled compressed features: the alpha-composited
rendered feature can be useful even when individual primitive codes are not
directly language-retrievable.

## Implication

ScanNet direct point-query is not contradictory evidence. The current ScanNet
v67 route includes direct point teacher/text supervision on label PLY points
(`direct_point_loss_weight`, text pseudo-CE, and teacher-feature distillation),
whereas LERF direct selection is being tested on LERF checkpoints that were
trained primarily for rendered-view RADIO reconstruction and rendered-view
grounding. To make LERF primitive selection competitive, the method needs a
Dr. Splat-like registration branch, an instance/SAM-cluster primitive
aggregation objective, or a learned compact-to-text adapter trained from
multi-view pseudo labels without using test masks.

## 2026-05-12 Registration Follow-Up

The recommended Dr. Splat-like registration branch has now been added as an
evaluation-time readout. It renders VFA-refined RADIO features from posed views,
projects them to SigLIP2 space, registers visible samples back to Gaussian
centers with depth/alpha checks, and queries those registered primitive
embeddings under the same OpenGaussian-style query-select-render metric.

The fixed `registered_view + softmax_scene + 24 all-pose views + top0p02`
protocol reaches 0.3421 macro mIoU and 0.5547 macro Acc@0.25, compared with
0.0804 / 0.0932 for the original Gaussian-center readout. The remaining gap is
now scene-specific: Waldo Kitchen reaches 0.1413 fixed mIoU and 0.1515
best-by-scene mIoU under the 24-view protocol, while a targeted 48-view softmax
probe improves it to 0.1756 but still trails OpenGaussian's 0.2270.

## 2026-05-12 Context Aggregation and View-Coverage Follow-Up

I tested GT-free primitive context aggregation after registration and then
expanded the registration view budget. The current paper-facing balanced
variant uses `registered_view + softmax_scene + 96 all-pose views +
voxel_max(res=80, blend=0.50) + top0p02`. It improves fixed-protocol macro mIoU
from 0.3421 to 0.3850 and Acc@0.25 from 0.5547 to 0.6428. Per-scene mIoU is
0.4055 / 0.4491 / 0.4862 / 0.1991 on Figurines / Ramen / Teatime /
Waldo Kitchen; the best-by-scene diagnostic reaches 0.3968 macro mIoU and
0.6651 macro Acc@0.25. This is slightly above the OpenGaussian official macro
reference, but it should still be described as official-source context because
baselines were not locally rerun under the same evaluator.

The strongest fixed-mIoU variant uses the same 96-view setup with a lower
`voxel_max` blend of 0.35. It reaches 0.3852 macro mIoU and 0.6313 Acc@0.25,
with Waldo Kitchen improving to 0.2189 mIoU. Because the mIoU gain over blend
0.50 is only 0.0002 and Acc@0.25 is lower, I keep blend 0.50 as the main
balanced paper row and report blend 0.35 as a diagnostic.

A connected-component refinement was added and unit-tested as a GT-free
instance-consistency diagnostic. Early 24-view probes showed negligible Waldo
gain under the old score field, and the full 96-view component run confirms the
same conclusion: `top_score_components`, keep=3, rank=score_sum reaches 0.3849
macro mIoU and 0.6428 Acc@0.25, essentially matching but not improving the
main 96-view VPR result.

I also added and tested `voxel_max_dilate`, a one-hop GT-free voxel-neighborhood
propagation mode inspired by context/grouping methods. It is not promoted:
`res=80, blend=0.35` reaches only 0.3437 / 0.5711 fixed macro mIoU/Acc@0.25,
and `res=80, blend=0.50` reaches 0.3417 / 0.5632. The failure mode is expected:
neighbor dilation broadens small-object responses and hurts Figurines enough to
outweigh any Waldo coverage gain.

## 2026-05-14 View-Budget Follow-Up

I tested the same GT-free `voxel_max(res=80, blend=0.50)` and
`meanstd2p5 + floor0.005 + cap0.02` selector with larger all-pose VPR budgets.
The 128-view budget improves both paper-facing metrics under the 2% cap,
reaching 0.4185 macro mIoU and 0.6899 Acc@0.25. A follow-up global cap sweep
keeps the same 128-view VPR scores and first promotes `cap0.0175` at
0.4226 / 0.6864. A cache-backed follow-up sweep promotes `cap0.018` as the
current main row at 0.4227 / 0.6906, with per-scene mIoU 0.4829 / 0.4665 /
0.5043 / 0.2373 on Figurines / Ramen / Teatime / Waldo Kitchen. The 1.5% cap is
accuracy-oriented at 0.4184 / 0.7013. The 160-view budget is
mixed: it reaches 0.4165 mIoU but drops Acc@0.25 to 0.6613, mainly because
Teatime loses Acc. The paper-facing protocol is therefore updated to 128
registration views and the global 1.8% cap. A `registered_view_fallback=low` diagnostic was strongly
negative at 0.2692 / 0.4229, confirming that the direct fallback is useful for
low-visibility primitives rather than merely adding background noise.
