# Reference identity, extent completeness and source writing, next74

This iteration keeps the existing map / typed-query / element-set framework.
It investigates source evidence before changing semantic compression or adding
new scene-specific teachers. Every reported score is development evidence.
Prediction uses authorized source RGB and the existing positive reference
prompt, not target RGB, target masks or additional negative scribbles.

## 1. Reference identity is not a Gaussian visibility fraction

Implemented `prepare_reference_anchors_next74.py`: lift official positive
scribbles onto the new next72 NVOS carrier, then locate their visible source
image positions. This is a new reference-prompt evidence provider; failures in
its first prototype are not automatically evidence of a bug in every retained
query path.

The broad prototype accepted Gaussians whose reference contribution fell mostly
inside a positive scribble. A weak Gaussian behind the actual foreground could
pass this ratio and become prominent from another source view. In orchids
source3 this put all three selected points onto background instead of flowers.

The revised provider requires maximum-contribution ownership in the reference
pixel AND in the receiving source pixel. It corrects the illustrated background
placement. Maximum contribution is not guaranteed physical ground truth or
necessarily the nearest surface; remaining geometry errors still matter.
The broad mode is retained explicitly for reproduction, not as the default.

Source-only visualization (cyan=mask, red=the initial three points):

![Source correspondence comparison](../../output/reference_cover_next74/source_correspondence_comparison.png)

These images contain no target GT. They confirm a correspondence failure and
its local correction; they do not prove improved final query accuracy.

## 2. Correct positive locations do not specify a complete object set

Implemented `decode_reference_anchors_next74.py` with two policies:

- Joint three-point SAM3 decoding, keeping the original reference mask and using
  the frozen all-source video output where no new candidate is accepted.
- Covering the transported positive set: decode the strongest uncovered anchor,
  remove anchors covered by its mask, repeat until all candidates are explained
  or locally rejected. There is no fixed cap on the number of proposed parts.
  Candidate quality and positive-point inclusion are checked on source inputs.

The same frozen field reconstruction evaluates both replacement and union with
video evidence. Replacement can replace a whole cluster by several individual
flowers and severely regress. Union avoids deleting the old extent but adds
false positives. This is why source SAM quality or a positive prompt cannot be
interpreted as certification of complete extent.

Implemented `validate_reference_parts_next74.py`: additionally compare each
new part's native SigLIP foreground and same-window context similarity to the
same authorized reference-image crop. This keeps identity evidence separate
from membership assignment. It is still a heuristic identity check, not a
verified label or a new learned multi-teacher student.

Two-scene pilot results (all use the new carrier):

| Source evidence | orchids IoU / AP | leaves IoU / AP |
|---|---:|---:|
| Frozen all-source video | 65.859 / 98.003 | 92.552 / 99.774 |
| Broad anchors, replace | 46.057 / 89.911 | 92.095 / 99.444 |
| Broad anchors, union | 66.822 / 92.713 | 92.784 / 99.532 |
| Dominant anchors, replace | 45.742 / 90.448 | 91.904 / 99.478 |
| Dominant anchors, union | 66.010 / 93.654 | 92.773 / 99.580 |
| Cover positive set, replace | 54.395 / 94.680 | 91.396 / 98.734 |
| Cover positive set, union | 66.460 / 96.127 | 92.762 / 99.143 |
| Reference-image identity filtered union | 66.304 / 96.478 | 92.643 / 99.110 |

**Decision:** no variant jointly improves IoU and ranking sufficiently to justify
extending the proposal route to all scenes. Do not promote it, and do not choose
per-scene winners. The correction to anchor ownership remains useful research
code, but has not established a benchmark gain by itself.

## 3. Does the video incorrectly exclude the known positive identity?

`condition_video_exclusion_next74.py` measures the weighted fraction of visible
reference-positive anchors covered by each source video mask. The experiment
scales only negative evidence by this coverage; it does not change mask logits,
positive evidence, geometry, target access or the0.5 evaluation threshold.
Unknown object presence remains unknown as in the retained video protocol.

Mean source anchor coverage is99.803% for orchids and99.995% for leaves.
The conditioned results are65.860/92.552IoU, effectively unchanged. Thus omission
of these KNOWN reference positives does not explain the low orchids score.
This does not establish completeness of the unprompted object parts.

## 4. Check information loss when masks are written into the field

The remaining strong alternative is the discretized source inverse problem:
SAM masks are available at much higher resolution, but the field sees96x128
visibility mixtures. A single coarse pixel can couple distinguishable surfaces.

Added an explicit `--fit-scale` to the existing reconstruction. The bounded
control uses192x256 source operators and masks, the same120updates, same frozen
all-source video, same geometry and same target renderer/scorer. The eroded
unknown-boundary radius scales1→2pixels with the raster; it is not narrowed
merely to add supervision. Actual observed-element support is recomputed at the
new raster, so this intervention includes both source projection and support
coverage changes. It costs more source pixels and is not an equal-compute claim.

Pilots improve orchids65.859→67.962IoU and leaves92.552→94.084IoU.
Orchids AP98.003→97.869 slightly falls; leaves AP99.774→99.792 slightly rises.
The old/new geometry by source-resolution comparison is being completed on all
eight queries before selecting a candidate. Final numbers are appended below.

## LERF and other tasks

Three full-source semantic fields have now completed:
figurines66.553IoU/84.908AP/57.479selected,
ramen61.659/77.485/43.722,
waldo62.950/78.424/54.992. Teatime is still reconstructing at this writing.
Use the complete four-scene aggregate when available, not these partial means.
The selected score is a2D projection of selected Gaussians, not independent3DGT.

No new ScanNet semantic model or UQIS run is claimed. No compact-student or
four-query validation claim is supported by these prompt-only pilots. Source
coverage is an input improvement; high-resolution inverse rendering is a
representation correction. Neither alone establishes a novel CVPR contribution.
The unresolved research requirement is reliable full-set supervision consistent
with typed query identity, rather than generic propagation of semantic guesses.

Machine-readable results: `output/next74/summary.json`.

## Completed NVOS8 source-write result

| Geometry | 96x128 IoU / AP | 192x256 IoU / AP |
|---|---:|---:|
| old | 80.110 / 94.835 | 82.347 / 96.115 |
| new | 83.395 / 97.107 | 86.489 / 98.014 |

All eight new-carrier queries improve IoU; no per-scene selection.

| Query | New carrier, coarse | New carrier, fine |
|---|---:|---:|
| fern | 78.943 | 82.183 |
| flower | 84.436 | 86.589 |
| fortress | 94.514 | 97.212 |
| horns_center | 93.294 | 95.730 |
| horns_left | 90.542 | 92.659 |
| leaves | 92.552 | 94.084 |
| orchids | 65.859 | 67.962 |
| trex | 67.019 | 75.494 |

Fine source writing reaches86.489IoU/98.014AP, exceeding the retained complete
next43 route83.484IoU/96.057AP. Compared with the SAME new geometry and full
video, the increase is+3.094IoU/+0.907AP. Old geometry also improves+2.237IoU;
the geometry gain grows3.285→4.142pp when source writing is finer. This supports
a coupling between carrier quality and the information resolved by supervision.
It does not establish that resolution is the only remaining limitation.

Keep this as the new NVOS high-capacity research candidate. Do not bundle any
failed prompt proposal, identity-union or prior-transfer variant into it. The
new geometry still lacks a rebuilt compact native/structural student; this
mask-only query reconstruction does not certify same-map four-query deployment.

The full LERF fusion execution was accelerated by reducing each source adjoint
to touched Gaussian rows before scattering into the full native field. This
preserves all weighted contributions and all source views; it is an execution
change, not teacher sampling, a new semantic objective or a research gain.
The unfinished teatime full fit was restarted with this equivalent expression.

## Closure: all planned jobs in this iteration completed

All four uncapped LERF cohorts now have complete native reconstruction, source
refinement, extent decoding and frozen-protocol evaluation. No pending scores
are included.

| Four-scene LERF | 2D IoU | AP | Selected-subset projection IoU |
|---|---:|---:|---:|
| Retained next68 | 64.882 | 81.021 | 51.835 |
| Fresh old-cohort control | 64.966 | 81.353 | 51.871 |
| Coverage64 control | 66.098 | 82.578 | 53.279 |
| All293/122/169/180source teachers | **65.958** | **82.631** | **53.612** |

Final teatime all-source72.669IoU/89.707AP/58.255selected. The all-source
recipe gains+1.076IoU/+1.610AP/+1.778selected over retained. Against coverage64,
its2D IoU slightly falls, AP and selected projection improve slightly. Additional
coverage has diminishing returns; do not infer monotonically better semantics
from more frames. All-source is the predeclared uncapped reference cohort, not
a scene-wise assembly of the best64/all results.

A bounded source-write convergence check at384x512, boundary radius4, gives
orchids67.474IoU/97.447AP and leaves91.010IoU/99.770AP. Both IoUs are worse than
192x256, so this larger raster is rejected without a full-eight expansion.
The recommended NVOS research candidate remains the full-eight192x256 recipe.
Higher resolution does not fix inconsistent source masks or guarantee more
accurate geometry; a fixed high raster alone is not a general method claim.

No candidate in the source-resegmentation chain is bundled with either retained
research result. Source masks/identity hints remain imperfect observations.
Next method priority is a shared feature-field supervision operator that resolves
geometry without confusing part membership, object identity and unknown scope,
followed by rebuilding/validating the native and compact multimodal maps on that
carrier. ScanNet public10 and independent3D-ground-truth LERF validation remain
separate unfinished project work; this closure only describes this iteration.
