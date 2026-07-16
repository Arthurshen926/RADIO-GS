# canonical-mpr-v3: Region-summary and official SAM3 analysis

## Freeze and evaluation policy

The exact method, official C-RADIO checkpoint, global readout checkpoint,
source hashes, prompt solver, and benchmark manifests are locked by
`canonical_mpr_v3_evaluation_freeze_20260716.yaml`.  No benchmark mask/label,
query-set statistic, per-scene threshold, or test-set calibration may change
the method after this point.  Failed jobs may only be resumed with identical
numerical semantics.

For LERF, two readouts are intentionally separated:

- independent raw cosine is the query-set-invariant open-vocabulary result;
- `softmax_scene` is a closed-taxonomy, paper-compatibility result and depends
  on the declared scene category set.

## Region-summary language alignment

### The teacher ladder identifies the correct interface

All teacher-oracle rows use the official C-RADIOv4 checkpoint, official
SigLIP2-G tokenizer/text tower, exact benchmark query strings, and no 3-D
field.

| representation/readout | LERF sample mIoU | LocAcc | interpretation |
|---|---:|---:|---|
| official SigLIP2 spatial, protocol aligned | 0.015317 | 0.043269 | spatial tokens are not directly language usable |
| official crop summary, protocol aligned | 0.150193 | 0.283654 | official summary route is viable |
| canonical crop-summary MPR, scene macro | 0.248048 | 0.431605 | multi-view lifting stabilizes the teacher |
| v3 global 3-D readout, closed scene softmax, scene macro | 0.343318 | 0.604467 | query-free surface aggregation improves the compatible closed-set row |
| v3 global 3-D readout, independent cosine, scene macro | 0.219432 | 0.615628 | invariant open-vocabulary result |

The invariant v3 per-scene mIoU is
0.279072/0.147029/0.313963/0.137666 for
Figurines/Ramen/Teatime/Waldo Kitchen.  Its sample-micro mIoU is 0.228940 over
208 annotations.  It is substantially above the official 2-D crop-summary
direct-cosine oracle (0.089531 sample mIoU), so the learned global readout is
not merely exploiting the closed scene category denominator.  The larger
closed-set number must nevertheless remain explicitly labelled
`softmax_scene`.

### What the readout reconstructs

On eight scene-disjoint ScanNet validation scenes, the frozen readout obtains:

| metric | untrained | trained |
|---|---:|---:|
| selection/mean-descriptor score | 0.761330 | 0.926473 |
| all-view descriptor cosine | 0.753990 | 0.916762 |
| exact raw summary-token cosine | 0.198874 | 0.683351 |

The official final SigLIP descriptor is much more reproducible than the exact
1280-D backbone summary token.  This is expected: a canonical surface set can
identify stable region semantics, but not all view-specific crop composition,
occlusion, and background context contained by one image summary token.

The physical-scale audit makes the tradeoff explicit:

| radius | summary token | mean descriptor | all-view descriptor | relation L1 |
|---:|---:|---:|---:|---:|
| 0.25 m | 0.672125 | 0.938987 | 0.925447 | 0.043394 |
| 0.45 m | 0.681939 | 0.938449 | 0.916383 | 0.067673 |
| 0.70 m | 0.702307 | 0.928581 | 0.905938 | 0.068803 |

Increasing context slightly helps exact-token imitation while hurting
view-stable semantics and pairwise relations.  Therefore v3 correctly selects
by descriptor/view stability rather than chasing raw-token cosine.  The
remaining language error is primarily region definition and context
ambiguity, not lack of a larger per-scene head.  The readout remains one
global, permutation-invariant, query-free model trained without annotations,
instances, text, or benchmark vocabulary.

## Official SAM3 local boundary relations

### Primitive and screen-space audits give opposite answers

The new primitive audit compares the frozen canonical SAM3 capability rows
against the official-SAM3-before-matched-MPR teacher on exactly the support
graph edges.  It opens no task labels or prompts and performs no rendering.

| ScanNet scene | primitive row cosine | primitive relation Pearson | primitive boundary retention | rendered boundary retention |
|---|---:|---:|---:|---:|
| scene0062 | 0.964235 | 0.962918 | 0.933522 | 0.032605 |
| scene0140 | 0.970030 | 0.978986 | 0.909226 | 0.016084 |
| scene0200 | 0.970525 | 0.972017 | 0.913445 | 0.011959 |
| macro | 0.968263 | 0.971291 | 0.918731 | 0.020216 |

Thus the old statement “the canonical field reconstructs official SAM3 local
relations weakly” is incorrect.  The canonical primitive field reconstructs
them strongly.  The approximately 45-fold retention collapse occurs in the
primitive-to-pixel observation operator.

The screen audit still has a seemingly reasonable official-SAM3 mean cosine
of 0.705126, while retaining only 0.020216 of the teacher boundary margin.
Mean pointwise cosine therefore hides the failure: alpha blending mixes
foreground/background raw RADIO descriptors near depth/visibility edges, and
the frozen nonlinear SAM3 MLP projection cannot recreate a discontinuity that
has already been averaged away.

The promoted rank-16 boundary-conditioned residual improves screen retention
consistently but only from approximately 0.029859/0.014094/0.010809 to
0.032605/0.016084/0.011959.  It is a valid low-capacity observation-fidelity
module, not a solution to the compositing bottleneck.

### Method consequence

Primitive-domain queries should continue to use the official SAM3 primitive
capability and typed support-graph boundary channel directly.  They should not
render a high-dimensional SAM3 map and then rediscover the primitive decision.
For 2-D outputs, the preferred path is score/selected-membership before
rendering, followed by scalar first-surface compositing.  An official SAM3 run
on the method's rendered RGB may remain a disclosed optional mask-boundary
readout, with the unrefined result reported separately.

Changing the canonical field, increasing its dimension, or training a new
SAM head is not supported by this evidence.  Any future observation operator
must be selected query-free on RGB/depth discontinuities and must preserve the
already strong canonical primitive relations; it cannot alter primitive
scores or use benchmark masks.

### Downstream registered-prompt closure

The frozen eight-scene NVOS strict-unseen run is consistent with the primitive
diagnostic.  With no target RGB, target-camera support, feature/score
calibration, or test-set calibration, the macro foreground IoU is 0.740130 for
the primitive unary and 0.749029 after the seeded random-walker.  Thus the
official SAM3 capability plus signed graph propagation provides a small net
gain without an RGB mask decoder.  The subsequently applied fixed connected
component readout lowers the macro to 0.726404 (pixel accuracy 0.953906),
primarily because Orchids falls from 0.700091 after propagation to 0.482042.
This is a component-prior failure, not evidence of weak primitive SAM3
relations.  All three stages must therefore remain visible in the result; v3
must not switch stages after inspecting benchmark masks.

## Verification artifacts

- `output/optimization_20260716/semantic_oracles/siglip2_spatial_teacher_lerf.json`
- `output/optimization_20260716/semantic_oracles/siglip2_crop_summary_teacher_lerf.json`
- `output/optimization_20260716/global_3d_readout/canonical_mpr_v3_text_summary.json`
- `output/optimization_20260716/global_3d_readout/validation_by_physical_scale.json`
- `output/evaluation_closeout_20260716/canonical_mpr_v3_lerf_cosine/`
- `output/optimization_20260716/boundary_residual/scene0062_primitive_sam3_relation.json`
- `output/optimization_20260716/boundary_residual/scene0140_primitive_sam3_relation.json`
- `output/optimization_20260716/boundary_residual/scene0200_primitive_sam3_relation.json`
- `output/evaluation_closeout_20260716/canonical_mpr_v3_nvos8/summary.json`

The final focused canonical/readout/solver/text-query regression set passes
49/49, including the primitive-relation audit regression test.
