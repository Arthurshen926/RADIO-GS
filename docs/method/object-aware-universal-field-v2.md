# Object-Aware Universal Field v2 candidate

## Status and boundary

Object-Aware Universal Field v2 is an opt-in development candidate.  Universal
Field v1 remains the frozen baseline until the field-frozen pilot passes its
source-only and benchmark development gates.  V2 does not add a second task
semantic field.  It extends each Gaussian's one deployment state with a small,
query-independent object-relation code:

\[
  \mathcal F_i^{v2}=(z_i^{\mathrm{RADIO}},a_i^{\mathrm{obj}},r_i),
  \qquad a_i^{\mathrm{obj}}\in\mathbb R^{16\text{--}32}.
\]

The RADIO D512/L512 state and five reliability scalars retain their v1
identity.  Object codes may express co-membership and scale structure, but may
not store a benchmark query, class vocabulary, target mask, or target RGB.

The 2026-08-21 Figurines B0 pilot is **not promoted**.  Its source-heldout
relation and language gates passed, and its affirmative multiview extent raised
LERF2D mIoU, but the same Gaussian posterior regressed LERF3D.  This establishes
that compact object structure is learnable while also rejecting proposal-graph
closure or positive-support noisy-OR as a complete 3D object model.  A later
pilot added per-view proposal/null marginalization, raw exact-MPR membership,
and a per-row visibility denominator.  It still improved LERF2D while
regressing LERF3D.  Three-scene leave-one-scene-out calibration of a geometry
cycle proxy did not repair the joint gate either.  Full-scene or full-benchmark
v2 training was then tested with an independent source-RGB authority: frozen
RADIO-DINOv3 mutual dense matches, fundamental-matrix RANSAC, and official
query-free SAM masks define an adjacent three-view cycle, while visible null
and occlusion remain distinct.  Its association calibrator excludes DINO and
geometry-overlap inputs.  A nested source-only temperature selection (`T=4`)
passed the heldout proper gate on ramen, teatime, and waldo_kitchen before
Figurines was opened.  The one authorized fixed Figurines pilot improved
LERF2D mIoU from `0.41424` to `0.42657`, but reduced LERF3D mIoU from
`0.57207` to `0.51544`.  It is not promoted or expanded to full4.  The evidence
now localizes the remaining problem to Gaussian-level physical-track purity,
not probability temperature, 2D mask extent, or the former geometry-label
tautology.  Benchmark threshold tuning and fixed-size track selection remain
unacceptable substitutes.

A subsequent source-only conservative-core audit did not authorize another
benchmark.  Reciprocal association plus a Wilson lower confidence bound on
shared exact-MPR Gaussian support reached `1.0` heldout purity on ramen and
teatime while retaining roughly `0.53` positive coverage.  Both the reciprocal
and complete three-view-triangle variants retained zero Waldo heldout
positives.  The candidate therefore fails the cross-scene coverage gate and is
not materialized on Figurines.  This narrows the next prerequisite further:
source proposal coverage or correspondence-lifted probabilistic Gaussian
support must improve before another benchmark evaluation is worthwhile.

The categorical analytic-track pilot is also not promoted.  On seven ScanNet
confirmation scenes its 19/15/10-class macro mIoU and mAcc changes were
positive, but split-10 still contained scene-level regressions.  It remains an
interface test for a future learned affinity, not a paper result.  Conversely,
the subsequent global `1536->16` compact-affinity pilot trained only on
scene0000 source-view same/different/unknown authority and passed its
source-heldout proper-score gate on scene0062/0070/0097.  However, the frozen
canonical proposal similarity already had heldout relation AUC 1.0 in all four
scenes; learning mainly repaired probability calibration.  Through the same
bounded sparse-track voter it changed confirmation 19/15/10 mIoU by
`+0.000061/+0.000075/-0.000062`, with split-10 mIoU regressing in every
confirmation scene.  This candidate is therefore rejected without affinity
threshold retuning or post-metric fallback.  The remaining ScanNet error is
categorical competition, not missing pairwise object-affinity rank capacity.
Separately,
the registered-prompt reliability gate produced a large, seed-stable NVOS
development improvement by invoking an independent region selector only when
proposal quality or field/proposal disagreement authorized it.  That result is
post-metric and target-RGB-assisted: it supports the risk-gated operator but
cannot promote the cached external selector or support a SOTA claim.
The subsequent fern carrier-native control removed the invalid cross-carrier
row correspondence: the released all-view LUDVIG scalar was rendered at the
registered prompt camera and transferred by normalized exact (W^\top) onto
the frozen current carrier.  At the single fixed `75/255` threshold, its target
IoU was `0.80463` versus `0.84480` for the native image-space selector, while
their coarse-mask IoU was `0.93086`.  Intersecting Method-v1 with the bridged
extent raised IoU from `0.83055` to `0.83713` (`+0.00657`).  Thus the exact
bridge is numerically and semantically valid, but a single prompt-camera
round-trip loses some target-view boundary/visibility information.  This is an
all-view, upstream target-RGB-assisted development result, not strict-unseen
evidence.
The independent horns-left confirmation preserved the important effect:
carrier-native bridge IoU was `0.86663`, native-selector IoU was `0.94337`,
and coarse agreement was `0.88839`; the fixed Method-v1 intersection improved
from `0.68984` to `0.91492` (`+0.22508`).  Across fern and horns-left, the
same threshold and operator improved macro IoU from `0.76020` to `0.87602`
(`+0.11583`), with both scenes improving.  This paired mechanism confirmation
justifies a fixed full-eight development run, while preserving the same
all-view/target-RGB-assisted non-paper boundary.
That full-eight run is now complete.  The unconditional fixed intersection
raised macro IoU from `0.81772` to `0.89790` (`+0.08018`; five scenes improved
and three regressed).  Reusing the already fixed prediction-time reliability
gate instead yielded `0.91623` (`+0.09851`; four improved, three replayed the
baseline, and trex regressed by `0.00157`).  Horns-center and leaves improved by
`+0.27731` and `+0.28484`; orchids regressed by `-0.15243` under unconditional
intersection even though bridge/native coarse agreement was `0.94189`.  This
distinguishes two failure sources: exact transfer is faithful on orchids, but
the upstream region authority is wrong or incomplete; across all scenes, the
single prompt-camera round-trip also trails the native selector by about
`0.04598` macro IoU.  Carrier-native exact transfer and risk gating therefore
belong in the registered-prompt method, while unconditional region veto does
not.  The result remains post-metric, all-view target-RGB-assisted development
evidence and is not a strict-unseen or SOTA result.

## Source authority

Frozen automatic SAM masks from legal source/mapping views are registered to
Gaussians with the exact front-to-back marginal responsibility used by the
capability field.  A pair is supervised only by affirmative joint evidence:

- `same`: jointly visible and supported by the same source mask or associated
  source track;
- `different`: jointly visible and supported by mutually exclusive masks with
  stable boundary evidence;
- `unknown`: occluded, missing, granularity-conflicted, or unsupported.

Unknown pairs are excluded from proper losses.  Absence from a mask is never a
negative label.  Mask scale and cross-view association are explicit training
inputs so one flat proposal partition is not treated as object truth.

Each source mask also binds a frozen masked-crop and expanded-context SigLIP
descriptor.  Sparse object membership pools the unchanged Canonical Capability
Feature into a mask/track descriptor and minimizes cosine loss to these two
teachers.  The object-language branch learns identity at the object level;
co-membership learns extent.  Neither branch may use its own prediction as a
training outcome.

## Staged optimization

Stage B0 is field-frozen: optimize only object codes, a globally shared
co-membership decoder, cross-view association, and object-language pooling.
RADIO codes, basis, capability heads, geometry, opacity, and reliability are
immutable.  This is the required pilot.

Stage B1 is allowed only after B0 passes.  It may update local RADIO codes and
one small object residual while keeping the RADIO basis and all frozen teacher
heads fixed.  Raw RADIO, DINO, SAM, SigLIP, geometry, and visibility fidelity
must pass the same hash-bound no-regression gates as v1.  Gradient projection
is required if object and capability gradients conflict.

## Typed posteriors

Text queries use one normalized primitive/object/null mixture.  Object identity
comes from mask-aligned language descriptors; object extent comes from the
learned affinity.  The field identity peak is immutable, and the null branch
falls back bit-for-bit to the v1 primitive posterior.  LERF2D and LERF3D consume
the same Gaussian posterior; output conversion may only apply legal monotone
calibration and visibility-aware rendering.

Categorical queries aggregate primitive class logits inside confident object
tracks, then fuse the object vote only for thing-like, low-margin primitives.
Stuff regions and low-confidence tracks replay primitive logits.  The operator
must be class- and track-permutation equivariant, zero-initialized, and cannot
introduce per-class or per-scene calibration parameters.

Registered prompts use the field posterior as an identity anchor, invoke frozen
SAM independently on each protocol-authorized registered RGB, lift masks with
the exact adjoint, robustly marginalize views and candidates, and apply the
independently confirmed component-local identity-density risk limiter.  A
region sidecar may veto foreground only through a prediction-time reliability
gate fixed before evaluation; otherwise the primitive/SAM posterior is replayed
unchanged.  Cached selectors from another carrier are development controls.
Deployment requires rendering their scalar evidence into registered views and
applying the normalized exact image-space adjoint to the current carrier; row
nearest-neighbour transfer is forbidden.

## Pilot promotion gates

The field-frozen pilot advances only if all applicable checks pass:

1. source-heldout co-membership proper score and object-language retrieval
   improve over v1/proposal controls;
2. LERF2D and LERF3D improve from one shared Gaussian posterior while LocAcc
   does not regress;
3. ScanNet mIoU improves without mAcc regression on an independent scene split;
4. NVOS/SPIn instance leakage decreases without foreground-recall collapse;
5. v1 capability fidelity and deployment-state hashes replay exactly.

Failure of a single-scene pilot stops full-cohort training.  Threshold search,
per-scene branch selection, larger RADIO capacity, and unfrozen basis training
are outside this candidate.
