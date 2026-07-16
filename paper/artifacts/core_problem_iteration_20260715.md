# Core-problem iteration — 2026-07-15

## Decision

This iteration followed the P0/P2 order from the method audit.  It did not
tune benchmark thresholds, text queries, target masks, or scene labels.  The
compact canonical field remains the main representation.

The only retained inference candidate is label-free multi-scale world-point
support selection.  Full-1280 capacity, raw-token norm restoration,
capability-private residuals, a local negative annulus, and heuristic
confidence-aware diffusion are not added to the method.

## P0: representation/render closure

### Full-dimensional render ceiling

An identity-decoder field with one free 1280-D vector per Gaussian was
initialized from the exact same canonical MPR and optimized with the formal
renderer.  On Ramen held-out frames 2/45/87/130:

| Field | Initial raw cosine | Best raw cosine |
|---|---:|---:|
| d384 render fitting (existing) | 0.740227 | 0.746503 |
| full-1280 free, 64 steps | 0.744410 | 0.749191 |

The full-dimensional advantage over d384 is only 0.002689.  Compact capacity
is therefore not the cause of the roughly 0.25 held-out cosine error.

### Official adaptor input norm

Original 2-D RADIO token norms have median approximately 31.  Feeding unit
tokens rather than original tokens into the official adaptors changes output
direction severely: raw-vs-unit adaptor cosine is 0.5571 for DINOv3 and
0.5216 for SAM3.

A query-free per-primitive log-norm scalar was tested.  Directly restoring it
left raw render cosine unchanged (0.73765 to 0.73761) but reduced held-out
DINO/SAM cosine from 0.82897/0.69647 to 0.28337/0.30140.  Joint raw+DINO+SAM
render fitting for 64 steps recovered only 0.28364/0.30512.  This variant was
removed: the current direction field has already adapted to the declared
direction-space contract, so post-hoc magnitude restoration is invalid.

The provenance must call the representation `radio_direction_unit`, not
unnormalized raw RADIO.

### Capability attainability

For 4096 query-free primitives, a free 1280-D vector was optimized against raw
and capability-first SAM MPR targets.  It reached raw cosine 0.998278 and SAM
cosine 0.993686.  Although 56.9% of initial per-row gradients had negative
cosine, the mean conflict was nearly zero and the shared vector attained both
targets.  A capability-private residual is not justified.

### Mixture/boundary error

Exact 3DGS hit contributions were replayed on the same held-out frames.
SAM cosine error correlates most strongly with official-teacher SAM boundary
strength (Pearson 0.3984): mean error rises from 0.1966 in the lowest boundary
quartile to 0.4335 in the highest.  Alpha entropy correlation is only 0.0883;
top-2 contribution ratio is effectively uncorrelated (-0.0013).

Therefore simply increasing view-residual rank or conditioning only on top-2
mixing is not supported.  A future 2-D-only residual must target observable
boundary/discontinuity signals and pass a multi-scene boundary-margin gate.

## P2: world-point granularity

Using fixed graph-relative candidate neighborhoods k={16,64,256} on ScanNet
scene0000, seed 42:

| Method | mIoU | Acc@.25 | Acc@.50 |
|---|---:|---:|---:|
| retained single k=64 | 0.299993 | 0.523810 | 0.126984 |
| label-free automatic scale | 0.300199 | 0.539683 | 0.142857 |
| GT oracle scale (diagnostic only) | 0.309143 | — | — |

The automatic selector uses the equal-weight geometric mean of unary
coherence, seed containment, and one-minus graph conductance.  The small oracle
gap shows that scale selection is not the dominant remaining bottleneck;
single-point feature/instance ambiguity is.

Two candidates were rejected immediately:

- local negative annulus factor 2: mIoU 0.124454, unary mIoU 0.028855;
- heuristic confidence-aware diffusion: mIoU 0.298038 and Acc@.25 0.507937.

Both were removed from the method.  The former often labels another part of
the same object as background; the latter is not a principled replacement for
a true constrained Laplacian/random-walker solver.

## Artifacts

- `output/optimization_20260715/render_matched_v2/ramen_full1280_renderft64.pth.json`
- `output/optimization_20260715/render_matched_v2/ramen_mixture_error_heldout4.json`
- `output/optimization_20260715/render_matched_v2/ramen_raw_sam_attainability_4096.json`
- `output/optimization_20260715/multiscale_world_support/scannet_scene0000_seed42_k16_64_256.json`

All formal results keep `test_calibration=false`.  Ramen and ScanNet scene0000
are development scenes; these numbers select or reject method components and
must not be presented as sealed benchmark results.

## Boundary residual and constrained solver follow-up

A 10,440-parameter, rank-8 screen residual was added as an experimental
module.  Its inputs are only rendered luminance, relative-depth, and alpha
gradients.  Its output is structurally multiplied by the maximum observable
discontinuity, so it is exactly zero on locally smooth observations.  It is
applied after canonical rendering and cannot change any primitive-domain text,
image, 2-D prompt, or 3-D point query.

On the Ramen development split (held out from MPR), a 32-step smoke run changed
raw/DINO/SAM cosine by only +0.000014/+0.000026/+0.000052 and the observable
boundary-vs-interior fidelity margin by +0.000052.  This confirms the contract
and gradient path but is far below the promotion gate.

The diffusion heuristic now has a true seeded random-walker alternative.  It
solves an eliminated symmetric-normalized Laplacian system with CG; positive
seeds are exactly one and negative seeds exactly zero throughout the solve.
With all other settings fixed, development diagnostics were:

| task/scene | diffusion | random walker | delta |
|---|---:|---:|---:|
| NVOS fern foreground IoU | 0.744685 | 0.741083 | -0.003602 |
| SPIn-NeRF fern foreground IoU | 0.948247 | 0.946817 | -0.001430 |
| ScanNet scene0000 point mIoU | 0.300199 | 0.294807 | -0.005392 |

Thus neither experimental component is promoted from the development-only
diagnostic.  The requested three-scene validation is reported below.

## Three-scene query-free validation (canonical-mpr-v2)

Fresh caches and fields were built with frozen hyperparameters and four fixed
held-out frames per scene.  No benchmark mask, label, text query, or
task-specific threshold was opened during construction or selection.

| scene | raw mean delta | DINO mean delta | SAM mean delta | p05 | primitive delta |
|---|---:|---:|---:|---|---:|
| scene0062 | +0.002538 | +0.002308 | +0.008022 | all improve | -0.000138 |
| scene0140 | +0.002417 | +0.001979 | +0.002204 | all improve | -0.000006 |
| scene0200 | +0.003906 | +0.002986 | +0.003046 | all improve | -0.000091 |

`canonical-mpr-v2` passes the query-free representation/render promotion gate.
It is the candidate canonical core: v1 MPR initialization, exact renderer
replay, raw RADIO render fitting, frozen official DINOv3/SAM3 capability
losses, local-affinity preservation, and a primitive replay prior.  The field
remains `radio_direction_unit`; no test-set calibration is used.

The absolute boundary result remains the main representation weakness.  SAM3
boundary-margin retention after v2 is only 0.02986/0.01409/0.01081 on the three
scenes.  V2 improves it consistently but does not yet reconstruct official
SAM3 local relations with high fidelity.

## Three-scene boundary residual decision

The same 10,440-parameter rank-8 residual was trained independently with
identical settings.  Its formal held-out delta relative to the frozen v2 field
is:

| scene | raw mean | DINO mean | SAM mean | raw/DINO/SAM boundary retention |
|---|---:|---:|---:|---|
| scene0062 | +0.000073 | +0.000100 | +0.000473 | +0.00058 / +0.00377 / +0.00117 |
| scene0140 | +0.000131 | +0.000189 | +0.000386 | +0.00083 / +0.00206 / +0.00066 |
| scene0200 | +0.000025 | +0.000033 | +0.000194 | +0.00070 / +0.00241 / +0.00050 |

The residual passes its narrow contract gate on all three scenes: observable
boundary fidelity improves, global capability means are non-inferior, it is
screen-only, and primitive queries are unchanged.  It is promoted as an
optional 2-D observation-fidelity module, not as part of the canonical
primitive descriptor.  Its small effect size must be reported; it does not
solve the SAM3 boundary-retention gap by itself.

The initially missing official aggregation/segment instance annotations were
subsequently fetched from the ScanNet v2 release and verified to align exactly
with the local label meshes: 51,610/372,941/83,291 vertices for
scene0062/0140/0200.  Semantic PLY labels were never substituted for instances.
The frozen one-click protocol gives mIoU 0.421774/0.271479/0.272792,
Acc@0.25 0.652174/0.531915/0.473684, and Acc@0.50
0.434783/0.063830/0.105263.  The three-scene macro mIoU is 0.322015.

## Surface-aware world-point lifting

The world-point compiler now supports query-independent support-graph hop
neighborhoods in addition to Euclidean kNN neighborhoods.  All candidate
scales use the same label-free coherence/containment/conductance selector; GT
is used only for the separately reported oracle diagnostic.  On the annotated
scene0000 development scene, adding hop radii 1/2/3 to Euclidean k=16/64/256
changes one-click mIoU from 0.300199 to 0.302333 and Acc@0.50 from 0.142857 to
0.158730.  The oracle scale is 0.317144.  This is a real but modest gain and
does not dominate the previously observed fixed local-64 prototype result
(0.304553 mIoU), so surface candidates remain an ablation rather than a new
default until cross-scene instance annotations are available.

The canonical evaluator previously accepted `--clicks 3` but silently compiled
only one point.  This protocol bug was fixed: multi-click world coordinates
now jointly define Mahalanobis seeds, appearance/boundary prototypes, and the
union of surface-hop candidates, and every seed is serialized in the result.
With three deterministic positive clicks, scene0000 mIoU rises to 0.349363,
Acc@0.25 to 0.619048, and Acc@0.50 to 0.190476 (one-click surface-aware:
0.302333/0.507937/0.158730).  This is a separate three-click protocol, not a
replacement for the one-click main result.

## Boundary-capacity follow-up on the development split

Teacher-extreme affinity weighting had already been tested on Ramen and only
changed SAM3 boundary retention from 0.037551 to 0.038128.  This rejects loss
imbalance as the sole explanation.  A larger but still low-capacity screen-only
residual (rank 16, 20,816 parameters, 256 steps) was then evaluated on the same
held-out frames.  Relative to the frozen v2 field it changes raw/DINO/SAM mean
cosine by +0.000869/+0.000961/+0.001604 and SAM3 boundary retention from
0.038128 to 0.039869.  Primitive queries remain unchanged.

This establishes that residual capacity and optimization duration matter, but
the absolute SAM3 retention remains low.  Rank 16 is therefore a stronger
optional 2-D candidate, not evidence that the nonlinear compositing problem is
solved.  It must pass the same three-scene frozen gate before replacing rank 8.

The frozen three-scene audit subsequently passed.  Compared with no residual,
rank-16 changes SAM3 mean cosine by +0.000863/+0.000401/+0.000536 and boundary
retention by +0.002746/+0.001990/+0.001150 on scene0062/0140/0200.  Compared
with rank-8, SAM3 retention is also higher on every scene:
0.032605 vs 0.031032, 0.016084 vs 0.014752, and 0.011959 vs 0.011309.
Rank-16/256-step therefore replaces rank-8 as the promoted optional 2-D
observation module.  The canonical primitive field and all primitive-domain
queries remain unchanged.

## Official crop-summary semantic lifting

The level-1 official SigLIP2-G spatial oracle was too weak for dense text
query.  The level-2 route was therefore tested without introducing a learned
query head: official C-RADIOv4/SigLIP2-G crop visual summaries were extracted
from Ramen training images and lifted directly to the canonical primitives
with the shared MPR responsibility contract.  All seven LERF annotation frames
and the four held-out observation-fidelity frames were excluded before cache
construction.  The resulting 1536-D cache covers 221,608 of 382,687
primitives from 120 training views.  No annotation mask, benchmark text, or
test-set calibration enters extraction, lifting, or selection.

Under the frozen `softmax_scene` (logit scale 50), peak-relative 0.60,
`polygon_argmax`, no-refinement protocol, rendering lifted features before
text scoring obtains sample mIoU 0.2007, category-macro mIoU 0.1916, and
localization accuracy 0.3803 over 71 annotated samples.  Scoring the same
canonical primitive descriptors first and then rendering scalar evidence
obtains 0.2079/0.2002/0.3803.  This is also stronger than the prior per-image
official crop-summary oracle (0.1502 mIoU, 0.2837 localization), indicating
that MPR preserves the official language-aligned descriptor and that
multi-view fusion improves its stability.

The promoted text-query readout is therefore `primitive_score`: every query is
compiled against the shared canonical primitive field before observation
rendering.  `primitive_query` remains a commutation audit.  This promotion is
method-level and label-free; it is not a scene-specific head or a benchmark
threshold choice.  The result is development evidence on one scene and must
still be reproduced on the other LERF scenes before it becomes a final table
row.

The frozen run was subsequently completed on all four scenes using the same
official summary extraction, canonical MPR contract, primitive-score readout,
and protocol constants.  Figurines/Ramen/Teatime/Waldo obtain sample mIoU
0.230223/0.207899/0.306575/0.247496 and localization accuracy
0.428571/0.380282/0.508475/0.409091.  The four-scene macro is 0.248048 mIoU
and 0.431605 localization; over all 208 samples, the micro result is 0.246087
mIoU and 0.432692 localization.  No mask refinement, RGB boundary decoder,
benchmark-text training, or test-set calibration is used.
