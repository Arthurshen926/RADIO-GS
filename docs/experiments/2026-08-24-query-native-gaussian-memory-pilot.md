# Query-Native Gaussian Memory pilot

## Question

Can the project replace foundation-feature reconstruction plus similarity with
a compact query-native memory that directly predicts Gaussian posteriors?

The first experiments deliberately freeze every L512 field.  They test the
decoder/query contract before authorizing a Universal Field v2 rebuild.

The categorical pilots consume the retained scalar baseline response as an
identity prior.  They do not reconstruct or output a 1536D descriptor, but an
unseen text still needs the v1 identity path to produce that scalar.  Results
below therefore validate posterior-level distillation and class-set
equivariance, not complete feature-decode-free deployment.

## Implemented contract

- `QueryPacket` separates decoder inputs from encoder identity.
- Text category decoding has no class-indexed parameter and is exactly
  permutation equivariant.
- Query cardinality is dynamic.
- Image/prompt posterior decoding separates identity retrieval and extent.
- Prompt seed probability clamps known evidence; NaN remains unknown.
- No experiment adds a per-Gaussian parameter or opens benchmark masks during
  fitting.

The v1 ScanNet decoder has `399,099` parameters (`1.52 MiB` FP32); its gate has
`124,587` parameters (`0.48 MiB`).  The larger sentinel has `531,051`
parameters (`2.03 MiB`).  These are constant per decoder, versus a
Gaussian-indexed high-dimensional capability sidecar.

## ScanNet per-scene query-native v1

Eight source-only scene decoders were trained in parallel.  The resulting
paper8 macro is:

| split | query-native v1 mIoU | retained compact mIoU | delta |
|---|---:|---:|---:|
| 19 | `0.36567` | `0.36401` | `+0.00166` |
| 15 | `0.36135` | `0.36189` | `-0.00054` |
| 10 | `0.46690` | `0.46716` | `-0.00027` |

This is not promoted because all three splits must be noninferior.  It does
show that removing the fixed 44-channel parameterization does not cause a
large accuracy collapse and improves the 19-class split.

## Cross-scene shared decoder

One decoder and one global gate threshold were trained across all paper8
scenes using the frozen source-distilled score caches as mapping-time teachers.
Its source gate passed all 24 scene/split checks.  Benchmark macro is:

| split | shared query-native mIoU | restored L512 baseline | retained compact |
|---|---:|---:|---:|
| 19 | `0.36118` | `0.35613` | `0.36401` |
| 15 | `0.36084` | `0.35300` | `0.36189` |
| 10 | `0.46434` | `0.45954` | `0.46716` |

The shared model improves the restored baseline but does not yet match the
scene-global compact student.  This measures the cost of removing all
scene-specific parameters.

A predeclared scene-query bipartite holdout improved 21/24 heldout pairs.  It
failed three pairs (`scene0000/split15`, `scene0062/split15`, and
`scene0590/split19`) by `0.0019--0.0029` MAE, so adapter-free arbitrary-query
transfer is not yet a valid claim.  In contrast, withholding the same words
from an individual scene degraded every split, demonstrating why global
cross-scene training is necessary.

## Larger per-scene equivariant sentinel

Increasing only the shared latent/query projections and pair MLP to `531,051`
parameters (`2.03 MiB`) gives:

| split | query-native v2 mIoU | retained compact mIoU | delta |
|---|---:|---:|---:|
| 19 | `0.36557` | `0.36401` | `+0.00156` |
| 15 | `0.36181` | `0.36189` | `-0.00008` |
| 10 | `0.46728` | `0.46716` | `+0.00012` |

The remaining 15-class gap is only `8.2e-5`, but strict promotion still fails.
A globally predeclared doubled 15-split sampling sentinel on scene0062 reduced
rather than removed the benchmark gap, so it is rejected; no outcome-based
split fallback is used.

## LERF direct posterior pilot

Figurines uses source-view residue 3 only for heldout evaluation.  The decoder
consumes official source SAM crop queries and frozen L512/reliability, and is
trained with BCE, Dice, Brier and identity ranking.  Invisible rows are
unknown.  Results are negative:

| variant | heldout proposal IoU |
|---|---:|
| primitive similarity control | `0.12457` |
| learned identity + SigLIP query | `0.04840` |
| learned identity + SigLIP/DINO query | `0.05174` |
| learned identity, 256D query | `0.02943` |
| identity-prior residual, SigLIP | `0.06404` |
| identity-prior residual, SigLIP/DINO | `0.04364` |

The identity-prior variant reaches about `0.197` on source-train calibration
but collapses on heldout proposals.  Direct posterior decoding therefore does
not remove the need for reliable cross-view object supervision.  Independent
same-view SAM masks teach proposal appearance and local support, not complete
physical-object extent.  Larger adapters or more loss weights are stopped.

## LERF v2 implementation audit

The first implementation was not a fair Query-Native test. It read raw
`local_codes`, divided normalized cosine similarity by `sqrt(query_dim)`, used
one scene-global anchor, had no 3D geometry, treated every visible row outside
one SAM proposal as negative, and selected checkpoints by the lowest observed
training loss. These are now corrected on the v2 branch:

- explicit raw-local / post-fusion coefficient / fixed RADIO-projection reads;
- learned cosine temperature constrained to `[0.02,0.2]`;
- top-K identity anchors, finite xyz/scale/opacity relations, and peak
  preservation;
- positive/explicit-negative/unknown supervision;
- disjoint train/validation/heldout view partitions and validation-only model
  selection/calibration;
- cached proposal-negative relations and GPU-resident identity-prior scoring.

The corrected figurines proposal-generalization sentinel remains negative:

| memory/query arm | heldout proposal IoU |
|---|---:|
| primitive similarity control | `0.12913` |
| raw local, K=6 | `0.06122` |
| post-fusion coefficient, K=6 | `0.06679` |
| post-fusion coefficient, K=4 | `0.06769` |
| post-fusion coefficient, K=8 | `0.07064` |
| fixed projected RADIO, K=6 | `0.07097` |
| post-fusion coefficient + DINO appearance, K=6 | `0.08519` |

Thus post-fusion memory, decoded-coordinate alignment, and appearance queries
all recover signal, but none closes the gap to the unchanged primitive prior.
This is evidence against field capacity being the primary LERF bottleneck.
It strengthens the requirement for real query-view to target-view same-object
episodes; unseen independent proposals are not a substitute. These heldout
proposal rows have now been observed during architecture development and are
not a clean final promotion gate.

## Memory-coordinate ablation on ScanNet paper8

The larger per-scene decoder was replayed with the post-fusion D512
coefficient as its memory coordinate instead of the raw L512 local code.  All
eight scenes completed under the unchanged evaluator and training budget:

| split | raw local | post-fusion coefficient | delta |
|---|---:|---:|---:|
| 19 | `0.36557` | `0.36584` | `+0.00027` |
| 15 | `0.36181` | `0.36155` | `-0.00026` |
| 10 | `0.46728` | `0.46718` | `-0.00010` |

The coefficient coordinate is therefore not promoted.  It removes one
obvious scene-gauge concern but does not improve all class splits.  Together
with the LERF representation ablation, this says that memory coordinates
matter at second order while supervision and query identity remain the
first-order bottlenecks.  A scene canonicalizer should only be tested inside
the shared cross-scene decoder, where gauge alignment is actually required;
adding it to independent per-scene models would not test that hypothesis.

The shared coefficient experiment consequently adds a rank-8 scene FiLM with
no Gaussian-indexed state and exact identity initialization.  Checkpoints and
the eligibility gate are selected on a fixed validation sample spanning all
8 scenes and all 3 class sets, rather than the last training minibatch.  On a
predeclared scene-query holdout, the results are:

| shared memory | heldout pairs improved | mean heldout MAE delta | all changed-row gates |
|---|---:|---:|---:|
| coefficient, no canonicalizer | `17/21` | `-0.00616` | `24/24` |
| coefficient + rank-8, seed 24 | `20/21` | `-0.00897` | `24/24` |
| coefficient + rank-8, seed 25 | `20/21` | `-0.00882` | `24/24` |
| coefficient + rank-8, seed 26 | `19/21` | `-0.00849` | `24/24` |

This validates scene-gauge alignment as a real shared-model improvement, but
not as a complete solution.  Every seed still fails the strict unseen-query
gate, consistently on `scene0062/split15`; the heldout identities there are
`chair` and `toilet`.  The same identities are mostly successful in split10,
so the remaining error is query-set competition/calibration rather than a
missing per-query identity alone.  No benchmark cache from these failed gates
is promoted or evaluated as a formal result.

Factoring the decoder into a pair-local identity residual and a centered,
permutation-equivariant query-set competition residual closes that source
gate on the predeclared seed: `21/21` heldout pairs and `24/24` changed-row
gates improve, with mean heldout MAE delta `-0.00967`.  This directly supports
the identity/competition separation; it is not a threshold-only change.

The authorized paper8 replay is nevertheless below the retained compact row:

| split | factorized shared mIoU | retained compact mIoU | delta |
|---|---:|---:|---:|
| 19 | `0.36275` | `0.36401` | `-0.00126` |
| 15 | `0.35922` | `0.36189` | `-0.00267` |
| 10 | `0.46533` | `0.46716` | `-0.00183` |

The factorization is retained as positive mechanism evidence but the weights
are not promoted.  Source score MAE and replay consistency are not sufficient
surrogates for opacity-volume-weighted categorical mIoU.  The next ScanNet
gate must add a source-only decision-preserving objective (top-class margin
and weighted confusion), while keeping benchmark labels closed.

## High-value gap closure: physical episodes and decision risk

The LERF trainer now consumes genuine directed episodes: a query crop in view
A is paired with a target proposal in view B only when frozen DINO mutual
matching, fundamental-matrix RANSAC and a three-view cycle confirm the same
physical object.  The target is B-view continuous exact-MPR Gaussian
membership.  Negatives require an explicit cross-view `different` proposal in
B; every other row remains unknown.  This replaces same-proposal
self-reconstruction with the requested query-view to target-view supervision.

| source scene | primitive heldout IoU | cross-view posterior | delta |
|---|---:|---:|---:|
| ramen | `0.25182` | `0.50093` | `+0.24912` |
| teatime | `0.46042` | `0.48275` | `+0.02233` |
| waldo_kitchen | `0.50968` | `0.00117` | `-0.50852` |

Ramen and Teatime pass.  Waldo has only 5 train, 3 validation and 2 heldout
directed episodes.  Initializing from Teatime raises Waldo to `0.45968`; also
freezing the transferred query adapter raises it to `0.49129`, but the strict
gate still fails by `0.01840`.  No Figurines benchmark was opened.  The next
LERF implementation must jointly train the three source scenes with a small
scene canonicalizer; further Waldo threshold or schedule tuning is stopped.

A separate continuous-support audit confirms that hard membership was not the
only Waldo failure: fuzzy exact-MPR support preserves Ramen/Teatime purity and
coverage, but Waldo still has no reciprocal proposal-level identity edge.
Direct DINO-cycle episodes, rather than a learned reciprocal proposal scorer,
are therefore the correct supervision authority.

For ScanNet, eight query-independent caches now bind each Gaussian's
`opacity * scale.prod()` without opening labels or masks.  Training adds
class-mass-balanced weighted top-class CE and weighted soft-IoU, and eligibility
truth is restricted to rows where the source teacher changes the final class.
A vacuous `threshold=1.01` replay was detected and the gate now requires
nonzero changed-row selection and positive weighted decision improvement.

The decision model improves every source scene/split containing a changed
decision and selects thousands of rows, but regresses already-correct heldout
queries.  Fixed low-margin (`0.04`) and minimum-margin-gain (`0.04`) risk
limits reduce activation to 1,444 rows but still fail strict query transfer.
No paper8 evaluation is authorized.  The remaining ScanNet component is a
selective-risk estimator trained with explicit beneficial/harmful
counterfactual labels; another global threshold sweep is not justified.

## Joint extent and counterfactual risk closure

The first joint LERF compiler incorrectly discarded a confirmed same-object
episode whenever that target view lacked an explicit different-instance edge.
That violated the three-state contract: absence of a negative means unknown,
not absence of a positive.  Compiler v2 retains positive-only episodes and
increases Ramen/Teatime/Waldo from `54/32/10` to `82/58/22` episodes.  With a
shared decoder, rank-4 scene canonicalizer, object/scene-balanced sampling and
checkpoint selection by worst-scene IoU gain, all six seeds pass every-scene
noninferiority. Mean heldout gains are `+0.2134/+0.1335/+0.2288`.

This large source improvement does not directly solve benchmark text readout.
The learned crop-query adapter obtains only `0.03677` LERF2D direct macro mIoU
and `0.04104` LERF3D macro mIoU, while localization remains `0.87159`. A fixed
deterministic cosine-preserving projection improves these to `0.06453` and
`0.05995`, respectively, and also passes 6/6 source seeds. Both rows remain
rejected. The remaining LERF failure is now specifically cross-modal extent
calibration: image-crop identity localizes the text target, but its learned
extent probability and source threshold do not transfer to text queries.

ScanNet now freezes the score decoder before producing an immutable
baseline/raw-candidate/source-teacher triplet. Counterfactual labels are
beneficial, harmful or neutral; neutral rows are not forced negative. A
three-way risk estimator is trained under scene LOSO with harmful weighted
mass as the primary gate and nonempty adoption required. Only `2/8` heldout
scenes pass at a maximum harmful fraction of `0.25`; paper8 remains closed.
The selector is therefore implemented correctly, but the present candidate
score decoder does not contain a sufficiently broad transferable improvement
region for a selector to recover.

## Current decision

The architecture remains a promising v2 candidate for categorical text
queries: it is efficient, structurally open-set, exceeds the retained result
on 19/10 classes and is within `8.2e-5` on 15 classes.  It is not a promoted
universal replacement because the strict three-split gate, LERF extent and
unseen-query transfer gates remain open, and the scalar identity prior still
comes from the retained v1 path.

The next justified LERF experiment must first create an independently gated
cross-view authority (for example, short-baseline mutual correspondence with
unknown handling).  Only then should the same Gaussian posterior be evaluated
on both LERF2D and LERF3D.  Training a larger direct mask decoder on the current
independent proposal carrier has low expected value.
