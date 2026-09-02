# Surface-Aligned Object-Centric Memory v4: first geometry ladder

## Decision boundary

This round changes only carrier geometry and registration.  It does not train
features, object tokens, query adapters, or posterior calibration.  Historical
SUGM-v3 method code is not imported.  The retained benchmark rows are sealed in
`paper/artifacts/v4_retained_baselines_20260830.json` and are comparators, not
inputs to the geometry experiment.

Milestone 1 is now complete on the preregistered three-scene development
cohort.  ScanNet, LERF Figurines, and LERF Ramen all pass with one shared
projection configuration (`maximum_splat_radius=1`, surface band 1.5 voxels,
at most eight contributors).  Only the oracle object-codebook stage is opened;
learned codebook, query encoder, and compression remain separately gated.

An earlier availability check was wrong: it relied on stale README and open
issue text.  The default Microsoft repository branch contains the MoGe-3 model
implementation, and `Ruicheng/moge-3-vitl` and `Ruicheng/moge-3-vitg` publish
pretrained MoGe-3 checkpoints.  The availability receipt has been corrected;
there is no longer an external release blocker.

## Implemented contract

- backend-independent `SurfaceCarrier` projection, lift, render, and adjacency;
- Gaussian exact-renderer transport baseline;
- triangle-mesh raycast oracle;
- sparse surface voxel/surfel z-buffer;
- positive/negative/unknown evidence and multi-view sufficient statistics;
- constrained affine depth calibration and confidence-weighted sparse fusion;
- hash-bound geometry and method receipts;
- static rejection of v3 imports and concrete scene identities in v4 source.

The sparse carrier retains only contributors in a front-surface band of 1.5
voxels and caps each pixel at eight deterministic contributors.  The initial
one-element z-buffer was rejected because it improved same-view metrics but
regressed tracked cross-view transfer.

## Coordinate-convention bug found and repaired

The prepared ScanNet `transforms.json` stores NeRF/OpenGL camera axes.  The
first mesh loader incorrectly treated those matrices as OpenCV poses.  A visual
mesh-RGB audit exposed the error: first-frame raycast coverage was about 31%.
Applying the same OpenGL-to-OpenCV conversion used by the training dataset,
`diag(1,-1,-1,1)`, raised coverage to 98.57% and aligned the rendered mesh RGB
with the source image (support-region PSNR 17.96 dB).  All pre-fix ladder
numbers were overwritten and are not valid evidence.  A regression test now
seals the conversion.

## ScanNet mesh-label oracle result

The diagnostic explicitly opens mesh vertex labels and records that access in
its receipt.  Labels are used only to provide shared cross-view correspondence;
they are not persisted as method state.

| Carrier | same-view soft mIoU | tracked cross-view soft mIoU | boundary leakage | element purity | oracle-surface coverage | effective contributors |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian exact renderer | 0.62363 | 0.43729 | 0.35075 | 0.87864 | 1.00000 | 24.2208 |
| Sparse surface, 4 cm | 0.78225 | 0.45461 | 0.22323 | 0.94768 | 0.94828 | 2.2420 |
| Mesh oracle | 0.94332 | 0.60167 | 0.04155 | 0.98414 | 1.00000 | 2.3434 |

Relative to Gaussian transport, the mesh oracle improves all four primary
metrics.  The sparse surface improves same-view round-trip by 0.15863, tracked
cross-view transfer by 0.01733, purity by 0.06904, and reduces leakage by
0.12752.  Coverage decreases by 0.05172 and is reported rather than used to
mask another regression.

Depth is unavailable in the Gaussian transport shards.  Against mesh raycast
depth, the sparse carrier has mean/median absolute residual 0.07136/0.02129 m.
Its mean unsigned normal cosine is 0.89854 with median angular error 5.12
degrees; the mesh self-oracle is exactly 1.0/0 degrees.

## Source-only SAM result

This second ladder uses 82 sealed query-independent SAM proposals from 16
source views.  It opens no benchmark image, mask, label, query text, or target
RGB.  Because proposals are not tracked, cross-view best-proposal overlap is a
non-gating diagnostic.  Purity is evaluated only over 153 exactly pixel-disjoint
proposal pairs; parent/part overlaps are excluded by construction.

| Carrier | same-view soft IoU | same-view leakage | mutually-exclusive purity | mask coverage | effective contributors |
|---|---:|---:|---:|---:|---:|
| Gaussian exact renderer | 0.57361 | 0.29523 | 0.99523 | 1.00000 | 24.2208 |
| Sparse surface, 4 cm | 0.71220 | 0.15232 | 0.99895 | 0.92933 | 2.2420 |
| Mesh surface | 0.84544 | 0.08887 | 0.99952 | 0.99819 | 2.3434 |

The source-only direction agrees with the oracle on round-trip, leakage,
mutually-exclusive purity, and registration ambiguity.  The untracked
best-target-proposal overlap is slightly lower and remains explicitly
non-gating because it does not establish object identity across views.

## LERF MoGe-3 inference and fusion

The revision-pinned official MoGe-3 ViT-L checkpoint was run on all 120 sealed
source frames in each LERF scene.  Both scenes have 100% valid point-map
coverage.  COLMAP calibration uses only sparse points whose complete tracks
remain inside the source split.  Rejected local affine fits fall back to the
single robust global scale rather than dropping the view.

| Scene | strict source-only COLMAP points | locally calibrated / global fallback views | fused 4 cm elements | median dispersion |
|---|---:|---:|---:|---:|
| Figurines | 1,104 | 76 / 44 | 84,357 | 1.78 cm |
| Ramen | 16,320 | 49 / 71 | 44,995 | 1.82 cm |

The model checkpoint SHA-256 is
`9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925`.

## Shared projection-footprint correction

The first LERF run inherited a maximum three-pixel surfel splat from the
ScanNet prototype.  It passed Figurines but failed all three primary directions
on Ramen.  A bounded shared ablation changed only this footprint and used the
same value for all scenes:

| max radius | Figurines round-trip delta | Ramen round-trip delta | Figurines gate | Ramen gate |
|---:|---:|---:|:---:|:---:|
| 3 | +0.09076 | -0.10959 | pass | fail |
| 2 | +0.13083 | -0.03700 | pass | fail |
| 1 | +0.24930 | +0.09507 | pass | pass |

The monotone Ramen degradation and simultaneous Figurines improvement identify
over-wide splatting on the coarse feature raster as the error.  Radius one is
not a relaxed gate: coverage remains non-compensatory and purity must still not
regress.  The same radius-one setting was rerun on ScanNet.  Its label oracle
still improves round-trip by 0.15019, tracked cross-view transfer by 0.00403,
leakage by 0.12422, and purity by 0.06923.  The source-only ScanNet arm improves
round-trip, leakage, and purity; its untracked best-proposal transfer remains a
non-gating negative diagnostic (-0.00569).

Final LERF source-only values are:

| Scene / carrier | round-trip IoU | leakage | exclusive purity | coverage | effective contributors |
|---|---:|---:|---:|---:|---:|
| Figurines Gaussian | 0.36052 | 0.51150 | 0.99005 | 0.99655 | 26.8425 |
| Figurines sparse surface | 0.60983 | 0.25011 | 0.99952 | 0.97488 | 2.1112 |
| Ramen Gaussian | 0.61592 | 0.25203 | 0.99932 | 0.99794 | 66.1851 |
| Ramen sparse surface | 0.71099 | 0.18324 | 0.99937 | 0.99560 | 1.4020 |

## Completed geometry gate

The aggregate report evaluates exactly three expected scenes.  Both LERF arms,
the ScanNet sparse arm, and the ScanNet mesh-oracle stop rule pass.  Milestone 1
is complete.  This authorizes only an oracle object codebook; it does not
authorize learned masks, text/image query, or compression.

## First oracle object-codebook result

The v4 codebook stores two token IDs and weights per surface element plus
explicit unknown mass.  An opt-in ScanNet diagnostic used direct 3-D instance
IDs only as oracle association keys.  With the exact oracle memberships, the
same element posterior obtains 0.67296 held-out 2-D soft mIoU and 1.0 3-D soft
mIoU/purity, so the carrier and sparse top-2 representation have sufficient
capacity.

A stricter source-lifted arm remains below gate: using 12 mapping and four
held-out views obtains 0.46701 2-D and 0.46016 3-D soft mIoU, despite 0.94878
purity and 0.86855 top-1 accuracy.  Only 41.03% of elements receive known
membership.  This localizes the next failure to object write coverage/completion,
not token capacity or token merging.  Learned soft-codebook training therefore
remains blocked until source-mask association can cover unseen object extent
without erasing unknown exterior.

## Object-write diagnosis after the oracle gate

The geometry/oracle state at git HEAD `a42cd82a24ea00b2599def75b63bcfb57506266a`
is sealed by a source-tree and evidence milestone receipt.  Carrier parameters
remain frozen.  Before any learned completion, the object-memory contract was
corrected in three ways: training assignments remain dense until explicit
deployment compression, token confidence uses local unknown/conflict rather
than scene-global unknown mass, and token probabilities must form a simplex
with an explicit null probability.

The original 12-mapping-view oracle-ID experiment decomposes hard unknown
elements as follows:

| Reason | fraction of hard unknown |
|---|---:|
| A: never visible without an observed token | 20.48% |
| B: visible without object-mask evidence | 1.20% |
| C: mask evidence without association | 0% in the oracle-ID arm |
| D: associated but rejected | 0% |
| E: unseen surface of an observed object | 78.32% |

This first result identifies view coverage as the dominant oracle-ID failure.
A label-free selection authority then greedily maximized newly visible surface
without opening instance labels or queries:

| Views | uniform visibility | greedy visibility | uniform oracle-mask coverage | greedy oracle-mask coverage |
|---:|---:|---:|---:|---:|
| 16 | 51.12% | 66.71% | 50.49% | 65.92% |
| 32 | 66.12% | 81.87% | 65.31% | 80.86% |
| 64 | 79.60% | 89.11% | 78.67% | 88.02% |
| all 279 | 92.69% | 92.69% | 91.69% | 91.69% |

The 64-view cohort was bound to a legal source-RGB authority and processed by
the official query-free SAM3 hierarchy on five GPUs.  Real proposals expose a
different downstream bottleneck:

| Greedy views | proposals | visible surface | any-SAM covered | safely associated | observed tokens |
|---:|---:|---:|---:|---:|---:|
| 16 | 163 | 66.71% | 28.71% | 25.12% | 35 |
| 32 | 353 | 81.87% | 42.88% | 38.65% | 47 |
| 64 | 743 | 89.11% | 52.69% | 47.42% | 54 |

At 64 views, 69.27% of the remaining unassociated elements are visible but
covered by no SAM proposal, 10.02% are SAM-covered but not safely associated,
and 16.44% remain unseen surface of an observed object.  Missing proposal
support includes object-like regions such as cabinets, curtains, shelves,
couches, and tables, not only wall/floor stuff.

Two bounded proposal-recall arms preserve the decoder and all quality gates:

| 16-view arm | proposals | SAM coverage | safe association |
|---|---:|---:|---:|
| grid16 / crop2 | 163 | 28.71% | 25.12% |
| grid32 / crop2 | 351 | 31.59% | 27.21% |
| grid16 / crop3 | 174 | 28.80% | 25.19% |

Both gains are too small to justify further point-grid/crop parameter search.
The next stage therefore uses an explicit `ObservedObjectEvidence` contract and
null-capable row-wise partial soft matching.  Ordinary mask exterior stays
unknown; only explicit different-instance evidence becomes negative.  Multiple
part masks may match one token, ambiguous masks may select null, and no
one-to-one Hungarian constraint is imposed.  Completion and learned codebook
training are still not claimed by this diagnostic.

## Real proposal-to-token association

A source-only online bootstrap was added after the evidence contract.  Only an
unmatched hierarchy root may create a token; part proposals may reuse and add
evidence to an existing token; every proposal has an explicit null alternative.
The method function has no label path.  Instance labels are opened only after
the complete method snapshot has been copied back to CPU.  A zero-proposal
source frame exposed an ambiguous empty-tensor reshape in the first 32-view
run; that bug is fixed and covered by a regression test.

The first bounded gate sweep used the original coverage-greedy source order:

| 16-view policy | tokens | safe association accuracy | mass-weighted purity | observed-object split impurity |
|---|---:|---:|---:|---:|
| strict null / overlap | 89 | 97.58% | 0.7809 | 0.3047 |
| permissive | 78 | 95.97% | 0.7727 | 0.2781 |
| aggressive | 71 | 93.55% | 0.7602 | 0.2714 |

The permissive policy is the local 16-view Pareto point.  At 32 views it
reaches 42.88% proposal-written surface coverage and observes 66 objects, but
creates 121 tokens; safe association accuracy falls to 91.76% and split
impurity rises to 0.3432.  This is not a coverage hallucination: token support
remains exactly bounded by real SAM support.

Two source-only appearance probes did not close the gap.  RGB moments plus
histograms and masked means of the official SAM3 256D visual backbone both
produce the same 76-token discrete assignment at 16 views, with 95.16%
association accuracy.  Lowering the overlap gate to exploit SAM3 appearance
reduces the count to 70 but also lowers accuracy to 92.74%.  The SAM3
descriptors themselves are not constant (pairwise cosine standard deviation
0.20); raw cosine is simply not an instance-identity metric.

A label-free diagonal metric was then trained from cross-view high-surface-
overlap positive pairs and same-view mutually exclusive root negatives.  On 16
training views the pair AUC rises from 0.99948 to 1.0, but on 16 held-out source
views it changes from 0.95550 to 0.95419.  Only six reliable training positives
and two validation positives exist.  This arm fails its held-out stop rule and
is not used downstream.

The scarcity of positives exposed an upstream conflict: the view order was
optimized only for newly visible surface and therefore suppresses the overlap
needed by association.  A label-free reorder of the same sealed 64-frame
cohort balances new coverage with predecessor overlap:

| Prefix | original visibility | reordered visibility | reordered predecessor overlap |
|---:|---:|---:|---:|
| 16 | 66.71% | 62.20% | 34.52% |
| 32 | 81.87% | 81.22% | 53.87% |
| 64 | 89.11% | 89.11% | 74.09% |

This does reduce splitting under the permissive policy at 32 views (121 to 105
tokens; split impurity 0.3432 to 0.3099), but safe association accuracy falls
to 88.58% and mass-weighted purity to 0.7312.  Tightening the same reordered
arm recovers accuracy/purity at the cost of severe splitting:

| Reordered 32-view gate | tokens | safe association accuracy | mass-weighted purity | split impurity |
|---|---:|---:|---:|---:|
| permissive | 105 | 88.58% | 0.7312 | 0.3099 |
| moderate | 131 | 92.91% | 0.7595 | 0.3514 |
| strict | 156 | 94.09% | 0.7718 | 0.3829 |

No gate dominates the original-order 32-view result.  Threshold tuning is
therefore stopped.  The failure is now localized to irreversible online token
updates: early false merges contaminate later prototypes, while avoiding them
creates duplicate tokens.  The next association implementation must freeze a
view batch, compute null-capable soft assignments jointly, and update token
prototypes only after the batch decision.  Learned completion and 64-view
association remain blocked until that batch association passes held-out source
diagnostics.

## Frozen-batch association and source-video identity

Frozen-batch association removes irreversible within-batch prototype updates and
enforces at most one root per view per token.  It is conservative at 16 views:
batch sizes 4 and 8 reach 99.19% safe association accuracy but create 94--95
tokens.  Reordering plus a 16-view frozen batch gives 77 tokens, 94.52%
accuracy, 0.7982 mass-weighted purity, and 0.2791 split impurity.  At 32 views,
however, the same family still creates 128--134 tokens and split impurity rises
to 0.3411--0.3541.  Scheduling alone cannot identify disjoint surfaces of the
same object.

Official SAM3 video memory was therefore tested as a query-free source-only
identity signal.  Across all 63 adjacent pairs in the sealed 64-frame cohort,
forward tracking seeded 240 roots and produced 73 unique-target edges at IoU
0.30.  The isolated post-seal audit finds 94.44% identity accuracy over 54 safe
edges.  Raising the IoU threshold to 0.70 improves this to 97.96% over 49 safe
edges, but even 0.90 retains a high-confidence error.  Threshold tuning cannot
guarantee identity consistency.

A first 64-view integration exposed and fixed a composition bug: must-link
identity cores had incorrectly behaved as closed tokens, preventing unlabelled
same-object observations from attaching.  Before the fix, tracking increased
the token count from 198 to 230.  After the fix it produces 201 tokens and
improves mass-weighted purity from 0.7361 to 0.7453, but safe association
accuracy falls from 94.48% to 93.19% and split impurity rises from 0.4924 to
0.5155.  Single-direction transitive identity is therefore rejected as the
working policy.

Reverse SAM3 tracking was run over the same 63 pairs.  Requiring exact proposal
return in both directions leaves 44, 40, and 34 edges at two-sided IoU 0.30,
0.50, and 0.70.  The post-seal safe identity accuracies are 96.97%, 100%, and
100%, respectively, with no duplicate-target conflict.  The 0.50 policy is the
selected precision/coverage point and is the only video identity input allowed
in the next 64-view integration.

That final integration does not pass.  With reciprocal IoU 0.50 edges, the
64-view result is 202 tokens, 93.92% safe association accuracy, 0.7371
mass-weighted purity, and 0.5200 split impurity.  The no-video control is 198
tokens, 94.48%, 0.7361, and 0.4924.  Thus locally correct sparse edges still
perturb global grouping without enough coverage to reduce fragmentation.
Reciprocal tracking remains valid as a soft score or training target, but hard
must-link use is stopped.  The next method change must preserve the baseline
partition and learn or optimize a globally constrained merge objective rather
than changing batch seed formation.

That partition-preserving merge has now been implemented.  Geometry-only
groups are frozen first; reciprocal identity may merge whole tokens only when
their root-view sets are disjoint.  At two-sided IoU 0.50 it gives 194 tokens,
94.48% safe association accuracy, 0.7317 mass-weighted purity, 0.2373
object-best-token recall, and 0.4876 split impurity.  IoU 0.70 gives 195,
94.48%, 0.7327, 0.2353, and 0.4898.  Relative to the 198-token control, both
preserve association accuracy and slightly improve fragmentation/recall, but
lose 0.0034--0.0044 purity.  This is a weak Pareto tradeoff, not a solved
association stage.  IoU 0.50 remains the working point because the stricter
gate does not recover enough purity to justify its lower coverage.  The next
bottleneck is source-video identity coverage (40 reciprocal edges from 240
seeded roots), not another threshold sweep.

Increasing the per-pair object-scale seed cap from four to eight was then
tested without changing the reciprocal IoU 0.50 rule.  On the same 16-pair
cohort, reciprocal edges rise from 18/61 seeds to 35/103; both arms retain
100% accuracy over diagnostically safe edges.  Across all 63 pairs, the wider
arm produces 82 reciprocal edges from 379 seeds, including 64 diagnostically
safe edges at 100% identity accuracy.  Its 64-view integration nevertheless
fails the joint criterion: tokens fall from 198 to 190 and split impurity from
0.4924 to 0.4853, but safe proposal association falls from 94.48% to 93.37%
and mass-weighted purity from 0.7361 to 0.7292.  The additional ambiguous edges
and their transitive token merges outweigh the fragmentation gain.  Root cap
eight is therefore stopped; root cap four with reciprocal IoU 0.50 remains
only a weak association baseline.  Further work must score identity clusters
or use tracking as soft dense-assignment supervision rather than increasing
the number of hard merge constraints.

## Development-only LERF closure

The first complete v4 LERF path now freezes a source-only scene state before
opening benchmark labels or text queries. It serializes surface elements,
observed and conservatively completed memberships, token descriptors, and
source proposal prototypes. The same element posterior is rendered for 2D and
thresholded in the carrier domain for 3D. Intermediate gates are warnings and
do not block this development run; this is not a promotion result.

Figurines source32 produced 135 tokens from 579 SAM3 proposals, with 571
assigned proposals. Direct observed element coverage was 14.34%; conservative
token-conditioned geometry completion added 39.02%, for 53.36% total nonzero
membership coverage. A polygon parser defect initially discarded the LERF
`[N,2]` single-polygon representation and was fixed with a regression test.

The initial global token-plus-null softmax made the fixed 0.5 gate empty. A
per-token null comparison and relaxed 0.2 development thresholds closed the
flow, but the unrestricted-token result remained poor: 2D macro mIoU 0.02270
and 3D macro mIoU 0.02265. Fixed-capacity retrieval exposed a semantic ranking
failure rather than a threshold failure: mean-descriptor top-1 scored zero and
top-3 scored 0.00500/0.00675 (2D/3D). Reusing an independently cached SigLIP2
query bank gave exactly the same top-3 result, excluding current text encoding
as the source of the discrepancy. Retaining every assigned source proposal as
a token-local visual prototype improved neither top-3 (0.01668/0.01698) nor
top-5 (0.01614/0.01685) enough to close the gap.

The validated conclusion is therefore narrower than an end-method failure:
carrier projection, serialization, coverage completion, null-aware query, and
shared 2D/3D posterior execution all run, but current SAM-root object tokens do
not rank the correct visual identity near the top. The next method experiment
must preserve surface-local semantic evidence and use it to localize identity
before a token supplies extent. Further gate relaxation or threshold tuning is
not an adequate response to this failure.

### Text-aligned surface identity correction (2026-08-31)

The first surface-local experiment exposed a feature-space contract violation.
The stored 1536D `siglip2-g` adaptor is explicitly a spatial vision space;
`SigLIP2FeatureProjection` documents that it is not text-aligned and directs
grounding users to `SigLIP2SummaryHead`. Direct comparison of that spatial
adaptor with SigLIP2 text produced zero identity-peak localization accuracy.
Frame mapping, raster aspect ratio, and target carrier support were separately
audited: all 56 annotated observations had 100% carrier pixel support, so the
zero was not caused by an empty target projection.

The corrected development branch starts from the persistent 1280D RADIO
backbone grid and applies the frozen official summary head before surface
lifting. This raises identity-peak localization from 0% to 21.43% and
identity-only micro IoU from 0.01874 to 0.08640. With surface-local peak pooling
and three retained tokens, full Figurines macro mIoU rises from
0.02358/0.02404 to 0.06540/0.06483 (2D/3D). Five tokens regress to
0.04313/0.04158. Thus the feature-space correction is causal and material, but
does not yet make the method competitive.

Two object-descriptor controls further isolate the remaining failure. Applying
the nonlinear summary head after a simple mask mean of backbone tokens fails
(about 0.006 macro mIoU), consistent with that mean not being a genuine summary
token. Averaging text-aligned per-pixel outputs inside each source mask reaches
0.04311/0.04314. An equal-weight fusion with surface-local identity is nearly
neutral at 0.06569/0.06272. The selected development baseline therefore remains
the corrected surface-local top-3 arm; the next bottleneck is increasing legal
source-only identity coverage and cross-view fidelity, not token-capacity or
threshold tuning.

Using every query-independent feature record in the source-only frame manifest
was then tested without adding SAM proposals or opening target RGB. Semantic
element coverage increases from 20.81% at 32 views to 32.68% at 295 views, but
identity-peak localization remains 21.43%; identity-only micro IoU changes only
from 0.08640 to 0.08821, while full macro mIoU regresses to 0.06193/0.06161.
This stops indiscriminate view averaging. Additional views are useful only if
the memory retains a bounded set of consistent/high-confidence per-element
observations instead of averaging every view into one descriptor.

A bounded strongest-projection-mass view bank was then tested with one and two
retained observations per element. On Figurines it raises identity-only micro
IoU from 0.08640 to 0.12298 and 0.16401, respectively, but lowers the complete
2D/3D macro results to 0.06281/0.06156 and 0.05980/0.05906. Projection mass is
therefore not a sufficient view-reliability signal: it broadens identity
support without improving the correct identity rank or object extent.

A second query-independent control selected, for each element, the retained
view closest to its cross-view mean. On Figurines this produces
0.06683/0.06453 versus the reproduced average baseline 0.06541/0.06483. On
Ramen it regresses to 0.04111/0.04241 versus the average baseline
0.04193/0.04425, and its identity-peak accuracy falls from 18.31% to 7.04%.
The small Figurines-only continuous-2D gain is not cross-scene evidence. The
selected default remains the cross-view average; both mass-ranked and
mean-nearest single-view selectors are stopped.

A two-pass soft-reliability arm was subsequently implemented without changing
the base projection measure. The first implementation accidentally replaced
projection-mass integration with equal-view integration; that confounded the
ablation and regressed Figurines to about 0.063 macro mIoU. The corrected arm
multiplies the original projection mass by a query-independent cosine
agreement weight, and uses a leave-one-view-out reference so a view cannot
certify itself. Elements with no peer view retain unit reliability rather than
being discarded.

With agreement floor 0.1 and power 1.0, the leave-one-view-out arm reaches
0.06545/0.06481 on Figurines versus 0.06541/0.06483 for the average baseline;
identity-only micro IoU improves from 0.08640 to 0.09174 and peak localization
from 21.43% to 23.21%. On Ramen it reaches 0.04285/0.04519 versus
0.04193/0.04425, while identity-only micro IoU is essentially neutral
(0.10502 versus 0.10479). Mean reliability is 0.813 on Figurines and 0.836 on
Ramen. This is the first cross-scene non-regressing reliability result, but its
gain is small relative to a second semantic projection pass. It remains an
optional validated arm; the cheaper average stays the default until the same
weight can be computed from cached construction statistics or a broader
cohort shows a material gain.

The same runs exposed and fixed evaluation-integrity issues: every semantic
feature raster is now checked for exact shape and finite values, manifest
frames must exist in the COLMAP registration, empty identity maps no longer
receive a spurious top-left argmax, JSON refuses non-finite output, and the
null posterior is recomputed after top-k token retention. Frozen positive and
negative text banks are now loaded directly by the v4 evaluator, avoiding an
unrelated legacy evaluator import and its optional dependencies. Average mode
also no longer allocates or serializes unused per-view prototype banks.

## Scene-disjoint learned object completion (2026-08-31)

The v4 main line now isolates the missing link explicitly: sparse, mask-supported
observations are immutable input facts, while only unknown element/token pairs
may be completed.  The cache distinguishes `source_visible` from
`membership_observed`, applies deterministic object/view mask dropout, uses
complete 3D ScanNet instance membership only as the unknown-row target, and
evaluates four RGB-free held-out cameras against the original annotated mesh.
The primary assignment decision is threshold-free argmax over all tokens plus
one learned null; the historical 0.5 threshold is diagnostic only.

On the initial four-scene leave-one-scene-out pilot, the F7 RGB+normal MLP
reaches scene-macro 0.45003 3D soft mIoU and 0.38797 held-out 2D soft mIoU,
improving over observed-only by 0.03276 and 0.00621.  Adding source-only frozen
C-RADIO JL64 features raises these to 0.49188 and 0.43187, gains of 0.07460 and
0.05011 over the same observed-only baseline.  This overall gain is real but
does not solve the central visible-mask gap: visible-but-unmasked soft 3D mIoU
falls from 0.08167 to 0.04740, correct-assignment recall from 0.24149 to
0.12528, and predicted coverage from 0.51753 to 0.25391.  The aggregate RADIO
gain is therefore concentrated in the never-visible stratum and calibration,
not in recovery of already visible surfaces missed by the mask teacher.

A matched-capacity RGB control closes the model-size confound.  F7/H213 has
55,169 parameters versus 55,042 for F71/H128, but reaches only 0.44737/0.38073
(3D/held-out 2D), slightly below the smaller F7/H128 control.  Network capacity
does not explain the RADIO improvement.  Two-fold directional sentinels also
reject three local repairs as the next method: visibility-stratum sampling,
an explicit normalized RADIO cosine residual, and availability-conditioned
dual experts.  The dual expert especially collapses one held-out scene, while
direct cache diagnostics show that RADIO VBU object/prototype top-1 cosine is
about 0.488.  Thus the source feature is not globally uninformative; the
current pointwise scoring/training organization fails to convert it into
stable completion.

The pilot was too small for a scientific conclusion, so a formal expanded
cohort is now frozen.  All 57 PFIR annotation scenes were audited; five native
low-resolution scenes that would produce a 30x40 rather than 60x81 RADIO grid
are excluded.  SHA-ranked, physical-family-disjoint selection yields 16 scenes
balanced 8/8 across the two field shards, with 12 training scenes and two
validation scenes per shard.  All 64 source frames have exact checkpoint-bound
C-RADIO features.  The 16 validated caches contain 734,094 elements and 286
retained tokens.  Validation contains 258,033 elements and 78 tokens; its
scene-macro observed fraction is 0.31866, visible-but-unmasked fraction 0.09509,
never-visible fraction 0.58626, and independent mesh-rendered 4 cm ceiling
0.67965.  The superseded v1 stage did not decode or content-hash held-out RGB,
but still exposed hardlinks.  The strict v2 stage removes that capability:
each scene contains exactly the four source images, all 64 held-out image files
are absent, their raw paths are not recorded, and all 16 rebuilt caches remain
tensor-for-tensor identical to v1.

A four-fold shuffled-RADIO diagnostic preserves the RADIO row multiset within
the observed and visible-but-unmasked strata while breaking element alignment.
Relative to the earlier aligned pilot, held-out 2D soft mIoU is lower by 0.04735
on average and is lower on all four held-out scenes; full 3D is lower by 0.03141
on average.  However, unknown-only soft 3D is essentially unchanged (-0.00273
for aligned minus shuffled), and shuffled features increase visible-but-unmasked
coverage.  This supports a limited conclusion: aligned source descriptors help
the aggregate cross-view result, but the current pointwise completion route does
not turn that alignment into stable visible-mask-gap recovery.  Because the old
aligned and shuffled reports were emitted by different reporting revisions,
this remains a diagnostic rather than a sealed single-factor causal result; the
formal v10 runner now requires a hash-bound same-implementation aligned report
for any shuffled control.

Two preparation adapter defects were fixed before training: feature manifests
are now matched to cameras as an exact duplicate-free set rather than requiring
extractor order, and numeric saved stems may drop leading zeroes only when their
integer camera identity remains equal.  A real scene replay verifies that
coverage order `000110,000527,000121,000000` resolves to the corresponding
`rgb_110,rgb_527,rgb_121,rgb_0` tensors without reopening or regenerating any
feature.

The frozen 12-train/4-validation run now provides the first formal completion
result.  Aligned F71/H128 reaches scene-macro 0.56284 3D soft mIoU versus
0.42805 observed-only and 0.55179 for the closest-capacity F7/H213 control.
The same-implementation, hash-bound shuffled-RADIO arm shows the more precise
feature conclusion: alignment improves visible-but-unmasked soft IoU by
0.01561, assignment precision by 0.12190, coverage by 0.06251, recall by
0.05364, and pooled held-out 2D token macro/micro by 0.01360/0.04087.  RADIO
alignment is useful, although its effect is heterogeneous across scenes.

The current pointwise decoder nevertheless fails the Object Gate.  Its
scene-macro held-out 2D score is 0.30747, below 0.32070 observed-only; its
scene-token macro falls from 0.31173 to 0.24144 even while union-micro rises
from 0.25842 to 0.50664.  The discrepancy is explained by object-size
imbalance and spatial over-completion: only 44 of 78 held-out scene-token pairs
contain the target, while the learned posterior creates 34 false-positive-only
pairs.  This rejects independent element-token scoring, not the carrier,
completion supervision, or RADIO.  The next bounded arm is token-conditioned
surface message passing with learned geometry/RADIO edge compatibility and
soft extent regularization; it must improve held-out object-equal 2D and reduce
the 34 false-positive pairs while retaining the demonstrated 3D gain.  Hard
extent envelopes, threshold tuning, and direct LERF deployment of the
oracle-identity checkpoint are not justified.

The bound expanded-cohort contract is
`paper/artifacts/v4_scannet_completion_cohort16_contract_20260831.json`
(`789553dc61ef0509728b8b006152d3bd0d1273e0434e270dee8359b1239c1fa3`).
The scored result is bound by
`paper/artifacts/v4_scannet_completion_formal16_v10_20260831.json`
(`32eabc4f8ead9d54ba7ffe2d02d37b56ff37312447e45b8e9432dffe321ac2be`).
Radius, root cap, tracking threshold, v3 feature partitions, text selection,
and LERF proxy tuning remain outside this stage.

### Shared-edge message passing result

The next bounded arm was implemented without changing the frozen aligned
pointwise unary.  A source-only 19-D geometry/normal/RGB/RADIO edge scorer
feeds two or three local propagation steps, observed token and null facts are
clamped exactly after every step, and a soft observed-seed extent residual
returns suppressed token mass to null.  The first implementation exposed two
contract defects before the formal run: CUDA and CPU `floor` disagreed at a
small number of voxel boundaries, and a zero-logit extent gate removed half of
unsupported mass per step before learning.  Adjacency validation now uses the
same CPU-float32 authority as `SurfaceVoxelCarrier`; all 16 caches and about
2.89 million directed edges pass.  The extent residual now starts at logit
-3, retaining about 95.3% per step, and the latest module/trainer tests pass
19/19.

The fixed six-neighbour topology is not the limiting factor.  Target-aware
diagnostics show that oracle same-token flood-fill reaches 0.96098 full 3D,
0.88733 visible-but-unmasked 3D, and 0.64053 held-out 2D.  Multiplying the
frozen unary by the oracle reachable extent still reaches 0.76040/0.50354
(full 3D/held-out 2D) and removes every false-positive-only held-out token.
Only 4,452 of 128,739 validation object-surface elements lie in true-token
components without any observed seed.  The carrier therefore has ample
completion headroom.

A scene-disjoint source-only edge probe finds real but insufficient boundary
signal: validation AUROC/AP are 0.74534/0.77713, but TNR at 0.5 is only
0.48229.  Threshold-free soft bottleneck extent changes full 3D from 0.56284
to 0.57082 and held-out 2D from 0.30747 to 0.31147, while visible-but-unmasked
3D falls slightly.  A post-hoc hard 0.75 diagnostic raises held-out 2D to
0.38175 and clears false-positive-only support, but collapses never-visible
3D from 0.36890 to 0.11198.  This is a purity--coverage trade-off, not a
threshold to promote.

The formal learned residual is negative.  The two-step arm reaches
0.55607/0.30250 full-3D/held-out-2D scene macro; the three-step arm reaches
0.55565/0.30193.  Both are below the identical frozen unary at
0.56284/0.30747, and all four validation scenes regress in held-out 2D.
Continuous prediction mass on target-absent held-out tokens falls by 12.2%
and 13.1%, and assignment precision rises slightly, but coverage and
never-visible IoU fall.  Exact false-positive-only support count remains a
diagnostic rather than a promotion direction because a strictly positive soft
posterior assigns non-zero support to almost every absent token.

The mechanistic ablation localizes the regression.  Retraining the two-step
model with the shared extent residual disabled leaves full 3D and held-out 2D
within -0.00028/-0.00037 of the frozen unary, improves visible-but-unmasked 3D
by 0.00178, and lowers continuous absent mass by 7.1%.  Learned local messaging
is therefore nearly neutral and locally useful; one query-independent edge
field plus one global extent strength is too coarse.  The rejected arm is
frozen.  The next minimal structure is token-conditioned edge transport with
a scene-disjoint learned soft full-mass prior and two global dual updates.  It
keeps the pointwise unary, carrier, and observed clamps, but allows the same
surface edge to be open for one scene token and closed for another.  It does
not introduce a hard radius, benchmark threshold, connected component, text
query, or v3 dependency.

This result is bound by
`paper/artifacts/v4_scannet_completion_message_passing_formal16_v2_20260831.json`
(`2a23f312b2ebf2d2b444d19d58ad56df0f0d419fc0b0a64f02a30fc4d4f22946`).

### Global physical-mass projection result

The token-conditioned local/dual arm was also negative: forty epochs of
structured extent training change full 3D and held-out 2D by less than
0.0002, even though its edge BCE is learnable.  Two local steps therefore act
as weak smoothing and cannot repair long-range coverage.  The corrected mass
contract treats physical object support directly as the pre-confidence-cap
posterior mass; the 0.95 unknown-membership cap is not a reason to inflate the
mass target.

A target-only global projection gives the decisive mechanism check.  Fitting
one continuous bias per token to its exact physical support mass raises full
3D by 0.09274, unknown-only by 0.10115, never-visible by 0.09466, and held-out
2D by 0.03377.  All four validation scenes improve.  Thus frozen unary ranking
contains useful spatial information; the primary unresolved problem is
estimating and spatially allocating object support, rather than another local
edge threshold.

A deployable source-only scalar mass calibrator does not close that gap.  On
the formal 12 training scenes, summary features improve full 3D by 0.00815 but
reduce unknown coverage by 0.01580; adding the pooled F71 prototype is worse.
Source-camera coverage geometry also fails to improve the summary model.  A
training-only leave-one-scene-out blend cannot rescue the result: it selects
full strength for summary and 0.95 for coverage geometry.

Ten additional physical-family-disjoint PFIR training scenes were then staged
under the strict target-RGB-isolated v3 contract.  Their 40 source images were
processed with the same checkpoint-bound C-RADIO model, and their caches add
165 retained tokens.  With 22 training scenes and 373 tokens, training-only
cross-fit selects a conservative 0.5 summary blend.  The frozen four-scene
validation deltas are +0.00377 full 3D, +0.00242 unknown-only, +0.00254
never-visible, and +0.00162 held-out 2D, but visible-but-unmasked falls by
0.00332 and unknown coverage still falls by 0.01685.  Per-scene token-mass
mean absolute relative error remains 0.269--0.603.  More examples reduce the
worst direction error but do not make one scalar mass identifiable enough.

The bounded conclusion is therefore structural: exact global mass is highly
valuable, but a token-level scalar regressor followed by uniform global bias
projection is not the complete method.  The next arm must predict structured
spatial support (for example multiple token-specific support slots or a global
set-conditioned support field) while retaining exact observed clamps.  More
gate, threshold, local propagation-depth, or scalar-MLP tuning is frozen.
This result is bound by
`paper/artifacts/v4_scannet_global_mass_calibration_20260831.json`
(`c5c96bd0aeb92c4749d47de91111271ef42c5ceea037d22ff2b4c517f4a05835`).

### Bias-free spatial support slots

The first structured spatial arm uses seven continuous slots in each token's
source-observed PCA frame: the observed centre plus a symmetric pair on every
principal axis.  A shared network predicts radius, anisotropic scale, and
mixture weight for every slot.  Joint rigid rotation leaves its result
unchanged, PCA sign ambiguity cancels through the symmetric pairs, and exact
observed categorical clamps remain immutable.  No target, query, threshold,
hard envelope, connected component, or v3 module enters its forward API.

The causal controls expose a material implementation/design defect.  A
token-bias-only arm lowers full 3D by 0.01976 and unknown coverage by 0.06579.
Allowing both token bias and slots is less negative in full 3D (-0.01100) and
raises held-out 2D by 0.00720, showing that spatial slots add real information,
but the learned bias still collapses total support.  Learned slot scales span
roughly 0.14x--7.39x while token-bias magnitude reaches 1.1--2.2, confirming
that global mass and spatial allocation became improperly entangled.

Disabling token bias exactly produces the first consistently positive
structured allocation result.  On the unchanged val4 split, bias-free slots
raise full 3D by 0.00525, unknown-only by 0.00588, never-visible by 0.00586,
held-out 2D by 0.00093, and unknown assignment precision by 0.01444.  Full 3D
improves on all four validation scenes.  Unknown coverage still falls by
0.02401, so this is a retained component rather than a completed stage.

Orthogonally imposing the source-only mass prediction selected by training
cross-fit keeps full 3D positive on all four scenes (+0.00528 scene macro),
raises held-out 2D by 0.00220, and reduces the coverage loss to -0.01978.  It
slightly reduces unknown-only gain to +0.00478, consistent with the known
scalar-mass prediction error.  The result validates the factorization:
spatial slots should control where support goes, while a separate calibrated
head controls how much support exists.  Token bias inside the spatial branch
is now forbidden.  The remaining bounded work is explicit mass-conserving
training of the bias-free allocation field and then association closure.

This result is bound by
`paper/artifacts/v4_scannet_spatial_slots_20260831.json`
(`e3bd9abae46badd03ddc9c9eb909902faffe77b08d53c9e3878d88a41ce68cc0`).

A final target-only upper bound separates spatial ranking from mass error.
Imposing exact physical token mass on the bias-free slot posterior raises full
3D by 0.09740, unknown-only by 0.10829, never-visible by 0.10165, held-out 2D
by 0.03588, and unknown coverage by 0.11890.  Full 3D improves on all four
scenes.  More importantly, this exceeds exact-mass projection of the frozen
unary by 0.00465 full 3D, 0.00714 unknown-only, 0.00699 never-visible, and
0.00211 held-out 2D.  The slots therefore improve spatial ordering under the
same mass authority; they are not merely another calibration bias.  This
oracle arm is non-deployable, but localizes nearly all remaining headroom to
the separate source-only total-mass estimator.

A training-only leave-one-scene-out non-parametric control is also negative.
Both summary and summary-plus-F71 kNN select eight neighbours on the 22
training scenes, but validation token-mass mean absolute relative error stays
at 0.36--0.64 and 0.51--0.63.  When imposed on the bias-free slots, full 3D
falls by 0.09209 and 0.10596 respectively.  Cross-object nearest-neighbour
transfer of complete/observed area ratio is therefore frozen; neither k nor
distance weighting will be tuned on val4.

A scene-balanced ridge residual is the first mass estimator to improve the
deployable 3D candidate materially.  It predicts log mass correction relative
to the frozen unary and selects both regularization and correction blend by
leave-one-training-scene-out loss.  Among three preregistered feature groups,
summary-plus-F71 is also selected by the lowest training cross-fit loss
(0.14340 versus 0.14589 for coverage geometry and 0.14653 for summary).  With
bias-free slots it raises full 3D by 0.01194, unknown-only by 0.00722,
never-visible by 0.01166, and precision by 0.01894; coverage falls only
0.01017.  It is not promoted because held-out 2D is still -0.00051 and the
visible-but-unmasked cohort is -0.00475.

A same-mass control attributes the result cleanly.  Ridge mass projection on
the frozen unary supplies +0.01013 full 3D, while slots under exactly the same
predicted mass add another +0.00181 full 3D, +0.00271 unknown-only, +0.00252
never-visible, and +0.00073 held-out 2D.  The remaining 2D regression is thus
caused by imperfect mass prediction, not by slot spatial ordering.  Aggregate
token features are now saturated; the next mass model must represent each
source view's object coverage explicitly rather than add another aggregate
MLP, ridge feature, kNN metric, or gate.

The explicit source-view arm reconstructs, from the sealed carrier and four
source cameras, only the visibility of already observed token support.  Its 21
fixed-dimensional statistics include per-view visible support, carrier
occupancy, visibility multiplicity, and kept/dropped mask-view coverage.  They
do not read complete token labels or held-out RGB.  Source-view features alone
are rejected because their training cross-fit loss (0.14513) is worse than the
summary-plus-F71 ridge (0.14340), despite positive held-out 2D.

Combining source-view coverage with F71 is selected by training-only cross-fit
at ridge 10 and correction blend 0.5, with a lower loss of 0.14003.  On frozen
val4 it raises full 3D by 0.01265, unknown-only by 0.01060, never-visible by
0.01243, held-out 2D by 0.00098, and assignment precision by 0.01300.  Three of
four scenes improve in both full 3D and held-out 2D.  Coverage still falls by
0.01580 and one scene regresses, so this is the relaxed development candidate,
not a formal completion claim.  Under identical predicted mass, bias-free
slots add +0.00179 full 3D, +0.00260 unknown-only, and +0.00064 held-out 2D;
the factorization remains causally supported.  With both aggregate and
per-view source coverage now tested, work advances to real-token association
closure rather than another mass-feature or gate sweep.

## Gate result and next action

The strict promotion path remains gated, while the user-authorized development
path continues without hard intermediate stops.  The shared-edge/global-gate
arm above is complete and rejected.  Its next action is one bounded ScanNet
ablation: retain the aligned pointwise unary, condition edge transport on each
observed scene token, predict a soft complete-surface mass from its observed
descriptor, and apply two differentiable local/dual updates with immutable
observed clamps.  Held-out rendered loss and continuous absent-token mass must
remain explicit.  Unbounded generic diffusion, threshold sweeps, connected
component heuristics, hard extent envelopes, target RGB during construction,
and historical v3 instance modules remain forbidden.

## Four-scene real-token completion closure

The selected ScanNet candidate is now connected to real source-only LERF
tokens.  Real soft association is hardened only where an element has a
categorical winner.  Tokens that never win any element are excluded from the
completion axis, then mapped back to the full semantic token axis without
receiving synthesized support.  This repaired a real interface defect exposed
by 11 unseeded tokens in Figurines and three in Ramen.  A second cross-scene
defect was fixed when Waldo Kitchen exposed a valid 45x62 RADIO grid while the
development pipeline had incorrectly hard-coded 46x62.

Teatime and Waldo Kitchen were completed with the same source-only MoGe-3
checkpoint and shared radius-one surface projection used by Figurines and
Ramen.  Their geometry gates pass without parameter changes.  The completion
A/B uses identical carrier, association, semantic evidence, text readout, and
evaluation settings within each scene; only completion differs.

| Scene | legacy 2D macro | candidate 2D macro | delta | legacy rendered-3D macro | candidate rendered-3D macro | delta |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.07084 | 0.08293 | +0.01209 | 0.06999 | 0.08276 | +0.01277 |
| Ramen | 0.05785 | 0.05255 | -0.00530 | 0.05954 | 0.05442 | -0.00512 |
| Teatime | 0.10690 | 0.11463 | +0.00773 | 0.10555 | 0.11585 | +0.01030 |
| Waldo Kitchen | 0.15037 | 0.15558 | +0.00521 | 0.15736 | 0.14507 | -0.01229 |
| Scene mean | 0.09649 | 0.10142 | +0.00493 | 0.09811 | 0.09952 | +0.00141 |

Scene-mean micro IoU improves by 0.01907 in 2D and 0.01944 in the rendered
element-thresholded proxy, with all four scenes positive.  Identity-only
metrics are exactly unchanged between arms, so the A/B difference is caused
by completion rather than semantic query drift.  Macro improvement is not
uniform: Ramen regresses in both macro metrics and Waldo regresses in the
rendered proxy.  Per-category inspection localizes the remaining failure to
small and rare object support being over-expanded or assigned to a dominant
token.  The arm is retained as a relaxed development candidate, not promoted
as a finished method.

The reported 3D value remains a development rendering proxy.  Formal LERF3D
is unavailable because the cohort still lacks an official persistent
element-domain target authority; no proxy is relabeled as formal LERF3D.  The
four-scene result is bound by
`paper/artifacts/v4_lerf_four_scene_scannet_spatial_mass_development_20260831.json`
(`14b18e6f7259527a60a3ae2ffa2f9e75f779cdbf760dddba6a16662f5784096f`).

## Bound artifacts

- `paper/artifacts/v4_scannet_spatial_slots_20260831.json`
  (`e3bd9abae46badd03ddc9c9eb909902faffe77b08d53c9e3878d88a41ce68cc0`)
- `paper/artifacts/v4_scannet_global_mass_calibration_20260831.json`
  (`c5c96bd0aeb92c4749d47de91111271ef42c5ceea037d22ff2b4c517f4a05835`)
- `paper/artifacts/v4_scannet_completion_cohort16_contract_20260831.json`
  (`789553dc61ef0509728b8b006152d3bd0d1273e0434e270dee8359b1239c1fa3`)
- `paper/artifacts/v4_scannet_completion_formal16_v10_20260831.json`
  (`32eabc4f8ead9d54ba7ffe2d02d37b56ff37312447e45b8e9432dffe321ac2be`)
- `paper/artifacts/v4_scannet_completion_message_passing_formal16_v2_20260831.json`
  (`2a23f312b2ebf2d2b444d19d58ad56df0f0d419fc0b0a64f02a30fc4d4f22946`)
- `paper/artifacts/v4_scannet_geometry_ladder_oracle_a_20260830.json`
  (`98680aad862bceaef30cfbfd4a7de99b339405cf2d2a84d7766b7c8c6cdd674b`)
- `paper/artifacts/v4_scannet_source_mask_geometry_ladder_a_20260830.json`
  (`111179443c46912461ebeb736be2ea9608640f36a8c71937d4cdde2db0545353`)
- `paper/artifacts/v4_geometry_gate_partial_20260830.json`
  (`5547fe78decb98ab9bd0775f5986779cb42108b0c2cb91cb61b25bbe1b800025`)
- `paper/artifacts/v4_moge3_availability_20260830.json`
  (`4a8af5662f55516f4427cc31ec623e33c76f53960a2bee4311f4d50843f88dd9`)
- `paper/artifacts/v4_scannet_geometry_ladder_oracle_r1_20260830.json`
  (`a7e80df177769d8c64374a170dcdd1b1cbe1c61e6d83599f0d272a6245e8cc44`)
- `paper/artifacts/v4_scannet_source_mask_geometry_ladder_r1_20260830.json`
  (`85a2e64d791af6bb101902c88dfa6c0405bd9ca0281972891daeda3d3c1f2a0a`)
- `paper/artifacts/v4_lerf_figurines_source_mask_geometry_gate_r1_20260830.json`
  (`b85d414ea54dd49c378b0f4359ae8fec994f878662abc96aba006ca00e9fa4d1`)
- `paper/artifacts/v4_lerf_ramen_source_mask_geometry_gate_r1_20260830.json`
  (`86974e34d52121e7e6686361e8ca1583b4a9378579ba2e16c9810d11b3e1c6af`)
- `paper/artifacts/v4_geometry_gate_complete_r1_20260830.json`
  (`a5b3f66a664dddda26896eaf9320d1307a019d082585d245bf4652809dbf2aec`)
- `paper/artifacts/v4_scannet_object_codebook_oracle_gate_a_20260830.json`
  (`0fd6a9eae6b73fc14387321e973571e3468688df3db03ad20d1a8f2d7f2ee42f`)
- `paper/artifacts/v4_geometry_object_oracle_milestone_20260830.json`
  (`d26af7c429c2043fac186c12910026818a5040b5e8a3ad0e6ce0d654556c5005`)
- `paper/artifacts/v4_scannet_unknown_reason_decomposition_16view_20260830.json`
  (`86a60f7e4a58f049140f810724741f51ebd70b14767f25de7dbb2eafcd807b0a`)
- `paper/artifacts/v4_scannet_view_coverage_ladder_20260830.json`
  (`bdf0c2c5572153164a4ec4e2efdc818bcc548e4645b18856dfcaccfd3d4ebf43`)
- `paper/artifacts/v4_scannet_geometry_source_view_selection_20260830.json`
  (`58c92fd8d39daa77eba5cfcf541561f7a8fe91ae60e9fb3bcf5e72aa3e33814a`)
- `paper/artifacts/v4_scannet_source64_geometry_greedy_rgb_authority_20260830.json`
  (`00ee187f510e2544281a2ae09f7ec6e2b8c8de7cd717020b6af87ff09f15a137`)
- `paper/artifacts/v4_scannet_real_sam_object_evidence_ladder_20260830.json`
  (`f849a0247383115d02bd01f689a8a69bf01e311331ae7033a1159f722eb369a7`)
- `paper/artifacts/v4_scannet_real_sam_object_evidence_grid32_16view_20260830.json`
  (`c9652caa4acf01373dc7684788f723e5d3584ecee57a0c93d82cbd1269239961`)
- `paper/artifacts/v4_scannet_real_sam_object_evidence_crop3_16view_20260830.json`
  (`bafaf81c44ba71e5593b894b9669ca6115201b74cbb12259227ad42f2f13fcdc`)
- `paper/artifacts/v4_scannet_real_sam_token_association_permissive_a_ladder_20260830.json`
  (`f3d3dd4a2b9a0005b6816644bea366f5591e2b310f181138b70d7d976d11560b`)
- `paper/artifacts/v4_scannet_source_association_metric_16train16val_20260830.json`
  (`a685caf24a91178e4203ba9b54553b7146e788c73e442c3ea038a127166b4cf2`)
- `paper/artifacts/v4_scannet_association_aware_source_order_20260830.json`
  (`39f6fbf4796cd6df01250ea2d6d2fa3fb3c9b23b168a45576f77ff29c10a5e4e`)
- `paper/artifacts/v4_scannet_real_sam_token_association_overlap_order_ladder_20260830.json`
  (`25d21be4b25cfac32d8f9fa50111e59fa8123c740da1038a408401180692d417`)
- `paper/artifacts/v4_scannet_real_sam_token_association_overlap_order_moderate_ladder_20260830.json`
  (`6317a7fbe4d7320d1e7e453870c266fc0d368de31a74ab8a6c902333aeb21b7f`)
- `paper/artifacts/v4_scannet_real_sam_token_association_overlap_order_strict_ladder_20260830.json`
  (`65b5d4b50ef52a796d5ea7bff2440f4b01798c6d319ab48337cc214e0b9b453d`)
- `paper/artifacts/v4_scannet_real_sam_token_association_batch64_birth020_control_20260830.json`
  (`8cffcecf3fdd8fca6013b685407c26466c86dd3e87d39119951b9246574fa278`)
- `paper/artifacts/v4_scannet_real_sam_token_association_tracking_iou070_batch64_birth020_fixed_20260830.json`
  (`7dfabe60422feaa4c2c9e506d9ea10c5dfc6c5fcbc9df0d417ceab1f8c5e828d`)
- `paper/artifacts/v4_scannet_sam3_video_pair_association_audit_reciprocal_iou050_20260830.json`
  (`7cf859579757fff1b18bdc688e1f2266e6823d20b38cf4c06d4078247083b255`)
- `paper/artifacts/v4_scannet_real_sam_token_association_reciprocal_iou050_batch64_birth020_20260830.json`
  (`e9e98408bf7e960f8665f0a5ebad14e4ca7bd16366c5ff9ac8aeb9b3d9f720e8`)
- `paper/artifacts/v4_scannet_real_sam_token_association_reciprocal_iou050_postgeometry_batch64_birth020_20260830.json`
  (`a0a226be36748184a3741b460fc6fb3f0e76e4fda97613d84d8ab8f978a3eea1`)
- `paper/artifacts/v4_scannet_real_sam_token_association_reciprocal_iou070_postgeometry_batch64_birth020_20260830.json`
  (`33a83ea1c1bbbfc6f4a1cb0aa71911ee2aa403e831acec84fadb27d636c9aef6`)
- `paper/artifacts/v4_scannet_sam3_video_pair_association_audit_16pair_root4_matched_reciprocal_iou050_20260830.json`
  (`36f3d0c7824b7460881966f9446021f7f9707fb5c9fa890aba46bcbfe2a5161f`)
- `paper/artifacts/v4_scannet_sam3_video_pair_association_audit_16pair_root8_reciprocal_iou050_20260830.json`
  (`93908a0ffcc7441e1648af748b5bdf40cfa49ca39286ef824f8dc91ab1d5f145`)
- `paper/artifacts/v4_scannet_sam3_video_pair_association_audit_all_adjacent_root8_reciprocal_iou050_20260830.json`
  (`75047d57da1b45fcfcefcaa3861b393cd3f4e97a47531afbc71914db302461a6`)
- `paper/artifacts/v4_scannet_real_sam_token_association_root8_reciprocal_iou050_postgeometry_batch64_birth020_20260830.json`
  (`adfeff8c54d2f10fd04316757c54b21357f782543b584a58a6870717ef903c04`)
