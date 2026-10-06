# Protect native semantics during structural learning

Status: Experimental implementation, not a promoted architecture.

2026-09-28 next41: source model selection must exclude unknown pixels from
supervised risk, while reporting unknown activation separately. Known-domain
IoU cannot establish complete extent. Optional visibility-weighted priors use
all visible contribution in the query's fitting views to constrain deviation
from the parent; they do not label unknown support negative. This improves all
three ScanNet splits relative to the corrected next40 control but not LERF.
Reviewed source identity/containment alone still over-expands extents. Additional
assistant-reviewed source enclosures help, but are an assisted diagnostic and
not equivalent to automatic benchmark performance. No promotion. See
[next41 report](../experiments/2026-09-28-partial-evidence-and-identity-next41.md).

2026-09-28 next40: optional source-query kernels reconstruct direct source mask
extent through frozen geometry, with missing detections unknown. Query coordinates
remain native, but output rankings can still change. Additive odds corrections
preserve original logits at zero residual; probability clipping alone destroys
low-score category rankings. Corrected full12 replay improves ScanNet19/15 but
hurts10 and does not improve LERF selection; no promotion. Reference-conditioned
scope selection adds independent authorized spatial identity evidence and improves
NVOS8 AP by2.064 points, IoU by0.225. Three global graph-cut controls fail to
improve the parent. Source pseudo-label identity and whole/part composition remain
unresolved. See [next40 results](../experiments/2026-09-28-source-extent-and-reference-identity.md).

2026-09-27 next38 extension: jointly decompose query responses into nonnegative
scope factors rather than independently averaging scopes or taking a maximum
union. An explicit task policy distinguishes standalone category-union images
from reference-grounded prompts. Partial-label category competition learns a
bounded native-query logit correction using only the unchanged source vocabulary;
unknown labels are excluded and overlapping positives remain set-valued. This
changes cross-category decisions, not native encoder coordinates. Full19-map
replay improves LERF extent, but ScanNet remains a tradeoff; defaults stay frozen.
A separate192x256 source reconstruction improves4-scene scope geometry while
retaining the old semantic field/support. It is not a full-resolution feature
rebuild or proof that geometry and teacher capability are fully preserved.
See the [next38 report](../experiments/2026-09-27-taskwise-scope-composition.md).

2026-09-27 next37 extension: fit continuous scope memberships through frozen
source rendering and retain unmatched proposals as explicitly local evidence.
This changes auxiliary range state, not geometry or original native coordinates.
The native-dictionary/text-scope composition improves LERF actual selection but
still loses ScanNet10; it is opt-in and not a default promotion. Image/prompt
paths remain unchanged in that composition. Separate multimodal propagation
and direct native max-union controls expose regressions and are not mixed into
its reported results. Original correspondence validation scores are retained as
correspondence confidence, not relabeled as post-refit independent validation.
See the [next37 report](../experiments/2026-09-27-rendered-scope-reconstruction.md).

2026-09-27 next36 extension: source-view-validated scope hypergraphs are auxiliary
map state over unchanged semantic/structural coordinates and geometry. They
regularize text/image probabilities and optionally reference-conditioned NVOS
responses; they do not replace the map with an object inventory. Fixed confident
unaries protect only part of the final score domain, not all query rankings.
Conflict veto and modality-specific support normalization prevent unsafe
averaging but yield negligible full-cohort gains; no default promotion. A
source-only IoU/ranking head over native scope relations also fails its two real
text pilots and remains diagnostic. See the [next36 report](../experiments/2026-09-27-validated-scope-hypergraph.md).

2026-09-27 research extension: `NativeScopeDictionary` retains the original
96-coordinate field and adds three sparse native-code distributions per element.
This explicitly increases map capacity: 48 serialized bytes per element plus a
shared native codebook. It is not an implementation of the same-capacity single
vector control described below, and does not replace that default. The original
semantic state is frozen, but the new text scores include a bounded orthogonal
native-space contribution, so final semantic rankings are not invariant. Image
and prompt queries keep their original paths. See the
[hierarchical native field experiment](../experiments/2026-09-27-hierarchical-native-semantic-field.md)
for complete development results and limitations.

The [next34 multiview study](../experiments/2026-09-27-multiview-scope-fusion.md)
also explores query-conditioned mixtures of the retained native codewords.
This preserves the original stored anchor but changes matching to a nonlinear
distribution readout; it is not a fixed decoded vector shared by all queries.
Full results improve some 2D/category metrics while degrading selected extent,
so the option remains experimental and default-off.

2026-09-17 experimental follow-up: [bounded image extent readout](../experiments/2026-09-17-bounded-image-extent-and-calibration.md)
reuses the existing global extent decoder for image queries with a bounded
native-logit residual. It adds no per-element state and leaves every parent
tensor unchanged. The image-only opt-in path bypasses text-specific routing;
it does not establish a unified text/image scope algorithm. Native cosines stay
exact, but low-margin final ranks may change. Source-selected strength improves
figurines image retrieval and slightly regresses ramen IoU, so it remains an
experimental candidate. A separate image-query threshold regression is rejected.

This refines ADR 0007's single compact feature field. Each Gaussian still owns
one vector, with explicit coordinate ownership rather than a second feature
map. The controlled prototype uses 64 native semantic coordinates and 32
structural coordinates. Both experiment arms have exactly the same capacity,
source initialization, frozen semantic decoder, support and training budget.
Only semantic-coordinate write protection differs.

## Problem

The current native decoder has full column rank over all 64 field coordinates.
With zero decoder bias, preserving the decoded vector leaves no latent null
space; preserving only its normalized direction leaves a radial degree of
freedom, not arbitrary independent structural capacity. Joint losses for native
semantic reconstruction, region membership, DINO/RADIO reconstruction and SAM
contours are allowed to rewrite these same directions. Weight tuning and extra
source labels do not define a reliable capability-preservation contract.

Text/image semantic affinity, class-union membership, reference-instance
membership and signed-prompt membership have distinct supervision requirements.
Sharing map elements and rendering a common scalar response does not make the
underlying targets interchangeable. Source partial-mask IoU is not full-object
IoU, and the present LERF reference protocol verifies category unions, not
physical instance identity.

## Controlled correction

The semantic prefix is initialized from source-native feature fusion and read
by a fixed native decoder. Structural teacher losses act on the suffix; prompt
queries may read the whole vector. In the protected arm, a write mask blocks all
semantic-prefix gradients and an exact end-of-training invariant checks that
no semantic coordinate changed. Text/image membership currently trains only
its calibration when semantics are protected. It cannot yet exploit structural
coordinates to refine object extent; this is an explicit remaining limitation.

This first prototype tests coexistence of preserved semantics and trainable
structure. It is not the final identity-to-extent inference module. Do not claim
that adding a protected prefix alone solves textual boundary localization or
open-vocabulary object membership.

The training domain is the measured source-native anchor support, held equal
between experiment arms. Unmeasured native coordinates are not exposed as
semantic features. Geometry is unchanged. Export still verifies training
updates in the structural field; validity and convergence remain distinct.

RADIO and DINO provide optional dense structural reconstruction. SAM currently
provides masks/contours/response-boundary supervision, not native SAM feature
reconstruction. SigLIP supplies the native source anchor for this particular
text/image encoder. This does not establish that all four native teacher
capabilities have been preserved, or that SigLIP is mandatory for every future
implementation. The prototype includes a no-RADIO execution control.

## Acceptance

Verify exact semantic preservation on all source-supported elements, equal
support across arms, structural gradients/updates, cold-load equivalence,
source-only parameter fitting, and all three real query modalities. Compare
96-dimensional protected and unprotected arms before comparing against older
64-dimensional runs. Report that element storage increases by 50% relative to
64 dimensions. Independent benchmarks, reliable instance protocols and a
structurally conditioned text/image membership stage remain unfulfilled.

## Follow-up: structural readout trial

The [structural membership control](../experiments/2026-09-16-structural-membership-and-related-work.md)
implements text/image inference from automatic native seeds plus frozen structural
relations. It preserves every parent tensor but reduces final query AP in both
scenes. The trial is rejected for promotion. This refines the acceptance contract:
exact semantic-state retention is necessary but does not certify a learned
membership readout's open-vocabulary decisions.

## Follow-up: registered structural query repair

The [registered structural readout experiment](../experiments/2026-09-16-registered-structural-query-repair.md)
retains direct signed affinity in the frozen 32-coordinate structural suffix
for registered point queries. The existing 64-coordinate native path and all
parent tensors remain unchanged. It improves prompt IoU and AP in both LERF
development scenes, but individual complete objects regress. Source validation
selects one global gate scale; this is not a learned physical hierarchy.
Extra affinity training and automatic native-seed text/image propagation are
rejected. This scoped candidate does not promote a joint benchmark baseline or
claim complete object extent, absence detection, or instance-union support.

## Follow-up: source evidence and query boundaries

The [source evidence investigation](../experiments/2026-09-16-source-evidence-and-query-boundaries.md)
keeps the native semantic anchor unchanged and tests a source-fitted query-to-scalar
decision boundary. The optional native query threshold preserves native element
ranks, parent state and image/prompt responses. Full source teacher-mask fitting
improves text IoU in both development scenes but leaves AP almost unchanged.
Source masks and their outside negatives remain pseudo labels; target absence
and independent benchmark generalization are unverified. Do not describe this
as a solution to missing fine-grained native localization. Anchor replacements
and source-region max pooling did not deliver a joint improvement.

## Follow-up: source-grounded concept extent

The [source-discovered grounded transfer experiment](../experiments/2026-09-16-source-discovered-grounded-transfer.md)
adds an optional global linear extent decoder over the unchanged 64+32 element
vector. Native text prototypes select source-supported concept heads; queries
outside source-template gates fall back exactly to the prior native readout.
There is no additional per-element semantic map. Source captions and SAM masks
are offline pseudo-supervision, unavailable to query execution.

This changes the accepted text query's response from native cosine to a
source-concept extent probability. It is explicitly a concept-conditioned
readout, not evidence that every new query shares a lossless universal metric.
All parent tensors, native cosines, image/prompt outputs and rejected-query
responses are preserved. Accepted queries can still regress through incorrect
semantic routing or teacher scope. Ramen text IoU/AP improves substantially,
but figurines AP falls slightly. Retain this as a scoped development candidate,
not a default joint benchmark replacement. Structure32 membership-code
replacement is rejected because it destroys prior prompt capability.

## Follow-up: render-consistent extent supervision and rejected region selection

The [routing and extent repair](../experiments/2026-09-16-routing-extent-and-region-selection.md)
retains a zero-initialized residual probability decoder with source-view forward
rendering supervision. Both the original semantic and structural element
coordinates remain frozen. Lifted-label regression is not an inverse of the
renderer and is not interchangeable with pixel response supervision.

This remains an opt-in development candidate with modest text gains and exactly
preserved image/prompt observations. Source-centered routing, vocabulary
competition and increasing source-mask resolution are not promoted jointly.
An experimental global bank of independently learned source-region extent heads
also preserves the single per-element field, but increases global state and
regresses real text retrieval. It is rejected despite much higher GT-selected
mask capacity. Neither shared source phrase identity nor native crop similarity
certifies cross-view object/part correspondence. Oracle selection must remain
separate from query execution and training supervision.

## Follow-up: cross-view candidate selection

The [cross-view selection control](../experiments/2026-09-16-cross-view-selection-and-historical-comparison.md)
adds a source-compatibility quality prior and positive-weight native-similarity
ranker without changing the parent field or extent heads. Candidate-origin and
episode-target views are excluded from the training quality prior; a missing
source detection supplies no negative evidence. Source validation holds out
selector supervision only, not the previously trained field and extent heads.

Combining concept and region extents in a global block-diagonal residual decoder
also preserves the single per-element feature field. Both controls remain
unpromoted: cross-view source selection improves, but real-query gains are not
jointly positive. The historical performance control remains available; aligned
GT scoring does not by itself equalize historical source information or render
contracts. Do not describe source compatibility as verified physical instance
identity or use a source validation score as an independent benchmark result.

## Follow-up: native support for previously rejected queries

The [identity / extent completion control](../experiments/2026-09-16-native-identity-extent-completion.md)
keeps geometry, native field tensors and learned candidate extents fixed. Mean
native cosine inside a candidate is insufficient for extent selection; global
native-support IoU also regresses source-supported queries in ramen. The new
opt-in candidate therefore preserves the existing source-accepted concept head
and applies support/extent IoU only to previously rejected queries, abstaining
when intersection is zero. Support is computed over all observed elements with
the existing native query threshold; row subsets cannot redefine selection.

Text IoU improves in both development scenes, but figurines AP decreases and
individual unknown queries still regress. This is an IoU-improving candidate,
not proof of ranking preservation or a promoted joint benchmark baseline.
The configuration name `trusted_concept_start` denotes the location of the
preserved source-accepted heads only: existing false gate matches remain.
Do not turn benchmark query traces into training labels or silently replace
historical controls. Image and registered-prompt observations remain identical.

## Follow-up: source scope admission with hard negatives

The [source scope gate repair](../experiments/2026-09-16-source-scope-gate-and-hard-negatives.md)
adds an optional veto after the existing concept proposal. Its fixed inputs are
native query/text cosine and native query/source-region mean image cosine.
Only two nonnegative weights and a bias are fitted, using source co-observed
mask scope. Nearest incompatible pairs supply hard negatives; missing source
detections remain unknown. Image prototypes exclude the supervised source
view during fitting, and validation splits query phrases. This validation does
not hold out the base field or all source image information.

Actual admitted source pairs are nearly all positive, so fitting only those
pairs cannot establish rejection ability. Arbitrary all-pair fitting also fails
to alter real decisions. The hard-pair candidate improves full-text IoU and AP
in both development scenes while retaining all parent field tensors and exact
image/prompt observations. Keep the previous controls. The new veto does not
certify physical object identity or calibrate a true correctness probability;
source-mask errors and weak fine-grained native responses remain limitations.

## Follow-up: preserve native ordering inside structural membership strata

The [matched compression and completion diagnostic](../experiments/2026-09-16-native-compression-and-ranked-completion.md)
shows little mean AP loss from the current 64D projection of the identical native
source fusion, but large query-specific losses from the final extent probability
readout. The opt-in `native_within_extent` completion retains the selected
membership partition and maps native probabilities into its two response halves.
It preserves native ordering within each stratum, not across the partition or
necessarily after raster mixing. Source-accepted concept heads remain unchanged.

This improves both development scenes' AP and slightly improves IoU; image and
registered-prompt observations stay exact. A single globally rank-preserving
cutoff cannot retain the same extent and is rejected after real IoU regressions.
Do not mistake improved ranking for resolved object extent, or the current small
mean compression loss for proof that every query or historical adapter is lossless.

## Follow-up: candidate unions and calibration forward order

The [native support union audit](../experiments/2026-09-16-native-support-union-and-calibration-contract.md)
shows that even a strictly improving union against the native positive support
can sharply reduce real object IoU. Native support is a query prediction, not
verified object truth. Adding a nondecreasing native precision constraint limits
some regressions but does not establish reliable correspondence; both union
variants remain experimental controls, not the default text candidate.

Source query calibration must use sigmoid at the element level before raster
mixing, as deployed, rather than treating rendered mean cosine as equivalent.
The exact-forward calibration correction changes only text threshold parameters
and leaves actual development scores almost unchanged. Retain it for forward
contract consistency, without claiming it explains the main performance deficit.
Historical mean-cosine calibration artifacts remain available for reproduction.

## Follow-up: image-supervised native residual (2026-09-17)

An opt-in global decoder may map the unchanged 96-coordinate element vector to
an image-supervised correction in the original native encoder space. Its decoded
norm is capped at 25% of the native anchor norm in the current source-selected
control. The anchor, query encoder, geometry and structural field remain frozen;
there is no extra per-element feature table. Both numerator and normalization
use the corrected native vector, not an independently learned query adapter.

Region-mean feature fitting and actual sigmoid-before-raster image mask training
are distinct controls. The latter improves image development retrieval in both
scenes but damages text when shared. Therefore the scoped candidate explicitly
applies this image-supervised correction to image queries only. Text and signed
registered prompts retain their original paths exactly. This is not one identical
final metric for all modalities, and does not certify semantic retention of a
shared residual. Image-only scoping was informed by development results; independent
scene/benchmark acceptance is still required. Do not present this candidate as
an independent test gain or as replacing the retained complete control.

## Follow-up: measured native image support (2026-09-17)

An opt-in image validity buffer may extend the original native-anchor domain only
with actual source-training-view native observations. The current bounded control
requires at least two contributing views, keeps original supported rows exact,
and leaves text/prompt validity unchanged. Newly measured rows have a native
prefix and zero structural suffix; learned residuals do not extrapolate to them.
No new geometry or per-element feature table is introduced. Each modality's
rendered numerator must use its matching evidence-mass denominator. Unsupported
rows cannot become valid merely by flipping a mask or copying a nearby feature.
This is a measured-support experiment, not verified object-identity completion.


2026-09-27 next39 research exception: three opt-in full-native region fitting controls deliberately allow new semantic directions in an additional1536D map; they do not satisfy the protected-prefix method guarantee and are not promoted. Same native coordinates do not preserve query rankings. Both region-feature losses and source regional-retrieval distillation fail reliable replacement in two-scene real evaluation. Existing mainline remains unchanged. See [results and limits](../experiments/2026-09-27-region-native-reconstruction-and-system-gap.md).


2026-09-28 next43 research exception: full-native dense text replacements are experimental and not protected-prefix-compatible replacements; raw spatial tokens are not pooled language vectors. Native local pooling and separately indexed surface carriers failed fixed-scene replacement and remain disabled by default. Legal-source temporal NVOS fusion is an optional additional query-budget variant, with explicit pseudo-negative and boundary-abstention semantics. No target RGB is permitted. See [method evidence](../experiments/2026-09-28-dense-surface-and-source-video-next43.md).


2026-09-28 next44 online-source evidence exception: a query-transient element posterior may use independent source-image identity evidence and teacher mask extents while leaving native coordinates and geometry frozen. This does not assert preserved final ranking or pure field-only inference. The extra input/compute budget, two supported set types and unresolved whole/part carrier issues are explicit in [ADR0010](0010-typed-source-evidence-and-transient-set-inference.md).
