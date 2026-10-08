# Recover v10 capabilities within one Gaussian field

Status: research direction authorized by the user on 2026-10-08; mechanisms below
are requirements or hypotheses, not a claim of recovered accuracy.
Supersedes overly restrictive implementation choices in ADR0013, not its core.

2026-10-09:[ADR0015](0015-functional-reference-before-compact-relation-learning.md)
permits an explicitly uncompressed partial-relation functional reference within
the Gaussian export. Verify conditional functions before compacting their
coordinates; neither packaging nor feature reconstruction establishes parity.

## Decision

The user explicitly permits adjusting the ideal architecture to absorb v10's
successful experience, while retaining its core and recovering full performance.
The design remains one scene-specific Gaussian capability field, three teacher
roles (SigLIP semantics, DINO local appearance, SAM grouping), shared conditional
membership, and standalone standard inference from the field plus current input.

Unification means shared scene state, evidence semantics and conditional spatial
relations. It does not require every task to discard its native evidence and
argmax the same scalar affinity. Typed query adapters and small task-semantic
readouts are permitted; they cannot own separate scene maps or silently call old
solvers. Category competition must consume native category evidence as well as
common structure. Instance scope must consume identity uncertainty as well as
positive/negative spatial evidence. One Gaussian posterior feeds 2D and 3D output.

The initial full-native, one-mean-vector-per-channel implementation is not a
capacity upper reference: averaging may already have removed decisive evidence.
If functional retention requires multiple appearance modes, reliability, or
scene-specific decoding capacity, these can belong to the same Gaussian codes
and declared shared decoder. Their utility and storage must be measured. Do not
choose a slot count or compress before identifying what information must survive.

## What is transferred

1. **Native query functionality.** Preserve response ordering and local matching,
   not only reconstruction cosine. v10's bidirectional patch matching and
   separate native-channel ranks are explicit capability references. A mean
   Gaussian vector with maximum query-patch cosine is not an equivalent transfer.
2. **Conditional membership evidence.** Learn view-balanced co-membership,
   joint visibility versus missing evidence, nested-mask disagreement and metric
   relations from legal construction observations. SAM masks are partial and
   hierarchical; unseen or unmatched support is not negative ground truth.
3. **Identity and extent.** Retain alternatives and native identity evidence
   through conditional scope selection. Known-positive point competence is
   measured separately before uncertain text/image conditions. Adding query
   statistics or blending wrong anchors is not an already successful solution.
4. **Typed decisions.** Category partition, category union and single reference
   instance are distinct set questions over the same field. A category logit,
   ordinal identity rank and member affinity are not interchangeable quantities.
5. **Training and rendering.** Match deployed observation/query conditions;
   cover all training regions and record exposure. Retain protected writes,
   valid affine query algebra and actual subset transmittance recomputation.

Source RGB, source masks, tracks and full per-view patch arrays remain construction
artifacts in standard mode. Source-derived relation targets may supervise the
field offline. Using the 22-room labelled v10 member model, including distilling
its predictions, would inherit that extra supervision and requires a separately
declared experimental condition. It is not part of the source-only track.
The attachment's explicitly described NVOS source-assisted extension can be
studied separately through the same core, with full dependencies and costs; the
current standard-mode comparison must not silently acquire those inputs.

## Evidence and experiment order

The [transfer plan](../method/v10-capability-transfer.md) maps each v10 mechanism
to its evidence, current deficiency, proposed realization and retention test.
The five-task baseline and full attribution are complete and immutable. A local
mechanism test may follow that coverage, but cannot replace joint validation.

First implement and verify the native identity primitives against the old exact
operators. Run a fixed, full-67-image UQIS intervention on existing Gaussian
fields to test ordinal fusion and bidirectional field-set correspondence.
This isolates query-side alternatives without retraining or changing resources;
it does not reproduce v10's source patch bank and cannot certify representation
adequacy. Existing same-eight-candidate oracle IoU22.13% already prevents claiming
that reranking those candidates alone can recover v10 image IoU39.90%.

Then establish a matched legal-source high-capacity reference and functional
construction losses, followed by known-identity relation learning and typed
decoding. Keep source budget, teacher/label conditions, representation and readout
interventions separate. Each accepted common method runs all five cohorts and
all four UQIS modalities; no per-query, scene or benchmark winner bundle.

Full v10 performance remains a measured goal. Changed contracts may limit what
is recoverable; no guarantee or top-conference novelty follows from this design.
Compression and speed optimization follow capability recovery, not precede it.

## Implementation record, 2026-10-08

The first full-image identity intervention was followed by independent
relation-training and typed-decision packages. Functional native construction
was subsequently implemented and tested below. This sequencing does not certify
the initial mean field as a sufficient native-capability reference.

The subsequent protected-native study implements the first source-only native
functional construction, using full-dimensional residual maps baked into the
same field. Group learning and shared acceptance parameters are separated in a
matched intervention. This is a method experiment, not a replacement of the
three-teacher/standalone contract or a capacity-upper-bound claim. Its fixed
three-arm full-cohort protocol is `2026-10-08-protected-native-protocol.md` under
`docs/experiments/`.

The fixed common program uses the same cross-view-trained field recipe across
all cohorts, global ordinal image candidates followed by bidirectional
Gaussian-set matching, and bounded native/member category evidence. Text unions
and registered support paths retain the existing common member operator. The
two-arm relation study matches all source targets and exposures; only the
auxiliary view's support condition differs. See the predeclared protocols:
[training](../experiments/2026-10-08-cross-view-relation-protocol.md),
[category evidence](../experiments/2026-10-08-typed-category-evidence-protocol.md),
[common program](../experiments/2026-10-08-common-query-program-protocol.md).
The standalone entry is `radio_gs.query_field.runtime.QueryFieldRuntime`.
The [complete five-task study](../experiments/2026-10-08-cross-view-relation-results.md)
now measures gains in image identity and category decoding, limited additional
benefit from cross-view relations, and a LERF actual-selection regression. It
does not establish a default promotion or full recovery. New disconnected false
members despite preserved native identity require better conditional learning
and functional evidence retention, not merely more steps of the same objective.
A posthoc all-query figurines parameter-block swap retains positive effects of
new grouping codes while localizing the severe selection regression to the
shared scale/temperature/affinity block. Separating representation learning from
shared acceptance calibration is the next controlled full-cohort intervention;
it is not yet a globally validated freeze policy or a licence for scene-specific
parameter selection.

The [completed protected-native study](../experiments/2026-10-08-protected-native-results.md)
now tests that intervention on all 30 fields and all five benchmarks. Posthoc
rollback improves LERF actual-set projection 29.5153 to 33.7203%, mainly from
figurines; training-time protection yields 32.8763% and is not a universal
improvement. Group representation and shared acceptance parameters co-adapt, so
permanent freezing is not established as the final training policy. Shared
native functional maps improve held-out source preferences but do not reliably
transfer to real queries: all four LERF 2D scenes regress relative to protected
training. All 134 registered UQIS point outputs and eight NVOS outputs are exact
under native-only refinement. These paths structurally require membership
learning changes to improve. Full-UQIS observed-target recall also correlates
with source projection contribution; this is an exposure association, not a
measurement of actual training gradients or a proven causal attribution.

The subsequent [full native-coordinate control](../experiments/2026-10-08-native-coordinate-results.md)
is also complete on all 30 fields, all five benchmarks and 268 UQIS inputs.
Per-Gaussian coordinate fitting retains the same exported field and source query
objective. Relative to protected training, LERF actual-set projection improves
32.8763 to 34.8311% and ScanNet10 45.4731 to 46.3277%, while LERF2D falls 52.2864
to 50.5530% and UQIS image falls 20.4974 to 19.4973%. Registered point outputs
remain exact. Better source semantic preferences in 29/30 fields do not establish
real-query recovery. Parameterization and optimizer both change, observed-row
exposure averages 86.32%, and this is not a sufficient capacity upper bound.
No arm is promoted. The next unimplemented method work should test native/geometry
conditioning and reliable element relations inside the common member mechanism,
plus native function supervision that exercises real text/crop-patch behavior.
It must retain the same one-field, three-teacher, standalone-query contract.

The next implementation now adds direct partial Gaussian relation supervision
and an optional zero-initialized native/geometry residual inside the same member
function. Its exported confidence is one training-derived scalar per Gaussian,
not a per-view source bank. Native coordinates, geometry and original shared
acceptance parameters remain protected. Two new arms use the exact source-pair
trace of the earlier protected rendered control, separating the changed element
objective from added decoder capacity. It transfers v10's use of native/metric
evidence and carrier-level decisions without its labelled member model. Full
performance and model promotion remain unproven; the active fixed protocol is
[element relations](../experiments/2026-10-08-element-relation-protocol.md).

The [complete element-relation study](../experiments/2026-10-08-element-relation-results.md)
now closes all60 exports, five families and268 UQIS inputs/arm. Conditional
membership improves source BCE and NVOS response (57.9518→74.6098%), while
LERF2D collapses52.2864→16.9454% and actual selection32.8763→9.4783%.
Correct-support point recall increases but precision and IoU decrease. Native
and geometry feature binding matters on source relations; scalar exposure is
not an adequate pair-visibility transfer. Neither arm is promoted. This does
not invalidate the one-field core: it rejects treating source relation fit as
sufficient calibrated selection. The next controlled repair jointly retains
element relations and actual-subset rendered supervision, with the same parents,
source pairs, sampling and inference, and explicit actual gradient coverage.
The [joint repair](../experiments/2026-10-08-joint-relation-results.md) is now
complete: all30 fields, five tasks,268 UQIS inputs, full point attribution and
five copied-export checks. Compared with conditional-only, LERF2D improves
16.9454→33.2817%, actual projection9.4783→13.0840, NVOS actual subset
71.9338→76.3384 and both UQIS point inputs by about2pp. LERF remains worse
than protected rendered training in every scene. Known-support precision
recovers44.4343→50.6241%, still below protected58.3788. No promotion or
task-specific model selection follows.

Actual rendered logit gradients reach94.5246% of observed rows on average;
source element/rendered errors improve in29/30 and28/30 fields respectively.
These measurements rule out a disconnected rendering branch, not supervision
bias, inadequate optimization or train/query-condition mismatch. All30 default
scales lie within source10–90% quantiles. Native geometry conditioning and
joint supervision remain implemented capabilities, not a validated final recipe.
Next controls must distinguish calibrated known-support extent, uncertainty and
union composition, and genuine native text/crop function retention. Partial
unknowns cannot become negatives merely to suppress false positives. Missing
same-decoder render-only training prevents a standalone synergy claim. The
one-field, three-teacher, standalone inference contract remains unchanged.
