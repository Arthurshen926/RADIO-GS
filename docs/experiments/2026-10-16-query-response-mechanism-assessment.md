# TMM scope assessment: query-response refinement on frozen representations

Date: 2026-10-16, Asia/Shanghai. This is a code/evidence assessment and proposed
research formulation, not a new implementation or experimental result.

## User-directed scope

One problem and one mechanism, reusable across suitable teachers and query tasks.
Freeze feature fields, geometry, teacher encoders and native query adapters.
Latent dimension, compression, storage and feature reconstruction are outside
this paper's research variables. Existing storage controls remain in force.
Keep complete five-benchmark feedback; do not select benchmark-specific winners.

## Assessment

There is enough implementation and positive evidence to extract a mechanism
prototype. There is not yet evidence that a single common refinement rule
improves all teachers and query modalities, or that the current components alone
constitute a sufficiently novel TMM contribution.

The defensible question is: when an incomplete or identity-ambiguous observation
disagrees with an initial query response, which parts of that observation may
increase or decrease the response? A reliable foreground region does not prove
that its complement is background. A high initial score is not immutable truth.

## What the implementation actually contains

| Component | Existing behavior | Remaining gap |
|---|---|---|
| `radio_gs/querying/source_identity.py:11` | Within-query one/two-Gaussian mixture model averaging; insufficient separation supplies no relative veto | Score groups can reflect viewpoint or crop differences; they are not calibrated identity correctness and have only text-specific positive evidence |
| `radio_gs/querying/typed_evidence.py:48` | Visibility, erosion, positive/negative domains and detection activation | A detected mask's dilated complement still becomes candidate negative evidence; identity and partial scope require additional validation |
| `radio_gs/querying/typed_evidence.py:58` | Other-view corroboration and several consistency controls for set updates | Pseudo-label agreement is not statistical independence or truth; no common all-task benefit established |
| `scripts/research/fit_consolidated_response.py:28` | Rejected regions remain unknown; admitted region responses are projected through fixed geometry | Native-positive protection can refuse legitimate negative correction; admitted nonempty mass can replace the prior regardless of its total strength |
| `scripts/research/fit_consolidated_response.py:43` | Visibility backprojection and retention of unsupported native responses | Fusion is conventional; it cannot create identity/extent evidence that providers did not supply |

Historical controlled evidence:

- [Response integration](2026-10-02-consolidated-response-integration.md): same
  map, source evidence and selected sets; LERF semantic IoU 50.5885 to 70.3775.
  Selected-set projection remains 66.6617. Primarily establishes that useful
  existing evidence was absent from the response output, not a novel solver.
- [Identity admission](2026-10-02-conditioned-identity-integration.md): fixed
  geometry, features and set solver; selected projection 64.1205 to 66.6617.
  Supports query-relative evidence admission, with regressions and no calibrated
  identity truth. Neither result is an independent final test.
- [Negative-likelihood controls](2026-10-02-partial-negative-likelihood-next101.md):
  uniform negative downweighting, robust loss and removing negative supervision
  did not establish improvement. A single global negative coefficient is not
  the missing mechanism.
- [Reference proposal controls](2026-10-12-reference-proposals-results.md):
  restricting candidate scope did not outperform the matched random control.
  Evidence filtering must beat equally restrictive baselines.
- [Latest compact controls](2026-10-15-unified-five-feedback.md): pooled region
  ranking reduced LERF/UQIS text performance. Correct point identity also fails
  to recover correct extent. These are diagnostics, not proof of the proposed
  refinement mechanism or a reason to keep redesigning the representation.

## Proposed common mechanism

Frozen backend yields the initial response r0. Frozen legal adapters supply
partial observations, their source provenance, query compatibility and geometric
correspondence. A common admissibility rule decides separately whether an
observation can support an increase or a decrease. A fixed bounded residual
update produces r*. Existing typed output contracts then produce a single
instance, a category union or a competitive category partition.

The research target is the admissibility decision. Residual fusion, the geometry
operator and output adapters are supporting machinery. They should not be sold
as three separate innovations.

Positive eligibility needs query compatibility and reliable spatial assignment.
Negative eligibility additionally needs actual exclusion evidence: a legal
negative prompt or a sufficiently supported incompatible alternative at the
requested identity/granularity. Missing detections, occlusion, identity rejection
and a partial mask's exterior are not enough. Cross-view checks should exclude
the emitting view and avoid counting near-duplicate views as independent proof.
These signals remain fallible and cannot imply guaranteed non-degradation.

A schematic scalar-response operator is

    r* = clip(r0 + eta D^-1 sum_e A_e^T[
             w_e^+ (y_e - A_e r0)_+ - w_e^- (A_e r0 - y_e)_+ ], 0, 1).

Here A is a fixed legal observation operator, y is the supplied response, w+/w-
are separately admitted positive/negative evidence weights, and D is positive
contribution normalization with a fixed prior term. This is a proposed interface,
not the implemented new method, not calibrated Bayesian inference, and not a
novelty claim about the equation. No admitted evidence must produce exactly r0.
Evidence rejection must not become negative evidence; valid counterevidence must
be able to reduce an incorrect initial response. Category competition and score
normalization remain explicit frozen output semantics.

Cross-teacher transfer operates on response/evidence semantics, not equal feature
dimensions. Text semantic matching, image patch correspondence and point support
remain different frozen frontends. A 2D ray is not a certain 3D positive; an
estimated text/image anchor is not a legal hard click. DINO alone is not a text
encoder. If a protocol supplies no usable additional evidence, preserving r0 is
valid abstention, but it is not evidence of successful refinement on that task.

Strict map-only UQIS cannot gain source RGB/depth/SAM access. Source-assisted
LERF/NVOS/ScanNet controls must expose the same evidence to every compared
method. This is a response-refinement study with declared access contracts, not
the previous source-free compact-field deployment claim. v10's extra 3D instance
training supervision must stay explicit and matched within a comparison.

## Novelty boundary checked on 2026-10-16

[LUDVIG](https://arxiv.org/abs/2410.14462) already lifts foundation-model features
and refines features/masks with geometry/appearance graph diffusion. Frozen
features plus response smoothing is insufficient differentiation.

[EviSplat](https://arxiv.org/html/2609.34853v1), submitted 2026-09-28, is an especially
close current reference: it preserves observation features, selects evidence
according to text queries, combines instance/local relevance using observation
reliability, and uses positive-only correction. Its appendix also compares
signed correction. Therefore query-conditioned fusion, retaining unsupported
priors, reliability weights and positive-only updates cannot individually be
claimed as new. A possible distinction is explicit query/identity/scope-dependent
permission for harmful-exclusion control across legal query types, but this is a
hypothesis requiring direct matched-evidence comparison and further literature
review, not a verified priority claim.

The [TMM scope](https://signalprocessingsociety.org/publications-resources/ieee-transactions-multimedia)
fits multimedia querying and signal processing. The
[SPS author guidance](https://signalprocessingsociety.org/publications-resources/information-authors)
requires novelty and sufficient experimental evidence; a combination of mature
components alone is not a sufficient contribution. No acceptance prediction is
made here.

## Minimal evidence needed for a paper

1. Same frozen teacher, field, evidence and supervision: initial response,
   simple fusion, graph diffusion/appropriate strong response refinement, then
   the proposed rule. Retain v10's strong legal task frontends as controls; do
   not use only the weak 96D baseline.
2. One-mechanism ablations: shared versus directional admission; remove identity
   compatibility; treat unknown as negative; match retained evidence amount and
   visibility with random or confidence-based rejection. Freeze other terms.
3. Actual transfer: a shared core/configuration across all five benchmarks;
   suitable additional teacher configurations and text/image/point inputs;
   report actual update coverage, regressions, empty/unsupported cases and
   unchanged tasks. No improvement may be attributed to an untouched branch.
4. Beyond mean accuracy: wrong-to-right versus right-to-wrong corrections,
   false exclusion versus false addition, coverage-versus-error curves, and
   query time/peak memory. Include a strong already-accurate baseline and
   controlled missing/wrong evidence stress tests. Real held-out scene results
   remain necessary; repeated development results are not a final test.

The intended contribution is one empirically justified correction mechanism.
Problem characterization and cross-task validation support that contribution;
teacher count, benchmark count and code/interface unification do not substitute
for it. No model changes or new benchmark jobs were made in this assessment.
