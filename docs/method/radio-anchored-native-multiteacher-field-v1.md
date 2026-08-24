# RADIO-anchored native multi-teacher field candidate

## Status

This document defines a development candidate, not a replacement for the
frozen Universal Field v1 contract.  Universal Field v1 remains the D512/L512
RADIO field plus five query-independent reliability scalars.  A native
capability becomes part of the method only after a source-only gate and a fixed
full-cohort benchmark confirmation pass.  Rejected teachers and benchmark
diagnostics do not change the field identity.

The candidate keeps exactly one row-indexed latent per Gaussian.  Native
teachers may add scene-global decoders or query-transient object evidence, but
may not add a second row-indexed semantic table:

\[
  z_i\in\mathbb R^{512},\qquad
  \hat f_i^k=D_k(z_i),\quad
  k\in\{\mathrm{RADIO,DINO,SAM,LANG}\}.
\]

RADIO is the shared visual anchor, initialization and fallback.  It is not
assumed to be a sufficient statistic for every downstream capability.

## Why the capability must be lifted at its native level

Exact marginal lifting is linear, while the official capability heads and
object compilers are generally nonlinear.  Therefore

\[
  \operatorname{MPR}(h(f_{2D}))
  \ne
  h(\operatorname{MPR}(f_{2D})).
\]

Every teacher is consequently constructed from legal source RGB at the level
actually consumed by the query:

- native DINO dense descriptors supervise correspondence and physical
  appearance;
- native SAM masks supervise object membership, boundary and unknown
  visibility rather than generic feature cosine;
- native SigLIP dense/category capability supervises mutually exclusive
  category competition;
- native SigLIP masked and expanded-context crops supervise object identity;
- RADIO reconstruction preserves the common visual carrier and supplies a
  fallback where a native teacher is unobserved.

The mapping contract is source RGB -> native 2D statistic -> exact front-to-
back marginal registration.  Applying a native head after 3D averaging is a
separate control, not an interchangeable implementation.

## Query-variable-aligned readouts

Text, image and registered-prompt queries share the persistent latent but do
not share one output formula.

### Text

Text identity is supplied by the field language unary and object-level native
SigLIP evidence.  Extent is supplied independently by native SAM membership.
The identity heatmap remains responsible for localization; extent cannot move
the identity peak.  A physical-appearance authority such as native DINO may
reject or rank object proposals, but may not replace text identity.

### Image

Native DINO or RADIO retrieves a physical identity anchor.  The object
membership decoder then expands the anchor to a complete instance.  Dense
correspondence quality and held-out object reconstruction are separate gates.

### Registered prompt

The field owns identity and registration.  Protocol-authorized RGB plus
official SAM owns transient target extent.  Source-view proposals are included
only through an independently reliable identity/transport rule; unconditional
view union is not the method.

For NVOS RGB-assisted evaluation, the promoted development compiler factors
the transient query into four typed operators:

1. the official signed prompt identifies the instance;
2. full-scribble selection among official SAM3 prompt proposals establishes a
   complete prompt-side extent;
3. official SAM3 memory transports that instance over registered RGB;
4. the frozen target SAM/field-agreement rule either authorizes memory or
   replays the sealed target-only extent.

This compiler adds no persistent Gaussian state.  It is specific to a contract
that authorizes registered/target RGB and must not be reported as strict
source-only NVOS.

## Object membership supervision

For source proposal `o` and Gaussian `i`, the target has three states:

- positive: `i` is visible and has exact-marginal support inside `o`;
- negative: `i` is visible in the same observation and outside `o`;
- unknown: `i` is not observed, is occluded, or has conflicted proposal
  granularity.

Unknown rows never enter a binary loss.  Training and held-out source views are
disjoint.  The decisive gate is source-heldout mask reconstruction, not pair
AUC:

\[
  P(i\in o)=D_M(z_i,e_o),
\]

where `e_o` contains normalized native language identity and, when available,
native physical appearance.  `D_M` is scene-global and adds no per-Gaussian
state.  Thresholds are selected on source-training proposals only.

## Capacity-matched A/B/C gate

Before any full-field rebuild, each native dense teacher is tested with the
same geometry, L512 capacity, random decoder initialization, minibatch order
and optimization budget:

- A: frozen RADIO latent plus a global native decoder;
- B: the same L512 latent updated by native loss with a RADIO preservation
  anchor;
- C: the same L512 latent updated by native loss without the RADIO loss.

Held-out source observations decide the gate.  B is promoted only when it
improves A without materially reducing RADIO fidelity.  If A matches or beats
B/C, the native teacher is already linearly recoverable from the frozen latent
or the latent update overfits; a full reconstruction is stopped.

The first matched DINO-only sentinels on Figurines, ScanNet scene0000 and NVOS
Fern all select A.  B/C preserve RADIO almost exactly but slightly reduce
held-out native-DINO cosine.  This rejects naive dense-DINO latent retraining;
it does not reject native object membership or language-region supervision.

## Current promotion boundary

- ScanNet now uses variable-aligned categorical distillation.  Native SAM
  extent and native SigLIP region identity are composed at the actually
  consumed category-score level: equal-view aggregation, class-symmetric
  centering, source agreement and structural replay precede distillation.  A
  zero-initialized scene-global decoder maps frozen L512 plus the deployed
  categorical score to a residual; it adds no per-Gaussian learned state.  The
  teacher-gate capacity row is `0.36575/0.36273/0.46817` mIoU for 19/15/10.
  The deployable candidate predicts eligibility from L512, the five existing
  reliability scalars and the already computed baseline categorical response;
  it reaches `0.36401/0.36189/0.46716` without retaining native eligibility
  bits or any other per-Gaussian side information.
- LERF retains the promoted identity--extent posterior.  Native DINO proposal
  association and native DINO+SigLIP membership are source-gated candidates;
  they are not allowed into the benchmark method unless held-out membership
  improves materially.  The first multimodal decoder, disjoint-view proposal
  retrieval, and dense-frame official-SAM3 video propagation gates all fail,
  so none changes the method.  Cross-view membership must be transported by
  explicit registered geometry and checked against independent native regions.
- NVOS retains registered field identity plus target-RGB official-SAM extent,
  augmented by prompt-side complete-region selection, video memory and the
  frozen region reliability veto.  The native DINO A/B/C sentinel remains a
  source-only representation diagnostic, not the source of the NVOS gain.
- SPIn is outside the short-term validation cohort because the available
  dataset is incomplete (9/10).  It neither blocks nor consumes resources from
  the four active benchmark contracts.

## Promotion rules

1. No benchmark label, mask, query or metric participates in teacher fitting,
   source threshold selection or A/B/C selection.
2. A fixed operator must improve both mIoU and its companion accuracy metric
   on the independent confirmation cohort; per-scene fallback is forbidden.
3. LERF2D and LERF3D must consume the same Gaussian posterior and both improve
   without localization regression.
4. Registered-prompt RGB assistance must be explicitly reported as such; it is
   neither leakage nor strict source-only evidence.
5. Every output is bound to its field, source authority, teacher and query
   cache hashes.  A rejected pilot remains a diagnostic and cannot silently
   redefine Universal Field v1.

## Categorical score distillation

ScanNet classification does not consume a 1536-D descriptor directly.  It
consumes mutually exclusive class responses after source-view region evidence
has been centered and reliability-gated.  Consequently descriptor cosine and
even fixed-vocabulary top-two-margin losses are insufficient: both passed
source reconstruction gates but failed to improve two independent benchmark
sentinels.

For split `s`, the promoted candidate teacher is instead

\[
 r_i^s=\operatorname{norm}(q_i^s-\bar q_i^s),\qquad
 y_i^s=\operatorname{norm}((1-\alpha)p_i^s+
 \alpha a_i^s r_i^s),
\]

on source-eligible non-structural rows, with exact primitive replay elsewhere.
Here `q` is the native SAM-extent/SigLIP-identity class response, `a` is
cross-view class agreement, `p` is the deployed primitive class response and
`alpha=0.25`.  The compact decoder optimizes centered score coordinates and
teacher top-two margins on a fixed Gaussian-row holdout.  It therefore
distills the query variable rather than a high-dimensional surrogate.

On paper8 the teacher-gate capacity diagnostic raises mIoU from the descriptor
student `0.36027/0.35957/0.46626` to `0.36575/0.36273/0.46817`.  The final
compact gate consumes L512, the five existing reliability scalars and the
already computed baseline category-score blocks.  Source-only calibration may
abstain on a split only when replay is exact, every split is noninferior in
teacher-score error and their aggregate is strictly better.  FP32 is mandatory
for the 44-channel query cache so abstention reproduces the baseline decision
exactly.

The resulting fully compact paper8 row is
`0.36401/0.36189/0.46716` mIoU and
`0.67287/0.66803/0.76741` mAcc.  Relative to the restored baseline it gains
`0.00787/0.00889/0.00763` mIoU and
`0.00312/0.00542/0.00385` mAcc.  All 24 scene/split mIoU values are
noninferior to baseline, and no benchmark-conditioned fallback is used.  The
remaining gap to the teacher-gate capacity row is the explicit cost of
compressing eligibility, not an unaccounted sidecar.
