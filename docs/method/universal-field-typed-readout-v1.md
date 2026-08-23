# Universal Field v1 with typed instance-aware readouts

The source-SAM-distilled structural upgrade is specified separately as the
opt-in [Object-Aware Universal Field v2 candidate](object-aware-universal-field-v2.md).
V1 remains the frozen baseline until that field-frozen pilot passes its full
promotion gate.

## Method boundary

Every Gaussian stores one query-independent D512/L512 Canonical Capability
Feature and five reliability scalars.  Official DINO, SAM and SigLIP modules
are frozen capability operators; their outputs are supervision or
query-workspace tensors, not additional persistent scene fields.

The field and readout solve different problems:

1. the field reconstructs reusable appearance, semantic and structural
   evidence;
2. a typed posterior interprets that evidence under the legal input and output
   contract of a query family.

Primitive Readout-v0 remains the causal baseline.  The following typed
readouts are the promoted method.

## Capability-before-aggregation compiler

Frozen capability operators are generally nonlinear, so their order relative
to multi-view lifting is part of the method rather than an implementation
detail.  For source-view RADIO tokens $f_{vx}$, exact compositor responsibility
$w_{gvx}$, and a frozen capability head $h$, the canonical capability target is

\[
c_g^{\mathrm{direct}}=
\operatorname{norm}\!\left(
\frac{\sum_{v,x}w_{gvx}h(\operatorname{norm}(f_{vx}))}
     {\sum_{v,x}w_{gvx}}
\right).
\]

The former primitive readout instead approximated this target with

\[
h\!\left(\operatorname{norm}\!\left(
\frac{\sum_{v,x}w_{gvx}f_{vx}}{\sum_{v,x}w_{gvx}}
\right)\right),
\]

which is not equal to the first expression for a nonlinear $h$.  This is a
task-statistic loss, not category calibration and not evidence that D512/L512
lacks raw reconstruction capacity.

The field-compatible realization retains the one persistent L512/D512 state
and learns a query-independent, scene-global residual decoder

\[
\hat c_g=\operatorname{norm}\left(c_g^{\mathrm{D512}}+r_\theta(z_g)\right).
\]

The output layer is zero initialized, so step zero reproduces the deployed
descriptor exactly.  Training opens only source-view frozen features and the
exact-MPR authority; benchmark labels, masks, text queries and evaluation RGB
remain closed.  A deterministic Gaussian-row holdout must improve both mean
and fifth-percentile cosine before the decoder can be evaluated.  Rows not
observed by the direct source teacher retain the deployed D512 descriptor
bit-for-bit.  The decoder adds shared readout weights but no per-Gaussian
state or second semantic field.

On the frozen ScanNet paper8 cohort, the direct target improves the
same-readout 19/15/10-class macro mIoU from
`0.33535/0.33370/0.42213` to `0.35535/0.35180/0.45769`.  The deployable,
source-gated residual decoder reaches `0.35613/0.35300/0.45954`, gains
`+0.02078/+0.01930/+0.03741`, and also improves macro mAcc by
`+0.00450/+0.01307/+0.02485`.  All eight deterministic held-out-row source
gates pass.  A matched LERF sentinel rejects using the same
descriptor as a task-agnostic replacement: Figurines LERF2D falls from
`0.37021` to `0.28964` and LERF3D from `0.44840` to `0.27927`.  The direct
capability compiler is therefore a **typed categorical compiler** for ScanNet,
not a new universal primitive descriptor.  Fine-grained LERF identity remains
on the D512 identity unary plus object-extent posterior.  This paper8 result
promotes the compiler into the method; it is not by itself a SOTA claim.

## Text identity and object extent

For text query $q$, the field supplies the identity unary

\[
u_i(q)=\cos(e_i,t_q),
\]

where $e_i$ is derived from the same Canonical Capability Feature and $t_q$ is
the frozen text embedding.  Query-independent source-view official-SAM
proposals are associated across views and lifted to Gaussian membership
$r_{ik}$ with exact marginal responsibilities.  Proposal identity is scored by
the official SigLIP masked-crop representation.  The extent posterior is

\[
p_i^{\mathrm{ext}}(q)=\sum_k p(k\mid q)r_{ik},
\]

subject to a field-peak anchoring constraint: a region may complete the object
containing the identity peak, but may not replace that peak.  Localization
uses $u_i$ and segmentation uses $p_i^{\mathrm{ext}}$; they are rendered and
resized independently.  This factorization prevents broad regions from
diluting sparse text identity.

The promoted full-four development result improves LERF2D sample-micro mIoU
from `0.31417` to `0.39584`, preserves localization accuracy at `0.87981`, and
improves LERF3D mIoU from `0.33450` to `0.39684`.  Every scene improves.

A development-only full4 intervention ladder fixes proposal identity,
association, membership and boundaries one at a time.  Rendered sample-micro
fixed IoU is `0.27245` for one oracle proposal, `0.40321` after oracle
cross-view association, `0.41514` after membership-threshold selection,
`0.85676` when the finite proposal-set constraint is removed, and `0.93784`
with an oracle pixel boundary.  These are ceilings, not method results.  They
make dense proposal coverage and association the first implementation gate;
threshold tuning or a boundary head cannot substitute for missing object
support.

A denser 16-by-16 source-SAM sentinel increased Figurines proposals from 277
to 579 but changed LERF2D only from `0.39719` to `0.40065` and regressed paired
LERF3D from `0.46574` to `0.45402`.  It is rejected.  This separates proposal
availability from proposal identity: raw density is not promoted without the
calibrated null posterior and cross-view same-instance association below.
The matched query-listwise arm is worse again (`0.39143` LERF2D and `0.42094`
LERF3D), excluding a mismatch between absolute and listwise identity gates as
the explanation.

### Calibrated proposal/null form

The deterministic promoted readout is the finite, peak-anchored realization of
a latent proposal posterior.  Its calibrated extension is registered as

\[
P(y_i=1\mid q,F)=\pi_0(q,F)p_i^{\mathrm{prim}}(q)
                 +\sum_{k=1}^{K}\pi_k(q,F)r_{ik},
\qquad \sum_{k=0}^{K}\pi_k=1.
\]

Here `null` state $k=0$ means that no source-view proposal is sufficiently
identified and the method must retain the primitive posterior.  It is not a
background object and cannot erase the field identity peak.  Proposal logits
may use only source-legal masked-crop identity, peak compatibility,
query-independent association stability and field reliability.  Any learned
logits must be trained once across scenes with a proper scoring rule; the
fixed analytic realization below is the non-learned alternative.  Target
masks, per-scene thresholds and target-selected proposal counts are forbidden.

Cross-view association uses a three-state edge authority

\[
z_{ij}\in\{\text{same},\text{different},\text{unknown}\}.
\]

`same` and `different` require simultaneous legal visibility and affirmative
SAM/geometry evidence; absence from a proposal is `unknown`, not a negative.
This prevents occlusion and proposal miss from becoming false instance
boundaries.  A learned scorer cannot replace the fixed analytic/deterministic
readouts until it passes a source-only leave-one-scene-out gate.

The current analytic development realization uses no learned target-side
parameters.  For proposal $k$, its source-only cross-view evidence is

\[
a_k=\max_{j:\,v_j\ne v_k}\max\{J(R_k,R_j),O(R_k,R_j)\},
\]

where $J$ is Gaussian-support Jaccard and $O$ is intersection divided by the
smaller support.  Valid proposals must have $a_k>0$, satisfy the frozen area
and query-listwise descriptor gates, and contain the immutable field peak.
Their logits are

\[
\ell_k=8(s_k^{\rm id}-b_q)+\log(a_k)-\log K_q,
\qquad \ell_0=0.
\]

The $-\log K_q$ prior prevents proposal multiplicity from overwhelming the
explicit null.  This analytic marginal raises LERF3D full4 mIoU from
`0.39684` to `0.43730`, Acc@0.25 from `0.61538` to `0.67788`, and Acc@0.50
from `0.40865` to `0.48077`.  It is the promoted LERF3D development readout,
but remains non-blind evidence; a learned replacement still requires the
source-only leave-one-scene-out gate above.

The probability marginal and ternary edge authority are implemented in
`radio_gs/querying/latent_proposal_posterior.py`.  The operator normalizes the
null and valid proposals in one softmax, assigns invalid proposals exactly zero
mass, rejects duplicate Gaussian--proposal memberships, and falls back to the
primitive posterior bit-for-bit when a query has no valid proposal.  This is
the method seam used by the source-trained scorer; implementing the exact
probability operator does not by itself promote an untrained scorer or change
the reported benchmark rows.

The learned seam is implemented by
`radio_gs/models/proposal_null_scorer.py`.  It is proposal-permutation
equivariant, has an explicit DeepSets null head, and zero-initializes both
output heads.  Its exchangeable prior subtracts $\log K_q$ from proposal
logits, so epoch zero assigns one half of the mass to null and shares the other
half across the complete valid proposal cohort regardless of proposal count.
Training uses a categorical log score plus multiclass Brier score; occluded or
unsupported queries are marked unknown and excluded rather than converted to
null negatives.  This implementation is not promoted until a source-only,
scene-disjoint authority supplies its checkpoint and calibration gates.

LERF2D and LERF3D must consume the same Gaussian-domain posterior.  An output
domain may apply only an order-preserving Platt map

\[
C_d(p)=\sigma(a_d\operatorname{logit}(p)+b_d),\qquad a_d>0,
\]

before its legal projection.  This cannot change proposal selection, Gaussian
ranking, or topology.  The evaluator implements the seam with identity
defaults $a_d=1,b_d=0$; non-identity values require source-heldout calibration
and may not be selected from LERF masks.

### Reliability and boundary ownership

The five persistent reliability scalars are evidence precision, not an
unconstrained semantic predictor.  A reliability-conditioned extension may
combine identity and extent logits only as

\[
\ell_i=w_i^{\mathrm{id}}\ell_i^{\mathrm{id}}
      +w_i^{\mathrm{ext}}\ell_i^{\mathrm{ext}},
\]

with monotone constraints: more observation evidence cannot reduce identity
precision, while lower visibility purity or lower association stability cannot
increase extent expansion.  Reliability never creates class or proposal
identity.

Object membership and rendered pixel boundaries are separate authorities.
The field/proposal posterior owns object support; a query-transient frozen-RGB
boundary decoder may sharpen that support only where the benchmark contract
permits RGB.  It cannot select an object or write back persistent scene state.
Before any boundary residual, a soft Gaussian membership is projected with the
same visibility law as color,

\[
P(y_x=1\mid q)=\sum_i T_i(x)\alpha_i(x)P(y_i=1\mid q),
\]

leaving $1-\sum_iT_i\alpha_i$ as explicit background mass.  It is not
alpha-normalized: normalization would amplify low-opacity tails and erase the
distinction between foreground evidence and missing opacity.  Benchmark rows
whose released evaluator requires a binary selected-support projection retain
that named compatibility operator instead of being silently reinterpreted.

The current-field SAM boundary closure has now also been tested on the analytic
latent proposal posterior. On LERF2D full4 it raises sample-micro mIoU from
`0.37080` to `0.37818` with LocAcc exactly unchanged at `0.87981`; it also
exceeds the earlier primitive+SAM composition (`0.36152`). The result confirms
that the boundary authority is complementary to identity/extent inference, but
does not replace the retained `0.39584` readout. It therefore remains a
source-gated boundary seam, not an unconditional method switch.

## Mutually exclusive categorical posterior

For ScanNet OVS, class identity remains the normalized categorical field
posterior.  Let $m_i$ be the top-two class margin and $L_{\mathrm{sam}}$ a local
query-independent SAM-affinity operator.  The accepted correction is

\[
\tilde p_i = p_i + \alpha\,\mathbf 1[m_i<\tau]
             \left(L_{\mathrm{sam}}p-p\right)_i.
\]

Thus confident rows are immutable and uncertain rows receive only bounded
topological evidence.  Fixed parameters are 8 neighbors, 0.1 m radius,
SAM cosine threshold 0.5, $\alpha=0.25$, and margin threshold 0.03.  On paper8,
19/15/10-class mIoU changes from `0.33535/0.33370/0.42213` to
`0.33810/0.33617/0.42548`; all corresponding mAcc values also improve.
Unrestricted SAM categorical propagation is explicitly excluded.

A stronger class-temperature/bias shrinkage screen reaches development mIoU
`0.41023/0.43269/0.52560`, but its parent and variant were selected with
paper8/heldout evaluation scenes.  It is mechanism evidence for constrained
class competition, not a promoted result.  A replacement must learn one
shared rule from legal source scenes and pass a leave-one-scene-out gate.

The subsequent scene-redundant categorical-calibration branch is rejected as
a method direction.  Its first legal three-scene LOSO test regressed source
mIoU by `0.24576/0.26730/0.23732`, and adding scenes would improve parameter
identifiability without addressing object membership.  The unified
replacement is query-independent region/instance co-membership distilled into
the Canonical Capability Feature.  ScanNet class prediction remains normalized
text similarity plus argmax; a region operator may alter membership support or
bounded low-margin consistency, but it may not introduce class-specific
temperature, bias, or background rejection.

## Registered prompt posterior

NVOS and Available-Nine SPIn-NeRF provide registered prompts and permit the
contract-specific transient RGB workspace.  The promoted compiler implements
the same identity--extent factorization as the text branch: the sealed signed
field owns query identity, while frozen official SAM3 geometric proposals own
object extent.  In each registered view the field support produces one padded
box; official SAM3 produces its complete proposal set; one proposal is selected
by signed positive inclusion and negative exclusion, with coarse-field overlap
and SAM score used only as target-blind tie breakers.  The selected observations
are exact-adjointed to the common Gaussian carrier.  Only positive detections
are composed by noisy-OR; an absent detection or invisible row is unknown, not
a negative vote.  Missing views, assignments, or nonfinite evidence abstain.

The positive/unknown fusion and the rejected deterministic point control are
implemented in
`radio_gs/querying/synchronous_multiview_candidate_marginal.py`.  Candidate and
view SHA-256 identities are bound before accumulation.  The complete official
region invocation layer is
`radio_gs/scripts/build_nvos_synchronous_multiview_box_sam3_inventory.py`; it
replays the native target-view signed margin exactly, rather than applying a
lossy same-view `W.T/W` round trip before proposal generation.  Every view binds
captured RGB and exact compositor assignment.  The carrier-native plan is
`radio_gs/scripts/build_nvos_synchronous_multiview_candidate_plan.py`.  It
exact-adjoints the sealed field prompt and official source scribbles to the
current carrier and reprojects them to the complete source/target cohort, while
preserving the sealed native target raster as the target observation.  The
streaming exact-adjoint consumer is
`radio_gs/scripts/materialize_nvos_synchronous_candidate_marginal.py`.  The
Fern cold-start sentinel reaches `0.83340` foreground IoU and `0.94626` pixel
accuracy, versus `0.83061` for its native target-only box observation.  It
selects proposal 78 of 200, matching the retained target-only mechanism.

The native full8 execution is complete.  Same-run target-only macro IoU is
`0.81743`, while unconditional exact-adjoint source+target positive/unknown
fusion reaches `0.81385`.  This rejects unconditional source union, not the
native compiler.  Applying the already frozen component-local identity-density
risk limiter raises the all-view result to `0.82624`.  That deterministic
authority-bound replay is positive mechanism evidence, but was executed after
the unconditional development metrics were opened and remains just below the
existing target-only component-local row (`0.82651`).  It is therefore not a
new promoted or SOTA result.  Future source-view evidence must earn local
identity and transport authority; availability alone is not precision.

The earlier point-only compiler is retained as a negative ablation.  Robust
log-odds view fusion reached only `0.49936` Fern IoU.  Correcting missing-view
semantics to positive/unknown noisy-OR raised it to `0.68301`, proving that the
unknown state matters, but it remained far below box-region decoding.  Denser
weighted-farthest point trials did not close the gap (`0.67118`).  Thus the
region proposal, not point density or another view-fusion weight, is the core
extent mechanism.

The already completed target-frame sentinel first seals positive and negative
field support.  Frozen official SAM then decodes box and signed-point
hypotheses independently.  Final consensus ranks masks by positive inclusion
and negative exclusion; field overlap and SAM confidence are tie-breakers.

Box and points are not concatenated inside the SAM prompt encoder.  The joint
call reduced NVOS macro IoU from `0.81776` to `0.68259`, showing that prompt
encoder interactions are not an additive posterior.  This independent
target-frame selector reaches `0.81776` versus the field-only `0.52687`, without
using target masks for prediction.  It is valid RGB-assisted development
evidence, but it is an incomplete all-view-contract sentinel and does not
replace the primary shared compiler.

A subsequent identity-supported extent candidate applies the same separation
inside the decoded mask.  Official SAM supplies extent, while the sealed field
unary and its frozen positive points supply identity.  A connected SAM
component is retained only when it contains a positive point or explains a
minimum fraction of coarse identity support; the operator may delete instance
leakage but can never add foreground.  On the NVOS full8 development cohort it
changes macro IoU from `0.81762` to `0.86858`.  The current support fraction
(`0.05`) was inspected after development metrics were available, so this is
mechanism evidence rather than a promoted frozen rule.  Source/reference LOO
cannot identify the fraction because NVOS reference authority contains
scribbles rather than full masks.  The numerical floor is instead bound by
analytic inheritance from the `0.05` minimum coarse overlap already frozen in
all eight pre-metric SAM selector receipts.  This supplies a target-independent
parameter authority for future evaluation, but does not retroactively make the
development-derived component rule prospective.

Independent SPIn confirmation rejects that first global-mass denominator: it
removes real fragmented support in orchids, horns, and leaves.  The corrected
rule measures identity density locally inside each connected component,

`overlap(component, coarse identity) / area(component) >= 0.05`,

or retains the component when a frozen positive point anchors it.  This rule is
scale-local, adds no foreground, and inherits the same pre-metric `0.05`
authority without a new threshold search.  After freezing on NVOS plus SPIn
lego/orchids development evidence, the untouched horns/leaves holdout improves
scene-macro IoU from `0.69879` to `0.70073`; both scenes improve.  Across all
four currently gated SPIn scenes, the local rule improves scene-macro IoU from
`0.72279` to `0.72504`.  Component-local identity density therefore replaces
the rejected global-mass rule in the development method.

SPIn field-gated evaluation uses the same current sentinel after all nine
Method-v1 fields pass their gates.  Its full9 result diagnoses the new field,
but the final shared-contract row additionally requires the synchronous
all-view compiler above.  A historical carrier score is not a Method-v1
result.

The registered prompt contract treats protocol-authorized target RGB as a
transient observation, not target supervision.  Benchmark masks remain
scoring-only.  Consequently RGB-assisted NVOS/SPIn rows are valid when named
as such; `strict-unseen` is a different diagnostic contract and must not be
silently substituted for the benchmark's registered-prompt contract.

## Promotion rule

A typed readout is promoted only when its frozen cohort gate passes and no
scene exceeds the allowed regression.  Development improvements are recorded
as development evidence; they are not automatically prospectively blind or
SOTA claims.  Failed candidates remain negative controls and cannot be selected
per scene.
