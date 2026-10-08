# Gaussian element supervision and native-conditioned membership

Fixed before new benchmark predictions. Preserve one scene-specific Gaussian
field, the SigLIP/DINO/SAM roles, the same v2 query program, geometry, native
identity channels, candidate policy, scales and thresholds. Standard inference
reads only the export and current input. No v10 labelled member checkpoint,
additional 3D labels, source query bank or task-specific solver is introduced.

## Matched controls

Use all30 original parents from `protected_relations_v1/design.json`. The
completed `protected_training` arm is the matched rendered-loss control: same
parents, source regions, paired views, original support anchors, scale choices,
two epochs, seed20261008, Adam .001, gradient clipping1, group parameters updated,
shared acceptance parameters frozen. Verify its complete source trace, not just
region counts. Earlier controls remain immutable and reproducible at32a79c6.

1. `element_relation`: same existing member formula and trainable grouping;
   replace the two-view subset-alpha objective with partial Gaussian relation
   supervision derived from precisely the same paired masks.
2. `conditional_relation`: exactly the same element targets and sampling, adding
   a scene-specific residual inside that same member function. Grouping and the
   residual train together; original acceptance parameters remain frozen.

This separates the changed supervision/objective from the added conditional
function. It does not isolate every difference between pixel rendering and
element learning, prove actual gradient causality, or validate permanent freezing
as the ultimate policy. The parents already contain earlier rendered learning;
new arms do not retain an additional rendered term during these two epochs.

## Partial element targets

Recover geometric per-pixel contributions C from the compiled normalized
operator and total alpha. In each view, an observed Gaussian is eligible only
if it owns at least10% of some pixel's total contribution. This fixed visibility
witness prevents arbitrarily tiny occluded contributions from receiving a certain
label. For eligible rows, divide positive and negative interior contributions by
that Gaussian's total visible contribution. Remaining raster area is unknown.
SAM nesting and comparable disjoint exclusions follow the existing source rule.

Use half the primary-view evidence and half the associated secondary-view
evidence; normalize the multiple secondary masks within their view using the
existing association weights. Missing evidence is zero known mass, not negative.
The soft target is p/(p+n), weighted by known mass p+n. Conflicting votes remain
soft. A repeated fallback view does not double confidence. This estimates partial
source region relations, not complete-object 3D ground truth.

Uniformly sample128 Gaussian rows with replacement from each nonempty target
stratum (<.5 and >=.5), using a separate generator seed20261009 so source pair
traces match the old control exactly. Average confidence-weighted BCE within
each stratum, then across available strata. Record missing supervision, soft
sample fraction, row exposure and a digest of every sampled row/target/weight.
Region coverage is not proof of full Gaussian supervision or optimization.

## Shared conditional member function

Keep the original group/physical-scale logit and add a residual6*tanh(h/6), with
a10→32→32→1 SiLU MLP, final layer initialized to zero. Inputs: scale-gated group
cosine; paired native semantic and DINO cosine; log(1+scale-normalized squared
distance); absolute normal alignment; log element size relative to query scale;
absolute log element/support size ratio; opacity; element and weighted-support
observation confidence. Log size terms are clipped to[-8,8] or[0,8]. Confidence
is the mean maximum pixel responsibility over training views, stored as one
scalar per Gaussian. It is exposure strength, not exact pair co-visibility.

Native support prototypes preserve their weighted uncertainty. Native vectors
and geometry remain protected, and all query types, including negative supports,
call the same function. Text/image identities are never clamped to positive.
The MLP and confidence scalar are part of the same standalone export. Neither
per-view features nor masks are required at inference. This is a small
conditional-decoding hypothesis, not a novelty or full-performance claim.

## Validation and completion

After final optimization, evaluate fixed element targets on held-out development
source views, using pair seed20261011 and sampling seed20261012. Compare original
parent and final model BCE; no checkpoint selection. Native identity, geometry,
support buffers and original shared parameters must stay exact. Check old-export
compatibility, identity initialization, uncertainty/unknown handling, physical
unit invariance, export replay and matching source/element samples.

Construct all60 fields before prediction. Save and verify all predictions
before any new scoring. Score all four LERF scenes (response and actual subset
projection), eight NVOS cases, ten ScanNet rooms and all268 UQIS inputs. Keep
19/15/10 class scores, all four query modalities, precision/recall and paired
scene intervals; do not suppress regressions or unsupported inputs. Compare
with the matched rendered control, the original parent and previous common
program. Copy one real export per arm for source-denied fresh-process replay of
four inputs and a category menu. No default promotion or per-task winner bundle.
Retain v10 resource/supervision/input contract differences and the fact that this
is a repeatedly used development cohort, not independent final testing.
