# Semantic field primary with MoGe-3 geometry support

Status: Accepted for implementation by user direction, 2026-09-14.

This changes the architecture priority of ADR 0005 and extends ADR 0006.
The Single Compact Feature Field on Gaussian primitives is the primary map.
Local field evidence owns identity. Query-independent region candidates may
supply extent. Sparse voxels and surface carriers supply geometric support;
object organization is not a prerequisite for semantic queries.

MoGe-3 is a source-only geometric prior. Its predicted geometry is not assumed
correct solely because the model is accurate on other data. Bind predictions,
source cameras, scale calibration and source frame authority before applying
it. Evaluation RGB, labels and queries do not participate in the write path.

The first implemented candidate is a bounded depth compatibility residual on
Gaussian semantic observation weights:

    w_new = w_exact * (1 - strength * confidence * (1 - compatibility))
    compatibility = exp(-0.5 * ((z_gaussian - z_source) / tolerance)^2)
    tolerance = 0.08 metres + 0.02 * z_source

All quantities are converted to the same COLMAP world units. The initial
strength is 0.5, fixed before task evaluation. Missing/invalid depth and views
without accepted bounded local calibration preserve original writes. Strength
zero reproduces original weights exactly. This is a compatibility kernel,
not a calibrated posterior. Validity is not called model confidence: currently
confidence is a binary accepted-calibration/valid-depth gate.

The exact compositor authority remains immutable. Only the derived teacher
weights change; saved metadata records the new contract and source hashes.
Frozen canonical lifting contracts reject this option rather than silently
claiming compatibility. Initial integration supports dense exact-marginal
contribution-mean teacher construction. Gaussian centre depth is an explicit
approximation, not the depth of a ray/surface intersection.

No Gaussian geometry, opacity, or query score is changed by this first step.
This retains existing observed support while evaluating whether depth can
reduce cross-surface semantic contamination. Normals, multi-view confidence,
geometric optimization, and surface-driven coverage insertion are subsequent
ablations, not implemented benefits. Do not snap semantic features to voxels
or introduce another persistent semantic field.

Promotion requires a matched old/new teacher build, disjoint source-heldout
semantic reconstruction and boundary/coverage diagnostics, then unchanged
identity–extent readouts and full-cohort cold evaluation. Preserve the restored
historical control. Unit tests and source-weight diagnostics are not evidence
of task-metric improvement.

2026-09-15 user-directed clarification: no VFM, including RADIO, is mandatory.
The persistent map may use a learned latent with compatible query adapters.
Teacher presence, validity, feature space and auxiliary losses must be explicit
and independently optional. Acceptance includes running without RADIO weights
or caches, not merely setting a RADIO loss coefficient to zero. The first
native-DINO image-retrieval candidate is implementation evidence, not promotion
of a three-modality method.

2026-09-15 implementation extension: the development candidate now trains text,
image and registered signed-prompt retrieval against cross-view source-region
pseudo labels, with independent RADIO/DINO/SAM/SigLIP auxiliary terms. Native
SigLIP text/image queries may explicitly share a learned projection because
these encoders share a verified native space; equal dimensions alone never
justify sharing. Prompt pixels supervise the composited response, not a
single maximum-contribution Gaussian. SAM contours constrain feature structure
without converting all outside-mask pixels into membership negatives.

This is Development Evidence on figurines, not a Joint Development Baseline.
Teacher-generated vocabulary labels and mask correspondences are fallible.
Turning off auxiliary teachers does not remove the annotation producer or the
selected native query encoders. See the four-teacher implementation experiment
for cold execution, coverage limits, matched no-RADIO training and next gates.

2026-09-15 feature-training integrity correction: geometric visibility permits
training writes but does not establish a usable semantic observation. Exported
candidate validity now requires a nonzero feature-training gradient as well as
visibility; this is a minimum evidence condition, not proof of convergence.
Unobserved rows retain an invalid flag and cannot receive a positive retrieval
score in either direct or conditioned-query execution. Geometry remains fixed.

Dense teacher view sampling now covers the legal training views independently
of matched retrieval episodes. Per-view pixel traversal covers valid pixels
before repeating them. A source view with no cross-view region match must not
silently lose all dense VFM supervision. These are default training-integrity
corrections, not performance promotion. Legacy sampling remains explicit for
controls. All auxiliary teachers remain optional; no RADIO access is required.

2026-09-15 signed-query development clarification: registered positive and
negative source pixels may condition a transient query vector using the
immutable visible-element contributions. The field, query adapter, opacity
and geometry remain frozen, and the resulting single element posterior is
shared by 2D and 3D outputs. Already-correct signed evidence at the actual
retrieval threshold preserves the initial query; unresolved constraints are
reported rather than hard-clamped. This adds query computation and applies
only where registered signed evidence is authorized. Source pseudo-label
experiments are not promotion evidence for other benchmark query contracts.

2026-09-15 text-query evaluation clarification: text has no reference-image
identity. Within a fixed evaluation split, one text and target view is one
query task even if several source correspondences supplied its pseudo labels.
The evaluator now unions the existing positive/negative evidence and retains
source-record provenance; overlapping positive and negative pixels remain
unknown. Correspondence-weighted text reporting is an explicit legacy control.
Canonicalizing training records remains an independently tested option, since
it changes task sampling and loss weighting. Expanded validation cohorts must
specify their frames explicitly and filter both query and target views.

2026-09-15 coverage-aware evaluation clarification: supported-only IoU is a
conditional feature-quality diagnostic, not end-to-end retrieval quality.
Evaluation must additionally retain all existing nonconflicting 2D labels,
zero unsupported predictions, and count uncovered known positives as misses.
Report positive-label support recall alongside both scores. Reference-field
support intersections apply only to the conditional comparison, never to this
all-label denominator. Unknown labels remain unknown; these source labels are
still not benchmark ground truth. This changes reporting, not training losses
or frozen geometry.

2026-09-15 real-label acceptance clarification: the new frozen-field native
text evaluator can consume the existing LERF 2D polygons without opening
training features or target RGB. Its full-scene rendering, fixed width 480,
fixed 0.5 threshold and verbatim query policy form a development contract,
not the historical selected-only or official 3D contract. It reports full-label
IoU and threshold-free AP/AUROC with tied scores grouped. A higher pseudo-label
IoU or fixed-threshold real-label IoU does not by itself justify promotion when
ranking quality or another query modality degrades. Fixed-vocabulary top-1
source labels remain unverified pseudo labels even at high training accuracy.
The tested native matching-relation, fixed SVD query-basis, soft-text-only and
capacity/exposure controls are optional and have not passed joint promotion.

2026-09-15 pipeline correctness gate: new retrieval-field training rejects
implicit legacy crop-top1 hard-text supervision and obsolete grounded-text
schemas. Grounded schema v2 validates axes, disjoint signs, train-only tasks,
and same-view positive-detection negative policy; legacy controls require an
explicit flag. Query/element contradiction abstention, signed-input protection,
and complete independent source-region teacher coverage are defaults. Their
retention does not depend on held-out metric gains.

Direct native-space alignment and per-modality scalar readout calibration are
explicit experimental options, not promoted defaults. Calibrated ordinary and
transient signed-query retrieval share one readout implementation; transient
optimization must not update calibration or field parameters. See
`docs/experiments/2026-09-15-pipeline-correctness-and-space-alignment.md` for
passed contracts, failed quality gates, and remaining method limitations.

2026-09-16 experimental native decoder coupling: an optional deployment readout
computes cosine(D z + b, native query) using exactly the SigLIP reconstruction
head. D and b are global deployed model parameters, not a second per-element
teacher feature cache. This explicitly extends the earlier training-only-head
interpretation for this option. Prompt retrieval keeps its separate adapter.
The algebraic equality is verified; mixed LERF results do not justify default
promotion. ScanNet uses prescribed raw-ID categorical point readout; NVOS uses
reference 2D signed prompts with target access governed by its sealed contract.
New ScanNet/NVOS field evaluations remain pending. See
`docs/experiments/2026-09-16-decoder-coupling-and-joint-query-contracts.md`.
