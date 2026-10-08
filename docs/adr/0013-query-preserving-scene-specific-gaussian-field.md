# Query-preserving scene-specific Gaussian field

Status: accepted research direction by the user on 2026-10-07; no accuracy promotion.

2026-10-08 refinement: [ADR0014](0014-v10-capability-transfer-within-one-gaussian-field.md)
supersedes overly restrictive first-implementation choices. One field and a common
conditional relation core allow necessary evidence capacity and typed decisions;
mean native vectors and a single scalar member score are not architectural goals.

The active research branch is `codex/query-preserving-gaussian-field`, in the
independent worktree `/root/RADIO-GS-query-field`. Commit
`0f40545c2b93fa390e37038af8315c24b5bc57b5` preserves the complete nonignored v10
working state, including previously uncommitted files. The original worktree and
its index were not changed. External data and ignored outputs remain separately
versioned assets; the source snapshot is not a claim of a fully portable baseline.

## Decision and conflicts with earlier decisions

The target is one scene-specific Gaussian capability field, with native SigLIP
semantic coordinates, native DINO local appearance, and SAM-supervised
scale-conditioned grouping. These are owned subspaces on the same Gaussian
rows, not benchmark-specific maps. A native decoder and its query rule must be
paired, including bias and decoded norms. Initially retain sufficient native
capacity; measure compression separately after the member mechanism works.

This explicitly replaces the RADIO-required reconstruction contract of ADR0002
and older Canonical Capability Feature implementations for the new candidate.
The project name does not impose RADIO as an intermediate teacher. ADR0008's
frozen geometry, ADR0009's protected writes and ADR0011's distinction between
semantic response and actual subset rendering remain in force.

ADR0012 remains the description of baseline A (v10). Sharing observation utility
functions is no longer sufficient for the new candidate's unification gate:
text, image and registered prompts must consume a common member operator.
Category competition and the final rendering operator can remain typed. The
full-cohort category reader compares common text-union member scores over the
complete menu; native semantic argmax remains a separate control. Categorical
benefit must be measured and must not be claimed from this interface alone.

## Data and deployment boundaries

Construction uses only manifest-authorized mapping RGB/cameras and native
DINO/SAM/SigLIP. Additional 3D instance/NR3D labels are a separate condition,
not an implicit part of this first implementation. A scene-specific fit never
authorizes use of evaluation labels or final query answers.

The standard scene package contains geometry, capability codes, grouping
parameters and any required affine codecs. Runtime accepts this package and
the current authorized input. Source photos, masks and tracking are construction
artifacts, not implicit query resources. Input encoders are declared shared
dependencies and their storage and latency are additional to per-scene cost.
Explicitly permitted source-assisted NVOS and other extensions must enter the
same membership core and have separate quality/cost reports. Existing v10
source-assisted tasks are preserved as comparators, not automatically approved
new standard-mode dependencies.

Estimated identities and 2D ray correspondences are not certain positive
Gaussian labels. Only explicitly registered positive elements can be clamped.
Alternative identity/scale evidence should be retained before selection.
The first implementation exposes alternatives and returns a coherent MAP
hypothesis; relational text binding and calibrated identity uncertainty are
still pending. The first image rule retains query patches but does not claim
to reproduce v10's full bidirectional source-patch matching.

## First bounded experiment

`configs/query_field/figurines_pilot.json` fixes six feature-training and two
development mapping frames, a 96x128 projection and 200 grouping updates before
new results are read. Geometry is the actual v10 Gaussian carrier. Geometry
was fitted on a larger mapping set, so this is a feature-view holdout, not
unseen-view geometry evaluation or an independent final test.

Recompile source assignments on this carrier. Do not reuse earlier 274805-row
memberships on the current 554467-row geometry. Cache receipts bind source
image, native region cache, SAM checkpoint/cache, DINO checkpoint, geometry,
camera report and configuration. Development features never initialize the map.

The control keeps the same native semantic/local channels with an untrained
group projection. The candidate fits only group codes and a scale gate with
actual Gaussian-rendered partial mask losses. Disjoint, comparable-scale
source masks can supply exclusions; nested masks and unobserved areas remain
unknown. These are teacher grouping targets, not verified whole-instance truth.

## Progression

1. Freeze A and its dependencies; record actual saved metric contracts.
2. Verify the Gaussian compiler, exact query algebra, and protected native field.
3. Establish conditional grouping/member competence on fixed development views.
4. Align the four query adapters and identity/scale inference; compare high-capacity B.
5. Complete all five development cohorts, then isolate the measured v10 gaps.
6. Compress under matched inputs, supervision and readout after method quality is
   understood; lock independent final evaluation separately.

The user's 2026-10-08 steering puts full-cohort execution before further local
method or engineering optimization. This supersedes the former ordering of
compression before expansion and any historical local gate that would prevent
running another benchmark. Earlier rejected variants remain rejected; expansion
uses the same accepted physical field. It does not select a different model per
scene or input. See the [full-cohort protocol](../experiments/2026-10-08-five-benchmark-architecture-protocol.md).

Each transition needs measured evidence. A failed bounded pilot remains a
diagnostic, not a promoted replacement or a reason for unrestricted parameter
search. Proposed one-percentage-point quality and fourfold semantic-increment
compression goals are planning values until the comparable B baseline is fixed.
Performance retention must also be measured against A; an A-to-B contract loss
cannot be hidden by good B-to-compact retention. No top-conference novelty or
accuracy claim follows from the package structure alone.

## First implementation evidence, 2026-10-07

The fixed 200-step pilot and one 2,000-step optimization-budget control are
complete. A third, matched 2,000-step control transfers the established
next67/68 distinction into the shared member decoder: fit a relaxed selected
subset's alpha through gsplat, allowing every observed Gaussian to receive a
gradient even if it was hidden under full-scene transmittance. Base geometry
and opacity remain frozen. No source assets are introduced at query time.

This improves actual subset diagnostics in all three evaluated input conditions
over the matched visible-membership objective; normalized membership-image
metrics can regress. Keep both output types explicit. This is one-scene
development evidence with SAM region targets, not a five-task promotion.
See the [complete experiment ledger](../experiments/2026-10-07-query-preserving-gaussian-field-bootstrap.md).

## Two-scene evidence and retained boundary

The [next ledger](../experiments/2026-10-07-query-field-two-scene-and-readout.md)
adds fixed source-budget controls and actual LERF scoring. The current shared
member decoder fails to retain the exported field's native text selection
capability on both scenes. A native-only bypass is retained as a diagnostic;
it does not meet the shared-member decision above and is not a new default.
Likewise, single-anchor selection is not an equivalent solution to genuine
category unions. No additional teacher, source-query fallback or label-trained
member component is introduced to disguise this gap.

Keep geometry and native coordinates protected. The next bounded mechanism
study should compare reliable registered and inferred native support under the
same source budget and operator, measuring both identity and actual extent.
Compression awaits a stronger matched high-capacity reference. This is a
refinement of the implementation priorities, not a relaxation of the accepted
architecture or a claim that the full research direction has failed.

## Training-supported inferred inputs

The [three-scene support control](../experiments/2026-10-07-training-supported-identity.md)
identifies a concrete input-domain mismatch: original native peaks frequently
have negligible training exposure and are not eligible member-training supports.
Allow a frozen train-only eligibility mask for inferred anchors. This condition
is derived from the original reliable SAM interiors/exclusions, never benchmark
queries or development masks. The observed output carrier remains complete;
explicit registered inputs remain unchanged, including inputs outside this mask.

The fixed local gate passes on figurines, ramen and a newly constructed teatime
field without changing native, group or geometry tensors. The third scene is
historically opened development evidence, not independent final testing. Keep
this as an opt-in supported-input policy. It does not establish full identity
correctness, complete member extent, compactness or five-benchmark replacement.

## Physical scale in the common member energy

The [fixed spatial-energy comparison](../experiments/2026-10-07-physical-scale-member-results.md)
finds that scale-gating feature coordinates alone leaves membership indifferent
to distance between equal group codes. Permit an opt-in Gaussian log prior on
distance to the weighted input support centre, normalized by physical query
scale. Construction world units must be explicit in the export. This modifies
the common member energy for all four inputs and negative supports; it does not
add another query branch, another teacher, source access or label supervision.

The coefficient and thresholds were fixed before results. A frozen post-fit
prior fails the image gate in one scene. Fitting the same member energy through
the existing source subset-alpha objective passes the text/image/point mean
gates in both initial scenes and unchanged third-scene confirmation. Use that
refitted arm as the current opt-in research candidate, preserving old exports
and v10 defaults. This demonstrates useful locality and training consistency;
it is not calibrated identity, object completeness, a novel prior by itself,
or proof of five-task/v10 equivalence. Native and actual member readouts must
still be reported separately, including observed semantic regressions.

## Protected native locality and full LERF development coverage

The [next fixed study](../experiments/2026-10-07-native-local-member-results.md)
tests native DINO distance inside the same member energy, using the already
stored native coordinates and unchanged source supervision. Both frozen and
refitted controls improve text subsets, but neither passes the declared common
input gate. In particular, the refitted ramen 3D teacher-region mean regresses
by 0.5225 points despite image/2D gains. Keep this metric as an explicit research
option; the physical metric remains the common candidate. Native coordinates
being immutable does not guarantee native object completeness after readout.
No per-modality mixture or stronger-looking text-only promotion is authorized
by this evidence.

The independent fixed physical run on waldo_kitchen completes the four historical
LERF development scenes. Its equal-scene actual subset mean is 32.3130%, versus
saved v10 66.9633% under different source-access/resources. This is full coverage
of the declared local LERF cohort, not completion of ScanNet OVS, NVOS, UQIS,
independent final testing or the desired accuracy target. Preserve these cohort
and contract boundaries in future claims and compare every new common method
without selecting different variants per scene or input.

## Real signed raster prompts

The [fern development probe](../experiments/2026-10-07-native-signed-prompt-results.md)
uses raw positive/negative scribbles as transient registered ray evidence for the
existing common member energy. Permit normalized multi-pixel support pooling,
explicit missing-evidence accounting and a separate positive-extent scale
estimate. No source tracker, query mask generator, extra 3D labels or hard 2D
element clamp enters deployment. Numerical pooling must replay deterministically.

Eight fixed arms pass copied-field source-denied replay. Source fitting improves
the retained response IoU but regresses actual selected-set IoU and AP on this
one availability-selected case. The extent estimate stays at the default and
provides no gain. Keep the common physical candidate; do not choose an untrained
model for fern or promote a new scale policy from this evidence. Further changes
must address signed support learning and false additions through the shared core,
with complete per-input/readout gates and honest full-cohort denominators.

## Shared support training and retained aggregation question

The [matched support study](../experiments/2026-10-07-support-training-results.md)
tests unsigned and signed ray mixtures in source fitting. Factor observed-support
filtering and signed exclusion into differentiable common helpers used by both
deployment and fitting. New support policies remain opt-in source experiments;
default single-support fitting and the physical member energy stay unchanged.
Neither mixed policy passes the common gate, so neither replaces the candidate.

The negative result does not authorize per-scene checkpoint selection or more
support-mixture tuning. Distinguish uncertainty over Gaussian associations within
one ray from multiple co-target observations and independent negative constraints.
Their current unit-normalized pooled prototype drops directional disagreement;
source-only concentration measurements establish that information loss, not a
causal accuracy explanation or a successful replacement. Any new aggregation
must remain over the same conditional member kernel with declared typed inputs.

Exact deployed-query replay does not imply bitwise training reproducibility:
two repeated executions of the original fitter differ in learned tensors.
Report numerical repeatability separately and do not treat small one-run metric
differences as statistically established architectural gains.

## Expected energy is not a resolved uncertainty model

The [next completed controls](../experiments/2026-10-07-expected-support-results.md)
retain uncertain support moments and separate negative-ray identities in one
shared member operator. The energy expectation is algebraically exact, without
new field tensors. Paired source training preserves singleton presentation count
while doubling rendered losses, with a matched legacy-aggregation control.
Neither fitted arm passes the common gate; both remain explicit experiments.

The loss of resultant length in the old prototype did not establish that its
length measures confidence about complete object membership. Expected energy
reduces 2D capability, and independent negative maxima erase most authorized
positive scribble coverage. Their monotonicity/idempotence does not make them a
valid signed-evidence model. Do not promote positive-only gains, higher soft
semantic scores with worse actual sets, or a response-mixture substitute based
on uncalibrated source statistics.

Further aggregation changes must distinguish association uncertainty from
object-member variation, preserve positive/negative compatibility and check
evidence-density/resampling effects using authorized prompts. The current
physical field remains the common candidate. No query-time source solver,
per-modality fallback, new teacher or 3D annotation is authorized by this result.
Cross-GPU hard-subset rendering has an observed <=5.9605e-8 numerical difference
despite exact memberships/selections; report that separately from exact
same-device replay and previously observed non-bitwise training.

## Relative signed evidence and remaining identity/extent separation

The [frozen competition study](../experiments/2026-10-07-competitive-evidence-results.md)
compares positive, negative and null evidence through the unchanged common
member logits. All negative rays can contribute by observed-mass-normalized
log mixtures, retaining the accepted per-ray prototype. No field tensor or
inference source right changes. Old exports and defaults keep product exclusion.

The ray control preserves prompt coverage and improves all three fern public
readouts slightly, but misses one fixed teacher-subset gate. The pooled control
also lowers fern AP/subset. Neither becomes the common candidate; no second
NVOS task or post-result coefficient sweep follows. Exact unsigned four-scene
LERF and six four-input copied-field replays do not substitute for accuracy.

The source-only post-study audit separates visible target-associated anchors,
visible off-target anchors and unknown invisible anchors. Both association
errors and member-extent errors remain. Most false additions are outside known
partial exclusions, including cases with target-associated anchors. This does
not authorize labelling unknown pixels as background. Further common-method
work should test identity-conditioned scope and source-view consistency, keeping
protected native evidence, one member operator and the full per-input/readout
gates. A stronger negative combiner alone is not a supported solution to the
remaining v10 gap. These are research hypotheses, not new validated capabilities.

## Complete development cohort, 2026-10-08

The [full results](../experiments/2026-10-08-five-benchmark-results.md) now cover
all five families: four LERF scenes with two readouts, eight NVOS cases, ten
ScanNet rooms, and nine UQIS rooms with 67 targets and 268 independent inputs.
Twenty-five new source-only fields join the four retained LERF fields and fern;
horns left/centre share a field. The common physical architecture is unchanged.
All new predictions precede new scoring, including unsupported empty outputs.

Explicit metric camera compilation and the two new benchmark adapters are
integration changes. They introduce neither labels into scene fitting nor a
new member decoder. Official trex scribbles require lossless collapse of three
identical RGB channels; the failed attempt and correction receipt are retained.
ScanNet/UQIS native/source caches and geometry are declared construction inputs,
not runtime source dependencies or old fitted member-model reuse.

Full execution is complete; quality remains substantially behind saved v10.
The ScanNet member/category result is below the native control in every room,
NVOS reveals large differences between ranking and thresholded response, and
UQIS exposes unsupported spatial prompts alongside supported-query errors.
First measure observation coverage, then separate native fusion, query identity,
member extent and typed readout losses. Do not compress, add per-benchmark
fallbacks or resume local sweeps before that diagnosis. This system comparison
has unequal source budgets and teachers; it is not an isolated architectural
effect, an independent final test or evidence against the research direction.

## Architectural attribution after full coverage

The [complete diagnostic study](../experiments/2026-10-08-architecture-attribution-results.md)
replays the accepted outputs and separates observation coverage, inferred anchor
identity, conditional member extent and typed readout. Its candidate oracles,
coverage ceilings and analytic scope bounds are evaluator-only evidence.

Correct 3D anchors still yield weak extent; the hard spatial envelope excludes
little of the supported-query target mass. NVOS negative removal barely changes
the failures, while LERF overselects and NVOS underselects. Native field direction
retention coexists with poor source-query ranking retention. ScanNet category
competition loses to the same field's native control in every room. A single
threshold, radius or negative-evidence repair is not supported by these results.

The next research design must preserve native query evidence beyond anchor
selection, learn identity-compatible scope from authorized source supervision,
and give typed category competition a defined semantic evidence interpretation.
One carrier and one shared member core remain requirements; they do not require
discarding query evidence or comparing uncalibrated instance scores across
classes. No particular new energy, learned head or scale policy is promoted yet.

Source query-retention and known-identity member controls precede combined
method changes. Match source budgets when making causal claims, record episode
exposure, and preserve complete-cohort reporting. The current high-capacity
storage prototype is not proof of a solved high-capacity query-preserving model.
