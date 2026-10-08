# RADIO-GS — active system context

Latest completed study: `docs/experiments/2026-10-08-joint-relation-results.md`
and JSON; runner `scripts/research/run_query_field_joint_elements.py`, root
`joint_element_relations_v1`. All30 exports, five benchmark scores,268 UQIS
inputs,67-point diagnosis, exact native replay and five copied-export cold
checks are complete. No pending jobs or model promotion. Frozen construction
7d9fcc8, complete tools7df5335;12 focused tests pass, including actual CUDA
rendered gradients outside element samples. Prior failure study is f364d25.

Joint .5 element +.25 primary/.25 secondary subset-rendered losses retain the
same original parents, full source traces/samples/two epochs and export capacity.
Against conditional-only: LERF2D16.9454→33.2817%, actual projection9.4783→13.0840,
NVOS actual-set71.9338→76.3384, UQIS2D26.8342→28.8602,3D26.2318→28.4509.
NVOS response falls74.6098→70.9966 but exceeds protected57.9518 (7/8 improve).
LERF still regresses in all4 scenes versus protected52.2864/32.8763; precision
is13.7906 versus34.2303%, recall90.4444. ScanNet19/15/10=37.9527/38.7972/45.6303;
UQIStext/image=13.2306/20.8428. No per-task winner bundle or v10 recovery claim.

Known-correct59-point precision44.4343→50.6241%, still below protected58.3788;
observed recall63.3705, actualIoU33.0234 (query means). Joint source element
BCE=.344084 (29/30 improve), rendered BCE=.545022 vs parent.638630 (28/30).
Actual weighted rendered-logit gradient coverage94.5246% observed rows versus
36.8539% directly sampled; coverage is not adequate optimization or causality.
Exposure-proxy recall gap18.5463pp remains;30 default scales lie within source
10–90% quantiles. Rendering is connected and partly repairs acceptance drift,
but incomplete pair evidence and train/query-condition mismatch remain open.
Next method work must separate known-support calibration, uncertain/union query
composition and native text/crop functionality; preserve unknowns and validate
all inputs. Do not infer a unique cause from these controls, merely tune global
thresholds, or resurrect labelled v10 solvers. Missing same-decoder render-only
control prevents an isolated synergy claim. Source feature permutations were
not rerun for the joint arm. Full results include paired-scene uncertainty.
The new member core reads native descriptors even for registered points. Earlier
native-update/point-output invariance only applied to the legacy member core;
future native refinement must revalidate point/signed queries and all five tasks.

Latest completed study: `docs/experiments/2026-10-08-element-relation-results.md`
and JSON. Both new arms have all60 exports, five-family scores,268 UQIS inputs
per arm,30 source feature-binding audits, all67 point attribution cases, and10
copied-export cold replays complete. Root `element_relations_v1`, construction
code4bdf77a. Both keep exact protected native/geometry/support/shared parameters;
full paired source traces and element samples match.70 related tests pass.

Direct element training barely changes the protected rendered control. Adding
the shared native/geometry residual improves source BCE .628311→.286285 (30/30)
and NVOS response57.9518→74.6098%, but LERF2D52.2864→16.9454 and actual-set
projection32.8763→9.4783 collapse (all4 scenes). LERF actual precision falls
34.2303→9.6658%, recall rises87.8739→91.5271%. NVOS actual-set IoU improves only
69.9469→71.9338%, much less than response. ScanNet19=37.9904%; UQIS text/image/
2D/3D=12.9180/20.8970/26.8342/26.2318%. No arm promotion or task winner bundle.

Known-correct59-point diagnostic: observed recall54.8275→67.8470%, precision
58.3788→44.4343%, IoU32.5077→30.4461 (query means). Within-query exposure gap
20.8757→15.5289pp, still not actual gradient evidence. Source native/geometry
binding permutations worsen BCE, but scalar exposure contributes almost nothing.
Direct sampled-row coverage averages36.8539% (LERF7.5087%,NVOS5.9546%); region
coverage is not complete Gaussian optimization. Scalar exposure is not pair
co-visibility. Total new payload42,312,688 bytes: one scalar/element +1441 weights.

The matched joint repair is now complete above. The element-only residual study
did not retain new rendered loss during its two epochs, but neither its failures
nor the partial joint recovery prove loss omission is the sole cause. Preserve
unknown/conflicting evidence, all-five evaluation and source-free inference;
do not select per-task winners.
Older frozen studies require their recorded code snapshots after changes.

Latest completed studies: `docs/experiments/2026-10-08-native-coordinate-results.md`
and `2026-10-08-protected-native-results.md`, with full JSON evidence. Four full
arms, each all30 fields and all five benchmarks including268 UQIS inputs, are
complete:120 exports total. Roots `output/query_field_20261008/native_coordinates_v2`
and `protected_relations_v1`. The v2 query program is fixed across arms. Main
construction revisions `ea23689`, `d67813c`, `083acc1`; no default promotion,
per-scene/task winner selection, or complete v10 recovery is established.

Decoder rollback restores LERF actual-set projection29.5153→33.7203%, mainly
figurines25.7584→43.8528. Training-time shared-parameter protection gives32.8763%
and is worse than posthoc rollback; it is not a universal recovery. Compared
with previous common program, protected LERF2D54.1849→52.2864, UQIS image
21.9248→20.4974,3D point28.4507→28.0598. All30 relation traces match the previous
joint fit; geometry/native/support buffers and protected shared parameters are
exact. No default promotion or per-task model selection.

Source-only native functional construction is now implemented. Shared full-native
affine residuals fit teacher query preferences and bake into the same field.
25,451 source regions visited twice; source-development semantic KL improves
.110428→.102544 (26/30 scenes), local .177713→.173917 (24/30), but transfer is
inconsistent: UQIS text12.5155→13.8000, image20.4974→20.4352, LERF2D52.2864→50.5804
(all four scenes decline). Source pseudoquery fitting is not proof of real query
retention. Shared maps cannot separately correct spatially different mixtures.
All134 registered UQIS point outputs and8 NVOS outputs remain bit-exact across
native-only refinement. The two native-protected controls each replay all268
native score vectors exactly; three copied real exports replay15 cold input/
readout cases with no forbidden opens (Python audit, not OS sandbox; one scene).

Full-UQIS post-score visibility diagnosis:59/67 3D inputs have correct supports,
yet protected observed-target recall is only54.8275% (query mean). In57 queries
with both exposure strata, high-contribution portions exceed low-contribution
recall by20.8757pp, clustered interval[12.337,27.994],8/9 scenes same direction.
This is association, not a measured training gradient or causal proof. Artifacts
under `protected_relations_v1/diagnostics/point_observation_mass`. Do not hard-label
weak/unknown observations or infer that a scalar threshold fixes membership.

Direct full-native per-Gaussian coordinate construction is complete, protocol
`docs/experiments/2026-10-08-native-coordinate-protocol.md`, runner
`scripts/research/run_query_field_coordinates.py`, construction code `083acc1`.
Same source queries, two region epochs and v2 inference; row-local Adam .001,
full FP32 construction states, original output dtype/dimensions. All30 source
recipes and export capacities checked;50,902 steps,60 development views/2,788
regions. CPU arithmetic trial v1 stopped before benchmark prediction after five
fields; attempts retained; all30 restarted uniformly in v2 after CUDA parity.

Relative to protected training, coordinates improve LERF actual projection
32.8763→34.8311% (3/4 scenes; paired interval[-1.1248,5.1302]pp), ScanNet10
45.4731→46.3277% (9/10;[.3047,1.5538]pp), but LERF2D52.2864→50.5530 and UQIS
image20.4974→19.4973 decline. ScanNet19=37.8615,text=12.7697. Source semantic
KL=.099348 (29/30 improve), local=.174727 (26/30), yet true query transfer remains
inconsistent. Observed row exposure averages86.3243%, range63.0192–97.3935;
region coverage does not prove full Gaussian optimization. Optimizer and
parameterization both differ from shared residuals; this is not a pure capacity
causal estimate or a sufficient one-vector capacity upper bound. All134 UQIS
registered outputs plus8 NVOS outputs are exact vs protected. One copied real
export replays all five input/readout cases without forbidden opens. Combined
four-arm cold checks cover20 cases, one physical scene.53 related tests pass.
All construction, prediction, scoring, source matching, output checks, summary
and plotting are complete; there is no pending experiment from this batch.

The initial native/geometry conditional member and joint-supervision studies
above are complete. Genuine pair visibility/disagreement, acceptance under
typed query composition and real text/crop-patch functional retention remain
unresolved. Do not infer recovery from source BCE/KL or ranking AP alone. Retain
one field, three teacher roles and standalone inference; no labelled-head reuse,
source-bank runtime fallback, compression detour, or per-task solver revival.

Earlier relation/common-program study remains reproducible at `dfd107b`; see
`2026-10-08-cross-view-relation-results.md`. Preserve the one Gaussian field,
SigLIP/DINO/SAM roles, common conditional membership, source-free runtime and
actual-set verification. No old solver, labelled-v10-head distillation, or
per-scene/query threshold tuning is part of the current track.

Latest direction, 2026-10-08: the user explicitly authorizes adapting the ideal
architecture to recover v10 capabilities while retaining its core. Read
`docs/adr/0014-v10-capability-transfer-within-one-gaussian-field.md` and
`docs/method/v10-capability-transfer.md`. Shared Gaussian state and conditional
relations do not require identical scalar task outputs. Necessary local modes,
reliability and small typed readouts are permitted; their value must be tested.
Standard inference remains standalone, with three teacher roles and no hidden
v10 solver or labelled-member distillation. Full-native averaged descriptors
are not automatically an adequate high-capacity capability reference.

First implementation complete: `query_field/identity_evidence.py` transfers
v10-compatible ordinal native fusion and weighted bidirectional matching, with
exact old-operator tests. The fixed67-image/four-control study is complete:
`docs/experiments/2026-10-08-native-identity-transfer-results.md` and JSON.
Parent image IoU/AP15.9861/24.7530%; same-eight rank15.8670/25.1978%; same-eight
Gaussian-set bidirectional17.6289/28.5899%; global eligible-anchor rank18.3874/28.2145%.
Correct anchors21/21/25/26 of67. Both stronger controls improve8/9 scenes, but
their paired-scene IoU intervals cross zero. No promotion or per-scene selection.
All predictions preceded scoring;67 parent tensors/candidates/sets replay exactly.
23 tests and a copied-field real-query four-control source-denied replay pass.
This is a research intervention, not a changed default or a new five-task result.
The completed five-task baseline and attribution below remain the reference.
Next construction work must preserve functional responses/correspondence and
learn visibility-aware conditional relations; do not treat reranking as sufficient
or sweep fusion weights/candidate counts. Existing fixed-eight oracle22.13% and
correct-point extent failures already require work beyond these query controls.

Updated 2026-10-08 on `codex/query-preserving-gaussian-field`: the user authorized
the scene-specific, three-teacher, query-preserving compact Gaussian research
direction. New entry: `python -m radio_gs.query_field`. Read
`docs/adr/0013-query-preserving-scene-specific-gaussian-field.md` and
`docs/method/query-preserving-gaussian-field.md` before developing this branch.
The v10 state below is baseline A, not the new architecture or a promoted student.
Source snapshot: `0f40545c2b93fa390e37038af8315c24b5bc57b5`. Original dirty working
tree and index are preserved in `/root/RADIO-GS`; development is isolated in
`/root/RADIO-GS-query-field`. Local candidate gains are measured below; no
five-task promotion or v10 replacement is established.

**Query-Preserving Gaussian Field Candidate**: one scene-specific Gaussian
capability field with protected semantic/local subspaces and a shared
scale-conditioned membership operator. Typed inputs carry uncertain identity
evidence or authorized spatial support; native matching remains paired with
its decoder. This refines the Canonical Capability Feature vocabulary without
requiring RADIO reconstruction or an identical scalar output across tasks.

Current priority (2026-10-08): architecture-level attribution and evidence-led
method design after the complete five-task baseline. The full diagnostic run is
complete: `docs/experiments/2026-10-08-architecture-attribution-results.md` and
its JSON, under `output/query_field_20261008/architecture_attribution_v1`.
All30 fields,60 feature-held-out source views/2910 partial regions,208 LERF
query-views,10 ScanNet rooms,8 NVOS cases and268 UQIS queries are included.
No runtime method, threshold or baseline prediction changed.

Measured findings: UQIS target observation coverage87.43%; most target misses
are on observed elements. Text/image top anchors hit20/67 and21/67 targets;
best-GT choice among the same8 candidates only raises actual IoU12.08/15.99
to16.45/22.13%. All59 supported3D anchors are target-correct, yet conditional
actual IoU31.73%; only1.16% target mass is outside the fixed analytic envelope,
while41.50% is observed, inside and still missed (conditional scene macro).
Two of61 available2D prompts lose all target support after observed filtering.
NVOS removing negatives changes macro IoU by only+0.0115pp; flower/orchids
remain severe underselection. LERF actual precision/recall33.81/87.60% shows
the opposite error direction. ScanNet member competition loses14.66pp relative
to native19 on the same field;57.77% observed valid points have every category
member score<=0.5. Source-partial-query AP after fusion/projection falls from
95.67 to49.14% in ScanNet and98.99 to46.78% in UQIS, despite high semantic cosine.

Interpretation: protected native storage is not functional query preservation;
identity evidence is discarded too early; correct-support member extent remains
weak; independent instance affinity is not calibrated category evidence.
Hard geometry extent or the signed negative rule alone does not explain the
failures. Soft scale/group/training effects remain coupled. Different source
budgets/teachers/3D supervision prevent attributing the entire v10 gap causally.
Next: matched observation/query-retention controls and known-identity member
learning, then evidence-preserving identity and typed category competition.
Preserve one Gaussian carrier, three native teacher roles and one conditional
member core; no source-query solver, labelled member-head reuse, threshold sweep
or compression work. Proposed joint identity/scope designs are not validated
replacements. Do not restart the rejected aggregation-only studies.

Completed prior priority (2026-10-08, explicit user steering): complete one frozen
architecture baseline on all five benchmarks BEFORE further local optimization
or staged gap diagnosis. Earlier local experimental promotion gates must not
delay full-cohort integration. No new mixture/negative-rule/compression/speed
study is authorized as a substitute for this coverage. The common physical
field at fb3c768 is fixed for this run. Full cohort: four LERF scenes (two
retained readouts), eight NVOS tasks, ten ScanNet rooms, nine UQIS rooms with
all67 targets/268 four-modal queries. Existing LERF and fern fields are reused;
all 25 missing fields are now built, with horns left/centre sharing one field.
New fields use18 uniformly selected legal training views and2 development views,
same2000-step physical recipe. Source budget and geometry differences from v10
remain explicit. ScanNet/UQIS use declared unlabelled-mesh surface Gaussians,
not the old labelled member model. First categorical member baseline is common
text-union membership followed by complete-menu argmax, with native argmax as
a separate control. All UQIS inputs call the same instance member core.
Protocol: `docs/experiments/2026-10-08-five-benchmark-architecture-protocol.md`.
Configuration: `configs/query_field/five_benchmark_v1/cohort.json`.
Run root: `output/query_field_20261007/five_benchmark_v1`. Status: all five
cohorts predicted, hash-verified and scored; all new predictions preceded new
scoring. Results: `docs/experiments/2026-10-08-five-benchmark-results.md` and
its JSON evidence. Candidate/v10 (%): LERF2D IoU49.4130/70.4007; local LERF3D
actual subset32.3130/66.9633; ScanNet19/15/10 member mIoU22.2853/23.3694/26.5462
versus61.0115/62.5276/72.4861; NVOS signed response IoU57.6865/86.5134,
AP94.8988/98.0222 and separately actual subset70.5360. UQIS text/image/2D/3D
AP21.3937/24.7530/47.9527/46.7809, actual set IoU12.0755/15.9861/28.3732/27.6913;
all-four IoU>=0.5 scene-macro success6.9444%. All268 queries remain scored,
including6 unsupported2D and8 unsupported3D prompts. No accuracy promotion.
ScanNet native19 control36.9452% exceeds the member result in every room;
this localizes a measurable additional loss to the member/category readout
path, without explaining the native-to-v10 loss. NVOS flower/orchids response
IoU7.7530/9.2527% despite AP91.5207/93.0105% requires separating ranked evidence,
member extent and readout calibration. Do not react by fitting target thresholds.
Next: quantify source/target observation coverage before native fusion, query
identity, member extent and final transport/readout interventions, one stage
at a time with this complete baseline retained. No compression/speed work yet.
The only post-freeze integration correction accepts exact replicated RGB
scribble channels in official trex; failed logs and correction hashes are saved.
22 initial integration tests and8 post-fix signed-raster tests pass; fern's four
prediction tensors replay exactly. Original worktree/head/index remain unchanged.

Historical local study (2026-10-07; superseded work order, unchanged results):
`docs/experiments/2026-10-07-competitive-evidence-results.md`.
The frozen relative signed-evidence study is complete: 6 exports, 351 teacher
regions x3 rules, 6 real fern prompt predictions/scores, 6 four-input cold
replays, 6 same-GPU NVOS cold replays and 68 LERF queries x3 exact unsigned audits.
63 tests pass. No fitting, scene-tensor growth or parameter search occurred.
Ray-mixture signed competition gives fern response IoU/AP/actual subset
76.7569/92.1666/76.4433% versus parent76.7399/92.0952/76.1453%. Both competitive
rules preserve99.9396% positive prompt coverage, but ray competition loses0.0242pp
figurines signed-teacher actual subset; pooled also loses ramen, fern AP/subset.
Neither passes the fixed gate, so keep the physical parent/product default.
The historical block on a second NVOS scene is superseded by the completed
full-cohort run; it still does not justify tuning strengths/thresholds. These small
differences are not statistical significance claims. Current accepted LERF
four-scene32.3130% and fern signed76.7399% remain unchanged.
Post-study source-only audit exactly replays all original image/2D/3D IoUs.
Image anchors have majority visible target mass in77/169 figurines,102/149 ramen,
31/33 fern cases;53/26/2 are not visible and cannot be called incorrect identity.
Even target-supported image anchors yield only42.6501/39.8360/44.2962% actual IoU.
Only11.74/16.15/1.83% of all image false-added pixels overlap current known
source exclusions; unknown pixels must not become automatic negatives.
Next: retain native identity evidence beyond anchor selection while learning
scope in the same member operator; test source-view scope consistency. This is
a hypothesis, not an accepted replacement. Do not continue aggregation-only
sweeps or add query-time source memory, per-input decoders or extra labels.
Evidence: `docs/experiments/2026-10-07-competitive-evidence-results.json`.

Previous candidate evidence (2026-10-07):
`docs/experiments/2026-10-07-expected-support-results.md`.
The expected singleton-energy / independent-negative-ray study is complete:
6 paired fits, 3 frozen controls, 9 teacher evaluations, 6 LERF and 6 NVOS scores.
Neither fitted arm passes; keep the physical parent and single-support defaults.
Actual LERF subset IoU (%), paired legacy / expected: figurines38.4655/25.3824
versus parent41.7884; ramen24.5118/19.1874 versus19.4642. Expected figurines
member-response IoU rises49.1988->56.4633 while actual set precision collapses.
Expected 2D teacher means regress on figurines/ramen; some image/3D means improve.
Fern expected positive response IoU77.4965 improves but AP91.4135 falls; signed
response IoU0.2478 and actual subset6.4326 collapse. Signed positive-prompt
coverage falls99.9478->8.8609% under independent negative maxima. Never select
the positive-only arm to hide this failure. All original thresholds/data remain.
Both fits retain2000 singleton presentations with4000 rendered losses, versus
2000 in the parent, at half loss weight each. Matched old/new full traces and
initializations agree. Expected conditions cost6660/6662/6664 versus4666 legacy;
no new scene tensors or learned parameters. No teatime/waldo follow-up is triggered.
46 tests and9 four-input copied-field replays pass. All6 NVOS cases replay exactly
on the original GPU. A cross-GPU actual-alpha comparison differs at22 pixels by
<=5.9605e-8; members, selected rows, retained response and thresholded mask agree.
The failed replay is retained; no universal cross-device bitwise claim.
Source-only32-episode audits show sigmoid(mean energy) differs from mean sigmoid;
the latter also fails to restore original source-interior response in all3 scenes.
Discarded support norm was not a calibrated confidence estimate. Neither an
energy average nor independent negative vetoes is a validated uncertainty model.
Next: compatible positive/negative association evidence and prompt consistency,
with evidence-density/resampling checks before target scoring. Do not repeat
unsupported threshold/mixture or hard-forward sweeps; historical next49/67
already bound those shortcuts. Current four-scene LERF32.3130% and fern signed
76.7399% remain unchanged; full five-task accuracy and compactness remain open.
Evidence: `docs/experiments/2026-10-07-expected-support-evidence.json`.

Previous candidate evidence (2026-10-07):
`docs/experiments/2026-10-07-support-training-results.md`.
Two fixed source-support training controls are complete on figurines/ramen/fern.
They share deployed observed-support filtering and signed member multiplication;
inference energy, field tensors, source data and 2000 update budgets stay fixed.
Mixed unsigned/signed actual LERF subset IoU: figurines33.5579/34.3990% versus
parent41.7884%; ramen22.1834/20.7247% versus19.4642%. Real fern signed response
IoU76.2658/76.2887%, AP91.0246/91.3218%, actual subset75.2162/75.2016% all regress
from the parent. Neither passes; no teatime/waldo extension, per-scene mixture
or support-count/weight sweep. Default remains the single-support physical fit;
accepted LERF four-scene actual32.3130% and fern response76.7399% are unchanged.
Four real NVOS source-denied replays and four four-input LERF model replays pass;
40 tests pass. The full episode/first-pixel sequence matches across new arms;
unsigned/signed fits use2000/2500 member evaluations with equal2000 optimizer
updates and no deployment tensor growth. All models and scores are complete.
Exact-old-code fern repeats expose non-bitwise training; inference remains exact.
Current repeat teacher means differ by <0.005pp, versus up to~0.19pp drift from
the earlier saved fit. This is not a significance interval or cross-scene bound.
A source-only post-study audit finds pre-normalization support norm around0.73
for four positive rays and0.47–0.48 for mixed negatives in figurines/ramen; the
decoder renormalizes all to1. This loses disagreement but is not causal proof.
Next: distinguish within-ray uncertainty, co-target positives and independent
negative constraints in evidence aggregation over the same member kernel;
preserve singleton competence and repeat consequential fits. Do not infer a
successful architecture or promote a variant from source-prompt diagnostics alone.
Evidence: `docs/experiments/2026-10-07-support-training-evidence.json`.

Previous candidate evidence (2026-10-07):
`docs/experiments/2026-10-07-native-signed-prompt-results.md`.
The unchanged common physical member field now accepts real registered signed
NVOS scribbles through normalized uncertain rays. A deterministic raster adapter
and optional positive ray-centre extent estimate add input evidence, not another
mask decoder. Fern has 12 training/2 teacher-development views, 329,038 Gaussians,
183 source episodes, 2000 steps and a 1.577GB self-contained export.
All 8 fixed untrained/trained x positive/signed x default/extent arms and copied
source-denied replays complete before scoring. The extent floor is inactive as
an intervention: all scales remain 2.6988m and outputs unchanged.
Fitted signed retained-response IoU is 76.7399% versus saved v10 fern 82.1851%
(5.4453pp gap), with different raw-prompt/source-access/source-budget contracts.
Fitting improves response IoU 74.8546 -> 76.7399%, but AP falls 93.5841 -> 92.0952%
and actual subset IoU falls 77.8284 -> 76.1453%. Do not select the untrained arm
for fern or claim across-readout improvement. All eight cold replays are bit-exact
after correcting CUDA raster-pooling reduction order; 36 distinct tests pass.
This is 1/8 historically opened cases, not full NVOS or a new promotion.
LERF four-scene actual subset32.3130% is unchanged; ScanNet/UQIS remain pending.
Next: common signed multi-support learning and false-addition control under
source-only supervision, with fixed text/image/point and actual-subset gates.
Evidence: `docs/experiments/2026-10-07-native-signed-prompt-evidence.json`.

Previous candidate evidence (2026-10-07):
`docs/experiments/2026-10-07-native-local-member-results.md`.
Keep the fitted physical metric as the common research candidate. A protected
native DINO cosine distance is now an opt-in shared member-energy control, with
no new parameters/tensors/teachers and paired affine/native scoring. Source-only
exclusion and updated readout audits precede the intervention; source labels,
sampling episodes, candidate unions and all thresholds remain fixed.
Native-metric refitting improves actual text subsets (%), figurines 41.7884 ->
47.6108 and ramen 19.4642 -> 26.9443; image diagnostics improve on both. However,
ramen 3D-point teacher diagnostics fall 38.8454 -> 38.3229, failing the declared
common-input gate. Frozen insertion scores 59.4702/41.5179 text but hurts image
and points substantially. Neither is promoted; no per-input mixture, third-scene
native fit or strength sweep. Constant native locality is not proven to preserve
complete objects; post-evaluation scope bounds are diagnostic, not causal proof.
Waldo kitchen independently completes the physical method's four-scene LERF
development cohort: native/member semantic IoU 44.4599/42.1092%, actual subset
35.1987%. Equal-scene four-scene means are native semantic43.8943%, member
semantic49.4130%, actual subset32.3130%; saved v10 subset66.9633% under different
source-access/resources. Do not mix native-intervention scenes into that macro.
Waldo has 1,496,561 Gaussians, 18 training views, 2000 steps, 7.17GB export;
158 teacher-region diagnostics. All four input adapters replay from copied
exports; 29 tests pass. All jobs are complete. Evidence:
`docs/experiments/2026-10-07-native-local-evidence.json`.
Next: real signed/registered point evidence and scope-dependent appearance
variation, then remaining ScanNet OVS/NVOS/UQIS integration. Full five-task
performance, compactness and independent final testing remain unestablished.

Previous candidate evidence (2026-10-07):
`docs/experiments/2026-10-07-physical-scale-member-results.md`.
The shared member energy now optionally subtracts physical-scale-normalized
squared distance to the weighted support centre. Explicit geometry world units
are stored in the export; no new parameters, teachers, source views or labels.
Two fixed arms separate a frozen inference restriction from source-only refitting.
The frozen arm fails the figurines image gate; the refitted arm passes text,
image, 2D and 3D mean gates in both initial scenes and unchanged teatime replay.
Actual LERF selected projection IoU (%), previous supported -> refitted metric:
figurines 14.6841 -> 41.7884, ramen 9.6614 -> 19.4642, teatime 21.9424 -> 32.8006.
Same-scene v10 gaps remain 29.3255 / 38.8432 / 42.2491 percentage points under
different source-access/resource contracts. Native responses remain exact;
ramen's member-response semantic IoU regresses 44.2349 -> 42.8233 despite better
actual subsets. Keep these outputs separate. Current opt-in fitting recipes:
`configs/query_field/{figurines,ramen,teatime}_spatial_metric.json`.
No tensor-storage growth beyond the previous supported-anchor exports; their
2.66 / 2.84 / 8.77 GB scene packages remain high-capacity, not compact students.
Next: nearby false additions and query identity/extent binding, real reference
image/official point checks, then full five-task expansion. Do not resume
benchmark-specific threshold/prior-strength sweeps or hide native-only bypasses.
Three source-denied four-input cold replays and 26 distinct regression checks
pass. All fixed jobs are complete. Sealed evidence:
`docs/experiments/2026-10-07-spatial-metric-evidence.json`.

Previous candidate evidence (2026-10-07):
`docs/experiments/2026-10-07-training-supported-identity.md`.
Native top-1 text anchors in figurines/ramen were 0/36 eligible source-interior
training supports and 0/36 source-view dominant contributors. An optional
train-only `anchor_eligible` buffer now restricts inferred text/image inputs,
while retaining the complete observed member carrier and all registered prompts.
No learned/native/geometry tensor changes; one Boolean byte per Gaussian added.
The compiler and fitter share `training_episode` to prevent rule drift.
Paired actual LERF subset IoU (%), original -> supported: figurines
11.5965 -> 14.6841, ramen 6.7424 -> 9.6614, newly constructed teatime
16.8013 -> 21.9424. Teacher-region image diagnostics improve on all three.
All registered-point diagnostic metrics and native text responses remain exact.
This passes the fixed local mechanism gate; real subset quality is still far
below the desired reference. Keep it opt-in, preserve the native-only diagnostic,
and investigate remaining member extent before compression or five-task claims.
Teatime adds a matched 18-view/2000-step fit on 1,829,615 Gaussians; no hyperparameter
change or extra teacher/label. Three source-denied four-input cold replays pass.
22 targeted tests pass; 14 affected tests rerun after the training-rule refactor.
All jobs complete. The original worktree has new parallel paper changes; its
branch, HEAD and index remain unchanged, and those paper changes are preserved.

Previous candidate evidence (2026-10-07):
`docs/experiments/2026-10-07-query-field-two-scene-and-readout.md`.
Three more source-only fits completed; figurines/ramen both have fixed 6/18-view
controls and real LERF prediction-before-scoring runs. More observations improve
point diagnostics but do not consistently improve real text subsets. On the
18-view fields, actual LERF subset IoU (%), figurines/ramen: shared union
11.5965/6.7424; single-anchor member control 20.3929/9.8131; native-only control
46.4535/25.4241. Native-only bypasses the member core and is not promoted.
The current identity-to-member conversion loses substantial available native
capability; neither scale selection nor cancelling union resolves it. Do not
compress this candidate as an established high-quality teacher or substitute
its source-SAM diagnostic scores for benchmark results. Next mechanism work
must test native inferred support versus reliable registered support while
preserving the common member operator and both identity/extent outputs.
19 targeted tests and a second-scene four-input source-denied cold reload pass.
All jobs complete; no five-task replacement, independent final test or compact
student. Full evidence: `docs/experiments/2026-10-07-query-field-two-scene-evidence.json`.

Updated2026-10-06. Sole development default: `configs/pipeline/system_v10.json`.
Common entry: `python -m radio_gs.pipeline.cohort`; its default is also v10.
v9 remains a numerically unchanged comparator, with default metadata disabled.
Latest research and promoted-system report:
`docs/experiments/2026-10-06-query-conditioned-membership-and-system-convergence.md`.

Latest phase complete:3 fixed fits, exact independent calibration and31 freshly
executed common jobs. No jobs pending. v10 converges the previously supported
image member component; it does not deploy the failed new query-trained heads.
UQIS actual setIoU text/image/2Dpoint/3Dpoint (%):
16.3865/39.8980/49.7357/49.0470. Image versus v9 gains5.0218pp, paired scene95%
CI[2.4038,7.7406],7/9 rooms improve. Image AP44.0336->59.3088%, purity60.00->57.17%,
coverage42.89->55.27%. Four-query success remains7/67: gains scene0030 target22,
loses scene0435 target35. No per-query fallback. All native outputs and every
text/actual-point rank and set remain exact; image outputs exactly reproduce the
previous image component. This is component convergence from opened development
evidence, not a new independent gain. Image/2D/3D now share the existing real
member model; text keeps its prior physical scope. Estimated image identity is
still uncertain; joint identity/member uncertainty is not claimed solved.
Fresh public replay: LERF70.400676/66.963274%; NVOS86.513417%;
ScanNet19/15/10=61.011491/62.527589/72.486061%. Public job inputs unchanged.
LERF selected projection differs from v9 by-0.001654pp; unassigned replay
difference, not an improvement or exact-retention claim.
Complete replay/comparison:
`output/query_conditioned_membership_20261006/image_convergence/{system/summary,comparison}.json`.

Research closure:1377 independent NR3D references,2,578,504 training samples.
Estimated anchors:263 on target,980 on other labelled objects,134 unlabelled;
unlabelled vertices never become training negatives. Actual query target labels
replace the assumed positive-anchor target. Same capacity, same22/3/4 rooms.
Four reused development rooms,175 queries, exact calibration: old bridge7.0088%,
supervision-only6.4847%, plus query evidence6.8550%, fully anchor-free6.6060%.
Exact calibration uses every score breakpoint, eliminating the inherited0.05
grid boundary; final cuts .095868/.137625/.064491, conclusions unchanged.
Query-mean interventions reduce the latter two models to1.9121/2.2848%: query
evidence matters, but does not resolve identity. On correct-anchor subsets the
old conditional member decoder48.37% falls to29.53/21.64%; wrong-anchor subsets
improve1.99->4.22/5.00%. Mixing uncertain identity into one local member learner
damages existing conditional competence. These statistics concern this scalar
evidence/tree implementation, not all possible query-conditioned learning.
Do not resume its feature/capacity/threshold scans. Preserve conditional member
decoding; concentrate next on independent target/attribute/relation identity
evidence and competition among physical instance/scope hypotheses. A field of
full-sentence cosine scores is not itself resolved reference identity.
Research results: `output/query_conditioned_membership_20261006/comparison.json`.
Default supervision remains the existing independent3D labels only; new NR3D
member supervision is not deployed. SOTA, compactness, independent final testing
and unified high four-query accuracy remain unproven.

Previous typed report: `docs/experiments/2026-10-06-typed-identity-and-inferred-membership.md`.
Typed identity/scope follow-up completed:3 fixed12-epoch fits,49 scene jobs
(31 common primary plus18 UQIS controls),804 UQIS query executions, and one
175-query external-development anchor repair. No default promotion or pending jobs.
Independent semantic and relational parameters remove the old shared-gradient
coupling, but frozen parameters do not protect final identity decisions. Four
development rooms: semantic9.347%, frozen relation10.704%, joint11.346%; the
relation gate fails. A fixed-identity bridge to the existing v9 member model
improves semantic9.347->13.636%, passing its declared development gate.
Full UQIS factorial comparison, actual setIoU text/image (%): v9 16.3865/34.8762;
lexical-only15.7051/34.8762; lexical+membership18.1561/39.8980;
native+membership20.0983/39.8980. Gains come from membership, not the new text head.
Native+membership text delta3.7118pp has sceneCI[-0.3818,8.0829]; image5.0218pp
has[2.4038,7.7406]. All-four success7->10/67 for this attribution control.
The nominated lexical+membership primary fails its text promotion gate; the
native control also has uncertain text generalization and is not a new default.
All original native and actual-point arrays/sets remain exact in all3 conditions.
Fresh public22 replay: LERF70.400676/66.962744%; NVOS86.513417%; ScanNet unchanged.
LERF selected projection differs from prior by-0.002183pp; cause not independently
established, so public exact-retention check fails too. No SOTA equivalence claimed.
UQIS diagnostics: text20/67 correct-majority seeds and24/67 correct estimated
anchors, image42/67 and44/67. The largest text loss guesses the wrong anchor
inside a mostly correct region; other correct-anchor losses involve missing
extent or false additions. One query-supported anchor candidate gives only
7.0088->7.0226% on four external development rooms, fails gate, no UQIS expansion.
Next core issue: estimated identity is not a guaranteed positive prompt. Keep
the query and its identity uncertainty through complete-carrier membership;
do not replace instance referring supervision with same-class recognition,
or resume point-selection/threshold sweeps. New lexical/relation heads remain
research controls; public typed readers and sole default v9 remain unchanged.
Results: `output/typed_referent_binding_20261006/comparison.json` and
`identity_diagnosis.json`; recipe `configs/recipes/typed_referent_binding.json`.

Previous native phrase report: `docs/experiments/2026-10-06-native-phrase-identity-and-relational-competition.md`.
Native phrase follow-up completed: one role extractor and four fixed retrieval
fits across two bounded rounds. No benchmark execution or default change.
Query-only predicted spans are independently pooled by frozen SigLIP; native
phrase recognition is not relearned through contextual-token projection. All
candidate landmark positions participate in spatial relation marginalization.
Four repeatedly opened independent-training-domain development rooms: native
full query6.325%, predicted target phrase8.659%, prior local9.786%, new uniform
control10.508%, native spatial witnesses11.110% actual set IoU. Gain over prior
local1.324pp has scene interval[-2.137,4.616]pp; no stable promotion established.
Uniform-witness intervention10.813% gives only0.297pp conditional spatial gain.
Correct annotated roles/landmarks only reach11.406% on the same frozen head;
this privileged intervention is neither an automatic score nor a system ceiling.
Second matched round directly supervises relation discrimination among same-class
instances in1113 training descriptions. Native/uniform10.305/9.359%; conditional
spatial effect0.919pp, but total worse than round1, so this gate also fails.
All175 development queries retained: wrong official-class seed identities fall
111->57 using native target phrases, but same-class wrong identities21->57.
Joint ranking recovers some referents yet raises wrong-class choices57->69;
stronger relation supervision raises them to75. Preserve semantic identity
decisions while learning relation-based instance reordering; do not resume loss,
temperature or benchmark threshold scans. These text candidates are not in v9.
Results: `output/native_phrase_binding_20261006/final_comparison.json`.
The following records the previous v9 promotion and its five-task validation.
Comparison: `output/real_direct_membership_20261006/comparison.json`.
Complete valid31-task collection: `output/real_direct_membership_20261006/final_validation.json`.
All work in this phase completed:4 fixed fits,58 executed scene jobs including18
UQIS reruns after isolating an auxiliary-array mutation. Valid conditions are31
real-training common jobs and9 rendered-training controls;536 retained UQIS
queries,1072 including invalidated executions. No pending jobs.

The retained mechanism is direct actual-anchor complete-carrier membership.
A single classifier consumes exact per-view nested-mask co-membership, ambiguity,
visibility, displacement and native appearance; no seed mask is assumed pure.
Additional official3D instance supervision comes from22 independent rooms;
real registered observations replace12-view synthetic training. Same anchors,
labels,1,415,632 vertex samples and learner in both domains. Global threshold.35
selected only on3 calibration rooms. Four separate real development rooms:
native32.399%, rendered learner37.480%, real learner41.498% actual IoU.
No benchmark source views, geometry, teachers or evaluation contracts changed.

UQIS9/67targets/268queries, actual IoU text/image/2Dpoint/3Dpoint (%):
v8 16.3865/34.8762/42.0902/41.5238;
v9 16.3865/34.8762/49.7357/49.0470.
Point AP52.7449/52.3589->75.7901/75.3253%.
Point IoU deltas+7.6455/+7.5232pp, scene95% CI[3.6305,11.0551]/[3.1609,11.3009].
Real versus same rendered learner+9.6183/+9.6180pp, both9/9 scenes improve.
Therefore deployment-domain training evidence matters; head complexity alone
is not the demonstrated contribution. Rendered learner degrades v8 set IoU.
Scene0030 still loses3.181/4.332pp versus v8; no per-query fallback.
All-four success remains the same7/67 targets. Text/image ranking and sets,
and all four native prediction arrays, are bit-exact. Point purity falls about9pp
while coverage rises about18pp; actual set IoU nevertheless improves.
New shared classifier2.629MiB; native maps unchanged, still not compact students.

Other four public-task replays (%): LERF semantic70.400676,
selected-Gaussian2D projection66.964928; NVOS retained positive-reference86.513417;
ScanNet19/15/10=61.011491/62.527589/72.486061. Public readers unchanged.
Valid final collection reuses this phase's22 public predictions alongside9 fully
regenerated UQIS jobs. Every execution predicted before scoring; the collection
is explicitly a split execution, not a new31-job simultaneous first evaluation.
SOTA equivalence, independent3D LERF GT and independent UQIS final testing remain open.

Failed text branch: explicit ScanEnts3D phrase-to-instance supervision added on
same22 rooms (3426 eligible mention labels). Matched scene model development
IoU6.786->8.893%, still below previous local9.786%; query-mean correction9.187%
and uniform scene witnesses9.409% outperform the learned conditional witnesses.
Fails declared gate: no new text benchmark run, no mixing this head into v9.
Privileged correct-phrase diagnostic: frozen phrase recognition45.020% versus
learned mention20.381%; projecting arbitrary contextual tokens through the
frozen pooled-output head gives9.146%. These use annotated phrase boundaries
for diagnosis, never deployed inputs. A token state is not a native text query.
Source: https://scanents3d.github.io/dataset.html . New external phrase labels
are additional independent training supervision, not automatic teacher truth.

The earlier priority was to preserve genuine native phrase recognition while
binding target and landmark identities. The typed follow-up above supersedes
that plan: scope reuse helps, but estimated identity remains uncertain.
Do not resume local loss/attention/batch/threshold sweeps, use annotated phrase
boundaries at benchmark inference, or claim all-query/high-accuracy completion.
UQIS2Dpoint remains pixel+camera only, no sourceRGB/depth/SAM;3Dpoint worldcoord;
image only its own crop; text full expression; no cross-query target pairing.
Category partitions/sets and reference instances retain distinct typed readers.

Previous query-supervision phase: `docs/experiments/2026-10-05-query-specific-supervision-and-transfer.md`;
6fits/67jobs/1340queries completed, no promotion. Original local correction was
98.43% query-common; named eligibility gain+.816pp uncertain. Bidirectional,
same-class and posterior-normalized variants rejected. No further scans.

Previous completed phase: real-observation grounding and identity/scope supervision.
Report: `docs/experiments/2026-10-05-real-observation-grounding-and-identity-supervision.md`.
Results: `output/real_referent_grounding_20261005/comparison.json`.
All29 independent rooms rebuilt with the deployed real RGB-D observation compiler:
2643 stride20 frames,97532 regions, no total frame/proposal cap. Same22/3/4
fit/calibration/development split; all1592 listener-correct training descriptions
now have positive support. Fit target-majority candidates175/246->243/246;
finite-family label-selected diagnostic IoU34.81->56.93%, NOT automatic accuracy.
Extra NR3D and official3D supervision remain explicitly declared.

Three fixed12-epoch fits,49 fresh scene jobs,804 UQIS queries all complete.
UQIS text actual IoU: v8 16.3865%, local_scope20.2121%, scene_scope17.0300%,
scene_identity8.2291%. Local four-query success7->11/67, but four rooms decline
and95% scene delta interval[-.3323,8.0292]pp crosses zero. No runtime promotion.
Independent real-room IoU6.3250->9.7857/9.1587/8.7238%; still near zero in some
rooms. Attention gives no reliable advantage; purity-only supervision rejected.
Local seed dominant identity count stays23/67:5 corrected,5 damaged. Stable
correct identities contribute another+2.0531pp through better range choices.
Frozen native coordinates do not guarantee safe downstream identity decisions.
All native semantic and image/point prediction arrays and sets remain exact.

Full31-job candidate is scene_identity, not the local_scope nine-room control;
do not combine their best scores into a new unified method. Public replay:
LERF semantic70.4007/selected-projection66.9633%, NVOS positive-reference86.5134%,
ScanNet19/15/10 61.0115/62.5276/72.4861%. Retained development results, not SOTA
or independent3D validation. Native maps stay high-capacity; compact student and
independent final testing remain open. All current jobs ended successfully.
Next: transferable instance identity and complete membership, with corrected vs
newly damaged native decisions measured separately. No attention/purity weight
search. Training observation repair is complete; prior two-room-only status below
is historical. Previous phase report:
`docs/experiments/2026-10-05-referent-learning-and-observation-domain.md`.

Previous completed phase: isolate referent identity before changing scope.
Report: `docs/experiments/2026-10-05-native-context-and-identity-bottleneck.md`;
results: `output/identity_bottleneck_20261005/comparison.json`.
All27 scene jobs/804 UQIS queries complete. No public-task replay this phase.

Post-prediction label intervention, NOT an automatic result: keep native seed
order but remove non-target-majority regions. Text actual IoU16.3865->41.0613%,
image34.8762->41.6638%, unchanged scope decoder. Best target-majority input
region gives50.9328%; original finite one-hop family oracle52.9952%.
Text top10 contains a target-majority region for only35/67 targets; all67 have
one somewhere. No short top-k reranker can reach every target on these features.

Two complete automatic information controls retain all1107 source frames and
39360 masks, native coordinates and geometry. Unmasked tight-crop text identity
IoU8.8611%; original object +1.5x context equal-rank identity14.9309%, versus
v8's16.3865%. Image/point outputs remain bit-identical. Neither promoted;
do not tune context expansion/fusion weights or call background masking a bug.
Added offline channels230.627MiB, no new training or source access at query.

An evaluator-category oracle improves category-majority seeds33->45/67 but
target-majority seeds20->19/67. Category recognition is not referring identity.
Next: explicit natural referring-expression/scene-conditioned instance binding,
with native semantics retained and full membership a separate supervised task.
NR3D exists for22/3/4 of the independent fit/calibration/development rooms
(1728/207/175 records); instance-ID alignment and grounding training NOT done.
No new accuracy gain, independent final test, compact student or SOTA claim.

Previous phase: instance evidence reliability, deployment-domain
calibration and explicit ranking/set readout. Report:
`docs/experiments/2026-10-05-evidence-domain-and-instance-reliability.md`.
84 fresh scene jobs: two full31-job systems and22 public ownership controls;
24 affected subset fits refreshed after active-view budget correction. All complete.
No unit tests, new teachers, geometry updates or benchmark protocol changes.

Concrete fixes: ignore unlabelled vertices only in training, not deployment-domain
calibration; threshold remains .25 (old labelled-domain calibration IoU19.5752%,
complete domain18.1841%). Surface readout now defaults to unary ranks plus explicit
MAP sets. Legacy conditional-score controls are explicit; no historical score
splicing. All268 local-condition sets exactly match the previous learned surface
condition; AP recovers to26.5959/44.0336/62.7097/62.5198% across four modalities.
This is a score-contract repair, not new identity information or new point IoU.

Completeness-discounted exclusions give UQIS actual IoU16.8277/35.1907/45.7644/
44.6299%; text/image gain only+.2141/+.2150pp over the same surface condition,
all-four success still7/67. Not promoted. Local/minimum-complement source weights
degrade ScanNet19 to60.5774%, and matched mass still60.4746%. Positive ownership
alone retains61.0328/62.5663/72.5466%, LERF70.4416/66.9347%, NVOS86.5134%:
approximately v8, no major gain. The source readers share a new instance-reliability
function, but confidence is not calibrated identity/purity/completeness.
NVOS supplementary actual-set projection: v8 85.2248%, candidates85.2268%; the
headline86.5134% remains unchanged. No independent3D or SOTA claim.

Study `output/evidence_convergence_20261005/comparison.json`; declaration
`configs/recipes/evidence_convergence.json`. Latest full candidate is `partial_recipe.json`
under that output root; default and retained-v8 validation unchanged. Stop scalar
confidence/complement/graph searches. Next ONE mechanism: native-space query
identity jointly with complete membership, retaining alternatives before hard
seed selection, with actual text/image conditions and reliable correspondence
beyond one-hop support. Point-only supervision does not establish unified training.

Previous phase: local boundary evidence and shared whole-instance set
optimization. Report: `docs/experiments/2026-10-05-surface-consistent-instance-selection.md`.
40 fresh scene jobs (31 common tasks +9 fixed-relation UQIS controls),536 UQIS
queries; independent calibration and eight previously opened rendered development
rooms. No new final scenes or protocol changes. Common instance graph is compiled
offline and used for all four map-only inputs; native/anchor/seed controls bit-exact.
Public typed readers remain unchanged and freshly validated. No candidate promoted.

Actual UQIS IoU text/image/2Dpoint/3Dpoint (%):
- v8:16.3865/34.8762/42.0902/41.5238.
- Same-unary C parent:16.3865/34.8762/43.0747/42.7865.
- Fixed surface relations + global binary Potts:16.5876/34.7004/45.5480/44.9747.
- Learned relations + same solver:16.6136/34.9757/45.7644/44.6299.
Learning edges has no consistent advantage over the fixed relation control.
The learned solver adds2.6897/1.8434pp over the exact parent point unaries; both
95% scene intervals exclude zero. Versus v8,3D-point interval still crosses zero,
scene0203/0700 regress; text/image AP drops. Four-query success stays7/67. Do not
splice parent C's62.71/62.52 AP into the new cut sets: actual cut AP56.283/55.186.
New point purity75.54/75.53%,coverage57.74/56.54%; useful false-member removal,
not reliable complete identity. Candidate graphs +55.76MiB, maps5760.01MiB plus
shared models; additional24-room instance supervision, not compact or zero-shot.

Domain diagnosis: source masks cover87.75% of target vertices; point candidate
unions72.54/72.15%, final sets57.74/56.54%. Text/image positive corresponding
region unions only23.75/47.61%. Final target-dominant instances23/45/58/58 of67.
Neutral-unary expansion outside the candidate domain was separately completed on
eight development rooms: learned IoU24.5238->23.0607%, purity39.45->34.59, coverage
45.19->49.62. Reject this expansion; unknown is not negative, but adjacency alone
does not establish identity. No benchmark tuning or run of the rejected expansion.
Stop graph-strength search. Next: identity-conditioned association beyond one-hop
support, including actual text/image referents, with independently matched evidence.

This full common run: `output/surface_membership_study/learned_system/summary.json`.
LERF semantic70.400676/selected projection66.963280%; NVOS86.513417%; ScanNet
19/15/10=61.011491/62.527589/72.486061%. Public results maintained, not new gains.
Study `output/surface_membership_study/comparison.json`; declaration
`configs/recipes/surface_membership.json`. The latest retained-v8 replay is still
`output/system_v8_membership_closure_20261005/summary.json`. No unit tests added.

Previous phase: query-conditioned observation/region/element membership.
Report: `docs/experiments/2026-10-05-query-conditioned-membership-evidence.md`.
Three complete UQIS candidates plus all public tasks:49 completed scene jobs,
804 UQIS queries; eight new rendered confirmation rooms,215 instance targets.
All candidates research-only; v8 stays default. Shared native/anchor/seed controls
and text/image outputs are bit-exact. No benchmark queries, GT or protocols changed.

Concrete method corrections: retain exact anchor co-membership and nested-mask
ambiguity; train downstream from four-fold room-disjoint upstream predictions;
condition regional targets on the actual clicked instance, not seed-majority
identity, and predict seed purity instead of forcing one. In512 fit episodes,
310 seed masks have target fraction<.8 and216<.5. The corrected regional head
uses9645 pairs; element models use the same887126 samples on24 independent rooms.
Additional3D labels remain a separate supervision condition; native coordinates
and geometry remain frozen. Query permissions stay strictly map-only for UQIS.

Real UQIS point2D/3D AP and actual IoU (%):
- v8:52.745/52.359 AP;42.090/41.524 IoU.
- Prior exact-anchor learner:60.557/60.457;38.292/38.150 (rejected).
- Joint-observation inputs:61.728/61.459;43.468/43.170.
- Add cross-fitting:62.104/61.774;42.954/42.669.
- Correct regional query target as well:62.710/62.520;43.075/42.786.
The last candidate improves7/9 scenes but scene0203 drops~4.26pp and scene0700
9.03/10.92pp. Its mean gains+.984/+1.263pp have95% scene intervals crossing zero.
All candidates reduce four-query simultaneous IoU>=.5 success7->6/67. None promoted.
C raises point coverage to57.41/56.28% but purity falls to70.62/71.97%; only~53%
of added elements are target members. Stop this statistics-head/threshold search;
actual added-element identity and spatial boundary evidence remain the bottleneck.
Do not combine a new AP with an old set IoU as one successful decoder.

New confirmation rooms (fixed next room hashes, rendered RGB, NOT real benchmark
final test): first4/122 targets geometry17.492%, old head19.418, joint19.639,
cross-fit19.358. Next4/93 targets geometry16.701, cross-fit20.010, corrected-target
20.796. The correction beats cross-fit in only2/4 of the latter rooms; do not assign
all gains over geometry to that correction. Real/matched observation supervision,
complete scope, relational text identity and compact deployment remain unresolved.

Current study: `output/learned_instance_additions/joint_evidence_comparison.json`.
Full31 common run: `output/learned_instance_additions/element_joint_system/summary.json`;
other two candidates each rerun UQIS9; unchanged public methods share that current
control.15 completed same-run predictions retained when remaining jobs moved to
three free GPUs; no historical answers. Public LERF semantic70.400676%, selected
projection66.962017% (-.005634pp repeat variation), NVOS86.513417%, ScanNet19/15/10
61.011491/62.527589/72.486061%. Latest full retained-v8 replay remains
`output/system_v8_membership_closure_20261005/summary.json` (31 jobs).
The corrected candidate's complete executable recipe is
`output/learned_instance_additions/element_query_recipe.json`; it does not read
comparison outputs at inference. Extra surface arrays2851.435MiB remain research
assets, not compactness. Previous phase report:
`docs/experiments/2026-10-05-independent-instance-membership-learning.md`.

Larger independent whole-instance training inventory (only the stated pilot trained):
`output/independent_instance_membership/{split,inventory}.json`, all1181 eligible
official-training scans after excluding19 current evaluation room prefixes and
their rescans; room-grouped fit945 scans/445 rooms, validation236/111.
30,295/7,315 per-scan instance annotation entries. Additional3D supervision is a
separate condition. Real-photo training observations, instance class mapping and
part labels remain missing. Prior nine-room reliability training
overlaps the current ScanNet cohort and must not be reused as an unseen split.
Current v8 enables neither that head nor the new rendered-room learners.

All31 common-entry jobs freshly completed in `output/system_v8/summary.json`:
LERF4 dual outputs, NVOS8, ScanNet10, UQIS9. All predictions precede scoring.
v7 remains the frozen comparator. No historical final answers are runtime inputs.
Read [implementation](docs/SYSTEM_IMPLEMENTATION.md), [current conclusions](docs/CURRENT_RESEARCH.md),
[latest report](docs/experiments/2026-10-05-uqis-entity-range.md), and
[ADR0012](docs/adr/0012-common-observation-architecture-and-query-contracts.md).

Latest completed phase: scope correspondence and independent entity priors.
Four full UQIS conditions completed (36 jobs/1072 independent query executions):
third-view witnessed majority/union, generic SAM3 object scope, and full cached
FC-CLIP/Mask-Adapter COCO panoptic scope. None promoted; v8 remains the default.
Native/anchor/seed controls match v8 exactly; all text/image identity seeds stay
fixed. Panoptic scope raises 2D-point coverage48.52→51.44% but purity80.06→64.86%,
reducing actual IoU42.09→38.71%. Image34.88→33.30%; text16.39→16.58% is not a
reliable gain. No new public-task runs or gains are claimed in this scope study.
After predictions, label-selected finite candidate diagnosis: old one-hop52.995%
IoU, third-view union52.485%, panoptic scopes45.758%, joint old/panoptic56.173%.
24/67 targets gain candidate capacity; label choices are never method inputs.
Stop these fixed rule/prior comparisons. Next need independent instance/part
membership evidence and supervision of actual additions/removals, with separate
training inputs if external 3D labels are used. No universal negative from a
missing mask, no generic object prompt presumed to be an instance predictor.
Report and exact methods: `docs/experiments/2026-10-05-uqis-entity-range.md`;
complete results: `output/witnessed_scope_study/comparison.json`.

Latest promoted phase: native local appearance before region pooling.
Three full UQIS conditions (27 jobs/804 query executions) plus the complete
31-job common system:58 jobs total,1072 UQIS query executions on the same67
known targets. Source frames1107, original regions39360, retained patches1782270;
no added views, region cap, GT fitting or per-query winner selection.
Image AP/actual-set IoU (%): v7 39.433/28.599; equal SigLIP/pooled-DINO identity
39.791/30.739; bidirectional patches44.034/34.876; reciprocal patches41.793/31.286.
Promote bidirectional matching as v8; stop this matching-rule search. Its IoU
adds6.277pp vs v7,95% scene CI[2.314,9.927],8/9 scenes up; vs same-teacher pooled
control adds4.137pp,CI[.801,7.794],7/9 scenes up. Reciprocal control not promoted.
These are opened-development results, not independent final/SOTA evidence.
All nine reconstructed pooled descriptor arrays exactly match the earlier
appearance maps. Rebuilt pooled control reproduces all four earlier dual-native
readouts exactly. Each candidate's268 native/anchor predictions match v7 exactly.
The complete v8 run reproduces all four candidate readouts and actual sets exactly.
Study: `output/local_appearance_study/comparison.json`; complete comparison:
`output/system_v8/comparison_v7.json`; fixed conditions:
`configs/recipes/uqis_local_appearance.json`. No unit tests or approval chain added.

Common architecture: frozen geometry, separate native teacher coordinates,
explicit observation membership/visibility, typed set readout. The shared
`querying.observation_readout` is consumed by dense native, compiled category and
registered mesh readers. Typed source-assisted LERF/NVOS solvers remain declared;
UQIS is strictly map-only. One architecture is not one trained checkpoint or one
input-access/target contract for all tasks.

UQIS v8 image identity retains DINO local patches and sparse region-area weights.
`pipeline.local_appearance_map` compiles them from original legal source frames;
`querying.local_appearance_match` computes bidirectional nearest-patch similarity.
Equal SigLIP/local-appearance percentile ranks select a regional seed. Original
SigLIP field, physical scope, text and point rules remain unchanged. Match scores
are identity evidence, NOT membership probabilities or verified3D correspondences.
The map reads no source RGB/depth/SAM at query time; image owns only its crop,
2D point supplies pixel/camera only,3D point supplies its world coordinate.
No cross-modality target pairing enters inference. Other optional failed modes
are not enabled by v8. Base study maps remain frozen for reproducing controls.

UQIS dataset: `output/scannet_uqis_9_v0_3_1_candidate`,9 scenes/67 targets/268 queries.
Original geometry/membership maps: `output/uqis_map_only_v1/maps`.
Local maps: `output/local_appearance_study/maps`; v8 selects the chamfer map specs.
Map arrays total2852.810MiB vs174.038MiB v7 or231.695MiB pooled-DINO. Added shared
DINO checkpoint330.333MiB (half parameters165.139MiB), original SigLIP~7.49GB extra.
Build186.664 summed seconds. This is a high-capacity reference, NOT a compact
multi-teacher student. Shared-GPU cohort timing is not isolated query latency.

Current full-system metrics (%):
- LERF semantic70.400676, latest actual Gaussian-set2D projection66.967651.
- NVOS positive-reference86.513417.
- ScanNet19/15/10:61.011491/62.527589/72.486061.
- UQIS text/image/2D/3D AP:26.595870/44.033566/52.744860/52.358886.
- UQIS actual-set IoU:16.386530/34.876210/42.090214/41.523825.
Public task accuracy is retained; projection variation+.000498pp is not a gain.
LERF projection is not independent3D GT; NVOS full signed-protocol equivalence
and public SOTA equivalence remain unproven. UQIS is official-mesh/RGB-D-assisted,
not pure RGB+MoGe. All rooms are opened development; all18 difficult images remain.

Remaining main bottleneck: complete instance scope and relational text identity.
Target is unique dominant instance in45/67 image sets vs37/67 v7 (9 corrected,
1 broken). Image IoU>=.5 improves17->23/67, but text remains11/67 and each point
mode32/67. All four succeed only7/67 (scene-macro10.384%, previously8.995%).
Same36 identity-correct cases cover55.20%->55.92%: range is NOT solved. Lamp
scene0246 chooses87 seed elements supported by6 views, yet covers only10.70%
of the target, versus81.46% v7. All DINO conditions retain this failure; no per-case
fallback. The one-hop head preserves a partial seed's granularity; mutual nearest
patches do not certify whole-object scope. Keep identity and whole/part membership
separate. Prioritize independent complete-instance evidence shared by all queries,
then reliable text target/landmark identity, rather than more coarse-region weights.
Independent final rooms, part/signed interaction and compact retention remain open.

Prior completed controls remain recorded, not defaults:
- [Query roles/relationships](docs/experiments/2026-10-05-uqis-referent-identity.md):
  four full cohorts all fail; full-query positive support can still promote a wrong
  candidate. Do not resume geometric relation coefficient sweeps on partial masks.
- [Identity/range](docs/experiments/2026-10-05-uqis-identity-and-range-evidence.md):
  physical means and exclusive tracks fail; standalone DINO AP44.928 belongs to a
  different candidate with IoU27.783, not the v8 result. GT-chosen finite one-hop
  family52.995% is diagnosis only, not automatic performance or a universal ceiling.
- [Local membership](docs/experiments/2026-10-04-local-observation-membership.md):
  local semantic competition and member identity gates regress; do not equate a
  purer part with a more complete instance or similarity with foreground probability.

The entries below retain historical phase conclusions; old UQIS pause statements
are superseded by the completed reconstruction and evaluation above.

UQIS benchmark construction resumed at the user's request (2026-10-04), with
the strict map-only contract unchanged. The new maintained implementation is
`radio_gs.benchmarks.uqis`; final v0.3 development candidate is
`output/scannet_uqis_9_v0_3_candidate/release_final` (9 scenes,67 targets,268 queries).
All67 point queries match their target; maximum vertex distance4.485cm, minimum
image margin19px, minimum post-exclusion coverage86.0261%, >=5 sampled mapping
views per target. Joint reference selection fixes the window visibility conflict.
All67 crops reviewed by the assistant,18 flagged; independent identity/uniqueness
validation remains open. Four-modality and single-modality scoring preserve old
metric values on identical predictions. That construction-only phase ran no model.
Old maps contain62 observations now withheld and must be rebuilt. Official-mesh/
RGB-D geometry track only; not a pure RGB+MoGe result or an unopened test set.
See [phase report](docs/experiments/2026-10-04-uqis-v03-query-reconstruction.md) and
[benchmark usage](docs/benchmarks/uqis.md). Earlier paused references below concern
model experiments; construction repairs have now been completed as a candidate.

UQIS location correction (2026-10-04): the existing benchmark was found and is
readable. `/mnt/pool/sqy/3d_understanding/ScanNet-UQIS-10/source` holds upstream
inputs; generated data are in `output/scannet_uqis_9_v0_1` and
`output/scannet_uqis_9_v0_2_candidate`. The v0.1 frozen manifests contain9 scenes,
67 targets and268 queries; all9 mesh-coordinate/instance-label arrays and67 image
crops were loaded, and both versions'268 prediction arrays opened successfully.
The complete builder/evaluator/standalone package is retained at local remote
tracking ref `origin/codex/uqis-mainline` (`a0b8235`); commit `dbc8fbe` removed it
from this mainline while separating benchmark branches. v0.2 has31 core targets
and36 relational-text targets; these remain non-formal candidate results.
Its2D point input provides camera geometry and a positive pixel, not captured RGB
for SAM. UQIS experiments remain paused; this discovery did not run an evaluation
or establish accessibility of every upstream reconstruction asset.

Follow-up UQIS usability review is complete; see
[report](docs/experiments/2026-10-04-uqis-usability-and-query-contract.md).
The user reaffirmed the strict map-only query contract; image queries may carry
their reference crop, but2D clicks receive no captured RGB/depth/SAM input.
All268 historical predictions were rescored with zero Core-metric difference;
AP matches sklearn within3.33e-16. All9 cached meshes/instance arrays match raw
annotations, and full-sensor view exclusions recompute exactly. However25/31
legacy Core expressions explicitly contain relations/comparisons; that split
does not establish a relation-free task. Three prompt points are15–39cm from
their target triangle surfaces; the constructor omits per-pixel correspondence.
18/67 clicks are within5px of an image edge,14 exactly on it. Reference crop
quality also needs review. Keep these as historical candidate results, repair
query construction/versioning before claiming a validated benchmark. No frozen
queries/labels were changed and no current-model UQIS result was generated.
Existing assets are readable; some original2D annotation ZIPs remain inaccessible.

Latest completed study: individual region identity and one-hop geometric links.
All78 public-entry scene evaluations completed: original10 x6, another reserved3
0670_01/0678_00/0685_02 x6. New reconstruction uses all453 source RGB,34828
query-independent regions and5672 detected-region nodes; all18 predictions saved
before annotation scoring. No fitting or parameter change. Final19 original10 /
new3: v6 61.011491/59.387189; single verified60.137917/58.683289; linked verified
60.188221/58.934898; verified union59.618051/57.878853. Individual vs union is
+0.519866/+0.804437pp; links add only+0.050304/+0.251608pp. Neither beats v6;
new3 linked also loses15/10. Keep both as research controls; no default promotion.
The multiplicative emitting-name gate prevents identity correction:11742/11896
old and5611/5672 new regions have a single emitting label. Links change287/215
native top1s, but only7/0 after that gate. New graph has1174/1897 edges with both
endpoints >=80% instance purity; all1174 connect the same GT instance. This is
conditional post-prediction diagnosis, not proof all edges or independent-view
transfer is correct. Only60.58/65.26/66.81% projected regions are >=80% class pure;
whole-region confidence cannot assign correct membership within mixed scopes.
Do not deepen this graph or tune the multiplicative gate. Next separate identity
alternatives/unknown from local element membership; naming a region is not dense
semantic supervision. The new rooms are now opened. LERF/NVOS unchanged and not
rerun this round; no public-SOTA/compact/four-query/CVPR contribution closure.

Previous completed study: local element fusion and dual native teachers.
Frozen geometry/native coordinates; nine official-training rooms,932 views for
577/705-parameter local single/dual and705-parameter global controls. Additional
six previously opened rooms and three reserved rooms0606_01/0629_01/0653_01 were
fully evaluated; new cohort515 source images,36587 automatic regions. All18 new
predictions preceded label scoring. A real auxiliary entropy-policy deployment
mismatch was fixed; all36 development evaluations were rerun before new-room
prediction. Superseded outputs are explicitly separate. No post-result fitting.
Training local dual65.261522%19 versus base60.307115. New3: original55.260616,
equal native prior57.608254, local dual58.180987, same-capacity global53.748283.
Local beats global here but is only+0.572733pp over simple averaging; it loses to
averaging on dev6 and on new3 15/10. Do not promote learned heads or claim their
full+2.92pp versus base as independent contribution.0599 window omission remains.

PROMOTED v6: equal SigLIP/Mask-Adapter native probability prior within existing
source vote, no new scene-label fitting, no per-scene routing. Original10 final
19/15/10=61.011491/62.527589/72.486061 (+1.418525/+1.658532/+1.055484pp over v5),
9 scenes up/1 down0.066pp19. Dev6=60.812414%19 versus59.314710. New3=57.608254
versus55.260616 (+2.347638pp),2 up/1 down0.463783pp. Regressions remain included.
All22 full-system jobs complete: LERF semantic70.400676, selected 2D projection
66.962427, NVOS positive-reference86.513417%; tiny replay changes are not gains.
No improvement claimed for LERF/NVOS. Extra original10 native files1.104306GiB
(total2.314331GiB); RGB/geometry/encoders/source caches extra. No compactness or
annotated four-query/public-SOTA/CVPR contribution closure. UQIS remains paused.
The individual identity/correspondence study above is now complete. Retain its
granularity lesson, but do not promote its weaker complete readout or reuse these
opened validation rooms as untuned evaluations.

Previous completed study: [consumed-element reliability](docs/experiments/2026-10-03-consumed-element-reliability.md).
Separate SUPERVISED branch:9 official-training rooms, all932 source views, same
frozen map/geometry; native mask-aware region/query statistics and actual deployed
vote differentiation with point-balanced CE/soft IoU. Linear22/30 parameters and
fixed32-hidden546-parameter controls completed. Training19:60.8580→63.0496.
Different reserved rooms0575_00/0591_02/0599_02:388 views/24817 automatic regions,
1873 category-union encodings.30 public-entry evaluations completed (27 initially
predeclared,3 post-result same-checkpoint matched-mass controls), plus3 label-using
diagnoses kept separate. New-room final19: SigLIP62.840361, adapter vote61.839701,
linear retrieval62.518993, MLP retrieval62.801336, same-MLP shared62.843615.
MLP vs same-map vote+0.961635pp19, two rooms up/one down; no promotion. Both
capacities fail to beat same-mass shared weighting on primary19. Map-only mean
52.535175→57.254767, but0575 regresses8.589249pp: teacher replacement is not
universally better.0599 window FP806→516 but FN2333→3940, IoU75.8538→64.9410.
Ground-truth quality diagnosis64.806771%19 is NOT method performance/upper bound.
Stop expanding whole-view/category scalar heads; next test local identity/extent
at native region/visible-element granularity on training data, then other reserved
rooms. Current validation is now opened. LERF/NVOS unchanged, UQIS still paused;
no compact/four-query/public-SOTA/CVPR claim closure. Default remains zero-label v5.

Previous completed study: [supervised scope quality on independent rooms](docs/experiments/2026-10-03-supervised-scope-quality-independent-rooms.md).
Three earlier rooms0050_02/0231_01/0412_00:554 sources/36940 regions,15 evaluations.
Map19 improves42.719635→50.880137 in all3, but full adapter vote55.603399 and
supervised typed quality55.764741; same-mass shared55.739275. No promotion.
True-quality LABEL-USING diagnosis63.999849, not performance. These rooms are
opened development data. The earlier claim of no readable independent scenes
was a search error: ScanNet-PFIR-Small/AGILE3D contain official validation rooms.
They are not a pool of new training labels. UQIS remains inaccessible and paused.

Previous completed study: [source likelihood, identity binding and resolution](docs/experiments/2026-10-03-source-likelihood-and-resolution.md).
All70 full ScanNet scene evaluations completed. Same learned map: cross-view-excluded
detector likelihoods give60.354337/60.351585%19 versus original vote60.273548;
fewer wrong overrides but fewer useful corrections. Three identity/scope readers
give59.730102/58.530258/59.405379%19; no promotion or coefficient sweep.
All1158 source frames rebuilt into paired96x128/192x256 rasters from the SAME
SAM forward, shared proposal identities, physical boundary radius preserved.
Low replay matches all old point predictions exactly. High final19/15/10:
60.653474/62.239107/72.525994, +0.379926/+0.408736/+0.387038pp versus low;
9 scenes improve19,1 regresses. Versus v5: +1.060509pp19,8 improve/2 regress.
Identity/projection storage6.958→20.563GiB; map/RGB/models extra. Keep candidate,
not default. Finer projection is useful but does not resolve identity errors.
Next requires independent identity/extent supervision and an untuned scene;
self-agreement and same-scene pseudo-rate estimation did not establish reliability.
Correction: the earlier data search missed readable ScanNet-PFIR-Small/AGILE3D
validation scenes. They are validation, not a pool of training labels. See the
[independent-room study](docs/experiments/2026-10-03-supervised-scope-quality-independent-rooms.md).
LERF/NVOS unchanged; no SOTA or claim closure.

Previous completed study: [task-trained region readout](docs/experiments/2026-10-03-task-trained-region-readout.md).
All120 fresh ScanNet predictions/evaluations complete:40 context readouts,40 external learned
region readouts,20 query-independent maps,20 matched outside-scope-negative controls.
Same ConvNeXt backbone/masks: pretrained Mask-Adapter gives direct category58.844875 versus
plain pooling53.140041%19. This is an external pretrained teacher improvement, not our novelty.
All73905 automatic regions /1158 sources rebuilt, W and geometry unchanged, no query labels.
Map-only19: v5 SigLIP50.147691; ConvNeXt plain49.601052; learned region57.737935.
Learned map improves9/10 scenes. Final19/15/10 only60.273548/61.830371/72.138956:
+0.680583pp19 over v5,8 scenes improve,2 regress. No default promotion.
Dropping all outside-scope negatives fails: old map final56.688070, new map59.602818%19.
New-map-correct/old-final-wrong points59032; source fusion retains27320 and loses31712.
Next focus: independently justified source identity/scope override, not another weight sweep.
Do not blindly force this mutually exclusive category reader onto LERF parts or NVOS instances.
[Earlier scope/identity study](docs/experiments/2026-10-03-scope-granularity-and-source-identity.md)
remains a completed comparator. LERF/NVOS unchanged; external teacher ≠ compact multi-teacher student.

Previous bounded experiment: [positive region grounding](docs/experiments/2026-10-03-positive-region-grounding.md).
Two rules × all10 ScanNet scenes finished. Source-assisted regional identity improves the intermediate19-class
score50.15→58.64, but final59.59→59.90 only;8/10 improve,2 regress, including−3.55pp.
Same-positive-pixel direct control final59.00. Neither becomes default; no coefficient sweep.
The intermediate improvement uses source detections already available to v5, not new offline-map capability.
Focus now on independently verifying identity/scope corrections; repeating the same source evidence is insufficient.

## Current system (v9)

- Freeze the declared geometry, compile legal native teacher observations, element associations and visibility, then decode the requested set type.
- LERF retains separate semantic and actual Gaussian-selection heads. ScanNet combines the native region dictionary and pretrained Mask-Adapter prior under its mutually exclusive category contract. NVOS retains declared positive-reference tracking.
- All four UQIS queries read the same native high-capacity map. Image identity uses native SigLIP/local DINO; text uses native full-expression matching; both retain physical scope. Positive points use the shared real-observation-trained complete-carrier membership head.
- Extra3D point supervision is explicit. No evaluator labels, cross-query target pairings, sourceRGB/depth/SAM enter UQIS inference. UQIS is active, not paused.
- Unified high four-query accuracy, relation-aware text identity, compact students, independent final testing and official SOTA equivalence remain incomplete.

## Current complete results

|Frozen local metric|v9|
|---|---:|
|LERF4 semantic IoU / AP|70.400676 /87.619278%|
|LERF4 selected Gaussian2D projection IoU|66.964928%|
|NVOS8 positive-reference IoU|86.513417%|
|ScanNet10 point scene-macro19/15/10|61.011491 /62.527589 /72.486061%|
|UQIS text/image actual IoU|16.386530 /34.876210%|
|UQIS2D/3Dpoint actual IoU|49.735675 /49.046983%|
|UQIS2D/3Dpoint AP|75.790097 /75.325251%|

Earlier findings and vocabulary below are historical references. They do not
replace the active recipe, access contract or completion state above.

## Preserved v5 mechanism conclusions

Same v3 fields and teacher/source assets: information weighting gives final
59.083685/60.298858/71.379009 versus v3 58.449245/59.652818/71.027043.
Detector-relative confidence at matched per-query known-pixel mass then gives
59.592965/60.869057/71.430577. Final19 improves9/10 scenes versus v3; scene0070
regresses0.853pp. No scene-specific routing or parameter sweep. Full22 v5
reproduces all ScanNet scene/split metrics from the control exactly.

Whole-view-excluded semantic support weights fail to improve the complete result;
leave that research switch inactive. Confidence and agreement are not truth.
The category-information rule is for complete competing menus C>1, not arbitrary
single open queries, overlapping whole/part concepts or reference instances.

LABEL-USING representability diagnosis, NOT METHOD PERFORMANCE:
fixed geometry/W with GT-pooled regional classes reaches80.292984%19class;
GT fitting region logits200steps reaches81.590436%. No fitted variables saved,
no production config reads these outputs. These are in-sample diagnostics, not
upper bounds or independent accuracy. They show a large semantic assignment/
selection gap under this point protocol, not proof of physically perfect geometry.
They do not yet distinguish teacher recognition vs query matching vs mixed scope.

## Earlier field construction and storage

v2 ScanNet:57.921506/58.972766/70.905986. Same masked native observations and geometry,
region response before mixing gives+0.527739/+0.680053/+0.121057pp final and+2.147149pp
prior19class. Five scene19 results improve and five regress. No threshold sweep or
scene-specific route. Precompilation preserves every evaluated scene/split metric
exactly; it is operator reordering/storage, not additional accuracy or established novelty.

Semantic deployment files:1.210025GiB versus4.493211GiB v2 element fields and11.029344GiB
uncompiled regions/masks. Geometry, encoders, source cache/RGB are additional. Old
assets remain for comparisons, not runtime dependencies. Fresh19 reworded map-only
queries have0 argmax/support differences versus uncompiled and max probability
error7.15e-7; no old field, mask, camera or source-evidence config is required.
This is execution/equivalence evidence, not four-query accuracy or new-scene generalization.

Natural-background input control completed all10: same bbox+3px, same model/masks,
73905 new encodings. Prior19 falls46.503650→34.880087; final19/15/10 reaches
58.595790/58.871840/69.404065. Mixed final effects and strong prior degradation;
keep masked default. Do not promote or start blend/temperature sweeps.

## Preserved findings

- v1/v2/v3/v4 recipes and all comparators retained. Earlier same-evidence source coverage
  and one-pass-vs-iterative comparisons: `docs/experiments/2026-10-03-complete-source-cold-system.md`.
- Uniform-prior control, ownership confidence, partial categorical failures:
  `docs/experiments/2026-10-03-native-prior-and-evidence-conflicts.md`.
- Uncompiled pooling-order localization: `docs/experiments/2026-10-03-native-observation-pooling.md`.
- Finest ownership, dual-scope fusion, extra source calibration/iterations, sparse
  RGB conflict and scope transfer have negative evidence. Do not restore blindly.
- Native int8 is precision storage, not low-dimensional capability distillation.
- SAM3 `masks_logits` already contains probabilities; never sigmoid it again.
- Root historical context remains in `CONTEXT-HISTORY-2026-10-03.md`. Historical
  TPAMI manuscripts do not describe the current implementation.

## Remaining research

Prioritize source-grounded region identity supervision while preserving native
query coordinates and unknown evidence. Existing geometry/W can express much
better point segmentation when given correct semantic assignments. Distinguish
teacher recognition, matching and mixed-granularity selection errors, and require untuned-scene
benefit beyond simple baselines before claiming a new mechanism. Native teacher
coordinates and region atoms do not guarantee correct identity or complete scope.
Need independent3D extraction truth, annotated same-map four-query benefit, fair
public protocols/input tracks and a useful compact learned multi-teacher field.
Do not claim large accuracy gains or CVPR readiness from this integration.

## Domain vocabulary

The definitions below include earlier representations; they do not override the active implementation and evidence above.

## Representation

**Canonical Capability Feature**:
The sole persistent semantic feature owned by each Gaussian and used as the source for every downstream query.
_Avoid_: task feature, capability bank, semantic sidecar

**Single Compact Feature Field**:
A scene representation in which every Gaussian owns exactly one Canonical Capability Feature; task-specific semantic fields are excluded.
_Avoid_: multi-head field, task field, hybrid feature bank

**Universal Field v1**:
The versioned Single Compact Feature Field containing one factorized D512/L512 RADIO feature and its five query-independent reliability scalars per Gaussian; query readouts are not part of its identity.
_Avoid_: Method-v1, final method, task field

**Deployment Scene State**:
The persistent per-scene state required to execute a query from a cold start, including Gaussian rendering state and the Canonical Capability Feature.
_Avoid_: deployment cache, training checkpoint

**Training Artifact**:
State used to construct or optimize the field but unavailable to cold-start query execution.
_Avoid_: deployment dependency, hidden scene state

**Capability View**:
A query-independent working representation derived from the Canonical Capability Feature for semantic, appearance, or boundary reasoning.
_Avoid_: capability bank, task descriptor

**Query-Native Universal Gaussian Memory Candidate**:
The proposed v2 representation contract in which one persistent Gaussian
latent is supervised at query-response/posterior level and is not required to
decode into a named foundation-model feature space.  It remains a candidate
until ADR 0003 promotion gates pass.
_Avoid_: Universal Field v2, promoted field, multi-feature field

**Structured Universal Gaussian Memory v3 Candidate**:
The proposed single-field representation in which one persistent D512 latent
and five reliability scalars per Gaussian jointly retain visual semantics,
scale-conditioned instance membership, boundary structure, and reliability.
The D512 is an explicit shared-private product space: a shared core plus
semantic, instance, and boundary-private blocks with partition-owned writes;
raw RADIO reconstruction is diagnostic while promotion uses source-capability
Pareto preservation.
It is trained through source-heldout 3D-membership rendering and remains a
candidate until ADR 0004 promotion gates pass.
_Avoid_: v2 readout extension, instance sidecar, benchmark memory

**Shared 3D Instance Membership**:
The one per-Gaussian object posterior selected by text, image, or prompt
identity anchors and consumed unchanged by both 3D evaluation and 2D rendering.
_Avoid_: 2D proposal, projection repair, separate 2D/3D posterior

**Surface-Aligned Object-Centric Universal Memory v4 Candidate**:
The geometry-first candidate in which query-independent source observations are
registered to a visible surface carrier before object organization or feature
compression.  Local visual-semantic evidence and sparse soft object
memberships are persistent scene state; text, image, and prompt queries select
object tokens and produce one element posterior for both 2D and 3D output.
_Avoid_: SUGM-v3 extension, Gaussian-only memory, benchmark readout

**Frozen Geometry Stage**:
Query-independent geometry construction and validation completed before feature
reconstruction. Its Gaussian positions, rotations, scales and opacities are
bound to camera/projection contracts and remain fixed during semantic training
(ADR 0008). A geometry change invalidates lifted feature and mask assignments.
_Avoid_: semantic geometry repair, opacity-only semantic confidence

**MoGe Geometry Prior**:
Query-independent calibrated source depth used to constrain semantic observation
writes and, in separately evaluated stages, geometric support. The Gaussian
Single Compact Feature Field remains the primary map (ADR 0007); sparse
surface state does not replace local semantic identity.
_Avoid_: geometry ground truth, object-only map, second semantic field

**Semantic Observation Validity**:
For the learned element-field candidate, an element is geometrically visible
and has received a nonzero semantic feature-training gradient. Visibility
alone cannot validate an untouched random feature. This minimum evidence
condition does not certify semantic accuracy or convergence (ADR 0007).
_Avoid_: visibility equals learned feature, semantic confidence equals opacity

**Surface Carrier**:
A replaceable geometric domain exposing projection, evidence lifting,
posterior rendering, and surface-local adjacency without exposing backend
element identity to downstream object or query code.  Registered backends are
Gaussian transport, sparse surface voxels/surfels, and mesh oracle.
_Avoid_: Gaussian ID interface, exact physical registration

**Geometry Receipt**:
A hash-bound record of geometry, cameras, source evidence, model identity,
coordinate conventions, legal information access, and derived outputs used by
a carrier experiment.
_Avoid_: experiment note, unsealed data path

**Soft Object Codebook**:
A mapping-time object organization consisting of scene-level object tokens and
validated instance memberships. Top-1/top-2 storage is a later, measured
compression option, not a prerequisite for development evaluation (ADR 0006).
Overlapping candidate support must not be implicitly converted into this
categorical representation.
_Avoid_: D48 expansion, connected component, query-time tracking

**Candidate Support**:
Query-independent observed support for a possibly overlapping object or part
hypothesis. Candidate Support is bounded evidence, not categorical instance
probability; it preserves source provenance and remains available before query
selection and compression.
_Avoid_: calibrated membership, exclusive instance label

## Querying

**Authorized Query Input**:
The text, scribble, reference mask, reference image, or point that an evaluation protocol explicitly gives to the query procedure.
_Avoid_: query hint, target assistance

**Query Workspace**:
Ephemeral state derived from one Authorized Query Input; it cannot update the scene field or carry information into another query.
_Avoid_: online scene memory, persistent prompt cache

**Canonical Query Interface**:
The shared boundary that turns Authorized Query Input and the field into one Gaussian-domain result before output conversion.
_Avoid_: benchmark query path, task head

**QueryPacket**:
A model-independent, query-transient packet containing adapter-aligned tokens,
confidence, optional registered seed probability and spatial metadata.  It is
the replaceable query-encoder boundary for the Query-Native candidate.
_Avoid_: raw VFM feature, persistent query cache, zero-shot arbitrary encoder

**Primitive Readout-v0**:
The frozen minimal causal baseline that converts primitive similarity into output with fixed thresholds and no learned extent, boundary completion, or category rejection.
_Avoid_: Method-v1, final query interface

**Typed Posterior Interface**:
A globally shared query interface specialized by query family that produces a calibrated Gaussian Query Posterior without introducing another persistent scene field.
_Avoid_: task field, benchmark-specific sidecar

**Identity--Extent Posterior**:
A typed posterior that preserves a query-to-field identity unary for localization
and derives object extent from query-independent instance/region membership.  The
extent branch may complete support but cannot move or replace the identity peak.
_Avoid_: SAM semantic classifier, unrestricted region averaging

**Bounded Categorical Region Residual**:
A low-margin-only topology residual for mutually exclusive category posteriors.
It may smooth uncertain rows using query-independent SAM affinity, but cannot
create a class identity or alter high-margin predictions.
_Avoid_: SAM category logits, global proposal propagation

**Independent Prompt Consensus**:
For registered image prompts, box and signed-point SAM hypotheses are decoded
independently and combined after mask decoding.  Prompt types are not assumed
to be additive inside the frozen SAM prompt encoder.
_Avoid_: joint box-point posterior, prompt concatenation as Bayesian fusion

**Gaussian Query Posterior**:
The calibrated per-Gaussian output of the Canonical Query Interface before raster or world-space conversion.
_Avoid_: benchmark mask, task score map

**Signed-Evidence SAM Selector**:
A query-transient selector that ranks frozen SAM proposals by inclusion of
sealed positive field points and exclusion of sealed negative field points;
field-mask overlap and SAM confidence are tie-breaks only. It is not eligible
for contracts that forbid opening the target RGB at query time.
_Avoid_: field repair, strict-unseen readout, target-mask calibration

## Evaluation

**Five-Benchmark Program**:
The user-confirmed joint objective covers LERF-2D, LERF-3D, ScanNet OVS, NVOS, and ScanNet-UQIS with one method family. UQIS must evaluate text, image, 2D prompt, and 3D prompt queries. Available-Nine SPIn-NeRF remains supplementary and does not replace UQIS. Historical sealed experiment scopes remain unchanged.
_Avoid_: replacing UQIS with SPIn-NeRF, claiming partial prompt tests certify all four query types

**Available-Nine SPIn-NeRF Cohort**:
The orchids, leaves, fern, room, horns, fortress, pinecone, truck, and lego scenes; the unavailable Fork scene is excluded.
_Avoid_: full-ten SPIn-NeRF

**Evaluation Contract**:
A frozen definition of a benchmark cohort, legal information, output domain, metrics, aggregation, and comparator.
_Avoid_: evaluation setting, benchmark setup

**SOTA Target**:
A dated numerical target tied to one Evaluation Contract and comparator identity.
_Avoid_: competitive result, SOTA-level

**Development Evidence**:
A result used to choose or revise the method before the final frozen evaluation.
_Avoid_: final result, blind confirmation

**Joint Development Baseline**:
One complete method identity evaluated across all five benchmarks and used as the common comparison point for subsequent candidates.
_Avoid_: per-task best bundle, virtual incumbent

**Native Decoder Readout (experimental)**:
A single compact element field whose deployed text/image score is
`cosine(D z + b, native query)`, sharing D and b with native teacher
reconstruction. The global decoder is deployment state, not a second element
field. Exact matching algebra does not imply preserved semantic accuracy.
_Avoid_: transpose-only adapter tying, globally promoted retrieval baseline

**Reference 2D Signed Query**:
Authorized reference-image points, box, or positive/negative scribbles that
produce a transient image query. Geometric signed conditioning requires a
registered reference and preserves the full original signs beyond the SAM mask.
Unregistered image encoding does not authorize invented geometric correspondence.
_Avoid_: target-RGB propagation under strict-unseen evaluation, SAM mask as GT
