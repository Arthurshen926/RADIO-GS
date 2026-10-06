# RADIO-GS

## Active research branch — 2026-10-07

This branch develops the [query-preserving Gaussian field](docs/method/query-preserving-gaussian-field.md):
native DINO/SAM/SigLIP supervision, protected Gaussian capability subspaces,
and a shared conditional member decoder. Start with `python -m radio_gs.query_field`.
The architectural decision and resource boundaries are in
[ADR0013](docs/adr/0013-query-preserving-scene-specific-gaussian-field.md).
This is a research candidate; five-benchmark replacement accuracy is not established.

The baseline source snapshot is `0f40545c2b93fa390e37038af8315c24b5bc57b5`.
The older executable and results below remain baseline A; their defaults do
not select the new candidate.

## Preserved baseline A — 2026-10-06

The legacy executable default is [system_v10.json](configs/pipeline/system_v10.json).
`python -m radio_gs.pipeline.cohort` uses it by default; v9 remains a numeric
comparator. The same31-job architecture covers LERF4 with two outputs, NVOS8,
ScanNet10 and UQIS9. UQIS stays strictly map-only.

The [latest system convergence](docs/experiments/2026-10-06-query-conditioned-membership-and-system-convergence.md)
completed3 research fits and a fresh31-job replay. v10 reuses the existing member
decoder for image-estimated anchors, retaining native image identity. Image set
IoU34.88->39.90%, paired scene95% interval[2.40,7.74]pp;7/9 rooms improve.
Every native output and text/actual-point rank and set is exact. Four-query joint
success remains7/67, with one gained and one lost target. This converges a prior
development result, not a new gain from this round's training. Text still uses
its prior physical scope; estimated identity remains uncertain.

New query-supervised member heads failed: exact independent calibration gives
development IoU6.48/6.85/6.61% versus old7.01%. Adding query evidence or deleting
the guessed point does not reliably restore instance identity; correct-anchor
cases lose conditional scope accuracy. These heads are not deployed. See
`output/query_conditioned_membership_20261006/image_convergence/comparison.json`
for the promoted component and the parent directory for research controls.

The [preceding v9 promotion](docs/experiments/2026-10-06-scene-binding-and-real-membership.md)
retains direct actual-anchor membership trained on real observations from22
independent rooms, with explicitly additional3D instance supervision. UQIS2D/3D
point actual IoU improves42.09/41.52->49.74/49.05%; AP52.74/52.36->75.79/75.33%.
Both scene confidence intervals for set-IoU gains are positive. Text/image arrays
and sets and all native outputs are exact; four-query success remains7/67.
Scene0030 regresses and remains included. The classifier adds2.629MiB to unchanged
native maps. This is not a compact-student or independent-final-test claim.

Matched real/rendered training improves both point IoUs9.62pp across all9 rooms;
the synthetic-training learner alone degrades actual sets. The proven gain is
therefore the actual-anchor decoder with deployment-domain evidence, not added
model complexity alone. Public replay retains LERF70.4007/66.9649%, NVOS86.5134%,
ScanNet19/15/10=61.0115/62.5276/72.4861%, under existing declared protocols.

All4 fits and58 scene jobs completed, including18 UQIS jobs fully regenerated
after removing a shared-array normalization side effect. The valid collection
contains22 freshly executed public tasks plus9 corrected UQIS tasks; the matched
control adds9 UQIS tasks. See `output/real_direct_membership_20261006/final_validation.json`
and `comparison.json`. No historical final answers or per-query winners are used.

The [previous typed identity and membership study](docs/experiments/2026-10-06-typed-identity-and-inferred-membership.md)
completed3 fixed fits and49 scene jobs, with no default promotion. Reusing the
existing member decoder while retaining native identity improves UQIS actual
text/image IoU16.39/34.88->20.10/39.90%; four-query success7->10/67. Image gain
is positive across the scene interval; text remains uncertain. A new lexical
head hurts the benchmark, so it is excluded. All native and actual-point arrays
remain exact. Fresh public replay preserves main scores, except a small LERF
selected-projection difference66.964928->66.962744% whose cause is not established.
The key remaining interface issue is treating an estimated regional anchor like
a guaranteed positive point. A query-supported anchor repair failed external
development validation; it is not deployed. These are development comparisons,
not independent-final-test or SOTA claims.

The [previous native phrase study](docs/experiments/2026-10-06-native-phrase-identity-and-relational-competition.md)
completes one role extractor and four retrieval fits, with no benchmark/default
change. Query-only phrases retain native teacher coordinates; development set
IoU6.325->11.110%, versus the previous local9.786%. Scene uncertainty crosses
zero, and spatial evidence alone contributes only0.297pp. Direct same-class
relation supervision raises spatial dependence but lowers total IoU to10.305%.
Both declared gates fail. Target identity and relational reordering still need
to be reconciled; the failed text heads are excluded from v9. All-query high
accuracy, compact learning and official SOTA comparisons remain incomplete.

The previous [query-specific supervision study](docs/experiments/2026-10-05-query-specific-supervision-and-transfer.md)
completed6 fits/67 jobs with no stable promotion; its failed loss/batch variants
remain controls, not additional branches of the default method.

The previous [real-observation grounding study](docs/experiments/2026-10-05-real-observation-grounding-and-identity-supervision.md)
completed the29-room real RGB-D training repair using2643 frames and97532 regions.
That input repair remains active for research; the new study reuses it unchanged.

The previous [natural referent study](docs/experiments/2026-10-05-referent-learning-and-observation-domain.md)
identified the synthetic training observation gap. Its two-room real RGB-D
control has now been extended to the entire fixed independent cohort above.

The previous [identity bottleneck study](docs/experiments/2026-10-05-native-context-and-identity-bottleneck.md)
completes27 scene jobs/804 UQIS queries. A label-assisted identity-only intervention
raises text IoU16.39->41.06% with the same decoder; this is a diagnosis, not an
automatic result. Two full native-context controls yield8.86% and14.93%, both
below v8. All image/point outputs remain identical; neither candidate is promoted.
Correct category names improve category recognition but not the specified
instance. Explicit referring-expression/scene-conditioned identity supervision
is the next priority; context-crop and scalar fusion searches stop here.
Public tasks were unchanged and were not rerun in this phase.

The previous [evidence convergence study](docs/experiments/2026-10-05-evidence-domain-and-instance-reliability.md)
completed84 fresh scene jobs and refreshed24 affected subset fits. It repairs
complete-domain calibration and separates unary ranking from actual MAP sets.
Point AP recovers to62.710/62.520% while retaining the preceding45.764/44.630%
set IoU; this is a readout correction, not new identity information. Completeness
weighting adds only~0.21pp text/image IoU. Local minimum-confidence exclusions
regress public results; a positive-ownership-only control approximately retains
v8 accuracy. No candidate promoted. Query identity and full membership beyond
one-hop observations remain the priority; scalar reliability tuning is stopped.

The previous [surface-set study](docs/experiments/2026-10-05-surface-consistent-instance-selection.md)
completed40 fresh scene jobs, including the31-job common system and a9-scene
matched control. Joint binary set inference improves actual UQIS2D/3D point IoU
from42.090/41.524% (v8) to45.764/44.630%; a fixed-relation control reaches
45.548/44.975%, so learned boundaries do not establish a consistent extra benefit.
Text/image IoU16.614/34.976% changes little and AP declines; two point scenes still
regress, and all-four success stays7/67. v8 remained default at that stage. All public tasks were
rerun and retained accuracy; selected LERF66.963280% is repeat variation.
An additional eight-room neutral-unknown completion experiment was rejected:
coverage rises but identity bleeding lowers IoU24.524->23.061%. Candidate support
and actual referent identity remain the bottlenecks; stop propagation-weight search.
All four registered instance inputs share the same optional graph and solver;
source-assisted category/instance readers retain their declared typed contracts.
Additional3D training and5760.01MiB candidate map arrays remain explicit.

The preceding [query-conditioned membership study](docs/experiments/2026-10-05-query-conditioned-membership-evidence.md)
completes49 scene jobs (31 common tasks plus two9-scene UQIS controls), and eight
new rendered confirmation rooms. It restores anchor co-observation evidence,
uses room-disjoint upstream predictions, and corrects region targets to the actual
clicked instance. Corrected UQIS point AP is62.710/62.520%, actual IoU43.075/42.786%
versus v8's42.090/41.524%. These gains are not stable: two scenes regress strongly,
scene confidence intervals cross zero, and all-four success falls7->6/67. None
promoted. Public-task rerun retains accuracy (selected LERF66.962017%, repeat noise).
The full results explicitly separate each ablation, additional3D supervision and
actual element additions; no source-image permission or evaluation changes.

The [independent membership study](docs/experiments/2026-10-05-independent-instance-membership-learning.md)
now trains actual region and element models using separately declared3D instance
supervision. Preserving the exact point improves rendered-room confirmation IoU
17.86→22.07%, but real UQIS point AP gains do not transfer to better selected sets:
2D/3D IoU42.09/41.52→38.29/38.15%. Equal-count selection also fails. These models
remain research options; v8 was retained at that stage. Historical full validation:
`output/system_v8_membership_closure_20261005/summary.json`,31 completed tasks,
LERF selected projection66.967651% (repeat noise), all UQIS controls bit-exact.
Candidate UQIS9 runs separately;22 identical public-task predictions are shared
from this current replay before scoring. This is40 necessary new predictions,
not two independently predicted31-job cohorts. Real-observation membership
generalization and complete scope remain unresolved.

The subsequent [complete scope study](docs/experiments/2026-10-05-uqis-entity-range.md)
finished four full UQIS conditions (36 jobs/1072 queries). Third-view association
and two offline entity priors did not deliver a net improvement; they remain
disabled research options; their v8 comparison remains available.

```bash
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python -m radio_gs.pipeline.cohort --output output/my_system_run --gpus 4,5
```

|Current frozen local evaluation|v10 result|
|---|---:|
|LERF4 semantic / selected-projection IoU|70.400676 /66.963274%|
|NVOS8 positive-reference IoU|86.513417%|
|ScanNet10 point mIoU19/15/10|61.011491 /62.527589 /72.486061%|
|UQIS text / image AP|26.595870 /59.308782%|
|UQIS2D /3D positive-point AP|75.790097 /75.325251%|
|UQIS text / image actual selected-set IoU|16.386530 /39.897960%|
|UQIS2D /3D actual selected-set IoU|49.735675 /49.046983%|

LERF selected projection differs from the v9 replay by-0.001654pp; the cause is
not independently established. Public protocols remain local frozen comparisons,
not verified official SOTA equivalence; LERF projection is not independent3D GT.

The [complete local correspondence study](docs/experiments/2026-10-05-uqis-local-correspondence.md)
compares pooled and native local appearance on the same1107 legal views,39360
regions, geometry and teachers. Bidirectional patch matching improves image
actual IoU by6.277pp over v7 and4.137pp over the same-teacher pooled control;
paired scene95% intervals are[2.314,9.927] and[.801,7.794]pp. Reciprocal matching
is not promoted. All four readouts and actual sets exactly reproduce in the
subsequent31-job common execution. Other modalities/public tasks retain accuracy.
Historical v8 replay: `output/system_v8_membership_closure_20261005/summary.json`.
Original promotion comparison: `output/system_v8/comparison_v7.json`.

The common architecture freezes geometry, preserves each native teacher space,
and separates identity, observation membership and task-specific set readout.
Image identity now retains local DINO patches before region averaging. These
matching scores choose an identity; the original SigLIP field and text physical
scope remain intact. Images now use an estimated regional anchor with the same
explicitly3D-supervised member decoder as positive geometric queries. Image
identity remains uncertain. Source-assisted tasks keep their declared rendered-evidence solver.
Class partitions, category sets and reference instances have different contracts;
this is not one universal trained checkpoint.

UQIS evaluates67 targets with268 independent queries on the same nine maps.
Query-time access remains strictly map-only: image owns its crop,2D point supplies
pixel/camera only,3D point supplies its coordinate. Source RGB/depth/SAM and
cross-modal target pairings are unavailable. All18 difficult image references
remain included; every room is opened development, not final test.

Complete range and relational text identity remain substantial gaps. All-four
IoU>=.5 is only10.384% scene-macro (7/67 targets), versus8.995% v7. Retained local
patches increase map arrays to2852.81MiB; shared SigLIP and DINO encoders are extra.
This is a high-capacity map-only system, not a validated compact student. Independent
final scenes, part/signed interaction and public-SOTA equivalence remain incomplete.
LERF selected projection is not independent3D GT; NVOS uses the retained positive
reference protocol. Tiny public rerun differences are not method gains.

See [implementation](docs/SYSTEM_IMPLEMENTATION.md), [current research](docs/CURRENT_RESEARCH.md),
[UQIS protocol](docs/benchmarks/uqis.md), and [architecture decision](docs/adr/0012-common-observation-architecture-and-query-contracts.md).

## Previous research state — 2026-10-02

[Same-map image/point integration](docs/experiments/2026-10-02-same-map-image-and-point-queries.md) now supports explicit RGB image, 2D point and 3D point reference queries through `--scene-input` / `--query-input`. Eight source-only diagnostic runs completed on the existing ramen map. Identity/scope ambiguity remains; these are not official benchmark gains or complete four-query validation.

Start with [the consolidated research overview](docs/CURRENT_RESEARCH.md) and
[the result/dependency registry](configs/recipes/research_consolidation.json).
The current direction is a frozen semantic map, typed query evidence, and separate
semantic-response / selected-set readouts, with optional legal source-RGB access.
There is **no completed unified compact four-query or public-SOTA model** yet.

Consolidation is closed for the existing-map LERF text baseline. The executable
recipe is [current_research.json](configs/recipes/current_research.json), consumed
by `scripts/research/run_current_research.py`. Its default `identity_response` profile is the integrated LERF text research route:
semantic IoU/AP70.38/87.64%, selected66.66%. The old `identity` profile remains
the50.59/86.05% fixed comparator. [Response integration](docs/experiments/2026-10-02-consolidated-response-integration.md) evaluated all four scenes with the same source evidence; simple fusion accounts for most of the gain. Seven text queries still regress. The previous `source` remains
the fixed64.12% comparator. Four scene means improve; bag still fails.
`map`, `regional` and `calibration` remain explicitly separate controls.
All67 labelled texts plus1unlabelled probe completed the underlying full runs;
this is not new map reconstruction, a four-query model, or a submission claim.

```bash
python scripts/research/run_current_research.py --scene all --evaluate \
  --output output/current_research_new_run --plan
```

`--plan` prints exact commands; remove it to run generation then evaluation.
Use `--scene ramen --gpu 0` for one scene. No cached predictions are selected
implicitly. The [overview](docs/CURRENT_RESEARCH.md#0-本阶段已收尾唯一入口固定对照与明确取舍)
records retained/optional/unresolved work. The [complete run](docs/experiments/2026-10-02-consolidated-text-runtime.md)
and [provider controls](docs/experiments/2026-10-02-consolidated-provider-controls.md)
record measured results. The subsequent [identity/scope stage](docs/experiments/2026-10-02-source-identity-and-scope-control.md) completed20additional scene evaluations: matched fusion45.67 / subset fit63.94 / identity candidate64.86. That historical candidate was not promoted. The subsequent [conditioned identity integration](docs/experiments/2026-10-02-conditioned-identity-integration.md) completed12further scene evaluations, recovered part recall and reached66.66%full-source selected IoU; it is now the text development default, not a public-SOTA or four-query claim.

The [reference-instance integration](docs/experiments/2026-10-02-reference-instance-integration.md)
now also runs through `run_current_research.py --benchmark nvos`: fresh source
tracking, response reconstruction and independent selected-set output. Eight
positive-reference queries reproduce86.49%IoU. The frozen SAM seed mask is reused;
this is not raw signed-prompt, image-only or same-map four-query validation.

Existing references are separate result families: LERF semantic IoU66.930%,
selected projection62.599% retained /63.113% optional; NVOS8 IoU86.489%; ScanNet8
Gaussian-development19/15/10 IoU59.247/59.485/73.219%. They are not one model's
joint results. Selected projection is not independent3DGT; ScanNet is not the
public point/mesh protocol. UQIS remains paused.

Reaggregate the 112 registered evaluation reports (no new inference or scoring):

```bash
python scripts/research/summarize_consolidated_results.py
```

Generic region retrieval/tracking and held-source scalar calibration now have
complete4-scene controls: selected65.37% for regional fallback; semantic55.54%/
AP84.69% for calibration. Both remain optional because of identity regressions
and the calibration ranking tradeoff; they were not jointly evaluated.
See [the provider controls](docs/experiments/2026-10-02-consolidated-provider-controls.md).
Next address identity and scope reliability within this explicit-input baseline;
keep historical references separate. The result
registry is not the runtime recipe. See [the research plan](paper/CVPR_RESEARCH_PLAN.md).

## Historical implementation notes (2026-09-18 and earlier)

The material below is preserved as historical context. Its references to
“current”, “active”, or submission readiness describe those earlier snapshots,
not the consolidated status above.

[LERF4 execution and query repair](docs/experiments/2026-09-18-lerf4-closure-and-query-repair.md) completes four-scene full development text/reference evaluations, exact additional-scene online/replay checks and selected-output diagnostics. Text IoU is43.81/33.47/53.59/33.43%. Automatic incompatible-query replay improves the two-scene candidate to44.50/33.68%, with image/prompt unchanged and reviewed bag IoU42.12%; no global promotion. [Typed reviewed transfer](docs/experiments/2026-09-18-typed-scope-transfer-diagnosis.md) separates another bottleneck: thin chopsticks have88.17% same-view cycle but0.615% reviewed cross-view transfer. P3/P4 and complete ScanNet/NVOS acceptance remain open; declared full-method coverage is8/50 protocol cells.

[Fixed-scope identity collision](docs/experiments/2026-09-18-fixed-scope-identity-conflict.md) verifies incompatible source queries with exactly identical final outputs. Disabling all lexical takeover repairs the reviewed bag (0.029→81.859% source IoU) but damages ramen development text (33.474→21.226%), so it is rejected. [Registered ScanNet geometry](docs/experiments/2026-09-18-registered-geometry-pilot.md) fixes mesh-reference occlusion and completes a real MoGe/fusion pilot; low heldout coverage prevents promotion. These are bounded diagnostics, not completion of P3/P4/P5.

[Selected-output supervision and pipeline repair](docs/experiments/2026-09-18-selected-scope-objective-and-pipeline-repair.md) completes matched head-only controls and two source-scope passes. V2 text IoU improves to44.32/35.42% and selected-subset projection IoU to28.14/20.35%; image/prompt observations stay exact. The reviewed bag still fails and visible 2D masks do not certify physical 3D completeness, so no global promotion. Calibration now batches queries; all deployed balanced-calibration tensors reproduce the old control exactly. Other LERF builds continue; ScanNet camera/RGB preparation is complete, not reconstruction.

[Selected-object output audit](docs/experiments/2026-09-18-selected-object-output-contract.md) exposes a separate system gap: identical v2 fields score 44.06/32.92% with full-scene probabilities at original resolution but 25.55/18.27% when independently rendering the selected 3D elements. These use local polygon GT; released JPEG equivalence is unverified, so they are not official comparative scores. Source-only matched scope-head objective controls are running.

[Actual-query scope controls](docs/experiments/2026-09-18-query-conditioned-scope-controls.md) give small figurines text gains but fail the reviewed harmful-takeover case; both candidates preserve all image/prompt observations. The old gate had already trained on this severe negative example. No candidate is promoted. Complete teatime/waldo rebuilds are running; P4/P5 remain incomplete.

## Historical development method (2026-09-18)

The retained development candidate uses frozen MoGe surface Gaussians and one
96D element field: 64 native semantic coordinates plus 32 structural coordinates.
Pixel-contribution evidence fusion supplies native identity; text scope heads,
image residuals and registered prompt affinity have distinct readout contracts.
The deployment includes global decoders, source-concept prototypes and calibration;
scope-head state grows with the source concept inventory. RADIO is optional.

The hash-bound evaluation recipe is
[`configs/recipes/current_retained_96d.json`](configs/recipes/current_retained_96d.json).
It identifies the retained **v1-cache performance controls**, not the v2 models
or a single-command training pipeline. Run from the repository root:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/audits/run_current_retained_recipe.py --scene figurines --device cuda:0 \
  --output output/retained_recipe_new_run
```

The output directory must be new. This runs full online and recorded-encoding
text/reference evaluation and records actual query routes. It requires local
model/data assets declared by the recipe; it performs no training.
 Full text development IoU is 46.20% / 32.97%
(figurines / ramen); reference image/prompt are separate GT-derived protocols.
See the [stage summary](docs/experiments/2026-09-18-session-stage-summary.md).
Historical methods below remain controls, not the current 96D recipe.

The complete caption-dependent **v2 rebuild** is separately bound by
[`configs/recipes/v2_scope_96d.json`](configs/recipes/v2_scope_96d.json).
Pass `--recipe configs/recipes/v2_scope_96d.json` to the same evaluation runner.
Its training runner, `scripts/audits/run_v2_scope_rebuild.py`, starts from
verified caption-independent base assets; it is not a raw-RGB training recipe.
V2 repairs cache identity but does not improve both scenes: full-text IoU is
43.81% / 33.47%, versus the v1 controls' 46.20% / 32.97%.
The [source review and complete rebuild report](docs/experiments/2026-09-18-reviewed-scope-and-v2-rebuild.md)
documents harmful scope takeover, rejected correspondence controls and remaining
family-validation/cross-benchmark gaps. Assistant-reviewed masks are diagnostic
evidence, not independent human ground truth or training labels.



> Branch recovery (2026-09-14): the v4 object-candidate pipeline is a research
> candidate, not the performance baseline. The retained identity–extent LERF3D
> control has been re-rendered and independently scored at 43.739% sample mIoU.
> See [the recovery protocol](docs/adr/0006-repair-v4-contracts-and-restore-performance-control.md)
> and [the verified control](paper/artifacts/v4_repair_historical_control_20260914.json).
> This does not promote a joint five-benchmark baseline.

Geometry is now constructed and checked independently before semantic training;
see [ADR 0008](docs/adr/0008-finish-and-freeze-geometry-before-feature-reconstruction.md).
Frozen geometry candidates and their heldout source-depth audits do not yet
constitute a promoted semantic benchmark result.

The 2026-09-15 retrieval-field repair rebuilds text, image and registered-point
supervision from consistent source supports, with optional RADIO supervision.
Matched-budget experiments improve image and point retrieval in two development
scenes; ramen text AP still regresses. Native-direction constraints are not a
default. See the [implementation and evaluation report](docs/experiments/2026-09-15-component-pipeline-repair.md)
for all 42 training controls, repeated seeds and cold-query validation.

Reference image/point IoUs are GT-derived **2D category-union development
tasks**, not official LERF image-query or instance-tracking scores. The
[query-contract and reliability audit](docs/experiments/2026-09-15-reference-query-contract-and-reliability.md)
documents the exact inputs, source-query coverage repair, and 12 further
matched-budget controls, including regressions.

The current development direction restores the Gaussian semantic field as the
primary map, with MoGe-3 depth supporting observation registration. The first
bounded depth-weighted teacher builder is an unpromoted candidate; voxel and
surface carriers remain auxiliary. See [ADR 0007](docs/adr/0007-semantic-field-primary-with-moge-geometry-prior.md).

The historical RADIO-GS route reconstructs one query-independent C-RADIOv4 feature field on 3D Gaussian primitives and exposes several frozen query interfaces over that same field.

The executable historical mainline is the **one-field route trained with
`canonical-mpr-v3`**: the `canonical-mpr-v2` compact core plus one global,
scene-disjoint 3-D surface-region summary readout. The historical five-benchmark
research target was a sidecar-free D512/L512 Single Compact Feature Field;
it did not complete an end-to-end cold-field benchmark run. The
older HCD, dual-stream screen decoder, and screen-space refiner remain legacy
baselines.

## Historical D512 method

```text
frozen C-RADIOv4 spatial observations
                 │
                 ▼
query-free multi-view primitive registration (MPR)
  canonical-mpr-v1: top-1 raster assignment + alpha/depth responsibility
  + per-view normalization + contribution-mean fusion
                 │
                 ▼
compact canonical Gaussian field
  local code → primitive-local residual fusion
             → affine RADIO basis → one 1280-D feature / Gaussian
                 │
        ┌────────┼───────────────┐
        ▼        ▼               ▼
 official     official      global frozen 3D surface-region readout
 DINOv3      SAM3 adaptor       → official SigLIP2 summary head
 adaptor                          (text capability only)
        └────────┬───────────────┘
                 ▼
 typed query compiler → primitive unary / seeds
                 ▼
 optional shared 3D support topology and fixed solver
                 ▼
 3D support or deterministic 2D rendering
  (optional rank-8 zero-mean view residual for 2D rendering only)
```

The canonical field reconstructs raw RADIO only. It never stores a text query, image query, prompt, benchmark mask, or task-specific score. Its affine decoder is point- and batch-invariant, so direct primitive reads and alpha-normalized rendered reads use the same decoded primitive truth.

All new fields require the versioned `canonical-mpr-v1` observation contract.
This is the lifting-layer contract.  The promoted field-training contract is
`canonical-mpr-v2`: initialize from a compliant v1 MPR field, replay the exact
renderer, jointly preserve raw RADIO and frozen official DINOv3/SAM3 outputs
and local affinities, and constrain primitive replay.  Promotion is fail-closed
through the held-out multi-capability Pareto gate; it does not use benchmark
masks, labels, query text, or test-set calibration.
The optional view residual is centered with exact replayed MPR weights and is
never read by primitive-domain text, image, registered-prompt, or 3D-point
queries. It only compensates view context when rendering a 2D feature map.

### Representation status

- `Canonical-D384` is a high-fidelity oracle, not the submitted compact design.
- The historical compact candidate uses a 128-D per-Gaussian local code, primitive-local fusion to a 256-D affine coefficient, and a shared 1280-D RADIO basis.
- The active five-benchmark target uses one 512-D Canonical Capability Feature per Gaussian and must remove historical MPR, semantic-score, descriptor, and topology sidecars before cold query execution.
- DINOv3 and SAM3 capability losses/readouts use the frozen adaptors shipped in the official C-RADIOv4 checkpoint.
- The text readout is custom and predicts a 1280-D RADIO backbone summary token from a physical-scale, surface-connected canonical primitive region. The predicted token is passed through the frozen official SigLIP2 summary head; the custom readout must not be described as an official adaptor.
- It is trained once, without labels, masks, text, or scene IDs, on 24 ScanNet train scenes and selected on 8 disjoint official-val scenes. The older 2-D crop bridge and scene-specific crop-summary MPR caches are diagnostic/oracle baselines only.
- A semantic descriptor cache is disposable: formal text descriptors are deterministic functions of the canonical core, global readout, and official head. Cold per-scene storage contains no semantic field.

The latest controlled experiment record is `paper/artifacts/canonical_compact_capability_iteration_20260714.md`.

## Three-layer organization

### 1. Canonical representation

This layer handles only multi-view lifting, compact storage, affine RADIO reconstruction, and render consistency.

Primary entry points:

- `radio_gs/scripts/build_gaussian_multiview_teacher_cache.py`
- `radio_gs/scripts/train_canonical_radio_field.py`
- `radio_gs/scripts/finetune_canonical_radio_rendering.py`
- `radio_gs/scripts/render_canonical_radio_cache.py`
- `radio_gs/scripts/train_zero_mean_view_residual.py`

### 2. Frozen capability readouts

The same decoded RADIO primitive can be viewed through:

- the official DINOv3 adaptor;
- the official SAM3 adaptor;
- a frozen global region-to-summary bridge followed by the official SigLIP2 summary head.

Primary entry points:

- `radio_gs/scripts/build_canonical_capability_views.py`
- `radio_gs/scripts/build_scannet_surface_region_cache.py`
- `radio_gs/scripts/train_surface_region_summary_readout.py`
- `radio_gs/scripts/build_surface_region_semantic_cache.py`

### 3. Query interfaces

Supported query types compile into primitive-domain evidence:

- text category query;
- pose-free real-image exemplar query;
- registered 2D scribble, point, or reference-mask query;
- world-space 3D point query.

The support solver is optional: category relevance can be read directly, while instance prompts normally use the shared topology for grouping.

Score normalization is also typed and explicit. Independent cosine/relevancy is
the default text scorer, so adding an unrelated query cannot change an existing
score. Scene-category softmax is retained only as a named closed-taxonomy or
LERF paper-compatibility evaluation variant. Every `QueryResult` records the
effective policy; no modality override may use target labels.

Primary entry points:

- `radio_gs/scripts/eval_lerf_grounding.py`
- `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- `radio_gs/scripts/eval_nvos_gaussian_first.py`
- `radio_gs/scripts/eval_scannet_3d_point_query.py`
- `radio_gs/scripts/build_posefree_image_query_cache.py`

The reproducible pose-free ScanNet benchmark builder and its method-independent
mesh-domain evaluators live in `radio_gs/benchmarks/scannet_pfir/`. See
`docs/benchmarks/scannet_pfir_small_v1.md` for the held-out-frame, private-GT,
ranking, and instance-selection contracts.

## Audit ladder

Do not infer field quality from primitive-to-MPR cosine alone. The repository includes separate checks for lifting, coverage, compression, compositing, capability preservation, and query behavior:

- `audit_mpr_view_consistency.py`
- `audit_observation_lifting_contract.py`
- `audit_canonical_render_ceiling.py`
- `audit_feature_compositing.py`
- `audit_canonical_capability_fidelity.py`
- `audit_capability_cache_fidelity.py`
- `eval_canonical_primitive_reconstruction.py`

Every main result should retain provenance for the RADIO checkpoint, geometry rows, field checkpoint, capability adaptor, semantic bridge and region policy, graph topology, query compiler, threshold, and solver.

## Evaluation contract

Benchmark protocols are task-specific, but method internals are allowed when they are frozen without target labels. The current default rules are:

- use the benchmark's official query text/prompt and target data;
- compute its published metrics in the published output domain;
- allow query-independent field training, official frozen adaptors, fixed prompt registration, and a globally frozen support solver;
- record unary, propagated, and connected-component stages separately;
- do not select thresholds, calibrations, bridge checkpoints, or graph policies on benchmark test labels;
- keep `test_calibration=false` for formal results.

Here, `test_calibration=false` means that target labels, masks, or metric feedback are never used. If a declared variant uses query-conditioned scores or unlabeled evaluation-scene statistics, those transductive inputs are recorded separately rather than hidden by the safety flag.

Current NVOS and SPIn-NeRF outputs are protocol diagnostics until all official scenes are present and evaluated. A single-scene result must not be copied into a full benchmark main table.

## Environment

The code targets Python 3.9+, PyTorch/CUDA, `gsplat >= 1.4`, and a local checkout of the official RADIO repository.

```bash
conda create -n radio-gs python=3.9 -y
conda activate radio-gs
pip install -r requirements.txt
pip install -e .
export RADIO_REPO=/path/to/RADIO
```

Geometry, feature, and benchmark paths are supplied through each script's arguments or the legacy YAML configs where needed. Large datasets, checkpoints, and `output/` artifacts are intentionally not versioned.

## Tests

```bash
python -m pytest -q
```

Focused tests cover canonical-field checkpoint contracts, official capability fidelity, contribution compositing, view residuals, query compilation, score calibration, graph channel mixing, deterministic support solving, generic bridge data splitting, and evaluator safety flags.

## Legacy path

The following modules are retained for historical and matched-baseline experiments only:

- `radio_gs/scripts/train_feature_field.py`
- HCD codec / dual-stream hybrid configs;
- screen-space feature refiners;
- benchmark-specific auxiliary heads from earlier iterations.

Legacy checkpoints must be evaluated with their matching legacy renderer. They must not be silently loaded as canonical-field checkpoints or used to support claims about direct 2D/3D query consistency.

2026-09-16 iteration: [native decoder coupling and joint query contracts](docs/experiments/2026-09-16-decoder-coupling-and-joint-query-contracts.md)
records six controlled LERF runs, optional exact teacher/query readout, ScanNet
raw-class point output, and eight NVOS reference scribble query compilations.
Mixed LERF results do not pass joint promotion; new ScanNet/NVOS field metrics
are pending. [Evidence index](paper/artifacts/decoder_joint_iteration_20260916.json).

2026-09-16 follow-up: [signed support and NVOS camera repair](docs/experiments/2026-09-16-signed-support-and-nvos-camera-repair.md)
fixes prompt/render validity mismatch and NVOS camera/frame identity contracts.
The first fern frozen geometry and source contribution assets are built;
rendered validation rejects its joint-scale candidate. Signed image conditioning
also remains experimental after mixed-to-negative cross-view results.
