# Current decision — independent identity evidence, 2026-10-03

The bounded source-identity experiment is COMPLETE:40 LERF scene evaluations plus16 positive-reference NVOS refits. Local Qwen3 verification, full-source ownership context and the final mean-contradiction policy are all NOT promoted. Original semantic/selected70.3775/66.6617 becomes70.0384/67.1082,69.9030/66.9983 or70.2268/66.9628. Tiled evidence improves relative to its weaker baseline, not the retained mainline. NVOS86.4891 is unchanged. [Complete report](../docs/experiments/2026-10-03-independent-referent-evidence.md).

The useful mechanism evidence is narrower: independent identity judgment rejects a sparse false Waldo without suppressing a correct sparse paper napkin; local crops remove part ownership; full context partly restores it but causes other regressions. A whole-region identity decision cannot certify pixel scope or cross-view correspondence. The two initial policies share teacher outputs; the contradiction revision was developed afterward. New Qwen3 and extra visual inputs are additional information/compute, not an equal-budget optimizer comparison. No novelty or SOTA claim follows from these controls.

Stop region-level verifier, prompt, crop/context and confidence/admission sweeps. Next gate: correct source pixel scope and correspondence AFTER identity is established, with diagnostic source annotations separated from automatic method results. Compare identical inputs with simple fusion and the existing solver, then use an untuned scene. Do not replace this with more local filter gains. Public3D/ScanNet integration, independent multimodal accuracy and compact deployment remain outstanding; UQIS remains paused. All56 reports are complete; registry236; executable default remains identity_response.

The chronological entries below are prior stages, not pending work.

# Current evidence and claim boundaries — source reliability, 2026-10-03

Finite-evidence control COMPLETE (2026-10-03):24fullscene dual-output evaluations. A finite native mixture repairs the single false Waldo proposal but loses correct rare paper-napkin evidence; original70.38/66.66→69.37/66.60 or68.35/65.78, no promotion. Logit paper-napkin AP99.42 withIoU0.17 identifies calibration/attainability loss, not vanished semantic ranking. Inside-fit exceeds final-readout-only mixing but neither earns a new mainline or novelty claim. Stop confidence/count/prior sweeps; separate referent correctness, scope completeness and response calibration with independently verified evidence plus an untouched scene. [Report](../docs/experiments/2026-10-03-finite-source-evidence.md). Default and other benchmarks unchanged; SOTA/compact/multimodal submission claims remain unproved.

Source scale and direct identity controls are now complete:16fullscene dual-output evaluations, all764legal sources. Four fixed image crops degrade both outputs,67.89/62.98; same-evidence viewport bounds68.14/62.96 do not rescue this. Direct actual detector-mask native encoding70.91/67.07 gives small mixed gains, while unmasked1.5x context70.52/66.70 fails broad improvement. Default remains70.38/66.66. [Report](../docs/experiments/2026-10-03-source-scale-and-direct-identity.md). Image slicing is established SAHI-style input expansion, not a novelty claim; extra teacher cost is recorded.

The most concrete failure is one new0.504-confidence false waldo detection collapsing selected68.08→0. A no-veto identity state is not confirmation, and the single-view confidence cancels in normalized source fusion; repeated subset fitting can also amplify sparse evidence. Next mechanism must retain finite evidence strength and identity uncertainty before overriding a native map, jointly for response and membership. No minimum-view/query-name patch or crop/context/threshold sweep. This is a reproducible failure mechanism, not evidence that a proposed Bayesian or uncertainty solution already works. No new NVOS/ScanNet/four-query/compact/unseen-scene/SOTA result;UQISpaused.

The executable default is now `identity_response`: identical declared LERF map/source evidence, semantic IoU70.3775/AP87.6429/selected-projection66.6617. Old `identity` remains the matched50.5885/AP86.0472 comparator. Sixteen full-scene evaluations plus two entry reproductions are complete. [Report](../docs/experiments/2026-10-02-consolidated-response-integration.md).

This fixes a system-level omission: admitted source scope evidence previously updated selection but not semantic response. Simple visibility-weighted fusion explains almost all improvement; inverse fitting adds only0.375ppIoU while losing1.063ppAP. Therefore the evidence does NOT establish a novel optimizer contribution. Seven texts still regress. No per-query winner assembly or new VFM input was used.

Supported claims: reusable typed response/selected-set contracts; a complete source-assisted text pipeline with substantial matched-input development improvement. Unsupported claims: compact multi-teacher student success, same-map four-query accuracy, independent3D correctness, public four-benchmark SOTA, or ready-for-submission novelty. Same-map image/point interfaces did run source-only probes; they are no longer missing interfaces, but remain unvalidated task claims.

The bounded incomplete-scope gate is now complete and has **not passed**: containment-unknown controls add only0.10–0.19ppsemantic and essentially zero selected benefit; native-query-preferred parent completion reduces semantic/selection to69.75/65.15. Twelve full-scene dual-output evaluations and all67query evidence diagnostics are complete, with no promotion or parameter sweep. Coffee improves, but nose/hand/cookie scopes collapse: native relevance cannot certify requested whole/part extent. [Report](../docs/experiments/2026-10-02-query-scope-evidence.md). Source positive witnesses also occur on false identities; their presence is not independent truth. Do not describe these controls as a successful new contribution.

Next scientific gate: obtain independent identity and part/boundary scope evidence, compare its use under matched evidence/budget, and validate on an untuned scene. Stop further containment/cosine/negative-weight gates. Next system deliveries: ScanNet common runtime with public point/mesh protocol, independent geometry/3D evaluation, full multimodal task evaluation and complete source/map/query cost. UQIS remains paused. Do not substitute additional source-fit or NVOS-only weighting controls for these gates.

The chronological plan below is historical context; current status is the paragraph above and CURRENT_RESEARCH.md.

# Research plan after consolidation — 2026-10-02

The source-of-truth implementation and disposition are [CURRENT_RESEARCH.md](../docs/CURRENT_RESEARCH.md), with executable recipe [current_research.json](../configs/recipes/current_research.json). Existing-map text interface consolidation is complete; the scientific and original multimodal/compact claims are not.

Current text default updated after [conditioned identity integration](../docs/experiments/2026-10-02-conditioned-identity-integration.md): full-source selected64.12→66.66, all4scene means improve; semantic unchanged. The same-query detector-score density model preserves a no-veto hypothesis instead of applying scene-wide top5% rejection. Heldsource+2.36pp; forcedsplit ablation−0.55pp and yellowdesk collapse. Toycat/miffy improve substantially, bear nose recovers, bag still fails. This is12new complete scene evaluations and an integrated default profile, not independent novelty or unseen-scene validation. Identity no-veto behavior does not establish calibrated correctness. Same-map reference/prompt integration and untuned-scene evidence are next; do not restart threshold searches.

Reference-instance integration is now complete for the retained positive-only NVOS contract: explicit Scene/Query, fresh source tracking, normalized-response reconstruction and shared subset decoder. Eight-query86.4891%IoU preserved. Wrong output interpretation gives6.8182%, while actual subset silhouettes85.0794%are diagnostic only; use dual typed outputs. This is interface/output correctness, not an accuracy or novelty gain. Frozen reference SAM masks were reused, not raw signed prompts; native image identity and3Dpoint queries on one map remain next. [Report](../docs/experiments/2026-10-02-reference-instance-integration.md).

Fixed comparator: native semantic map plus fresh exact-text source masks and independent set fitting;67 labelled queries plus1unlabelled probe over4LERF development scenes. Semantic50.59/AP86.05, selected64.12. Regional fallback65.37 is optional due severe identity regressions; scalar calibration55.54/AP84.69 is optional due ranking loss. They were not jointly evaluated. Historical66.93 semantic is a different query-dependent chain.

Completed bounded follow-up: [identity/scope control](../docs/experiments/2026-10-02-source-identity-and-scope-control.md),20scene runs,2core candidates/1revision. Matched fusion45.67 vs subset fit63.94; conflict abstention61.45 rejected; native identity64.86 has toycat+50.09pp but bear nose−58.84pp. Default unchanged; no untuned scene or novelty validation. Next deliverable: preserve native identity evidence without allowing whole-region feature weakness to exclude correct parts. Nine-track descriptor diagnostics do not independently resolve drift. Diagnose source evidence before selected-set fitting, then validate decisions in unused source views. Same-teacher agreement is not truth; assistant/VLM source review is diagnostic only. At most two candidate mechanisms and two parameter revisions; no per-query winner assembly. Evaluate complete67query outputs and an untuned scene before contribution claims. The already completed two integration controls are not a reason to repeat an unconstrained sweep.

Contribution comparison must hold evidence and budget fixed across simple aggregation, tracking plus standard fitting, and the proposed mechanism. More source views and correct rendering are useful system fixes but not isolated novelty. Public protocol alignment and independent3D validation remain open. NVOS/ScanNet have legacy references, not this new shared runtime; UQIS stays paused. Same-map four-query evaluation and compact deployment remain required future deliveries, not abandoned goals.

The latest full evidence is [the text baseline](../docs/experiments/2026-10-02-consolidated-text-runtime.md) and [the provider controls](../docs/experiments/2026-10-02-consolidated-provider-controls.md). Do not restart documentation consolidation as the next research task.

# Historical next107 evidence — preserved for reference

Current method evidence comes from frozen geometry/native semantic memory plus
online typed source evidence and selected-set optimization. It is not evidence
for the historical compact multi-teacher student, nor a four-query SOTA system.
The historical manuscript must not inherit these numbers under its older claims.

Next107 completed24new scene controls on all four LERF development scenes, with
13difficult queries changed and54retained. Text plus automatically supplied core
boxes gives63.113%selected projection, versus62.236%matched core and62.599%retained
candidate. Same Box evidence with weighted backprojection gives57.569%: this
supports the utility of the downstream optimization/regularization package, not
novelty or a renderer-only causal attribution. Multiple component boxes and
prompt-compliance weighting do not improve the full cohort. Source confidence
must not be mistaken for correct physical identity or set completeness.

Three measurable requirements govern the next contribution:

1. **Identity correction and set completion must coexist.** Independent text
   evidence recovers plastic ladle (0→47.02; visual Box48.87), but an incorrect
   core can veto useful discoveries or seed more wrong instances. Preserve typed
   anchor, independent-detection and scope hypotheses rather than collapsing them
   into uniform positive masks. Spatula40.17→16.49 under the same Box rule is a
   required failure case, not an exception to hide with query-specific routing.
2. **Separate evidence gains from map-inference gains.** Keep identical-input
   backprojection and source-only teacher controls. Claim a compact feature-field
   or multi-teacher learning benefit only after a matched learned-field control
   demonstrates it. Current unchanged semantic scores establish preservation,
   not improved semantic recognition.
3. **Show transfer before publication claims.** Use independent development/test
   separation and fixed rules; evaluate current-map text/image/2Dprompt/3Dprompt
   paths and NVOS/ScanNet under their own allowed-input contracts. Category-set
   unions cannot be silently applied to reference-instance prompts. UQIS remains
   paused. Strong baseline comparisons must match source RGB/depth/teacher access
   and include online cost; existing published-context tables are not proof of SOTA.

Retain next107 Box only as an optional higher-mean development candidate; next104
is the rollback and next80 stable. No current CVPR-ready claim is established.
[Complete next107 evidence](../docs/experiments/2026-10-02-set-discovery-transfer-next107.md).

---

# Current research direction — 2026-09-29

The existing `radio_gs_tpami.tex` is a historical manuscript. Its compact-field
claims and numbers do not describe next44's online source-evidence method.
Do not update its tables with online results while retaining its deployment
claims. No current CVPR-ready or SOTA claim is established.

Candidate question: recover directly selectable 3D element sets from 2D
observations whose identity, extent completeness and geometric association
have different uncertainties. A mask can identify the right concept while
omitting valid instances or parts. Its complement is not automatically a
reliable exclusion label.

The next47 experiment first separates detector confidence, native identity
confidence, cross-view consistency and map prior under identical detections,
source cameras, renderer and 60 update steps. The new mechanism is a minimal
one-sided censor: other-view positive extent reduces exclusion strength.
It does not establish completeness, repair systematic missing detections,
verify physical identity, or constitute a proven novel contribution.

Evidence required before a submission claim:

1. Improvement over simple fusion with identical foundation-model evidence,
   not merely improvement from invoking a stronger detector.
2. Independently reviewed incomplete-range cases and interventions distinguishing
   false exclusions from geometric transport failures. Assistant review and
   synthetic masking are not human ground truth.
3. An official-code strong baseline and a separately labelled geometry/semantics
   exchange study. The official VALA checkout is being used; the first geometry
   pilot reuses existing RGB Gaussians and is not an author-protocol reproduction.
4. A held-out generalization experiment and complete online cost accounting.

Keep offline-field retrieval and online source-evidence inference as separate
deployment tracks. Preserve the original image/prompt paths without claiming
that unchanged branches validate a new four-query method. UQIS remains paused.
Delay compression, more prototypes and a new carrier until a mechanism passes
the same-evidence comparison and independent extent diagnosis.

Next47 matched-confidence control: preserving query-level total evidence weight while removing per-view native confidence variation gives IoU53.759/selected37.816, versus53.914/37.710 for the full method. Current evidence therefore does not establish per-view identity weighting as a strong contribution. Query-level confidence and the map prior explain much of the gain; neither factor establishes novel completeness inference.

The official VALA three-level figurines pilot completed with mIoU57.53 and Acc@0.5 64.29 using its own renderer/scorer, historical RGB geometry and 295 source views. This is a runnable mechanism anchor, not an equal-input comparison or full author-protocol reproduction. Next isolate geometry, semantic evidence and readout through controlled exchange before attributing the gap.

## Next48: concrete mechanism and remaining claim boundary

The four-scene official-code exchange now exists. VALA selected-projection IoU
on historical RGB geometry is51.854 with all source views and49.527 with the
current20/25/23/22-frame semantic subset. Identical subset/teacher/readout on the
current MoGe carrier gives35.187. Geometry training budgets/splits are not matched;
this is a carrier/interface diagnosis, not a new published-baseline reproduction.

GT-assisted fitting separates the visibility-normalized semantic image from the
selected-subset silhouette. On figurines/ramen, directly fitting the latter
raises MoGe selected IoU from43.98/34.39 to67.79/50.40; RGB reaches81.12/80.08.
These are deliberately GT-fitted diagnostics, not method scores or upper bounds.

The source-only candidate keeps semantic response fixed and decodes a separate
transient set. Detector/field conflicts on BOTH positive and negative evidence
become unknown. Fixed-recipe LERF selected improves37.710→43.523, all four scenes,
while the retained probability branch stays53.914 IoU/71.211 AP. This is a real
set-readout improvement, not improved semantic recognition. No per-scene winner
selection is used. Severe pikachu collapse in the unprotected decoder is removed,
but coffee/refrigerator regressions remain. Geometry-only RGB fitting is mixed
and rejected as a global replacement.

Do not claim first discovery of rendering/retrieval or opacity/occupancy mismatch:
[LESV](https://arxiv.org/html/2604.01388v1) and
[CIF](https://adreamwu.github.io/Consistent-Instance-Field/) address closely related
issues; [ILGS](https://github.com/DCVL-3D/ILGS_release) addresses identity and
boundary consistency. The candidate contribution must be demonstrated in handling
partial/conflicting set evidence under matched input and on held-out data. Current
threshold agreement is a useful baseline, not yet a novel calibrated completeness
model. Prioritize that question and carrier footprint/support, not more teachers.

## Next49: reliable extent evidence remains the research gap

Eight fixed-recipe four-scene controls and two-scene fourfold-carrier pilots
completed. Censoring negatives inside native-compatible enclosing proposals
restores coffee; requiring proposal availability in two source views otherwise
retains the previous set and restores refrigerator. Typed selected IoU changes
43.523→43.678; semantic IoU/AP remain53.914/71.211. This is a small development
reliability result, not a new recognition improvement or publication breakthrough.
The rule uses no query-name routing, but was developed on these results. Two-view
availability is not verified cross-view correspondence or complete-object support.
Source proposal/descriptor caches are additional inference state.

Positive expansion, biased Bernoulli straight-through optimization and refitting a
fourfold selection carrier fail to improve the cohort reliably. Decision-matched
alpha loss has small mean gains but does not combine reliably with extent evidence.
Do not stitch per-query winners or present these controls as independent novelty.

Next substantive experiment must distinguish partial visibility from actual object
exclusion using independently supported scope hypotheses and views withheld from
correspondence decisions. Report both missed extent and outside-object selection,
with genuinely independent validation. A full RGB+MoGe constrained reconstruction
is still uncompleted; splitting fixed Gaussians is not a substitute. Matched-input
strong baselines, identity/completeness evidence and generalization remain required
before a CVPR-level contribution claim. No global default promotion in next49.

## Next50: multiple hypotheses are insufficient without better evidence

Fixed-geometry, fixed-output-contract four-scene controls complete: soft-min
43.928, hard-min44.059, cross-view-prior43.974 selected IoU versus43.523. All
regress on figurines/ramen. Coffee improves substantially in hard-min but the
refrigerator regression persists. Do not stitch per-query winners. Retained2D
53.914/AP71.211 is not a recognition gain. Energies over differently qualified
partial masks are not calibrated completeness likelihoods.

Removing the native-field positive veto when other views corroborate a detection
yields43.473 with no newly active queries. This does not establish that the veto
alone explains missing semantics. Source evidence remains the limiting hypothesis;
exclude alternative carrier/correspondence causes via credible scope interventions
before introducing more latent selection machinery. Other-view priors exclude
self-votes, but do not provide independent held-out validation. No new default,
other-benchmark result or publication-readiness claim. See next50 report.

## Next51: reopen the carrier bottleneck

Annotation-assisted scope fitting on alternating labeled views, evaluated on the
remaining views, yields MoGe40.99/39.21 versus historical RGB52.86/76.33 on
figurines/ramen. These are diagnostic results, not method scores; geometry itself
was not rebuilt on this split and RGB training budgets differ. Figurines also has
four unseen-query observations; seen-query-only scores are47.83/61.67.

Source-only MoGe-anchored centre, anisotropic-scale and opacity refinement improves
the diagnostic transfer to44.64/43.57. With recomputed source projection and the
same pseudo-label decoder, full4 selected43.817 vs43.523 improves3scenes but hurts
teatime; original2D branch remains fixed. Transporting old membership without
re-decoding hurts both pilot scenes. No default promotion or major gain claim.

This is a two-carrier causal intervention, not a rebuilt unified semantic map.
Stop treating geometry as solved or attributing residual error only to scope
candidates. Prioritize full RGB+MoGe reconstruction followed by consistent semantic
reconstruction, with visibility/footprint and partial-evidence constraints. Current
bounded refinement lacks densification, explicit multiview depth consistency and
scale correction. Its small mixed gain is not a CVPR-level novelty result.

## Next52: calibrated depth helps modestly, distribution loss does not

New RGB+MoGe reconstruction initially missed accepted recovered scales. Fixing
this interface restores all20/25/23/22 training depth views; source files remain
unchanged. Full4 selected44.0367 vs43.5225, all4 improve, but essentially tied with
next50 hard44.059. Ray-distribution first/second-moment objective43.7185 does not
improve the depth-only recipe. These are fixed-budget development controls, not
novelty or independent-generalization evidence.2D remains frozen53.914/AP71.211.

The unified reconstruction milestone remains open: no densification/pruning,
new joint scale estimation or four-teacher semantic rebuilding on the new carrier.
Do not rebrand bounded geometry and set refitting as a completed full field.
MoGe consistency/conditional coverage metrics do not establish true geometry
accuracy. See next52 report for complete results and input-access boundaries.

## Next53: unified rendering helps; copying the old field does not

Rendering both outputs on the refined next52 carrier yields LERF4
56.574IoU/74.150AP versus53.914/71.211; selected44.037 stays fixed. Three scenes
improve and waldo regresses. This unifies output geometry, not raw-teacher field
training: old online source-evidence responses still generate the semantic input.
Do not label it a fully rebuilt offline map or independent benchmark result.

Source-response refit lowersIoU/AP while raising localization. Old latent source
projection distillation lowers nativeAP in both pilot scenes despite improved
source MSE. Preserve this negative result: direct old-field imitation is not a
sufficient learning objective. Prioritize reconstruction from raw teacher evidence
with independent query-relation/extent/transfer checks. Full teacher retraining,
densification, unified four-query evaluation and matched strong baselines remain
open. No per-query winners or CVPR-ready claim.

## Next54: native ranking and source-visible extent mass

Raw1536D native SigLIP features rebuilt on refined geometry do not suffice:
mixture36.052IoU/73.550AP, finest37.300/69.863. Existing source scope inference
raises mixture47.420/75.876. Retaining candidate ordering while transporting the
old response distribution using source visibility mass gives58.087/74.965 versus
next53 56.574/74.150; three scenes improve, figurines slightly declines. This is
real development2D gain with extra capacity, not a stand-alone new compact field.

Selection remains44.037. Reinitializing the set decoder with the new response
reduces it to42.206; reject that combination. Do not claim3D improvement. The native
field is query-independent, but source detection/calibration/scope and confidence
transport are query-time operations. Old extent mass may itself be wrong or zero;
this dependency limits the contribution and cannot be hidden in an offline claim.

Next research question: estimate reliable complete extent/support independently
while preserving native ranking. Require held-out evaluation and resource-matched
baselines. Four-teacher rebuilding, all-query validation and independent SOTA
comparison remain open. No default promotion or CVPR-readiness claim.

## Next55: independent confidence replacement failed

Source-only shared monotone calibration in pixel and element domains completes
four-scene controls but lowers IoU58.087→51.891/51.081 despite AP74.965→75.805/75.740.
Neither is promoted. Source Brier improves on three scenes while true IoU falls;
this is not a suitable proxy for target extent. Preserve next54 and prioritize
qualified identity/positive extent/exclusion evidence with held-out transfer and
actual selected-subset rendering. Do not present calibration experiments as a
CVPR contribution or claim mask incompleteness is uniquely identified.
See docs/experiments/2026-09-29-independent-source-extent-next55.md.

## Next56: strengthen the set-decoding baseline before claiming evidence novelty

Retained-membership continuation gives47.346 selected projection IoU versus44.037;
this is an additional120updates/query, not a novel completeness mechanism.
Other-view contribution qualification47.485 barely improves that control.
Annealing alone47.603 and deployment-aligned coverage47.863 separate membership
sharpening from the alpha/coverage objective. The final decision candidate improves
all4 scenes versus next54 but has coffee/refrigerator/spoon regressions. Semantic
IoU58.087/AP74.965 is unchanged. Evidence-mass semantic consensus is essentially
neutral and not promoted. No large new semantic breakthrough is established.

Use this stronger uniform set decoder for subsequent research controls. The
novelty target remains independent identity and whole/part/union completeness,
with held-out validation and fair source/input budgets. Do not attribute the
entire3.826pp to the coverage objective: its matched contribution is0.260pp beyond
annealing alone. No new ScanNet/NVOS/UQIS score or global default promotion.
See docs/experiments/2026-09-29-evidence-mass-and-deployment-set-next56.md.

## Next57: teacher mistakes can validate across source views

Five full4 controls do not improve next56. Full/subset visibility-ratio update
limits reach45.201 selected; half-view optimization47.498 and held-source gate
47.496 vs retained47.863. Source coffee masks mix the tea glass and white-mug
contents, while the inspected GT denotes mug extent. Held-source pseudo-IoU
improves and accepts an update that lowers true selected IoU. No GT alias or
query-specific fix is introduced. This supports a concrete entity/scope problem,
not an all-query causal attribution. Prioritize explicit per-instance identity
and requested-extent evidence, with a stronger independent source model only if
its additional input/compute and generalization are measured. Same-teacher
cross-view validation is insufficient. Keep next56 candidate unchanged.
See docs/experiments/2026-09-29-occlusion-and-source-validation-next57.md.

## Next58: local identity weighting and native teacher exchange are insufficient

Seven full4 controls complete. Per-instance identity47.871, area balancing47.845,
unknown enclosure47.816 and local VALA OpenCLIP identity47.958 selected vs retained
47.863 do not establish a material gain. Native OpenCLIP512D field with scope and
old confidence is54.793IoU/71.446AP vs58.087/74.965. Direct source scalar calibration
collapses fixed-threshold IoU1.415 despite64.623AP; native generic-negative scoring
35.691/60.830 does not recover the retained result. Crop precision and dimensionality
differ from SigLIP, so this is not a clean universal teacher comparison.

Keep next56. Avoid further local weighting sweeps or equating encoder replacement
with a contribution. Explicit entity/requested-extent evidence and a coherent
method contract remain open. No four-query completion, independent test, new
other-benchmark gain or publication readiness is established.
See docs/experiments/2026-09-29-instance-evidence-and-native-clip-next58.md.

## Next59: scope capacity helps cases; requested granularity remains missing

Matched120-step selected decoding: joint-view choice48.284, independent-instance
choice48.252, broader containing candidates48.354 versus next56 47.863. Do not
claim factorization itself improves performance. Refrigerator65.15→87.00 comes
from candidate expansion; coffee46.77→56.79 already improves with joint choice.
Bear nose, bag and three cookies regress. Semantic2D58.087/AP74.965 stays fixed.

The proposed variable must identify requested scope/relationship (entity, part,
contents/container or collection), rather than just maximize agreement with the
model's current rendering. Candidate hierarchy alone is not that variable and
can reinforce existing errors. Preserve whole/part/set cases in an independent
comparison, avoid query-specific GT aliases and per-query winner stitching.
Hierarchy is a development candidate, not a global or CVPR-ready promotion.
See docs/experiments/2026-09-29-factorized-scope-hypotheses-next59.md.

## Next60: admitting missing concepts does not establish correct extent

Four full4 controls: frozen semantic guidance48.320 and joint guidance48.361
selected versus next59 48.354 are neutral. Empty-query source admission plus
backprojected initialization gives48.493, recovering only corn0→7.881; the
other three empty queries lack eligible evidence. Separate scalar semantic
response reconstruction yields58.102IoU/74.986AP but localization falls
82.245→81.893. No broad improvement or default promotion.

The combined admission/initialization control removes an old-map dependency;
it does not prove source identities or complete extents. Any-instance margin
currently admits a query-view union, leaving mixed-instance supervision open.
Prioritize independently supported entity/requested-scope hypotheses before
fusion; do not market frozen-map scope guidance or tiny gains as a contribution.
No native feature retraining, final test or new other-benchmark result here.
See docs/experiments/2026-09-29-semantic-scope-and-empty-bootstrap-next60.md.

## Next61: direct recognition evidence is insufficient for extent

Two full4 controls replace self-referential scope choice with frozen native
foreground/complement similarity in a shared source RGB window. Selection48.136
versus hierarchy48.354; per-instance source admission with alternative extents
unknown47.829. Semantic58.087/AP74.965 remains frozen. Refrigerator improves
to95.179 but collection/object cases regress. No promotion or contribution claim.

Next research needs distinguishable referential entity/part/contents/collection
supervision, not a renamed crop-similarity criterion. Extra source model or labels
require explicit budget reporting. No such referential model was trained here.
See docs/experiments/2026-09-30-source-native-extent-next61.md.

## Next62–63: temporal extent evidence, not semantic confidence

Reciprocal source-instance scope graphs48.387/native-constrained48.345 versus
next59 48.354 are neutral. Source-only SAM3 tracking adds actual temporal evidence
from54/67eligible query seeds. The uniform temporal-union selected decoder gives
48.518 versus matched next56 47.863 (all4up), but only+0.164 over next59 and
not all scenes improve against that stronger candidate. Extra online compute.
Semantic reconstruction with this evidence loses IoU/AP/localization to
57.706/74.256/79.147; reject it. Retain the selected-only research candidate.

The distinction between extent persistence and semantic identity now has a
concrete counterexample. Neither seed identity nor referential whole/part/set
supervision is solved. No large gain, global promotion or publication-ready
contribution is established. Avoid claiming graph correspondence alone solved
identity or that temporal propagation trains a native feature field.
See docs/experiments/2026-09-30-instance-graphs-and-text-source-tracking-next62-next63.md.

## Next64–65: do not mistake retained information for proven retrieval gain

Typed temporal abstention58.085/74.854 nearly restores IoU but does not beat
retained58.087/74.965; adding temporal positives56.518/73.635 fails. Three native
area-rank slots cost3×storage and give raw29.916/70.611 versus mixture36.052/73.550.
Final max57.433/74.544 and source-selected level57.247/74.287 are not replacements.
Selected48.518 is frozen from next63. Do not call early mixing the proven main bug
or scale storage a contribution on these results. No global promotion.

Stop prioritizing variants of the same weak scope labels. Independently grounded
referential scope remains missing; full source-only RGB+MoGe reconstruction with
matched source/input budget is also still an open system-level comparison. Old
RGB-carrier exchange gains were confounded and must not be counted as new results.
No such full reconstruction was done here. See docs/experiments/2026-09-30-typed-temporal-and-scale-factorization-next64-next65.md.

## Next66: source reconstruction is a real bottleneck; set membership remains unresolved

Joint source-only RGB+MoGe reconstruction and fresh native features give LERF4
64.882IoU/81.021AP/87.328localization versus58.087/74.965/82.245. Expanded legal
RGB293/122/169/180 is explicitly an increased input budget; matched sparse
adaptive61.942/78.281 and fixed-count60.447/76.831 controls separate some effects.
This enabling reconstruction is not sufficient methodological novelty.

Selected-only projection falls48.518→43.748. A matched240-step source continuation
recovers45.862, outperforming actual-old-output distillation44.206. Neither soft
state transfer nor relaxing complement negatives solves the mismatch. Do not
promote a joint map or silently combine old selection with new response.

Prioritize identity-consistent, visibility-aware membership reconstruction on
changed Gaussian support and actual subset-render exclusion, with source
unknown/occluded evidence distinct from negatives. Source-resolution and
opacity/support factorization are hypotheses, not established solutions.
Waldo prior-depth disagreement remains; RGB PSNR is not physical-geometry truth.
No new other-benchmark/four-query/independent-test/SOTA evidence.
See docs/experiments/2026-09-30-joint-rgb-moge-reconstruction-next66.md.

## Next67: membership must be optimized in the visibility domain it is consumed in

Freezing members hidden in full-scene transmittance was incompatible with
selected-only rendering, where removing nonmembers exposes them. Let the same
source subset losses update all semantic-observed elements: matched selected
45.862→51.414, all4 scenes improve; semantic64.882/AP81.021 preserved.
Precision improves with almost unchanged recall. Hard-forward alone45.870 and
4×raster45.833 do not explain the gain; exposed4×51.220 adds no benefit.
Retain the simpler exposed-support candidate. Old-carrier48.518 is exceeded
on macro and3scenes, but teatime59.116→55.283 remains a regression.

This correction supports distinguishing semantic-response visibility from
explicit-subset membership optimization; it alone is not CVPR novelty. Next
need identity/whole-part supervision and transfer to the remaining query tasks,
with the same map/input protocol. No new other-benchmark or independent-test
evidence. Source RGB budget and query-time supervision costs remain explicit.
See docs/experiments/2026-09-30-subset-visibility-support-next67.md.

## Next68: source admission is necessary but does not supply missing identity evidence

Reject visible-membership-only33.505 and joint visible/subset49.665 versus
next67selected51.414.14/67queries lack eligible range supervision:11have no
detector candidate on the original source cohort,3have candidates but no
eligible positives. Other-view geometric corroboration newly admits corn;
admission alone remains0IoU, source initialization20.362. Overall51.835
versus51.414 is modest, with two small scene regressions. No global promotion.

Prioritize new correctly grounded source observations using a uniform source
selection/query recipe, and compare native recognition on query-independent
region proposals where text detection is absent. Full RGB geometry currently
uses many more source views than semantic teachers. More teacher views alone
are not a proven fix for named-entity/part identity. No such expansion or
new other-benchmark/four-query/independent-test result was completed here.
See docs/experiments/2026-09-30-source-admission-and-dual-observation-next68.md.

## Next69: source relevance is useful for ranking, insufficient for extent supervision

Native source top1regions with foreground/complement recognition improve semantic
AP81.021→83.482 and localization87.328→89.091, but IoU64.882→64.485. Cold/augment
set supervision51.641/51.576 and removal of supplemental negative loss labels
50.701 fail to beat next68 51.835. Eleven previously untrained queries become
eligible; some recover, others lose strong native fallback performance.

Do not promote proposal relevance to verified entity/whole-part supervision or
claim absent detector evidence proves absent native capability. Next needs
independent source entity/extent verification, not more loss terms on the same
uncertain masks. Source-view expansion remains untested here. No new four-query,
other-benchmark or independent-test evidence; no joint default change.
See docs/experiments/2026-09-30-native-region-supervision-next69.md.

## Next70–71: sharper segmentation and full RGB crops do not solve referential scope

Six full4controls fail replacement: cropped SAM3 cold51.463/augment51.502/
replace51.506/local-exclusion51.210selected vs51.835; semantic scope64.647/81.049
IoU/AP; actual full-crop feature reconstruction63.536/79.544 vs64.882/81.021.
Retain existing candidate. Full-crop bounds/resolution differ as well as masking;
no pure mask-effect attribution. Source and completed-eval visual diagnostics
show genuine entity/extent mismatches, not a reason to change benchmark labels.

Prioritize independent source referent/whole-part/content verification before
turning relevance into membership supervision. This remains unimplemented.
No GT-derived aliases, new other-benchmark or independent-test evidence; this
round is not a publication-ready contribution or performance gain.
See docs/experiments/2026-09-30-zoom-grounding-and-native-crops-next70-next71.md.

### next72: uncapped sources and geometry-transfer mechanism (2026-10-01)

- Preserve multimodal query tasks and frozen retained recipe; UQIS remains paused.
- LERF legal teacher eligibility now293/122/169/180, independent of source32 and
  RADIO-cache intersections. Original-resolution frame caches, explicit96x128
  first-fit raster, matched-old/coverage64/all native reconstruction controls.
- Initial64semantic pilots: figurines67.403IoU/56.947selected, ramen61.876/43.682;
  retained66.711/56.263 and61.477/43.079. Modest, not a SOTA breakthrough.
- Additional64extent evidence is not consistently beneficial: old-semantics
  selected55.225/44.336; expanded-semantics55.482/44.945. Do not extend this
  particular extent rule blindly or select per-scene winners. Completeness and
  independently established identity remain unresolved.
- NVOS full8same-video old/new geometry control79.562→83.244IoU,94.704→96.993AP.
  Retained full route83.484IoU remains stronger overall. Geometry gain is real
  but needs native/structural-map integration; no standalone-default promotion.
- ScanNet eight-scene nearest-point diagnostic53.938/54.503/67.680; described
  LESV K64transfer57.665/58.214/72.437. These are distinct readout protocols,
  not learned-model gains. Public10cohort additionally requires0200/0645;
  prepare legal sources without introducing a new implicit view cap.
- Still required before publication claims: full-cohort matching, reproducible
  arbitrary-query confidence prior, native ability retained after compact
  distillation, independent identity/whole-part evidence, multimodal regression.

See `docs/experiments/2026-10-01-uncapped-teachers-and-geometry-transfer-next72.md`.


## 2026-10-01 next73: coverage improves; old-prior transfer is not completion

Completed LERF64 four-scene macro66.098IoU/82.578AP/53.279selected,
versus retained64.882/81.021/51.835. The gain follows expanded teacher coverage;
do not present it as a novel mechanism or a compact multi-teacher student gain.
All-source SAM293/122/169/180 is complete; full native fusion remains in progress.
NVOS source-rendered prior transfer changes no geometry indices or target access,
but joint8new81.629IoU is below video-only83.244. Even unsupported completion
can introduce false membership: lack of video evidence is not positive identity.
Prioritize independent instance support for completion, not generic stronger
semantic regularization. Full controls and limitations:
[report](../docs/experiments/2026-10-01-source-coverage-and-prior-transfer-next73.md).


## 2026-10-01 next74: source writing must resolve the carrier

NVOS8same-full-video new-carrier source resolution96x128→192x256 improves
83.395→86.489IoU and97.107→98.014AP; every query improves IoU. The old-carrier
control80.110→82.347 shows both a general discretization effect and extra benefit
from improved geometry. Unknown-boundary radius scales with resolution; target
threshold, geometry and teacher tracks remain fixed. This is a stronger research
reference than retained83.484IoU, not a compact-student or four-query result.
Treat source supervision discretization as a core information bottleneck. Do not
claim novel methodology from resolution alone. A principled carrier-aware
supervision scheme would still need equal-compute and independent generalization
evidence, rather than a large collection of teacher/proposal variants.

Reference-owned source resegmentation corrected a prototype's background anchor
transport, but joint decoding, part coverage and native-image identity filtering
did not produce joint accuracy/ranking gains. Do not bundle them with the above
result. See the [next74 report](../docs/experiments/2026-10-01-reference-identity-and-source-write-next74.md).

Next74 closure: all-source LERF4 complete65.958IoU/82.631AP/53.612selected,
versus retained64.882/81.021/51.835. Coverage64 remains66.098/82.578/53.279;
full coverage has diminishing returns, so do not continue scaling teacher count
as the main research contribution. NVOS384x512pilot regresses both scenes; keep
the validated192x256 full-eight candidate86.489IoU/98.014AP. All iteration jobs
complete. Four-query new-map validation, compact student and public protocol
closure remain open; these scores do not establish publication readiness.
