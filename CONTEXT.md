# RADIO-GS

The [partitioned scope supervision audit](docs/experiments/2026-09-18-partitioned-scope-supervision-validation.md) completes fourfold correspondence-isolated source validation. Full-cohort fusion IoU is38.97/51.72; strict/bridged selection and local abstention have no shared gain. Fixed group-fold oracle reaches only41.91/53.97, so current scope hypotheses remain limiting. Zero-training-support misses (8/4) are restored to the denominator. Source COLMAP tracks expose cached-label conflicts independent of MoGe/Gaussian projection, but figurines has only3 distinct positive tracks. Coarse-only native-mask dedup is fixed for new v2 caches; bounded same-weight replay removes53/8 identity substitutions, yet their minimum native IoU is98.80/99.25%, so this is not demonstrated as the main bottleneck. Existing full caches/models remain unchanged; fifteen tests pass and controls/replays finish, sealed in `paper/artifacts/partitioned_scope_controls_20260918.json`. Retained final development text IoU remains46.20/32.97; no new retrieval gain or independent-test claim.

The [scope conflict and decoder precision audit](docs/experiments/2026-09-18-scope-conflicts-and-decoder-precision.md) traces59.32/67.40 percent of source false-positive response mass to conflicting strong views, versus10.98/9.61 percent to weak-view conflicts. One-pixel core tests retain37.61/49.67 percent internal conflicts; three-pixel erosion destroys many small96x128 scopes and cannot establish boundary causality. Depth-consistent mutually visible pairs still show15.54/9.33 percent containment-pattern mismatches, not verified physical part labels. Source-selected concept bias0.2 plus freshly refit admission improves ramen text IoU32.97→34.77 but regresses figurines46.20→45.56; reject shared replacement, retain previous candidate. Native state and image/prompt observations remain exact. Fourteen focused tests, complete real evaluations and cold checks finish; evidence is sealed in `paper/artifacts/scope_conflict_repair_20260917.json`.

The [scope-stage localization and union repair](docs/experiments/2026-09-17-scope-stage-localization-and-union-repair.md) measures matched source-query loss across 2D scopes, projection, fusion and decoding. Conditional candidate IoU98.35/99.47 falls to same-view cycle85.49/93.39 and leave-view-out33.54/44.83. Pixel-union-before-lifting fixes a real noncommutativity defect in current initialization/lifted supervision; its macro effect is small. Actual contribution weighting raises uncompressed leave-view-out IoU40.26/51.93, but matched frozen-feature final-layer refits fail source acceptance and are rejected. This is teacher pseudo-mask diagnosis, not physical-identity truth or a final retrieval gain. Both repaired initializations execute,16 focused tests pass, and retained complete fields remain unchanged. Evidence: `paper/artifacts/scope_stage_repair_20260917.json`.

The [scope fusion controls and selection ceiling](docs/experiments/2026-09-17-scope-fusion-controls-and-selection-ceiling.md) reject caption-augmented fusion, fixed robust cross-view aggregation, a depth-two admission tree, and semantic-compatibility admission as shared replacements. The last improves source validation but changes real text IoU46.20→45.93 on figurines and leaves ramen32.97 unchanged; image/prompt and all parent tensors remain exact. Retain `counterfactual_extent_benefit_admission`. GT-only fixed-query selection between the current native and extent outputs reaches only48.67/36.69 versus retained46.20/32.97: this is a nondeployable binary-choice ceiling, not a bound on new candidates or local fusion. Prioritize correct cross-view candidate extents over further global gate tuning. Fourteen focused tests, complete real evaluations and cache-denied cold checks finish; evidence is sealed in `paper/artifacts/scope_fusion_controls_20260917.json`.

The [granularity and extent-benefit follow-up](docs/experiments/2026-09-17-granularity-and-extent-benefit-admission.md) completes two source-only admission controls and an area-stratified fusion diagnostic. Actual routed source queries contain almost no parent/part errors; adding co-observed nearest-concept counterfactual extents supplies missing incompatible-range examples. A query-level gate learned from actual rendered IoU benefit raises figurines full-text observation IoU/AP44.35/62.58→46.20/66.80; ramen remains32.97/49.75. All image/prompt observations and all parent field tensors remain exact. This is an explicit development candidate, not a default rewrite or independent test. Area-octave fusion fails source AP acceptance and is rejected. Thirteen focused tests, real queries and cache-denied cold loading pass; source phrase holdout applies to admission fitting only. Evidence: `paper/artifacts/extent_benefit_controls_20260917.json`.

The [pixel-evidence fusion control](docs/experiments/2026-09-17-pixel-evidence-fusion-and-extent.md) now completes matched full readout rebuilding and all real query evaluations. Conditional region membership had discarded view observation mass and counted overlapping masks repeatedly. Splitting each covered source pixel's unit evidence among proposals and lifting actual contributions improves full-text observation IoU37.92/31.62→44.35/32.97 and image38.49/26.23→49.95/29.77 on figurines/ramen, with every prompt observation exact. Geometry, source views, native64 basis and structural32 remain unchanged; dependent heads are refit. Ramen image AP regresses58.47→56.69, so this is a development candidate with a tradeoff. Applying native-cutoff calibration to all accepted extents recovers figurines ranking but regresses IoU in both scenes and is rejected as a shared replacement. Reports explicitly separate observation and category aggregation. Nine focused tests pass; identities and controls are sealed in `paper/artifacts/pixel_evidence_rebuild_comparison_20260917.json` and `pixel_evidence_extent_control_seal_20260917.json`.

All scheduled clean-source rebuilds and bounded controls are complete and verified in `paper/artifacts/clean_repair_round_seal_20260917.json`. This seals execution, not full research acceptance. Clean OpenCLIP replacement now runs the complete field/readout/query pipeline on both scenes, but regresses text/image and is not promoted. Clean explicit/MLP/hash/MoGe structural controls and fixed unified extent controls are rejected. Physical identity, dense geometry truth, shared open-vocabulary extent, all signed constraints and same-complete-method ScanNet/independent test remain unpassed.

The source-pose-coverage rebuild now completes both LERF scenes, with16 mapping-only training views and fully rebuilt fields/readouts. Under the same scoped image residual, reference image IoU/AP improves33.58/53.08→38.49/61.41 (figurines) and21.36/52.86→26.23/58.47 (ramen). Full text becomes37.92/61.47 and31.62/49.31; prompt becomes46.56/64.39 and32.94/49.20, so ramen prompt regresses and joint promotion is not established. New views help previously weak poses but can damage old-view native fusion; more inputs do not resolve evidence aggregation. Mapping split isolation is enforced; the first rejected source plans remain archived. Removing untrained structural channel-scale weighting has no shared retrieval gain in four clean controls. NVOS fortress matched corrected-query base signed improves89.10→93.86. All eight NVOS tasks now finish: fixed-branch signed IoU/AP is78.26/90.38 (base) and81.36/91.35 (image residual); most gain comes from horns_left, ordinary image regresses, and several reference signed constraints remain unmet. See the clean rebuild report for development-only boundaries.

The [clean-source native capability repair](docs/experiments/2026-09-17-clean-teacher-rebuild-results.md) now completes both LERF multiscale rebuilds and bounded controls. A source-text/image-supervised global native residual, applied to **image queries only**, improves reference image IoU/AP from25.55/45.91 to33.58/53.08 percent (figurines) and12.00/37.57 to21.36/52.86 (ramen); every text/prompt observation and retained field tensor is exact. Full-text IoU/AP remains37.64/60.46 and28.48/47.21. Shared text/image correction and shared view-balanced reconstruction fail joint acceptance. Source precision calibration fixes a cold concept-gate discontinuity without changing real-query scores. The protected base field's text/image membership terms update calibration only, not element coordinates; actual image learning is explicitly in the new residual. This is a scoped development candidate with corrected SAM inputs, not independent test or complete ScanNet/NVOS acceptance. Its verified fields and evidence are sealed in `paper/artifacts/clean_lerf_native_repair_seal_20260917.json`.

A newly confirmed [SAM3 complex-RoPE casting defect](docs/experiments/2026-09-17-sam3-complex-rope-pipeline-fix.md) invalidates treating legacy point-only SAM outputs as faithful teacher inference. `to(dtype=...)` discarded the imaginary rotary phase, including the old FP32 point-only reference-query path. Float/bfloat16-only casts now preserve complex buffers; automatic-cache contracts bind the corrected loader. Matched same-weight source inference has only 40.08% corresponding-mask IoU between corrupted and corrected phases, with exact corrected repeatability. Old scores remain recorded development evidence, not clean-teacher architecture acceptance; corrected source and query pipelines have been rebuilt; execution receipts distinguish subsequent bounded controls. Non-point-only FP32 text grounding is a separate, unaffected path.

The [NVOS scale admission and local geometry audit](docs/experiments/2026-09-17-nvos-independent-scale-admission.md) identifies a failing scene-wide ±20% scale assumption. Independently admitted per-view scales recover fortress (41/41 views). Fixed bounded local corrections admit fern 19/19, trex 34/54, horns 15/61, orchids 6/24, leaves 3/25 and flower 0/33; low-coverage and failed scenes remain explicit. On the same fern fusion-heldout sparse observations, coverage changes 94.84→96.18% and depth p90 23.69→5.23%; dense truth and final retrieval remain separate gates. The [complete fortress/fern rebuild](docs/experiments/2026-09-17-nvos-geometry-rebuild-and-real-retrieval.md) now finishes real query evaluation: fern base image IoU 67.67→71.03 and image residual 66.55→72.66; fortress base image/signed 80.12/85.91. Cold native encoding and support-retention checks execute; FP16 encoder batch numerics are reported separately, not claimed bit-identical. A separately bounded 2.0 local-scale control recovers flower 33/33 and orchids 24/24 views, but those candidates have no new final retrieval scores. Source-region prompt validity is now explicit: missing signed seeds cannot crash the scene or become fabricated prompts; missing positive geometry cannot become an all-negative image task. See execution receipts, not stale scores, for completion.

The [joint native supervision control](docs/experiments/2026-09-17-joint-native-supervision-and-extent-handoff.md) improves native-fallback text IoU from 33.60/8.20 to 35.32/22.72 on figurines/ramen, but the complete readout still fails joint acceptance, including after source-only recalibration. Keep the prior complete text model and image-only candidate. Actual MoGe encoder/refiner structural training with matched cosine loss also regresses registered-prompt retrieval; bounded direct-feature distillation is rejected, rather than left as an unexecuted extraction-only idea.

The [NVOS source-support investigation](docs/experiments/2026-09-17-nvos-native-field-and-signed-query.md) identifies source raster undersampling as a real semantic coverage loss on fern. At identical frozen geometry, rebuilding source contributions at 384×512 and measuring native source tiles on previously unsupported rows raises target foreground support from 90.96% to 98.00%, and base image IoU/AP from 67.67/83.78 to 70.58/87.85. Signed reference-image conditioning reaches 71.78 IoU (71.94 with the image residual). Existing rows and text/prompt validity domains remain unchanged; image rendering requires its own support denominator. This is a single-scene development control, not full NVOS or complete-method joint promotion.

The [native image retrieval residual](docs/experiments/2026-09-17-native-image-retrieval-residual.md) is a new development candidate on the retained complete field. A bounded global native-space decoder residual trained through actual sigmoid-before-raster source image retrieval improves reference image IoU/AP from 33.12/54.52 to 39.41/57.66 in figurines and 13.81/34.21 to 19.91/47.68 in ramen. Sharing that image-supervised correction with text regresses text. The explicit image-only candidate therefore retains original text/prompt paths; cold evaluation matches their observations exactly. No per-element state, geometry or original parameter changes. Architecture selection was development-informed; this is not independent test or joint benchmark promotion. Region-mean feature fitting and recalibration controls do not achieve the same shared gain.

The [system controls and structural retention report](docs/experiments/2026-09-17-system-controls-and-structural-retention.md) completes the scope-consistent full reconstruction and bounded explicit/MLP/hash/MoGe/feature-assisted-geometry controls. None supersedes the complete control jointly. An affine-depth normal transport bug is fixed with separate geometry rebuilds; it is not the main observed score deficit. Actual MoGe intermediate extraction and OpenCLIP base-field replacement run, but do not establish independent teacher superiority. See the checklist execution records for unresolved acceptance gates, rather than interpreting executed rejected controls as a completed architecture.

The [bounded image extent trial](docs/experiments/2026-09-17-bounded-image-extent-and-calibration.md) connects image queries to the existing structural extent bank with a bounded native-logit correction, selected on source validation. Reference-image IoU/AP changes 33.12/54.52→35.41/55.63 in figurines and 13.81/34.21→13.56/34.45 in ramen; text/prompt observations and every parent tensor remain exact. It is an opt-in candidate with mixed outcomes, not a promoted joint baseline. A source-fitted image-query threshold regressor is rejected. All reference-query projection ceilings exceed the original cutoff, ruling out universally unattainable thresholds; nori nevertheless has only seven above-threshold elements despite high image AP. Scope selection and calibration generalization remain open. Thirty-five related tests pass; no new field training or geometry changes occurred.

The [source scope and correspondence control](docs/experiments/2026-09-16-source-scope-and-correspondence-consistency.md) rejects uniform fine/coarse mask assignment as a native-retrieval replacement. Current training correspondence graphs contain 12/68 same-view nonduplicate scope pairs, including 10 disjoint pairs in ramen. An opt-in compiler control permits near-duplicate proposals but blocks inconsistent same-view scopes across merged tracks, retains 64/73 and 199/230 edges, and rebuilds both scene episode bundles. This is source-supervision consistency, not verified physical identity or a retrieval improvement. Older strict correspondence admission already existed and previously regressed; it was not enabled in the retained training lineage. New full training, episode-dependent grounding/initialization rebuild and real-query acceptance remain pending. Mainline is unchanged; 41 related tests pass.

The [local geometry repair and native-field rebuild](docs/experiments/2026-09-16-local-geometry-calibration-and-native-rebuild.md) implements bounded spatial/depth-aware source-track calibration and consistent normal warping. On ramen points excluded from every new fit and views excluded from surface fusion, depth AbsRel median/p90 improves 5.82/10.91→3.12/6.15 percent at unchanged 99.90 percent coverage. Existing calibration/reference dependence still limits independence. Figurines loses coverage. Fresh contribution authorities and native64 fusion were built for both geometries: native-path text IoU/AP changes 8.20/30.97→11.76/31.82 in ramen and 33.60/58.48→31.45/56.89 in figurines. These are diagnostic native paths, not complete-method gains. Hard/soft source-consensus cleanup and removing recovered-scale sources are rejected. Complete mainline remains unchanged; 31 related tests pass. G2/G3 remain partial; reliable scope/granularity supervision is still required.

The [first next-stage geometry execution](docs/experiments/2026-09-16-geometry-distribution-and-expanded-track-audit.md) expands validation beyond the sparse semantic-source track overlap. On fusion-heldout views with majority observed semantic support, Gaussian depth AbsRel median/p90 is 3.13/29.56 percent in figurines (262 observations, 239 points) and 5.73/10.35 percent in ramen (4816 observations, 4166 points). Reference tracks remain sparse and old calibration is not independently held out. Maximum-parallax filtering does not remove the tails. Discrete contribution and native-sampling controls do not explain away transport error. New source reprojection scale fitting improves over median-ratio fitting but worsens matched old-MoGe tail/mean; it is rejected. Geometry and retrieval checkpoints are unchanged; no new final retrieval scores. G1/G2 remain open.

Next-stage work is tracked in the [repair and architecture checklist](docs/experiments/2026-09-16-next-stage-repair-checklist.md), accepted for planning by the user on 2026-09-16. Priorities are geometric transport, identity/granularity-consistent supervision, and native-semantic-preserving extent readout. Functional teacher substitution, explicit/hash structural residuals, and MoGe geometry-feature distillation are bounded controls, not adopted implementations. Feature-assisted geometry has an explicit bounded ADR 0008 preprocessing exception and a completed rejected control; semantic field training remains geometry-frozen. The checklist does not promote a new checkpoint or report new scores.

RADIO-GS reconstructs a query-independent compact Gaussian scene field for reusable scene understanding. The historical canonical field reconstructs RADIO features; the current retrieval-field candidate uses optional foundation-model teachers and does not require RADIO. This glossary defines the small set of domain terms that active method and benchmark work should share.

The [source-track correspondence audit](docs/experiments/2026-09-16-source-track-correspondence-and-scale.md) revisits the assumption that geometric transport is settled. On 674 strict source-only ramen COLMAP points (1,684 directed pairs), Gaussian contribution-centroid reprojection median/p90 is 2.45/6.28 pixels on the 128-wide grid, versus 0.39/0.74 for a snapped-ray true-depth control. This diagnostic includes carrier/pixel mixing and is not exact distribution transport or final IoU. Native teacher masks also conflict on 261/2,760 positive shared track-concept pairs. Figurines has too few strict shared points for general conclusions. A new source-only scale fit, globally point-held-out, improves median but worsens matched MoGe p90/mean and is rejected. Mainline and geometry remain unchanged; inspect full contribution transport and depth-edge/tail failures before further semantic loss tuning.

The [partial-negative supervision controls](docs/experiments/2026-09-16-partial-negative-supervision-controls.md) test abstention on conflicting source observations in both lifted-3D and actual rendered-pixel supervision. All four candidates are rejected. Rendered abstention improves over its matched continuation by only 0.13/0.04 text IoU points, and remains below the retained figurines mainline (40.80 versus 43.52; ramen 28.61 versus 28.42). Detectable conflicting negatives are only a subset of possible teacher errors; agreement does not prove complete scope. Native/query/routing state and independent region responses remain exact. Stop tuning these abstention controls; source scope completeness and correspondence require direct evidence.

The [visibility-weighting and depth-transfer audit](docs/experiments/2026-09-16-visibility-weighting-and-depth-transfer.md) finds that equal-view membership fusion gives tiny rendering contributions a full vote. Contribution weighting improves source cross-view mask IoU from 43.20/43.81 to 46.61/52.73, but matched concept-head refits remain below the retained mainline (42.02/27.88 versus 43.52/28.42 text IoU). Both refits are rejected. On the same calibrated-source-view subset, direct depth mask transfer obtains 42.42/53.57 versus weighted Gaussian 46.41/49.75, with different pixel support; it is a diagnostic using target source depth, not a deployable score or independent geometry proof. Source mask completeness, scope consistency and geometric correspondence remain unresolved.

The [source-stage and cross-view mask audit](docs/experiments/2026-09-16-source-semantic-stages-and-cross-view-masks.md) finds severe source mask transfer loss even without semantic features: same-view cycle IoU 87.88/93.81 versus leave-view-out 43.20/43.81 (figurines/ramen), despite positive support 95.32/96.57. These are teacher pseudo-label diagnostics, not independent geometry accuracy or final test. Source region coverage, teacher scope consistency and geometric correspondence remain confounded. A render-consistent native feature inversion reduces source MSE by about 73/72 percent but worsens real native AP from 58.49/30.97 to 44.68/22.99; it is rejected. The retained complete candidate remains `exact_render_query_threshold96`, unchanged. Repair and validate source cross-view correspondence before further feature reconstruction optimization.

The [native support union and calibration audit](docs/experiments/2026-09-16-native-support-union-and-calibration-contract.md) rejects unconditional native-IoU-improving candidate unions: the proxy can improve while true object IoU collapses. Precision-preserving unions give mixed results and remain controls. It also fixes source query calibration to use the deployed sigmoid-before-raster order instead of thresholding rendered mean cosine. The correctness-repaired single-candidate checkpoint `exact_render_query_threshold96` obtains text IoU/AP 43.52/62.91 and 28.42/42.81 (figurines/ramen), effectively unchanged; image/prompt observations remain exact. This mismatch was real but small in current measurements. Reliable native support and source object correspondence, rather than more candidate unions, remain the next primary issue.

The [matched compression and completion repair](docs/experiments/2026-09-16-native-compression-and-ranked-completion.md) reconstructs the current native source fusion before projection: removing 64D compression improves mean AP by only 0.63/0.94 points (figurines/ramen), while the previous structural completion severely damaged jake/waldo ranking. The new `native_within_extent96` text candidate retains the selected membership partition and uses native ordering within each stratum: IoU/AP 43.50/62.93 and 28.42/42.81 versus preceding 43.39/59.77 and 27.76/40.23. Image/prompt observations remain exact. A source-fitted admission threshold and a globally rank-preserving native cutoff are rejected due IoU regressions. Fine-grained native weakness already present before compression and incorrect selected extents remain separate unresolved problems; this is development evidence, not joint benchmark promotion.

The [source scope gate repair](docs/experiments/2026-09-16-source-scope-gate-and-hard-negatives.md) identifies a supervision gap: actual admitted source pairs contain almost no incompatible extents, while arbitrary pair training does not change real query decisions. Source-only nearest compatible/incompatible co-observed mask pairs train a two-feature veto without changing parent field tensors. The new text development candidate `hard_source_scope_gate96` achieves IoU/AP 43.39/59.77 and 27.76/40.23 (figurines/ramen), versus preceding 42.28/57.75 and 26.30/38.15. It rejects bag→table and chopsticks→noodles while preserving valid alias mappings; corn regresses to zero. Image/prompt observations remain exact. This is a development improvement, not independent final-test or joint benchmark promotion; weak native fine-grained ranking and harmful structural completion remain unresolved.

The [native identity / extent completion investigation](docs/experiments/2026-09-16-native-identity-extent-completion.md) finds that replacing all extent selection with native mean cosine or native-support overlap is not reliable. Preserving the existing source-accepted concept readout and completing only rejected queries with native-support/extent overlap yields a new development candidate: text IoU 42.28/26.30 versus 36.51/25.10 (figurines/ramen). AP is 57.75/38.15 versus 58.97/37.84, so the previous common baseline remains available and there is no unconditional promotion. All reference image/prompt observations remain identical. Remaining primary failures are false semantic gate acceptance, concept-dependent native support/calibration, and ranking damage from hard structural completion; source-accepted does not mean verified correct.

The [cross-view selection and historical comparison](docs/experiments/2026-09-16-cross-view-selection-and-historical-comparison.md) implements source-supervised candidate ranking with candidate-origin and episode-target view exclusion. It improves the rejected independent-region text control, but neither that ranker nor the combined concept/region candidate ranker supersedes the retained residual extent candidate jointly. The combined ranker obtains text IoU 35.76/26.50 versus retained 36.51/25.10 (figurines/ramen), with unchanged image/prompt observations. Historical retained LERF3D binary predictions rescored on the current 480-wide GT grid obtain 57.17/30.61; scoring is aligned but source/calibration/rendering contracts are not. Current repair is substantially better than the recent broken version, not proven better than the historical normal performance line.

The [routing, extent and region-selection repair](docs/experiments/2026-09-16-routing-extent-and-region-selection.md) retains a source-render-supervised residual extent readout over the unchanged 96D field. Full-text development IoU/AP improves from 35.37/57.34 to 36.51/58.97 in figurines and 23.09/37.24 to 25.10/37.84 in ramen, with exactly unchanged reference image/prompt observations. This is a modest improvement, not a new major breakthrough. Source routing controls and higher-resolution masks do not improve both scenes. Independent region extents preserve greater diagnostic mask capacity, but actual native-similarity selection regresses; they are rejected. GT-selected unrestricted versus concept-restricted region diagnostics localize remaining failures to concept access and within-concept ranking. Those oracle results are not deployable scores, training labels, or joint benchmark evidence.

The [source-discovered grounded transfer experiment](docs/experiments/2026-09-16-source-discovered-grounded-transfer.md) preserves every parent 96D field tensor and fits a source-only global concept-extent readout with native-query gating and exact fallback. Full-text development IoU improves from 33.63/8.59 to 35.37/23.09 percent (figurines/ramen); AP changes from 58.49/30.97 to 57.34/37.24. Image/prompt observations remain exactly unchanged. This is a scoped text candidate selected through development comparisons, not a universal open-vocabulary solution or joint benchmark promotion. Nearest-prototype semantic routing remains unreliable (e.g. chopsticks to noodles); source-discovered crop fusion alone, structure32 replacement and the MLP alternative are not retained as joint improvements.

The [source evidence and query-boundary investigation](docs/experiments/2026-09-16-source-evidence-and-query-boundaries.md) rejects query-free-only, view-balanced and dense local crop anchor replacements as joint improvements. A source-only query-conditional text boundary fitted from complete source teacher masks improves full-text IoU from 33.14/4.07 to 33.63/8.59 percent (figurines/ramen), with nearly unchanged AP and exactly unchanged image/prompt responses. This is a limited membership-readout improvement, not recovered fine-grained semantic localization or joint benchmark promotion. The larger ramen gain from partial-label calibration regresses figurines and must not be cherry-picked per scene.

The latest [registered structural query repair](docs/experiments/2026-09-16-registered-structural-query-repair.md) replaces the learned prompt adapter with direct signed structural affinity for registered points. On the same development cohort, prompt IoU/AP improves from 21.60/40.28 to 40.71/57.83 in figurines and 21.88/32.92 to 24.77/51.94 in ramen (percent). Every parent tensor and text/image response is preserved. Additional affinity training and automatic native-seed text/image extension are rejected. Complete object extent, semantic seed reliability, and joint benchmark validation remain unresolved; individual objects can regress substantially.

The [full retraining of the preserved source inventory](docs/experiments/2026-09-16-preserved-inventory-full-training.md) is complete for both scenes and all three development query protocols. Native semantic preservation and cold-load checks pass; text ranking improves, but membership and prompt results are mixed, so the model is not promoted jointly. Raw image calibration is all-negative and requires the explicitly evaluated source-calibration stage.

The latest [source inventory repair](docs/experiments/2026-09-16-source-inventory-preservation-and-group-diagnosis.md) fixes grounded-region augmentation that discarded original training regions, and inconsistent visibility gating of lifted labels. Both scene inventories and episodes are rebuilt; exact-common-support native fusion improves, and full retraining is reported above. Do not reuse old inventory-bound supervision artifacts unchanged.

The current system-level investigation is [native semantic preservation versus structural learning](docs/experiments/2026-09-16-system-level-native-structural-control.md), with the experimental coordinate-ownership contract in [ADR 0009](docs/adr/0009-protect-native-semantics-during-structural-learning.md). A protected 64-coordinate semantic prefix and trainable 32-coordinate structural suffix share one element vector. Exact native retention is verified; text/image structural membership inference and joint benchmark promotion remain incomplete. SAM currently supplies mask/contour supervision, not native feature reconstruction. Do not equate four enabled teacher names with four query-accessible retained capabilities.

The [structural membership follow-up](docs/experiments/2026-09-16-structural-membership-and-related-work.md) is implemented and evaluated but rejected: a frozen native field can still lose final query ranking through an unrestricted learned membership readout. Native parameter preservation and retrieval preservation are separate acceptance requirements.

The current source-support and three-query repair is documented in
[the component pipeline report](docs/experiments/2026-09-15-component-pipeline-repair.md).
Its unified source regions are a development candidate, not a promoted joint
benchmark baseline. Independent unit-region targets can be incompatible on
overlapping elements; native direction constraints remain experimental.
The [reference-query contract audit](docs/experiments/2026-09-15-reference-query-contract-and-reliability.md)
distinguishes category-union development scores from verified instance tasks.
Source-region query coverage and cross-view correspondence admission are
separate from auxiliary teacher coverage.

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
The joint RADIO-GS objective covering LERF-2D, LERF-3D, ScanNet OVS, NVOS, and Available-Nine SPIn-NeRF with one method family.
_Avoid_: six-task program, UQIS program

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
