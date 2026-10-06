# System refinement: preserve semantics, recover identity, validate actual selection

This round uses the frozen geometry, checkpoints, input permissions and evaluators. All results are development evaluations; UQIS remains paused. Parameters are registered before scoring and no per-scene winning arms are combined. The original 1773-source snapshot remains unchanged. Isolated opt-in implementations do not replace the frozen default.

## ScanNet: bounded scope evidence is a substantial method improvement

All eight provided-unlabelled-mesh scenes and all 19/15/10 class splits finish. Baseline arrays and reported metrics replay exactly. Fixed bounded correction is `native_logits + clamp(scope_logits - native_logits, -1, 1)`, after the existing native calibration. This gives scope evidence authority over uncertain native decisions while protecting calibrated native class margins greater than two. It does not guarantee raw cosine decisions or solve object/category-union mismatch.

| Fixed arm | 19-class mIoU | 15-class mIoU | 10-class mIoU |
|---|---:|---:|---:|
| Frozen native | 39.12% | 39.98% | 51.96% |
| Calibrated native alone | 39.90% | 40.54% | 51.90% |
| Bounded scope correction | **46.45%** | **47.42%** | **60.38%** |
| Gain over frozen | **+7.33pp** | **+7.44pp** | **+8.42pp** |

All eight scenes improve in all three splits. Equal-logit averaging is a separate control: 46.60/46.31/58.24%, but regresses scene0400 on every split; it is not stitched into the bounded arm. Teacher self-consistency is diagnostic, not independent ground truth. Previously source-rejected scope scenes0070/0140 remain included and flagged: improved task scores do not retroactively certify their source supervision.

The reusable opt-in implementation is `radio_gs/querying/bounded_native_scope.py`, disabled by default. Six focused tests pass. A real scene0000 replay is bitwise identical to experimental logits and saved predictions, without reading GT. Across all eight scenes, the calibrated-margin protection audit finds zero violations. Execution, per-scene results, replay and hashes: `output/scannet_system_refinement_20260919_v2/seal.json`.

## Native feature fitting: rejected before target scoring

A separate source-only control fixes geometry, native decoder, observed domain and all other heads. It minimizes rendered source teacher residuals with a regularized linear solve, then bounds per-element updates to 10% of the original norm. Twelve original source views fit the update; four reserved original source views assess it. Because baseline initialization saw these four views, they are not an independent holdout.

| Scene | Fit MSE before → after | Reserved-update MSE before → after |
|---|---:|---:|
| 0000 | .0002940 → .0002232 | .0003386 → .0005393 |
| 0062 | .0007612 → .0006490 | .0005835 → .0009126 |

Both updates overfit: lower training residual does not establish better cross-view semantics. Neither is evaluated on benchmark GT or promoted. This rejects this particular frozen-renderer fitting recipe, not all feature-field learning. Seal: `paper/artifacts/native_reprojection_system_20260919.json`.

## LERF: distinguish missing semantics from rejected good rankings

The registered controls modify graph edges using source multi-view SAM region membership, retaining the preceding final-scope unary and structure32 graph as baseline. One arm reweights original edges, another ranks32 spatial candidates and retains16. These are local graph controls, not independently verified physical object correspondence or full-object completion.

The convex fixed-unary propagation cannot create a peak above the maximum supplied unary. Four existing queries have all probabilities below0.5; changing edges alone cannot produce a nonempty selected object at the fixed threshold. Crucially, their causes differ: `ottolenghi` has90.28% AP but maximum probability0.203 and zero IoU, while `onion segments`, `ketchup`, and `plastic ladle` have AP1.45/2.02/0.50%. The former exposes query-dependent calibration rejection; the latter require better query-semantic evidence. Uniform threshold lowering is not an adequate solution. These GT-linked findings diagnose failure only and do not fit this round's method.

Detailed controls and full-cohort results are recorded in `docs/experiments/2026-09-19-lerf-source-relation-refinement.md`.

## Next acceptance priorities

1. LERF calibration: use source-only held-out-query supervision to protect well-ranked but low-confidence concepts; evaluate actual selected elements as well as probability maps. Do not tune query thresholds on target masks.
2. NVOS identity and completeness: retain prompt-anchored exclusion gains, but separately measure lost reference SAM-positive support. A connected component is not guaranteed to contain the whole object.
3. Feature writing: require reserved-view improvement before rebuilding all query heads. Existing training residual reductions do not justify another full rollout.
4. Preserve a joint-map/four-query acceptance gate. Separate successful benchmark candidates do not yet establish one universally improved model, and these development cohorts cannot support an independent-test or SOTA claim.

## NVOS: identity-anchored spatial exclusion improves the full cohort

All eight tasks finish with the same registered spatial connected-component rule. Starting from previous positive-prompt/SAM refinement, components must connect to the original positive prompt through geometric adjacency. A subsequent80-step positive-only repair preserves supported reference scribbles; it can reintroduce nodes, so the final score field is not claimed to be a strictly connected component. Outside the reference SAM mask is never used as a negative training label.

| Fixed arm | Eight-task mean IoU |
|---|---:|
| Previous prompt/SAM refinement | 68.6122% |
| Spatial component + positive repair | **79.3864%** |
| Structure-filtered component + repair | 78.0175% |
| Structure smoothing + component + repair | 78.0519% |

The spatial arm gains10.7742pp over the previous round and28.8857pp over the original50.5007% frozen image branch. These are different comparison points, not two new gains. Four tasks improve and four regress: horns_center +23.32pp, horns_left +28.93pp, leaves +45.64pp, trex +14.73pp, but orchids -17.02pp and fern -8.36pp. Hence retain an opt-in candidate rather than replacing every query unconditionally. No per-task winning graph variant is selected.

The leaves control is especially diagnostic: original positive evidence was already satisfied; adding positive supervision alone changed nothing. Spatial identity exclusion removes broad false positives while keeping target response nearly unchanged. Conversely fern loses about20.6% of SAM-positive evidence, demonstrating that reachability from a prompt is not a sufficient completeness guarantee. The learned structure suffix's hard similarity threshold adds no overall benefit beyond spatial connectivity. A future source-admission policy must protect completeness without using target masks to choose its threshold.

All baseline binary masks and baseline task IoUs replay exactly; tiny GPU optimization floating differences are documented rather than described as bitwise identity. Detailed input rights, failed execution attempts, task metrics and runtime replay are recorded in `docs/experiments/2026-09-19-nvos-system-refinement.md`.

SAM retention alone is not a validated admission rule: trex removes17.24% of SAM-positive mass yet improves14.73pp. Thus low retention can indicate either lost true parts or removed overextended teacher support. Independent cross-view identity evidence is needed to distinguish them; a threshold selected from these eight target outcomes would not be a legitimate fix.

Additional cold-runtime validation reveals a numerical boundary: the spatial runtime reproduces the scoring-grid binary result exactly, but rerunning the baseline optimizer produces one threshold-crossing pixel after bilinear enlargement (maximum score difference2.38e-7). The joint evaluator correctly fails closed. This does not change the authoritative eight-task execution, whose baseline checks all passed; it does preclude claiming unconditional bitwise cold-optimizer reproducibility. The failed check is retained rather than relaxed.

## Final LERF cohort decision

All four scenes and208 observations, source supervision checks and actual selected renders finish. Scene-macro2D IoU/AP/selected IoU:

| Arm | 2D IoU | AP | Selected-render IoU |
|---|---:|---:|---:|
| Retained structure32 graph | 44.5466% | 63.0429% | 31.3680% |
| Source membership edge weights | 44.5396% | 63.0490% | 31.3749% |
| Source membership neighbor ranking | 44.4940% | 63.1105% | 31.4676% |

Neither new arm is promoted: the small selected gain of neighbor ranking accompanies lower2D IoU and scene regressions. Source selected macro39.2288/39.2280/39.2366% is essentially unchanged. Retain the preceding structure32 graph result. These selected-render numbers are not physical volumetric3D IoU. Full evidence: `paper/artifacts/lerf_source_relation_control_20260919.json`.
