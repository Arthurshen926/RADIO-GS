# Expanded-source complete semantic-field reconstruction

This experiment follows explicit user authorization to rebuild the complete field, rather than limit improvement to frozen-map readouts. It is a registered reconstruction control, not a claim that rebuilding guarantees higher scores.

## Four-scene results

All4 scenes and208 text observations are scored with the fixed native-grid and selected-render protocols. Final integrity sealing is complete: 275 artifact bindings, exact new-field cold/replay checks, and all1773 frozen source hashes unchanged; see `paper/artifacts/full_field_rebuild_next19_complete.json`. No scene/query winners are stitched; all source admissions pass, without a per-query native-harm guarantee.

| Scene | Final IoU before → after | AP before → after | Selected-render IoU before → after |
|---|---:|---:|---:|
| figurines | 46.69% → 49.59% | 62.27% → 69.73% | 27.20% → 29.66% |
| ramen | 35.02% → 35.35% | 51.54% → 52.41% | 19.56% → 19.95% |
| teatime | 60.96% → 60.55% | 79.57% → 80.48% | 46.16% → 46.97% |
| waldo_kitchen | 34.03% → 53.47% | 54.72% → 71.30% | 25.84% → 37.69% |
| **Scene-equal mean** | **44.17% → 49.74%** | **62.03% → 68.48%** | **29.69% → 33.57%** |

With the same optional structure32 graph on each corresponding field, macro IoU improves44.55→49.84%, AP63.04→69.06%, and selected-render IoU31.37→36.23%. Teatime graph2D IoU declines61.20→59.70%; the new field is retained in the denominator. Selected-render IoU is not volumetric3D IoU.

Reference category-union development diagnostics, on matched images/clicks/target cohorts, give text44.18→51.33%, image40.18→45.90%, prompt43.42→42.89% IoU. These are not official four-query results. The field is therefore not promoted as an unconditional all-modality replacement. ScanNet OVS/NVOS prior candidates are unchanged by this LERF reconstruction; UQIS stays paused.

The strongest recovery is waldo: pot0→89.03%, plate1.23→72.57%, and pour-over vessel0→58.02% IoU. Ramen final improves only0.34pp despite a larger native-path gain; full rebuilding does not remove every semantic/range bottleneck. The complete recipe uses64→90 total source training views and71,142→127,482 base updates; it is not an equal-data/equal-compute architectural ablation.

## What had actually been reconstructed

The retained baseline did have a complete source/field/readout build. Its geometry uses81/107/96/93 mapping views across figurines/ramen/teatime/waldo, while semantic training uses16 views plus2 validation views per scene. The base training budgets were16560/20304/19476/14802 steps. Native64 is initialized by source fusion then protected; structure32 and query/range components are learned. Subsequent pixel-evidence reconstruction updates native fusion and refits dependent readouts, inheriting structure from that base. Recent graph, calibration and support controls did not repeat this complete process.

Thus it is inaccurate both to say that the map was never reconstructed, and to describe recent readout improvements as joint full-field reconstruction.

## Registered change

All eligible views in the existing source32 RGB authority intersecting the frozen geometry mapping split and verified RADIO source assets are used, preserving the original two validation frames. Training counts are20/25/23/22. This is not all registered RGB, nor expanded access to benchmark images.

Rebuild order:

1. Reuse only compatible immutable2D SAM payloads; generate missing ones and a complete new manifest. If runtime binding is incompatible, regenerate instead of rewriting old metadata.
2. Build new geometry-indexed responsibilities and four-teacher source assets, episodes, source grounding and component inventory.
3. Recompute native basis/fusion and observed support; train a new structure32 field and query supervision from the new assets. Protected native64 is newly reconstructed, not copied from the old map and not freely optimized.
4. Refit image calibration, source captions, region/extent heads, rendered residuals, native thresholds, routing, query residuals and bounded fallback admission.
5. Cold-load text/reference queries and replay encodings; independently render native-grid LERF2D and the frozen JPEG selected-render protocol. Optional structure graph is rebuilt from the new field and observed domain.

Geometry, opacity and evaluation contracts are unchanged. Input count, source ordering, training steps and the last-four-training-view range-admission cohort can change. This control therefore measures a complete expanded-source recipe, not isolated architectural benefit or equal-data comparison. Original two-view validation remains fixed, but it would be false to claim every source validation cohort is identical.

## Implementation and integrity

New roots are `output/full_field_rebuild_next19/{scene}`. Shared preparation and orchestration are isolated scripts `prepare_full_field_rebuild_next19.py` and `run_full_field_rebuild_next19.py`. The1773 frozen source files are not overwritten, and old checkpoints remain at their original paths. Source-stage input roots and asset hashes must be audited; the legacy label-directory guard alone is not proof of complete information isolation.

A concrete evaluation hazard was caught: `evaluate_frozen_lerf2d_native_grid.py` hard-codes the previous field root. The new explicit-root adapter `evaluate_full_field_rebuild_next19.py` binds the new field, encoder, geometry and observed support instead. The route audit's480-width diagnostic is not substituted for native-grid official LERF2D, and reference image/prompt category-union diagnostics are not described as an official four-query benchmark. UQIS remains paused.

Original source template text sometimes says "extended to16"; actual new `teachers.json` train frames and receipts are authoritative. New source acceptance failures, if any, remain reported and are not erased to promote a high target score.

Ramen and waldo had legacy SAM logical-device bindings of `cuda:1`, so the compatibility rule regenerated all27/24 frames under `cuda:0` rather than editing metadata or reusing incompatible manifests. A source-only comparison found all top-level tensor values identical on each scene's18 common old frames (`sam_regeneration_equivalence_next19.json`). The final recipe therefore changes their source inventories without observed SAM tensor drift on the overlapping frames.

The expanded teatime run exceeded GPU memory. The successful recovery preserves the original Adam, losses, source inventory and full35,856-step budget: GPU-interpolated frozen RADIO/DINO teacher tensors are stored on CPU, and each step's selected rows are copied back to the GPU. This frees2.34375GiB; selected-row values match exactly in the dedicated check. Training completed without further OOM. Earlier sequential/chunked/CPU-Adam experiments were discarded, not stacked into the retained recipe. All failed attempts and execution changes remain preserved in the recovery receipts and `2026-09-19-teatime-full-rebuild-memory-recovery.md`. The adopted training process was reparented, so its original exit code is unavailable; completion is established from the final report, full update counts and artifact hashes rather than a fabricated return code.

Evaluation also records the existing AP definition, actual selected-element count, and target-region observed semantic support for every query. These are post-hoc diagnostics, never admission signals. `compare_full_field_rebuild_next19.py` pairs every observation with the old final and old structure-graph arm separately, including the frozen selected-render IoU. It does not stitch per-scene winners. Nonzero alpha remains an opacity-support diagnostic and cannot certify that the correct target surface exists.

A fixed calibrated-native control, using the same source-fitted query threshold and scale but bypassing extent readout, is evaluated on all208 observations alongside final and structure-graph scores. Its formula must replay the old calibrated-native score artifact before applying it to the new field. This separates changes in the fused native response from changes introduced by range readout; it is a diagnostic arm, not a benchmark-informed route switch.

Existing negative controls remain relevant: globally clipping every concept's native response and uniformly bounding every text range correction were already rejected in earlier cross-scene trials. The successful ScanNet categorical bounded residual does not justify silently applying that rejected text policy again. See the protection-boundary discussion in `2026-09-19-benchmark-closure-and-semantic-ladder.md`. The present native arm is a measurement, not automatic fallback routing or a repeated policy search.

Source vocabulary and native calibration populations are reported separately. For figurines, caption phrases grow791→919, but actual native-threshold supervised queries grow only33→34. Caption range evidence is not automatically threshold-calibration supervision. This is a continuing generalization constraint, not evidence by itself that any particular failed benchmark query is caused by calibration. The paired report records these populations for every scene.

## Completion criteria

All four scenes, full208 text observations, frozen selected-render scoring, source-admission status, image/prompt reference diagnostics and cold/replay checks must finish before reporting a final candidate. Every dependent head must trace to the new field; identical geometry and actually changed semantic/structural state must be verified. No per-scene winner stitching or benchmark threshold search.

## Scene-level diagnosis: figurines

All56 observations and all3 diagnostic arms complete, with cold/replay equality, source acceptance and full lineage checks. In the native annotation grid, native-only AP stays67.87→67.92%; final AP improves62.27→69.73% and IoU46.69→49.59%; final plus the same structure graph reaches50.43% IoU/70.61% AP, versus47.50%/63.26% before. Graph selected-render IoU improves30.19→34.55%. Reference image/prompt diagnostic IoU slightly decreases49.39→49.01% and47.47→47.38%. These are not a complete four-scene promotion.

This localizes the mean text gain mainly downstream of the native response; it does not isolate structural training from all changed range supervision. For frame00041/jake, final AP gains62.09pp while native AP changes−0.78pp; for frame00195/toy elephant, final AP gains68.24pp while native changes+0.44pp. Conversely, frame00195/bag loses30.75pp IoU and21.72pp final AP despite native AP improving0.94pp. Range readout can therefore recover native ability or still damage it, depending on the query. Miffy also loses native ranking, so not every remaining error can be attributed to the range head. No query-specific routing changes are made from these diagnostics.

Source acceptance is not per-query semantic protection. The new figurines admission report explicitly keeps `native_harm_protection_certified=false`: its source validation-all diagnostic admits37 of55 harmful candidates, and routed validation admits2 of4. The former also includes counterfactual routes. These are source pseudo-supervision diagnostics, not independent target estimates; old/new populations differ, so the rates cannot establish a controlled improvement. They nevertheless demonstrate that the remaining harmful-overwrite problem exists even when average source admission passes. The paired report retains both complete diagnostic sets.

The deployed admission is query-level: `extent_agreement_features` reduces full-element agreement to8 statistics, and `ElementRetrievalField` uses `torch.where(accepted[None], proposed, original)`. A query can need completion in one region and native preservation in another, which this binary admission cannot express on its own. This is a representational limitation, not proof that an element-wise gate will improve the benchmarks. Any follow-up must distinguish supported positive, reliably visible negative and unknown evidence on source-held-out views, and evaluate actual selection as well as probability rendering; generic global clipping has already failed.

Granularity counts must separate actual routed queries from constructed counterfactual neighbors. In new figurines, actually routed, concept-accepted source observations contain2049 aligned cases,2 selected-part cases,8 unknown and only3 disjoint cases (2 harmful). The much larger719 disjoint count in the aggregate granularity table includes synthetic neighbor proposals; it is not a deployed routing error rate. The comparison now stratifies kind, concept acceptance and scope separately. This makes limited genuine difficult-case coverage visible rather than treating synthetic negatives as proof of independent reliability.

The native-only arm includes the existing source-fitted query threshold and sigmoid scale before rendering. It therefore measures the calibrated native path, not a pure raw-feature ablation: source fusion, compression basis and calibration all change in the complete recipe. Native-grid AP after rendering cannot isolate these changes from each other.

Waldo failure separation remains necessary despite the large mean gain. In the new final readout, frame00053/ottolenghi has AP87.77% and positive support100%, yet selects0 elements and yields0 IoU: this is a selection/calibration failure, not evidence that the semantic ranking is absent. Frame00066/plastic ladle has positive semantic support0%, AP0.50% and0 selected elements despite nonzero geometry alpha100%; spatula has only9.69% semantic support and selects8308 elements but still0 IoU. Nonzero opacity cannot distinguish correct target geometry from background carriers. These cases require different remedies; neither a uniform threshold change nor semantic retraining alone is justified by all three.

Across all208 observations,32 improve final probability-render IoU while worsening selected-render IoU (figurines9, ramen6, teatime15, waldo2). Every scene improves mean selected-render IoU, but that average does not remove this output-contract mismatch. Future source admission should measure actual element selection alongside the probability-render objective. Native calibration supervision also barely expands: caption counts791/422/556/705→919/514/688/859, yet native-threshold query counts33/15/31/25→34/15/31/25. More caption supervision does not by itself repair open-query calibration.


## Completed prompt controls and decision

Two fixed four-scene,12-pass controls were completed on the rebuilt fields: fitting the existing affinity cutoffs with region-derived training scales, and fitting them with the exact fixed deployment scale. Reference prompt IoU changes42.89→40.86% and42.89→39.65%, respectively. Both worsen source validation in all four scenes and are rejected. Text/image reference observations remain exactly unchanged; geometry, latent features and unrelated heads remain unchanged. Correcting the scale mismatch alone is insufficient; the present metric and partial source-region supervision do not establish complete object extent. See [metric control](2026-09-19-prompt-metric-alignment-next19.md) and [deployment-scale control](2026-09-19-prompt-deployment-scale-next19.md). Their separate complete seals are `paper/artifacts/prompt_metric_only_next19.json` and `paper/artifacts/prompt_metric_deployscale_next19.json`.

A second teatime memory correction stores the4127×1347420 grounded probability matrix on CPU and transfers one original row at a time. It avoids a20.72GiB float GPU allocation without reducing regions, changing mask unions or changing the closed-form solver. First/middle/last transferred rows match exactly. Both memory recoveries, unavailable original process exit status, and preserved failed attempts are documented in [recovery report](2026-09-19-teatime-full-rebuild-memory-recovery.md).

All registered reconstruction and prompt-control experiments are complete. Retain the rebuilt text/map recipe as a development candidate, without automatic default or all-modality promotion. The next priority is source-validated local scope evidence and deployment-aligned actual element selection, separating missing semantic support from good-ranking/empty-selection failures. No benchmark-specific threshold fitting or per-scene result stitching is warranted.
