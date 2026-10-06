# Source-region compatibility does not yet supply a large range repair

This bounded experiment preserves the verified final-plus-structure32 graph and tests two source-only relation changes. No native feature, range head, query encoder, geometry, official renderer, or fixed threshold is changed. It is a development diagnostic, not independent test evidence.

## Two fixed controls

A (`boundary_soft`) multiplies the original16 neighbor affinities by reliable multi-view agreement of SAM membership vectors. Per-source-view memberships are lifted with the existing exact contribution authority. Each mask coordinate is weighted by inverse square-root area before vector normalization, equivalent to inverse-area weighting in the inner product; this reduces domination by coarse masks without treating a contained part as a negative. Only elements with above-median positive visibility mass and membership>0.5 in some mask covering at most25% of the source image provide informative evidence. A pair needs at least two informative source views. Conflicting views (at least one cosine<0.1 and another>0.5) abstain. Uncovered pairs retain the baseline edge. No region exterior is assigned a negative element label.

B (`source_rerank32`) keeps16 edges but chooses them from32 spatial candidates by the same structure/spatial/source affinity. This tests a modest increase in reach at equal edge capacity. It is **not** a global same-object graph: a32-neighbor candidate set cannot establish long-distance object identity. We must not claim this tests all forms of LUDVIG-style global feature-neighbor propagation.

All constants are declared before scoring, with no target-GT parameter fitting. Original16 neighbors are explicitly retained: cKDTree's33-neighbor query changed941 equal-distance neighbor slots relative to its17-neighbor query in figurines. The first failed replay (max probability error0.001401) is preserved under `figurines_failed_knn_ties`; the corrected experiment restores the exact previous graph, replaying within4.18e-7. This avoids attributing kNN tie differences to source evidence.

## Evidence quality

On figurines only17.26% of32-candidate edges have two nonconflicting informative source views. The fraction of original edges attenuated by more than half is0.049%. The source evidence therefore does not certify widespread strong boundaries among already-local neighbors. B changes28.49% of neighbors but increases average physical edge length only11.45%; it is a local relation reorganization, not full-object completion.

Source acceptance is measured using the existing deterministic last-four-source-view/12-phrase-per-view caption pseudo-mask cohort. These are teacher regions already participating in the pipeline, not independently confirmed physical object GT or held-out validation. Both normalized probability IoU and actual selected-only rendered IoU/precision/recall are reported. No parameter is fit to these scores.

Figurines source probability/selected IoU: previous graph44.940/30.272%; A44.958/30.277%; B44.975/30.195%. The small probability improvement of B does not improve actual element selection. Real56-query-observation2D IoU/AP: baseline47.501/63.257; A47.482/63.260; B47.385/63.519. Ranking gains do not justify replacing the retained method.

Complete fixed four-scene and source results are sealed below. Neither control is promoted.

## Why local graph refinement cannot repair missing semantics

This positive-conductance convex Laplacian solver obeys a maximum principle (up to numerical tolerance): it averages the supplied unary evidence and cannot create a new positive peak above every supplied probability. In the frozen final input, all elements are below0.5 for ramen `onion segments` (maximum0.1465) and waldo `ketchup` (0.1168), `ottolenghi` (0.2030), `plastic ladle` (0.2080). Altering neighbors or their weights cannot recover a positive selected set for these queries at the unchanged threshold. This is a direct no-GT diagnostic, saved in `unary_absence_diagnostic.json`.

Likewise, a locally coherent but semantically wrong positive island need not become correct through smoothing; cutting its boundary may simply protect its mistaken unary. Therefore the next large improvement must supply or repair trustworthy query-specific evidence (or an independently supported object identity/range), not merely increase graph iterations or edge strength. This conclusion does not rule out richer graph algorithms that add legitimate new unary evidence; it explains the limitation of the present fixed-unary controls.

The absence trace further separates calibration from ranking. All four empty queries reject concept routing and retain the native branch; native and final maxima agree within1e-6, so this particular failure is not scope takeover. `ottolenghi` nevertheless has90.28% pixel AP on its one annotated observation while max membership is0.203 and IoU is zero: the existing query-dependent membership calibration rejects a well-ranked target. In contrast, onion segments/ketchup/plastic ladle have only1.45/2.02/0.50% AP. Uniformly lowering a threshold would mix a genuine calibration defect with three poor-ranking cases. The native trace is saved separately; these annotations are used only for diagnosis, not fitting or choosing this experiment's parameters.

## Complete four-scene results

All208 benchmark observations and all deterministic source cohorts complete. Frozen final masks replay with zero pixel disagreement. No field/geometry/encoder or official evaluator changes.

| Scene / arm | 2D IoU | AP | selected-only IoU |
|---|---:|---:|---:|
| figurines / final | 46.690 | 62.272 | 27.196 |
| figurines / baseline_calibrated | 47.501 | 63.257 | 30.191 |
| figurines / boundary_soft_calibrated | 47.482 | 63.260 | 30.190 |
| figurines / source_rerank32_calibrated | 47.385 | 63.519 | 29.965 |
| ramen / final | 35.017 | 51.540 | 19.556 |
| ramen / baseline_calibrated | 35.230 | 52.407 | 21.012 |
| ramen / boundary_soft_calibrated | 35.233 | 52.419 | 21.036 |
| ramen / source_rerank32_calibrated | 35.250 | 52.410 | 20.884 |
| teatime / final | 60.958 | 79.572 | 46.158 |
| teatime / baseline_calibrated | 61.204 | 80.696 | 47.354 |
| teatime / boundary_soft_calibrated | 61.193 | 80.707 | 47.358 |
| teatime / source_rerank32_calibrated | 61.071 | 80.669 | 48.069 |
| waldo_kitchen / final | 34.032 | 54.722 | 25.836 |
| waldo_kitchen / baseline_calibrated | 34.251 | 55.811 | 26.915 |
| waldo_kitchen / boundary_soft_calibrated | 34.251 | 55.809 | 26.915 |
| waldo_kitchen / source_rerank32_calibrated | 34.270 | 55.845 | 26.952 |

| Scene-equal macro | 2D IoU | AP | selected-only IoU |
|---|---:|---:|---:|
| final | 44.1744 | 62.0265 | 29.6863 |
| baseline_calibrated | 44.5466 | 63.0429 | 31.3680 |
| boundary_soft_calibrated | 44.5396 | 63.0490 | 31.3749 |
| source_rerank32_calibrated | 44.4940 | 63.1105 | 31.4676 |

Source selected IoU macro is39.2288% for the retained graph,39.2280% for A and39.2366% for B. Neither establishes meaningful source-range repair. B gains about0.100 points selected-output IoU but loses0.053 points2D IoU, with per-scene regressions; A is effectively unchanged. Retain the previous final-plus-structure32 graph, reject either new control as a uniform replacement.

The core remaining issue is query-specific evidence rather than local edge tuning. Treat high-AP/all-negative query calibration separately from low-AP semantic ranking. Preserve unknown and occlusion semantics when developing reliable exclusion; this experiment does not supply physical whole/part truth. Do not turn missing source masks into negatives or present the modest32-neighbor reach change as global same-object propagation.

Seal: `paper/artifacts/lerf_source_relation_control_20260919.json`. All four bound episode and responsibility assets passed SHA checks. The tie-order failed attempt and its original code are retained for audit; all current baseline replays pass. No parameter search or benchmark-GT fitting occurred.
