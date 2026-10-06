# Preserved one-sided evidence strength and query-route ceiling

This is a fixed-map, full-four-scene LERF development control. It isolates a concrete error in the previous cross-view candidate: foreground-only evidence used numerator=denominator=foreground mass, so any nonzero overlap became confidence 1. The new `positive_scope_evidence` preserves the original per-element foreground fraction. For containment relations, only a source element with foreground/visibility at least .75 provides one-sided positive evidence; smaller overlap is unknown. The .75 cutoff is the existing consensus reliability cutoff, fixed before target evaluation. Mutual support and abstention relations are unchanged. The original field, concept prototypes, source grouping, text encoder, geometry, alpha rendering, evaluation frames and GT are unchanged.

Four source jobs and four target evaluations completed. Each target arm has 208 observations across four scenes, with fixed native-grid probability IoU/AP and the existing selected-element rendering metric. The latter is not volumetric 3D IoU. No per-scene winner is assembled into a virtual model.

| Arm, scene-equal mean | 2D IoU % | AP % | Selected-render IoU % |
|---|---:|---:|---:|
| Frozen final | 49.7408 | 68.4795 | 33.5676 |
| Previous bad one-sided candidate | 49.4534 | 68.5999 | 33.4460 |
| Preserved-strength local correction | 49.9330 | 68.4952 | 33.8634 |
| Previous all-source local correction | **50.3637** | **68.8626** | **34.1887** |
| Preserved-strength direct replacement | 48.3994 | 66.1198 | 33.4066 |

The correction recovers 0.48 point 2D IoU and 0.42 point selected IoU versus the bad candidate, yet still loses to the prior all-source local method in every aggregate metric. Scene IoU relative to the previous local method: figurines 50.7945→49.6568, ramen35.9660→35.5885, teatime60.9620→61.3070, waldo53.7323→53.1797. Hence no promotion. A same-graph composition is not warranted after this failed no-graph prerequisite.

The old frozen target masks replay exactly; source adjoint accumulation replays its old all-source cache within 2e-6 before relation filtering. All 1773 frozen source files retain their hashes. Five focused tests pass, including a test that weak positive tails cannot become certain votes. The source masks are SAM pseudo labels; the source diagnostic uses frames seen by the parent grouping and field, so it is in-sample even though each view's numerical mass is excluded for its own relation check. In-sample source gains did not predict target gains: waldo's local source diagnostic improves both probability and selected IoU, while the target IoU falls from 53.7323 to 53.1797.

## System bottleneck exposed by the query contract

The deployed scope route admits only 40 unique concepts and 132/208 evaluation observations. The remaining 76 observations cannot receive any of these scope corrections. Their frozen final mean IoU is 30.1%; admitted observations average 58.7%. This is a measured limitation of this scope module's reach, not proof that every missed concept is unknown to every teacher. The native compact response alone scores lower than the final field on the 76 unadmitted observations in all four scenes, so simply falling back to that response is not a demonstrated fix.

Among the 40 admitted concept routes, 25 have query-to-source cosine above .99 and 15 are approximate. Approximate routes include `pirate hat→santa hat` and `miffy→bunny`. Both are semantically risky; overlap with a benchmark target is not identity evidence. For the santa-hat source group, 8/9 tested source views have weak projected visibility and one is disjoint, so the cross-view candidate abstains and makes zero local changes for pirate hat. The prior all-source method changed 60,535 elements for that route and scored better on the evaluated pirate-hat instances, but that alone does not validate the underlying label. The same-caption group can mix object identities and part extents before this candidate receives it.

The strongest next step is to build and test query-independent element evidence for concepts absent from the current route, while carrying source instance/scope identity separately. Evaluate source phrases on views excluded from both grouping and fusion; check original teacher query ranking, geometry-transported teacher ranking, compact field ranking, and final selected elements under the same query/output contract. Candidate source admission must jointly guard full probability masks and actual element selection. Do not use target GT to choose a route, threshold, scene-specific arm or training label. This round does not update ScanNet OVS or NVOS, and UQIS remains paused by prior user instruction.

Full paired results and source bindings: `output/view_strength_next24/summary.json`; sealed receipt: `paper/artifacts/view_strength_next24_complete.json`.
