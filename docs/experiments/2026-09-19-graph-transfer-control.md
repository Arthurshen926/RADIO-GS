# Source-only graph transfer on frozen LERF carriers

This experiment tests a graph mechanism on the existing frozen field; it is not an exact reproduction of LUDVIG. No frozen implementation or checkpoint is modified. It compares direct source DINO lifting with the frozen learned 32-coordinate structure suffix, keeping the same native query probabilities and geometric neighbor graph. Direct DINO has 768 coordinates versus 32 learned coordinates and different training objectives, so differences do not isolate distillation alone.

## Registered first control

Two complete scenes: figurines 56 observations, ramen 71. Fixed 16 spatial neighbors among observed elements. Direct DINO uses only the 16 source-training views and exact existing front-to-back contributions. Source DINO maps are bilinearly sampled at the authority grid, normalized per pixel, then contribution-weighted into elements and normalized. No target RGB is read. Structure uses the frozen learned suffix. Each arm uses its own median positive neighbor squared distance for feature scaling and the same spatial scale, symmetric degree normalization, continuous convex solver, lambda=1, unary confidence=0.5, 64 CG iterations. Query compatibility is the full frozen calibrated native response; there are no strongest-point seeds or hard positive/negative assignments. Parameters are fixed before target scoring, not target-selected or source-fitted.

The existing `continuous_convex_v2` solver is reused. This is a soft-unary graph control, not the released LUDVIG logistic-reference/binary-edge algorithm. Low semantic probability attenuates pairwise conductance and remains a soft unary, not a certified negative label. Unknown unobserved elements stay outside the graph.

2D evaluation reuses native annotation grids and fixed 0.5 membership, including exact baseline pixel replay checks. Actual selected elements are saved before the selected-only renderer; historical JPEG GT, byte threshold10, all-scene denominator, and all fixed cameras remain unchanged. These are development controls, not independent test results or volumetric 3D IoU.

| Scene / arm | 2D IoU | AP | selected-only IoU |
|---|---:|---:|---:|
| figurines native |44.31|67.87|25.68|
| figurines native + DINO graph |44.79|68.14|27.28|
| figurines native + learned structure graph |45.02|68.49|28.94|
| figurines frozen complete final reference |46.69|62.27|27.20|
| ramen native |17.11|48.60|18.67|
| ramen native + DINO graph |14.74|49.20|15.86|
| ramen native + learned structure graph |14.26|49.43|16.06|
| ramen frozen complete final reference |35.02|51.54|19.56|

Both feature sources support small AP improvements; learned structure is not universally inferior to direct DINO. But fixed-threshold membership and actual selected output regress in ramen. Graph smoothing reduces aggregate selected elements from18,386 to17,514/16,769 in figurines and14,672 to13,711/13,123 in ramen. Thus ranking improvement does not establish a correct membership calibration or complete object support. Do not promote either native replacement.

## Explicit follow-up

After the first native-unary figurines result, a separate follow-up was registered to apply the same two graphs and unchanged parameters to the current complete final soft response. This preserves the previously effective range head and asks whether graph refinement provides additional value. This follow-up is not retrospectively presented as the original registration. Full two-scene probability and selected-output results will be sealed separately under `output/graph_transfer_final_20260919`.

## Checks and limitations

All19 existing query-conditioned diffusion tests pass. No target GT selects graph parameters. No frozen field is changed. Relationship diagnostics sample100,000 fixed-seed identical graph edges and compare DINO/structure distances and top4 neighbors; these are descriptive feature relations, not verified object identity. No claim is made that foreground-background graph edges are physically correct, that lambda=1 is optimal, or that this resolves scale/whole-part grouping. Only two scenes are evaluated; no complete benchmark-wide promotion is supported.

The completed final-unary follow-up gives:

| Scene / arm | 2D IoU | AP | selected-only IoU |
|---|---:|---:|---:|
| figurines frozen final |46.69|62.27|27.20|
| figurines final + DINO graph |47.73|62.68|29.62|
| figurines final + structure32 graph |47.50|63.26|30.19|
| ramen frozen final |35.02|51.54|19.56|
| ramen final + DINO graph |35.36|52.28|20.69|
| ramen final + structure32 graph |35.23|52.41|21.01|

Preserving the effective range head changes the conclusion: both graphs now improve both scenes on all three reported metrics. Gains are modest and are not evidence of solved object identity or large SOTA recovery. The structure32 branch is compatible with deployment without direct teacher caches; the DINO branch is a diagnostic/reference that requires a separate feature lift. A four-scene unchanged-parameter follow-up is running as explicitly subsequent development validation, not an independent test.

The fixed sampled neighbor-distance Spearman correlation between direct DINO and structure32 is0.4995/0.3555 in figurines/ramen, and top4 neighbor overlap44.47/42.72%. Learned structure has different relations, but its downstream results do not support a blanket claim that it is less suitable for graphs. Correlation is not an object-membership accuracy measure.

The reusable opt-in structure-only entry point is `scripts/audits/graph_transfer_20260919_structure_runtime.py`. Its graph construction reads only deployed geometry, the frozen latent structural suffix and observed mask; its unary is the current query's final probability. It does not open source teacher/episode/responsibility caches. A separate cold process reproduces figurines graph probabilities within3.28e-7 and has zero hard-selection disagreements. Optional int64-neighbor/float32-affinity cache is39,061,056 bytes for203,443 observed nodes (3,255,088 directed edges); it can instead be reconstructed from the map. This additional graph cost is not included in the original96-dimensional field storage claim. The larger diagnostic DINO feature bank is not required by this runtime.

## Four-scene final-unary extension completed

All208 observations complete. Baseline final prediction masks match the frozen reference exactly in all four scenes. No threshold, graph weight or neighbor count was changed.

| Scene / arm | 2D IoU | AP | selected-only IoU |
|---|---:|---:|---:|
| figurines / final | 46.690 | 62.272 | 27.196 |
| figurines / direct_dino_calibrated | 47.732 | 62.678 | 29.621 |
| figurines / distilled_structure32_calibrated | 47.501 | 63.257 | 30.191 |
| ramen / final | 35.017 | 51.540 | 19.556 |
| ramen / direct_dino_calibrated | 35.363 | 52.283 | 20.693 |
| ramen / distilled_structure32_calibrated | 35.230 | 52.407 | 21.012 |
| teatime / final | 60.958 | 79.572 | 46.158 |
| teatime / direct_dino_calibrated | 61.238 | 79.564 | 47.276 |
| teatime / distilled_structure32_calibrated | 61.204 | 80.696 | 47.354 |
| waldo_kitchen / final | 34.032 | 54.722 | 25.836 |
| waldo_kitchen / direct_dino_calibrated | 34.265 | 55.615 | 27.064 |
| waldo_kitchen / distilled_structure32_calibrated | 34.251 | 55.811 | 26.915 |

Scene-equal macros: frozen final44.174/62.027/29.686 (2D IoU/AP/selected IoU); directDINO44.649/62.535/31.164; deployed structure32 **44.547/63.043/31.368**. The structure32 graph improves all three measures in each of four scenes. It adds0.372 points2D IoU,1.016 pointsAP and1.682 pointsselected IoU on average. This is a modest, consistent development candidate, not a major performance-gap resolution. DirectDINO has slightly larger2D IoU gain but requires separate source lifting and is weaker on AP/selected macro; no per-scene winner stitching is performed.

Seal: `paper/artifacts/graph_transfer_final_four_scene_20260919.json`; initial failed native replacement and two-scene follow-up remain sealed in `paper/artifacts/graph_transfer_control_20260919.json`. No default or frozen checkpoint is changed.
