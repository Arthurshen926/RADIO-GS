# Query calibration: source held-out query controls

The frozen native membership cutoff is a linear ridge prediction from normalized query embedding, clamped to the minimum/maximum source cutoff. It predicts an **absolute cosine value** and does not inspect the query's distribution over the current map. The existing ridge selection uses leave-one-query cutoff MSE; it does not certify held-out-query actual selected-element output. Consequently high target ranking can coexist with a cutoff above the entire map response, as previously observed for `ottolenghi`.

This experiment keeps the map, native encoder/decoder, scope head, structure graph, cameras, official output thresholds and all evaluation observations unchanged. It does not apply per-target-view min/max scaling: each query has one global map cutoff and one selected element set across views. No benchmark GT is used for fitting.

## Fixed controls and fair source crossfit

Each query's cosine mean and population standard deviation are computed over **all observed map elements**, not a minibatch and not unobserved zeros. Standard deviation floor is1e-6. These statistics are element-count weighted and therefore sensitive to Gaussian sampling density; this is a limitation, not a density-invariant score model.

Existing source-only exact-render optimum cutoffs are transformed to z units. A uses their median; B predicts z from the normalized query embedding with fixed ridge penalty1, bounded to training-source z range. Both convert back to absolute cosine via each new query's global map mean/std. Neither forces the maximum to1 nor mathematically guarantees any selected positive element.

Calibration validation leaves out an entire near-duplicate query group (connected normalized-embedding cosine>=0.95), not individual repeated query IDs. A fair absolute-cutoff ridge control uses the same groups and fixed penalty1; the currently deployed in-sample calibration is reported separately. Source teacher masks and the semantic map are not independently held out. Both normalized probability masks and actual selected-only masks are evaluated for every eligible source query-view pair, using the original confidence>=0.7 and minimum20-pixel mask rules. No parameter is selected using these results.

For integration, only queries whose existing complete final probabilities equal native probabilities within2e-5 are recalibrated; all scope-active query columns are retained. The existing structure32 graph is then applied unchanged. This avoids erasing previously effective scope inference. It also means these controls do not repair scope-active calibration failures.

## Early source evidence

Figurines195 eligible source observations: absolute heldout25.755/15.231% (probability/selected IoU); A18.582/13.817; B22.597/15.390. B slightly improves selection but damages probability IoU, so this is not joint source acceptance. All21 real figurines queries are scope-active and therefore remain unchanged.

Ramen current in-sample47.664/28.116 is compared with33.625/18.292 under absolute query-group holdout. The former uses the old selected ridge0.01 while the latter uses fixed1, so this raw gap mixes regularization with holdout; a matched ridge1 in-sample diagnostic is reported below. B improves the fair heldout control to35.602/19.744, whereas A gives29.576/18.827. The fair heldout comparison establishes a scene-dependent normalization benefit, not permission to select a different method by benchmark scene.

Complete four-scene fixed evaluation is now sealed below; neither candidate is promoted.

## Failure attribution boundary

The high-AP/all-negative `ottolenghi` case is a genuine membership-calibration diagnosis. Other all-negative queries must be separated by ranking **and support coverage**: in particular, root's independent audit finds zero observed support for the evaluated plastic-ladle target. Its near-zero AP cannot be interpreted as proof that the teacher does not recognize ladles. Score calibration cannot create missing geometry/semantic support.

## Complete source and target evidence

All208 target observations and all eligible source query/view pairs complete.

| Scene / arm | 2D IoU | AP | selected IoU |
|---|---:|---:|---:|
| figurines / final | 46.690 | 62.272 | 27.196 |
| figurines / baseline_calibrated | 47.501 | 63.257 | 30.191 |
| figurines / zmedian_calibrated | 47.501 | 63.257 | 30.191 |
| figurines / zridge_calibrated | 47.501 | 63.257 | 30.191 |
| ramen / final | 35.017 | 51.540 | 19.556 |
| ramen / baseline_calibrated | 35.230 | 52.407 | 21.012 |
| ramen / zmedian_calibrated | 35.230 | 52.407 | 21.169 |
| ramen / zridge_calibrated | 35.286 | 52.407 | 21.215 |
| teatime / final | 60.958 | 79.572 | 46.158 |
| teatime / baseline_calibrated | 61.204 | 80.696 | 47.354 |
| teatime / zmedian_calibrated | 60.207 | 80.364 | 46.615 |
| teatime / zridge_calibrated | 60.256 | 80.374 | 46.675 |
| waldo_kitchen / final | 34.032 | 54.722 | 25.836 |
| waldo_kitchen / baseline_calibrated | 34.251 | 55.811 | 26.915 |
| waldo_kitchen / zmedian_calibrated | 37.123 | 55.424 | 29.167 |
| waldo_kitchen / zridge_calibrated | 37.004 | 55.373 | 29.041 |

| Four-scene macro | 2D IoU | AP | selected IoU |
|---|---:|---:|---:|
| final | 44.1744 | 62.0265 | 29.6863 |
| baseline_calibrated | 44.5466 | 63.0429 | 31.3680 |
| zmedian_calibrated | 45.0154 | 62.8632 | 31.7854 |
| zridge_calibrated | 45.0120 | 62.8526 | 31.7804 |

`ottolenghi` improves from0 to59.79/57.36%2D IoU under A/B, without an evaluation-query special case or target fitting. This directly confirms a repairable calibration failure. But teatime regresses, macroAP drops, and most source-heldout comparisons reject the standardized schemes. The small macroIoU gains (~0.47 points2D/~0.42 pointsselected) are not a robust replacement or a large gap closure. Retain the existing structure graph and native calibration.

Matched penalty1 source comparisons (probability/selected IoU) isolate calibration holdout from old hyperparameter differences:

| Scene | absolute fit1 | absolute group-LOO | zmedian group-LOO | zridge group-LOO |
|---|---:|---:|---:|---:|
| figurines | 28.032/17.028 | 25.755/15.231 | 18.582/13.817 | 22.597/15.390 |
| ramen | 36.492/21.916 | 33.625/18.292 | 29.576/18.827 | 35.602/19.744 |
| teatime | 46.954/26.592 | 41.462/24.922 | 20.200/17.605 | 26.460/21.599 |
| waldo_kitchen | 40.951/28.383 | 38.238/27.533 | 17.398/12.298 | 19.849/15.536 |

Thus the earlier large current-fit-to-heldout difference was not entirely a query-generalization effect: the old selected ridge varies by scene (10/0.01/0.1/0.1). The matched table supplies the appropriate attribution. Query-wise source calibration remains incomplete, and first-two-moment map normalization is too crude to reliably express target scope/granularity. No target outcome selects a per-scene winner.

Checks: frozen final masks exactly replay all208 observations;61 protected scope-active query columns have zero changed selected elements (max probability replay error<5e-7). Source cutoff target native prefix/decoder/observed buffers exactly match the current maps in all four scenes. No frozen source/checkpoint changed. The initial unsupported torch mmap loading attempt failed before any fit/evaluation and is preserved.

Seal: `paper/artifacts/lerf_query_calibration_next19.json`. Both bounded candidates remain experimental and unpromoted. Next work should improve source query calibration objectives and test them separately from support loss/semantic ranking; this result is not a reason to force a positive selection or apply target-view min/max normalization.
