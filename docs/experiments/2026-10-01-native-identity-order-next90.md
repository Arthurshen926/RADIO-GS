# Native identity ordering versus generative categorical margins (next90)

## Matched mechanism

Freeze next88's16pair proposal bank, all source images, cached Qwen decisions,
geometry, tracking model, scope solver and evaluator. Compare native pair cosine
ranking without categorical admission and native ranking among pairs admitted by
both Qwen views. The pair's seed view stays the higher cached Qwen-margin view,
so the first arm is NOT teacher-free. No new provider inference or target labels
are used. Both arms change13regional queries, retain54query outputs, and use the
same source initialization. This tests the division between native retrieval
ordering and categorical identity evidence, not a new field representation.

## Completed four-scene comparisons

| Scene | next88 | Native rank | Native rank after admission |
|---|---:|---:|---:|
| figurines | 67.225 | 66.631 | 66.631 |
| ramen | 49.552 | 48.033 | 47.304 |
| teatime | 62.034 | 62.137 | 62.137 |
| waldo_kitchen | 58.093 | 55.995 | 55.995 |
| Macro | 59.226 | 58.199 | 58.017 |

All numbers are selected Gaussian 2D silhouette projection IoU percentages.
Neither replacement is accepted. Native ranking recovers red spatula28.71 and
improves the handle to59.01, but selects cookware for pour-over (65.72→2.04),
wrong sake scope and a poorer pikachu extent (93.28→67.47). The two arms differ
in their selected sake candidate; filtering is not a reliable correctness gate.
Neither a native cosine nor an uncalibrated answer-token margin alone reliably
orders both referent identity and extent quality.

A launch initially failed before inference because the nested output root was
absent; creating the experiment root resolved it. All eight final evaluations
completed. No core defaults, map geometry or semantic response changed.

Artifacts: output/next90/summary.json, source_choices.jpg,
output/native_order_seeds_next90/{native,qualified_native}/lerf_*/contract.json.
No new other-benchmark, independently annotated3D, compact-student or SOTA claim.
