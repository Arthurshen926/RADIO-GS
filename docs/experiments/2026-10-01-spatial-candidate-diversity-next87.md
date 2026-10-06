# Same-budget spatial candidate diversity (next87)

## Intervention

Keep all source frames, the native query space, Qwen identity judgments, tracking,
source-initialized decoder and evaluator. Replace next83's support-Jaccard>0.8
near-duplicate suppression with intersection/min>0.2 exclusion. Native pair
ranking and the16pair verifier budget are unchanged. Pair proposals still come
from the hard equivalence graph. This is a candidate-stage experiment, not a
new source-frame limit or a verified physical-instance partition.

## Full four-scene result

Selected Gaussian 2D silhouette projection IoU (%):

| Scene | next85 | Spatial diversity |
|---|---:|---:|
| figurines | 67.099 | 62.919 |
| ramen | 49.594 | 46.300 |
| teatime | 62.033 | 60.200 |
| waldo_kitchen | 57.586 | 58.511 |
| Macro | 59.078 | 56.982 |

Reject the replacement; preserve next85 research candidate/next80 stable reference.
No per-query winner assembly. Source review shows a false meat candidate for
kamaboko and a wrong waldo candidate. Stronger support exclusion discards useful
alternative views along with redundancy. It cannot substitute for physical
identity or separate view quality from instance diversity.

The source-only selection changes the spatula shortlist from zero to one
camera-covering pair, but that pair is a broad support (52424centres in camera),
not evidence of precise target recovery. Coverage counts are post-hoc diagnostics,
not used to select hypotheses. All source-frame permission boundaries are retained.

## More specific cause exposed by next86–87

Post-hoc SOURCE-crop inspection of high-native-score regions that project into
the spatula scored camera reveals repeated clear red spatulas (source frames15,
16,63,64,68). Their native scores0.1326–0.1490 exceed or rival the old leading
hard-equivalence candidates. Their hard-support intersection/min spans0.246–1.0
across the six examples (same-frame overlaps included). This is not proof of
annotation correspondence or IoU, but demonstrates that relevant-looking source
semantics were available before strict geometric candidate admission.

Next88 therefore tests soft geometric proposal association rather than further
spatial deduplication: a bounded128region native shortlist across all source
views, strongest positive-overlap other-frame partner, then16native-ranked
near-deduplicated pairs and the SAME two-view identity classifier. Geometry proposes
correspondence rather than certifying identity. No camera/GT-conditioned candidates.
The new region budget and removal of the third-view admission are explicit; this
is not a single-threshold ablation or guaranteed physical correspondence.

## Artifacts

- output/next87/summary.json
- output/next87/source_choices.jpg
- output/next87/candidate_coverage.json
- output/next86/camera_covering_source_crops.jpg (diagnostic only)

No new semantic-response, other-benchmark, compact-student or SOTA claim.
