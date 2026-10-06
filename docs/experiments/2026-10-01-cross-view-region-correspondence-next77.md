# Cross-view region correspondence before native writing, next77

## Implemented mechanism

Next76's scale slots did not establish region correspondence and failed
replacement. This iteration uses projected support rather than area rank.

For every legal source image, project every native SAM region onto the frozen
Gaussian carrier with the full alpha-contribution adjoint. Normalize each
Gaussian's regional contribution by its total visible contribution in that
image. All nonzero contributions are retained. Matching uses192x256 projection,
the same as next75's native writing. No query text or evaluation annotation is
used to build these correspondences.

For each source view A, choose two distinct neighboring source cameras B and C
using translation distance normalized by the scene's median camera separation,
plus viewing-direction distance. No mask or query score chooses these views.
Use A/B common visibility, weighted by the geometric mean of their alpha masses,
to compute regional soft intersection and both marginal supports. Pick B's
maximum symmetric-Dice region for each A region. Record directional coverage too:
partial containment must not be interpreted as equal range merely because one
direction overlaps well.

C does not participate in the A/B assignment. Confirm the pair by requiring both
members to have support for the SAME region in C: maximum over C regions of the
minimum A/C and B/C Dice. The write confidence is the geometric mean of the A/B
Dice and this confirmation. If a necessary shared support is absent, the region
retains its original weight1; missing visibility is unknown. C is an independent
observation for this local decision, not an independent dataset/test split, and
nearby images remain correlated.

Use this query-independent confidence to reweight descriptors BEFORE the
per-pixel native-region mean, then perform unchanged alpha fusion into the
single1536D map. If all covering region weights are zero, retain the original
mixture. Uncovered areas are not manufactured negatives. Geometry, full source
cohort, encoder, native-space query matching, calibration and scope inference
remain fixed. First freeze selected membership; then use the unchanged extent
decoder to measure actual selected-Gaussian projection.

This is correspondence-weighted observation fusion, NOT verified object IDs,
an explicit instance graph, or a complete solution to whole/part ambiguity.
Averaging still happens after weighting, and geometric agreement can consistently
support a semantically incorrect teacher region. Neither transitive component
merging nor unobserved-surface completion is introduced.

## Artifacts and scope

- `scripts/research/build_region_correspondence_next77.py`: full-source regional
  projection, A/B assignment and third-view confirmation.
- `scripts/research/rebuild_expanded_semantic_next72.py`: optional region weights;
  no-argument default behavior is unchanged.
- `scripts/research/run_correspondence_write_next77.py` and
  `decode_correspondence_next77.py`: frozen-protocol execution.
- `output/region_correspondence_next77/{scene}/correspondence.json` and
  `weights.pt`: source-only correspondence evidence and confidence.
- `output/next77/summary.json`: completed retrieval results.

All results are development evidence. Selected projection is not independent3D
GT evaluation. No NVOS/ScanNet/UQIS, compact-student or four-query completion
claim is made. No benchmark-specific aliases or scorer changes are introduced.

## Correspondence construction coverage

| Scene | All source views | Region observations | No joint visibility (neutral) | Construction seconds |
|---|---:|---:|---:|---:|
| figurines | 293 | 24486 | 184 | 67.2 |
| ramen | 122 | 8719 | 202 | 36.1 |
| teatime | 169 | 11681 | 1275 | 53.7 |
| waldo_kitchen | 180 | 13009 | 1251 | 57.1 |

The table describes source evidence, not correspondence accuracy. No independent
instance labels were used to measure precision/recall of these assignments.
The third-view check has not been separately ablated from pair-only weighting;
any observed gain belongs to the whole intervention.

## Completed four-scene results

| Scene | Previous IoU | Correspondence IoU | Previous AP | Correspondence AP | Previous selected projection | Recomputed selected projection |
|---|---:|---:|---:|---:|---:|---:|
| figurines | 68.517 | 68.716 | 87.626 | 87.824 | 59.432 | 59.409 |
| ramen | 62.609 | 62.847 | 78.300 | 78.321 | 44.018 | 43.748 |
| teatime | 73.031 | 72.966 | 90.202 | 90.423 | 58.333 | 58.444 |
| waldo_kitchen | 63.563 | 63.417 | 80.456 | 79.759 | 55.386 | 55.673 |
| **Macro** | **66.930** | **66.987** | **84.146** | **84.082** | **54.292** | **54.318** |

All8evaluations completed: four semantic-only controls with frozen memberships,
then four unchanged-decoder selected-set evaluations. The former preserve the
old selected scores exactly. No scene/query-wise best-result stitching.

**No promotion.** IoU and selected-projection macro changes are negligible,
AP falls slightly, and two scenes lose selected accuracy. Keep next75 as the
current complete research candidate. This is not a statistically established
improvement, a SOTA gain, or a solved correspondence/identity problem.

## Method conclusion and next boundary

This experiment addresses *which regional observations receive more weight*.
It does not preserve distinct region identities through fusion, nor independently
establish whether a region contains the requested object or only part of it.
Geometrically repeated semantic errors remain possible. Together with next76,
these controls show that area slots and this geometric-confidence reweighting
are insufficient replacements for the retained method; they do not prove that
all region-level or graph-based representations must fail.

Further work should retain native region descriptors plus sparse Gaussian
support and directional containment as auxiliary, queryable evidence. Candidate
scope inference must distinguish equivalent observations, part containment and
unresolved overlap; query compatibility must be checked before transferring a
coarse region's score to all its elements. Evaluate that mechanism against the
current native field on the same geometry and output contract, and validate
selected members as well as response maps. This larger representation/inference
change is still OPEN, not implemented by the scalar-weight prototype above.

Generic unseen-query calibration, compact multi-teacher rebuilding and same-map
four-query validation also remain open. NVOS/ScanNet/UQIS status is unchanged.
