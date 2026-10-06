# Next60: frozen semantic scope guidance and recovery from empty membership

## Conclusion

Four controls across all four LERF development scenes completed. Frozen semantic guidance does not solve requested scope. Source-evidence initialization recovers one previously empty query, corn, but its selected projection IoU remains only 7.88%. Separate semantic response reconstruction gives tiny IoU/AP gains and lowers localization. No large performance improvement or globally promoted replacement is established.

## Mechanisms and causal limits

The next59 hierarchy chooses candidate extents using agreement with current subset rendering. We tested replacing that criterion by local soft IoU with the frozen next54 semantic response, and by the product of semantic and subset-rendering agreement. Geometry, source frames and the 120-step subset decoder stay fixed. The semantic response already includes source scope and old confidence priors; this is not independent identity evidence or a learned whole/part/contents/collection variable. Both controls remain essentially neutral.

A separate concrete failure mode is circular admission: old semantic positives qualify new source positives, so a concept missing from the map can fail to acquire evidence even when source detections exist. For queries with no observed membership above 0.5, the bootstrap candidate admits source views with a positive native foreground/context margin and a nonempty eroded source mask. It initializes membership by visibility-weighted backprojection of source probabilities, then applies the existing 120-step annealed, deployment-aligned subset objective. Nonempty queries retain next59 predictions. This combines admission and initialization changes; their individual contributions have not been isolated.

Eligibility uses an any-instance positive margin for a query-view, while masks still union source detections. It therefore does not certify every instance identity. Detector complements and boundaries are pseudo-supervision, not verified complete object masks. Of four empty queries (corn, onion segments, ketchup, plastic ladle), only corn has eligible evidence and changes. Its fit uses ten source views. The other three remain unrecovered.

The response control retains bootstrapped membership and independently reconstructs a scalar semantic query response through visibility-normalized linear inverse rendering, using 60 solver steps and prior strength 0.1. It does not use subset opacity as semantic confidence and does not retrain native feature vectors or the four-teacher compact field. Only corn changes. The solver retains consistency-based unknown handling. Existing response metadata was corrected to describe these actual solver settings, without changing predictions.

## Results

Percent; macro average over figurines, ramen, teatime and waldo_kitchen. Identical existing calibrated evaluator and geometry.

| Candidate | 2D IoU | AP | Localization | Selected projection IoU |
| --- | ---: | ---: | ---: | ---: |
| Parent next59 hierarchy | 58.087 | 74.965 | 82.245 | 48.354 |
| Semantic scope choice | 58.087 | 74.965 | 82.245 | 48.320 |
| Semantic × geometric choice | 58.087 | 74.965 | 82.245 | 48.361 |
| Empty membership bootstrap | 58.087 | 74.965 | 82.245 | 48.493 |
| Bootstrap + separate response | 58.102 | 74.986 | 81.893 | 48.493 |

Bootstrap improves selected projection by 0.139 percentage points over next59. Ramen selected macro rises 35.092→35.647; other scenes remain unchanged. Corn selected annotated-view IoUs are 12.585, 11.183, 11.681, 2.617 and 1.341 percent (mean 7.881), versus zero previously. This is recovery of a missing response, not accurate object reconstruction.

The response arm increases semantic IoU by 0.015 and AP by 0.021 percentage points, but localization drops by 0.352 points. This is a tradeoff, not a joint improvement. Keep next56 as the stable comparison and next59 as the preceding development candidate; no global defaults change.

## Remaining core problem and research direction

The system still lacks independently supported entity identity and requested extent. Reusing the existing semantic map to choose an extent does not add this missing information. Removing an old-map veto allows learning to begin, but does not make the source evidence correct or complete. Cross-view consensus among the same mistaken teacher cannot provide that guarantee.

Next method work should introduce source-grounded entity/part/container/collection hypotheses with explicit identity evidence before fusion, preserving separate extents until query intent resolves them. Evaluate that mechanism against the fixed next59 decoder and the empty-bootstrap control, rather than attributing extra optimization or another teacher-weight sweep to new semantic capability. Source supervision should score both rendered probabilities and actual selected elements. No query-specific GT aliases or per-query best-arm stitching are justified by these results.

## Execution and evaluation scope

- Frozen RGB+MoGe next52 carrier, unchanged source frame lists, source RGB and cached teachers only; target RGB/GT does not generate predictions.
- These are repeatedly used four-scene development results, not independent final test or matched public SOTA comparisons.
- Selected projection IoU evaluates hard selected Gaussians rendered onto 2D annotations. It is not true 3D-ground-truth IoU.
- No new ScanNet, NVOS or UQIS results. No new tests or audit framework.
- Post-run eligibility expression was aligned to observed support; a read-only comparison found no input query changed eligibility, so outputs remain applicable.

## Reproduction and artifacts

Scripts under `scripts/research/`: `decode_semantic_scope_next60.py`, `run_semantic_scope_next60.py`, `decode_empty_bootstrap_next60.py`, `run_empty_bootstrap_next60.py`, `reconstruct_empty_response_next60.py`, `run_empty_response_next60.py`, `summarize_semantic_scope_next60.py`.

Summary: `output/semantic_scope_next60/summary.json`.
Candidate artifacts: `output/semantic_scope_next60/{semantic,joint}/lerf_<scene>/typed_readout.pt` and `output/empty_bootstrap_next60/{bootstrap,response}/lerf_<scene>/typed_readout.pt`.
Each evaluated arm has `anchored/pipeline_calibrated/evaluation.json`; bootstrap and response directories include source fit metadata.
