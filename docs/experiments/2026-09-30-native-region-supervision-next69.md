# Next69: native region recognition helps ranking but does not certify target extent

## Result and decision

Four full four-scene controls completed after building new source-side recognition evidence. Supplementing semantic scope inference improves AP **81.021→83.482** and localization **87.328→89.091**, but IoU **64.882→64.485** declines. AP improves on every scene. Three membership-supervision variants do not beat next68 selected51.835. There is **no joint replacement or major overall performance gain** this round. Preserve the existing candidate; keep the ranking improvement as a research result, not a promoted complete model.

This is existing-source evidence expansion, not expanded-view teacher reconstruction. Geometry/native map features remain next66, and all source images belong to the original20/25/23/22teacher-frame cohorts. Additional native encoder compute is explicitly used. No target RGB/masks train the candidates. Existing query texts remain runtime inputs, not a new closed vocabulary used for map training.

## Method change

The previous evidence path requires the text detector to produce a mask. The new path starts from existing **query-independent source region masks** and their native1536D SigLIP descriptors:

1. For each source view and runtime query, select the region with highest native query/region cosine. Each query is scored independently; no competition against the benchmark's other query texts.
2. On the source RGB, encode the selected region's foreground and complement in a common crop window. Admit the candidate only if foreground similarity exceeds complement similarity for that same query. This is a relevance criterion, not verified identity.
3. Add admitted masks to the original detector evidence. Reuse other-view geometric positive admission and the source-based initialization from next68, then optimize membership on all semantic-observed elements for240steps.

All queries received source recognition computation. The cold control supplements only queries that lacked eligible views under the predeclared next67 source-fit history. It does not select queries based on annotation scores. The augment control supplements every query. Supplemental masks are unioned with the original candidates; this remains a potentially ambiguous union of extents.

Of14previously untrained queries,11now become eligible under the cold control: figurines pikachu/tesla door handle/waldo; ramen corn/hand/kamaboko/sake cup; teatime dall-e brand/paper napkin; waldo_kitchen ottolenghi/pour-over vessel. Onion segments, ketchup and plastic ladle remain without eligible supervision. Eligibility is not evidence of correct identity or scope.

Evidence generation accepted350/271/280/309view-query pairs; scene runtimes were47.5/44.4/41.9/40.5seconds on the current machine with cached regions/features. These timings exclude original teacher/map preparation and subsequent fitting, and are not a general efficiency claim.

## Full recipe results

Four-scene macro percentages:

| Recipe | 2D IoU | AP | Localization | Selected projection IoU |
|---|---:|---:|---:|---:|
| Next68 candidate | 64.882 | 81.021 | 87.328 | **51.835** |
| Native masks for untrained queries (`cold`) | 64.882 | 81.021 | 87.328 | 51.641 |
| Native masks for all queries (`augment`) | 64.882 | 81.021 | 87.328 | 51.576 |
| Cold native masks without supplemental negative loss labels (`positive`) | 64.882 | 81.021 | 87.328 | 50.701 |
| Native evidence in semantic scope inference (`semantic`) | 64.485 | **83.482** | **89.091** | 51.414* |

*The semantic arm deliberately inherits next67 membership to isolate the semantic change.51.414is unchanged, not a newly trained selected-set result and not a comparison claiming selection degradation caused by semantic inference. No best-of-arm stitching is reported as a combined model.

The semantic control starts from the same next66 native map response, adds the new candidates to the60step source scope solver and applies the same source-visible-mass confidence transport from next53. It changes semantic spatial ordering; it does not retrain the underlying native features. The confidence transport is retained dependence on an older response distribution.

| Scene | Original IoU / AP | New semantic IoU / AP |
|---|---:|---:|
| figurines | 66.711 / 84.722 | 68.393 / 88.196 |
| ramen | 61.477 / 76.117 | 61.262 / 77.098 |
| teatime | 71.328 / 88.146 | 69.117 / 92.082 |
| waldo_kitchen | 60.012 / 75.097 | 59.167 / 76.551 |

This establishes a ranking gain for the tested evidence path. It does not establish whether the remaining IoU loss is entirely calibration, incorrect extent, or both. No benchmark thresholds were tuned to turn that tradeoff into a reported win.

## Concrete gains and failures

Query-average selected projection scores, next68→cold:

- pikachu57.58→75.89; waldo8.37→14.46; kamaboko12.18→14.18; sake cup12.24→16.52.
- corn20.36→7.64; dall-e brand21.85→10.37; paper napkin39.79→32.71; ottolenghi81.97→74.49.

These show why source-supervision absence is not itself permission to overwrite the map. Ottolenghi was already retrieved well without range-fitting views; native fallback had useful capability. A positive region similarity does not establish requested object identity, whole/part extent or reliable negative evidence outside the mask. New candidates can mix incompatible extents and erase existing capability.

To isolate one such risk, the positive control requires original detector/temporal foreground presence and its complement before a negative pixel enters the subset loss. Supplemental regions supply no new negative loss pixels. It still uses detector complement evidence inside the preexisting cross-view qualification and seed statistics; this is **not** a completely negative-free reconstruction algorithm. Removing those new negative loss labels still lowers selected IoU to50.701. Therefore simply treating new regions as positive-only is insufficient; positive identity/extent itself remains uncertain.

The modest recognition gain for some names does not justify using native top1regions as universal object masks. Nor should the failed controls be hidden by selecting winners per query.

## Research conclusion

The tested native source recognition path supplies useful ranking information and bypasses some missing text detections, but its inferred scope is not reliable enough to serve as full-object supervision. The main unresolved method question is how to distinguish **semantic relevance** from **evidence that this exact region is the requested entity and extent**.

The next substantial experiment should compare source-only entity/part verification or independent source segmentation against this relevance-only proposal path, then check cross-view identity and extent agreement. More source views remain an untested option; repeated views of the same wrong semantic region would not solve this problem. Neither another loss over the same labels nor automatic pseudo-negative completion is currently supported by the controls.

For the paper, do not present this as a new SOTA system or a completed four-teacher universal field. The new evidence is SigLIP-specific in this experiment, uses runtime source-image access, and has been checked only on reused LERF development scenes. No new image/prompt-query, ScanNet, NVOS, UQIS or independent3D-GT result was produced. The selected metric is2D annotation IoU after selected-only3D Gaussian rendering.

## Artifacts

- `scripts/research/build_native_proposals_next69.py` and `run_native_proposals_next69.py`: source evidence.
- `scripts/research/run_native_membership_next69.py`: cold/augment membership controls.
- `scripts/research/run_native_positive_next69.py`: supplemental-negative-loss exclusion.
- `scripts/research/run_native_semantic_next69.py`: semantic scope and confidence-transport control.
- `scripts/research/summarize_native_proposals_next69.py`.
- `output/native_proposals_next69/summary.json`, `cold_query_results.json`, and per-scene fit files.

All4evidence jobs and16scene-level evaluation recipes finished, with no failed subprocess. Existing artifacts were not overwritten. No defensive tests or audit framework were added; validation was the actual source teacher, reconstruction and benchmark execution.
