# Next70–71: source zoom grounding and native RGB crop reconstruction

## Scope

This iteration tests two sources of error: whether text segmentation needs a better source search window, and whether the native teacher representation benefits from unmasked RGB crop encoding. The geometry stays frozen at next66 adaptive_allrgb. All reconstruction uses the original20/25/23/22legal source frames, not target views. The expanded RGB geometry input budget remains part of the earlier candidate. No new other-benchmark, four-query or independent-test result is asserted.

## Next70: native proposals supply search windows, SAM3 supplies new masks

Next69 treated native region recognition as possible mask supervision and did not improve joint performance. Next70 instead uses each accepted native region only to define a1.5×bounding-box source RGB crop. SAM3 sees that crop at its1008input resolution with the unchanged exact query. Its probability masks are mapped back to the original source image and then to the96×128source fitting raster. SigLIP foreground/complement contrast qualifies the predicted masks. The native proposal itself is not the new mask.

This adds crop-based model computation; it does not add source views, a new pretrained teacher, or verified semantic identity. Local processing can truncate targets outside its window and increases only the detector's effective input scale; the final source label raster remains96×128. The experiment therefore does not test native-resolution source label reconstruction.

Completed controls use the same240step exposed-support set fitting and source-only admission/initialization as next68:

| Recipe | Selected projection IoU (%) |
|---|---:|
| Retained next68 | **51.835** |
| Cropped masks added only to previously untrained queries | 51.463 |
| Cropped masks added to all queries | 51.502 |
| Accepted cropped masks replace old detection/temporal union locally | 51.506 |
| Replacement plus local-window exclusion authority | 51.210 |

In the replacement arm, a view/query with an accepted crop mask uses it instead of both the original detector masks and the old temporal union. Where no accepted crop mask exists, original evidence is retained. The exclusion arm additionally allows crop-complement negative pixels *inside that source crop window* to override the old native-positive veto; it does not globally label missing regions as negatives. Neither revision improves the retained candidate. Membership arms preserve semantic64.882IoU/81.021AP/87.328localization by construction.

A separate semantic scope arm adds the crop masks to the same60step scope solver and existing confidence transport. It yields64.647IoU/81.049AP/85.746localization, versus64.882/81.021/87.328. Its51.414selected score is inherited next67, not a retrained membership result. It fails as a semantic replacement and also loses the83.482AP achieved by next69 native-region scope evidence.

All five full four-scene controls completed. No query-specific winner selection or threshold tuning was applied. No replacement is promoted.

## Concrete supervision mismatch, not just inference resolution

[Source-only visual examples](../../output/zoom_grounding_next70/source_examples.png) use the highest qualified crop confidence for six predeclared illustrative queries; no qualified crop detection existed for dall-e brand. They are not a statistically representative sample. Original detector and local detector often return nearly the same shape for pirate hat/bear nose, and both local relevance stages can accept a visibly tea-bag-bearing glass for coffee.

[Post-evaluation extent diagnostic](../../output/zoom_grounding_next70/reference_extent_diagnostic.png) juxtaposes reference annotations and the retained prediction for the first annotated instance of bear nose, coffee and miffy. These target images/annotations are used only to explain completed results, never as training inputs, crop selection, aliases, or acceptance thresholds.

The diagnostic distinguishes three mismatches:

- The selected bear-nose source example supervises the small brown tip; the displayed benchmark annotation covers the larger pale muzzle. The retained set also includes extensive false-positive bear surfaces. Both source scope mismatch and over-selection are present.
- Coffee source evidence can point to a tea glass or liquid surface, whereas the displayed reference selects the white mug extent.
- The highest-confidence miffy source crop is a white rabbit-shaped object; the displayed reference selects a different small orange/red figure. Crop confidence and foreground similarity do not establish the intended named entity.

These examples do not prove all labels are wrong, that the benchmark is invalid, or that all remaining loss is label ambiguity. They do show why repeated source views or sharper segmentation of the same incorrect referent need not improve recall. Annotation definitions and evaluator behavior remain unchanged. No GT-derived prompt aliases were added.

## Next71: full native RGB crop field reconstruction

The original teacher preparation (`prepare_four_teacher_retrieval.py`) encoded native-mask foreground crops with gray background and a tight native-resolution box plus3pixel padding. Next71 re-encodes the same source region cohort using unmasked RGB within a1.5×box derived from the stored96×128region support, resized to384×384. The pretrained SigLIP vision/text space is unchanged.

This is a practical alternative representation, **not a pure masking ablation**: context area, crop bounds and the mask resolution used to derive them also differ. A gain/loss cannot be attributed solely to removing background masking.

The new query-independent descriptors rebuild the native1536D map on the same geometry, followed by the same240step two-scalar source calibration, original60step source scope inference and original confidence transport. Source masks and queries do not change. Explicit selected membership is inherited from next68 to isolate semantic-field effects. This is an actual new native feature field, not just another query-conditioned pseudo-label readout; it is not four-teacher student retraining.

Completed full4 result: **63.536IoU /79.544AP /85.473localization**, below the retained64.882/81.021/87.328. Selected51.835is inherited next68 and is not a new gain. This full-crop recipe is rejected.

| Scene | Retained IoU / AP | Full-crop field IoU / AP |
|---|---:|---:|
| figurines | 66.711 / 84.722 | 66.441 / 84.450 |
| ramen | 61.477 / 76.117 | 58.642 / 72.715 |
| teatime | 71.328 / 88.146 | 69.678 / 86.703 |
| waldo_kitchen | 60.012 / 75.097 | 59.382 / 74.306 |

## Research decision

None of the six full4controls provides a joint improvement. Retain next68 membership and its64.882/81.021/51.835semantic-IoU/AP/selected recipe. The previous next69semantic branch remains only a ranking tradeoff, not a jointly promoted model. No scene/query winners are stitched together.

The evidence does not support another iteration of crop enlargement, unqualified mask union, or giving the same detector more authority over contradictory regions. It also does not support unmasked1.5×crop replacement as a way to preserve more usable native capability. These are failures of the tested recipes, not universal impossibility results.

The more substantive remaining requirement is **referent and extent supervision**: determine whether evidence denotes the intended entity, its whole object, a part, or its contents before converting that evidence into positive/negative membership labels. Multiple source views can confirm the same wrong interpretation. Source-only independent referential verification or task-aligned scope supervision remains unimplemented; this round did not solve it. Any future learned verifier must use separate training/source evidence, not the displayed evaluation masks or GT-derived query aliases.

For CVPR, this round adds causal exclusions and illustrative failure evidence, not a new publishable performance claim. Strong independent-test, multi-query and cross-benchmark evidence is still missing. Do not portray unchanged selected scores in semantic-field controls as gains, or this projection metric as independent3D-GT IoU.

## Reproduction and artifacts

- `scripts/research/run_zoom_grounding_next70.py`: crop SAM3 in the existing sam3 environment, then native qualification in cybersim_agent. Initial environment-incompatible launches were corrected before the successful evidence runs; no model/environment upgrade was needed.
- `scripts/research/run_zoom_membership_next70.py --gpus 0,1,2,3`: cold/augment/replace controls.
- Same runner with `--modes local_exclusion --gpus 4,5`: local exclusion authority.
- `scripts/research/run_zoom_semantic_next70.py --gpus 4,5`.
- `scripts/research/run_full_crop_field_next71.py --gpus 0,1,2,3`: source encoding, actual native field rebuild, scope, transport, evaluation.
- Summaries: `output/zoom_grounding_next70/summary.json`, `output/full_crop_field_next71/summary.json`.
- Source and reference diagnostic scripts: `visualize_zoom_sources_next70.py`, `visualize_reference_extent_next70.py`. The latter is post-evaluation only.

All six four-scene evaluation recipes and their prerequisite jobs finished. No defensive tests or additional audit framework were added. Existing retained artifacts and evaluator were preserved.
