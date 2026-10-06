# Next62–63: source-instance graphs and text-seeded temporal extent evidence

## Outcome

Four controls × four LERF development scenes completed, plus four source-only tracking jobs. Reciprocal instance graph assignment is essentially neutral. Temporal extent evidence improves selected projection on all four scenes against the matched next56 decoder, 47.863→48.518 (+0.655pp), but only +0.164pp over next59 hierarchy and is not better on every scene against that stronger candidate. Reconstructing semantic responses with the same temporal evidence worsens semantic IoU, AP and localization. Keep temporal selection as an optional research candidate; reject semantic replacement. No large benchmark breakthrough, global default promotion or CVPR-readiness claim.

## Next62: joint scope choice before fusion

Reuse next59 candidates: each source detector instance retains its literal mask and up to three small SAM enclosures, with existing fallback relations. Lift every candidate through the frozen observed-support-normalized source operator. Compare candidate memberships on co-visible Gaussians, weighted by the minimum source visibility mass of the two views. Pairwise score is weighted soft IoU.

For each pair of source views and the same query, link only reciprocal best literal instances with literal soft IoU >0.1. This is a fixed geometric heuristic, not verified identity; weak co-visibility can still create false matches. Same-name instances with no reciprocal link remain separate. Optimize weighted pairwise candidate agreement with at most ten coordinate-ascent passes, initialized from literal scopes; isolated instances keep literal scope. This uses no current field prediction to choose candidates. The identity control additionally restricts choices to positive native foreground/complement margins when any are available. It uses next61 cached source descriptors, not new target input.

Keep next54 initialization, next56 120-step decoder, old-native supervision qualification and frozen next52 geometry. This does not create referential whole/part/contents/collection labels. Coordinate ascent is local optimization, not a globally optimal or learned scope model. No-edge or disconnected instances do not become validated by this graph.

| Selected projection IoU (%) | figurines | ramen | teatime | waldo | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Next59 hierarchy | 47.475 | 35.092 | 59.762 | 51.087 | 48.354 |
| Reciprocal graph | 47.644 | 35.303 | 59.379 | 51.220 | 48.387 |
| Graph + native identity | 47.729 | 35.223 | 59.614 | 50.812 | 48.345 |

Semantic IoU58.087/AP74.965/localization82.245 is unchanged by construction. Neither graph control establishes a useful new performance improvement. This rejects these correspondence/assignment rules as replacements; it does not prove all cross-view instance methods are ineffective.

## Next63: new temporal evidence from authorized source RGB

Load the existing SAM3 video tracker with its detector backbone. For each query independently, choose the training view with highest detector score among views with nonempty masks and positive existing native foreground/context margin. Seed all admitted instances in that view as separate tracker objects. Track forward and backward over the sorted authorized source frames; no target RGB or new prompts. Keep objects only when tracker presence logits are positive and union them for the text-union query. Saved probabilities are resized to the existing source feature raster. Seed-frame output is excluded from additional evidence.

Only54 of67 queries have eligible seeds (figurines18/21, ramen11/14, teatime12/14, waldo13/18). Others retain detector-only evidence. Tracking does not certify seed identity, discover every unseeded instance, or ensure precise boundaries. Noncontiguous source views, temporal drift and coarse seed masks remain limitations. Extra online teacher compute is explicit; this is not equal-compute map-only inference.

**Selection control:** take the union of original per-frame detection and positive-presence tracked probabilities. View confidence is the maximum of the original confidence and the seed's confidence when tracking is present. Retain old-native positive/negative qualification, eroded boundary abstention, frozen geometry and the matched next56 120-step subset decoder. Missing tracking does not become negative supervision. Source detector masks are not replaced, allowing instances absent from the seed frame to remain represented.

**Response control:** additionally reconstruct semantic responses from the same temporal union, starting at the same next54 native mixture field, with the same60-step source consistency solver and unchanged next53 source-visibility-weighted confidence transport used by next54. Inherit the temporal selection artifact unchanged. This tests semantic ranking changes with the existing confidence-distribution procedure; it is not native-feature retraining or a new calibration fit to GT.

| Four-scene macro (%) | 2D IoU | AP | Localization | Selected projection |
| --- | ---: | ---: | ---: | ---: |
| Matched next56 decoder | 58.087 | 74.965 | 82.245 | 47.863 |
| Next59 hierarchy | 58.087 | 74.965 | 82.245 | 48.354 |
| Temporal selection | 58.087 | 74.965 | 82.245 | 48.518 |
| Temporal selection + response | 57.706 | 74.256 | 79.147 | 48.518 |

| Selected projection (%) | Matched next56 | Temporal selection |
| --- | ---: | ---: |
| figurines | 47.633 | 49.095 |
| ramen | 35.063 | 35.275 |
| teatime | 58.820 | 59.116 |
| waldo_kitchen | 49.935 | 50.584 |

Against matched next56: pirate hat14.326→26.912, jake28.009→37.856, spoon in ramen42.870→47.051, sheep62.560→67.657. Regressions include pink ice cream80.751→76.835 and miffy6.706→6.034. Do not compare these query gains to next59 without changing the stated comparator. Against next59, teatime and waldo selected scores decline.

The semantic response control declines0.381pp IoU,0.709pp AP and3.097pp localization. Its scalar probabilities are not improved just because temporal masks help some selected extents. Reject that replacement; do not stitch query-specific winners.

## Method interpretation

The useful evidence is modest but concrete: appearance-based temporal propagation can supply some missing extent information beyond independent frame detections. Merely reassigning existing candidates with source-native similarity or a co-visibility graph has not yielded a meaningful gain.

Temporal spatial persistence and semantic query identity must remain distinguishable supervision. The result supports that distinction, but is not proof of which tracker errors account for the semantic losses. A detector-seeded tracker can consistently propagate a wrong entity. Positive native margins are still recognition heuristics, not referential identity labels.

Remaining central work is source-grounded referential supervision with explicitly distinguishable entity, part, contents/container and collection scope, plus reliable seed identity. Existing automatically generated evidence does not provide verified labels for these distinctions. New source annotation/model capability would change the supervision/compute budget and needs to be stated as such. Do not claim this round trained that missing model. Preserve the best coherent candidates and evaluate future changes uniformly; a collection of per-query wins is not a method.

## Artifacts and boundary

- output/instance_scope_next62/summary.json: both complete graph controls; per-scene fit metadata includes graph counts.
- output/text_video_next63/summary.json: matched baseline, hierarchy baseline, temporal selection and response control; per-query source tracking coverage.
- output/text_video_next63/tracks/lerf_<scene> contains source seed and probability artifacts.
- output/text_video_next63/decision/lerf_<scene>/typed_readout.pt is the optional temporal selection candidate.
- Scripts under scripts/research: decode/run/summarize_instance_scope_next62.py; track_text_sources_next63.py; run_text_video_next63.py; decode_text_video_next63.py; run_text_video_decode_next63.py; refine_text_video_next63.py; align_text_video_next63.py; run_text_video_response_next63.py; summarize_text_video_next63.py.

All launched jobs completed successfully. No new tests/audit framework. Same repeatedly used four-scene development cohort and frozen evaluator. Selected projection is hard selected Gaussian rendering compared to2D annotations, not true3D-GT IoU. No target RGB or annotations generate predictions. No new ScanNet/NVOS/UQIS results, independent test or public SOTA equivalence. No global defaults changed.
