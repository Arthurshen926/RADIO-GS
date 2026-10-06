# Next61: direct source-native extent evidence before fusion

## Outcome

Two method controls completed on all four LERF development scenes. Neither improves the retained hierarchy candidate. No promotion. This round tests actual source-image candidate evidence, rather than another weighting of the old reconstructed semantic response. It does not solve referential extent or establish CVPR readiness.

## Method

Candidate scopes and geometry match next59: each detector instance has its literal mask and up to three enclosing SAM masks covering at least 90% of it (with the existing fallback relations). Query encodings remain frozen native SigLIP. No benchmark-query competition or GT aliases.

**Native selection:** for each instance family, take a single window enclosing all candidate masks, enlarged by the existing 1.5× crop rule. Upsample saved 96×128 masks to source RGB resolution and create foreground/complement images with gray masking. All family alternatives use the same window and native 384×384 encoder preprocessing. Choose the candidate maximizing query cosine(foreground) minus cosine(complement), before Gaussian fusion. The source RGB encoder is frozen. Keep next59-style old-native positive qualification and the existing 120-step annealed selected-set decoder; initialize from next54 as in the matched next59 experiment. Semantic response is unchanged. This adds source image encoding compute and is not a map-only inference result.

**Source supervision:** reuse exactly these candidate scores, admitting individual selected scopes only if their foreground/complement margin is positive. Union admitted instances; rejected identities and alternative extents remain unknown. Negatives must be outside the entire candidate union and not old-native positives. Admit positive masks without requiring old-map positives. For empty observed memberships initialize from confidence-weighted source probability backprojection; otherwise retain next54 initialization. Run the same 120-step subset decoder. This is a combined admission/unknown/initialization control, not an isolated proof of any one component.

Per-view confidence still comes from the existing detector/native confidence aggregation. Positive crop margin is a heuristic identity signal, not verified identity. Candidate masks remain coarse saved masks, not new high-resolution segmentation. Scoring alternatives directly does not parse whole/part/contents/collection semantics.

## Results

Percent, scene macro; same frozen geometry and existing calibrated evaluator.

| Method | 2D IoU | AP | Localization | Selected projection IoU |
| --- | ---: | ---: | ---: | ---: |
| Next59 hierarchy | 58.087 | 74.965 | 82.245 | 48.354 |
| Direct native candidate selection | 58.087 | 74.965 | 82.245 | 48.136 |
| Native per-instance source supervision | 58.087 | 74.965 | 82.245 | 47.829 |

Semantic metrics are identical by construction, not evidence of an improved semantic field.

| Scene | Parent | Native selection | Source supervision |
| --- | ---: | ---: | ---: |
| figurines | 47.475 | 47.434 | 46.967 |
| ramen | 35.092 | 35.123 | 34.481 |
| teatime | 59.762 | 59.116 | 58.160 |
| waldo_kitchen | 51.087 | 50.870 | 51.706 |

Source supervision helps refrigerator 86.997→95.179 and red toy chair31.168→35.713, but hurts three cookies73.090→65.135, sheep69.160→62.340, bag40.145→33.035 and old camera47.763→44.078. Corn becomes1.331 from zero, below next60 bootstrap7.881. No per-query winner stitching.

Across957 source instance observations,811 have multiple scope alternatives. The native rule selects an enclosing alternative538 times;735 selected scopes have a positive margin. These are teacher decisions, not correctness statistics. The distribution alone does not prove systematic overexpansion. Real development regressions establish that this rule is insufficient as a replacement.

## Interpretation and next research boundary

The experiment rejects the simple claim that giving native query similarity direct access to candidate foreground/complement is enough to infer correct extent. Recognizing a concept inside a crop and selecting the requested entity/part/set are different tasks. Relaxing old-map supervision admission without reliable extent evidence also fails globally. It would be incorrect to claim this controls away all geometry or semantic-writing bottlenecks, or proves every failure is scope ambiguity.

Further method work needs a source-grounded referential prediction objective with distinguishable entity, part, contents/container and collection targets, coupled to cross-view identity before fusion. Candidate similarity alone is not that objective. A new model or annotated source supervision would change information/compute budgets and must be reported explicitly; no such model was trained here. Avoid repeating local identity-weight or crop-score sweeps as a purported contribution. Preserve next56 comparison, next59 development candidate, and next60 optional missing-concept experiment; no global replacement.

## Execution and scope

All eight decoding/evaluation jobs completed successfully. Source evidence cached under output/native_extent_next61/source_evidence; no target RGB or annotations generate predictions. Four repeatedly used LERF scenes are development evidence, not independent final test. Selected projection is hard Gaussian selection rendered against 2D annotations, not physical 3D-GT IoU. No new ScanNet, NVOS or UQIS results, tests, or audit framework.

Scripts: scripts/research/decode_native_extent_next61.py, decode_source_extent_next61.py, run_native_extent_next61.py, run_source_extent_next61.py, summarize_native_extent_next61.py.

Summary: output/native_extent_next61/summary.json. Each native/source scene includes typed_readout.pt, fit.json and anchored/pipeline_calibrated/evaluation.json. Source fit metadata drops unused inherited choice counters; real choices reside in source_evidence and summary.json. No prediction changed by that reporting cleanup.
