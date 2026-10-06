# Next64–65: typed temporal evidence and scale-factorized native fields

## Outcome

Four complete four-scene method controls and four additional native-field evaluations finished. No new candidate beats the retained semantic IoU/AP combination. Keep next63's optional selected-only temporal candidate and its unchanged next54 semantic response. No global default change, significant gain or CVPR-readiness claim.

## Next64: temporal extent is not complete semantic supervision

Next63 unions tracked and detected probabilities, then uses a query-view confidence for both foreground and complement evidence. This couples temporal continuation to exclusion decisions. We implemented two alternatives with the same source frames, frozen carrier, native prior,60-step consistency solver and source-visibility confidence transport:

- Abstain: retain detector positives, targets, confidences and visibility; remove a detector negative when it conflicts with an eroded positive tracked region. No new temporal positives or negatives.
- Positive: retain that abstention and additionally admit eroded temporal positives outside original detector positives. Use seed confidence only for those new entries; retain original confidence elsewhere. Extend visible support only to new positives. Never interpret the tracked complement as new negative evidence.

The confidence is therefore a pixel-query evidence weight, not an indiscriminate increase for the whole source view. Tracker presence still does not verify text identity. These are experimental supervision policies, not default repairs.

Added positive entries across source pixel-query observations: figurines6299, ramen11076, teatime36709, waldo2725. Entries contradicting eroded detector negatives:1032,39,1325,47 respectively. These counts are pseudo-supervision disagreements, not measured true/false labels. Most new tracked positives were in previously unlabelled observations or boundary bands.

Abstention largely avoids next63's IoU decline (57.706→58.085), but AP74.854 and localization81.397 remain below retained74.965/82.245. Adding temporal positives worsens IoU56.518/AP73.635. This shows that removing new background supervision alone is insufficient; those positive additions are not a reliable semantic replacement either. Do not equate tracking an object with proving the requested textual scope. The experiment does not identify every erroneous tracked pixel.

## Next65: replace early region mixing, then separate storage from readout

The existing native rebuild averages all overlapping SAM-region descriptors at each source pixel before lifting to Gaussians. The new field retains three relative area ranks among covering source masks: smallest, middle (floor median rank), largest. Each slot lifts the corresponding frozen1536D SigLIP descriptor independently through the same Gaussian contribution weights, aggregates over the same source frames, and normalizes its feature vector. Geometry and query encoder are unchanged. Features are query-independent; this is native teacher storage, not four-teacher student retraining.

Two readouts were evaluated:

1. Elementwise maximum cosine across the three slots. Refit the same two scalar source calibration parameters for240 steps; then apply the existing60-step source extent inference and unchanged next53 confidence transport.
2. One global level per query, selected using the greatest rendered native-cosine separation between source detector positive and negative regions. This uses source pseudo-supervision, not test annotations, and adds query-time source selection. Apply the same scalar calibration, extent inference and confidence transport. This arm reads stored float16 slot descriptors; the initial maximum arm used the pre-save normalized float32 descriptors. Tiny differences cannot be interpreted as a precision-matched readout ablation.

Relative source area rank is not a verified entity/part hierarchy or a physically aligned cross-view scale. A slot can still contain different entities or granularities in different views. Keeping three slots does not solve that correspondence problem. Taking the maximum can select context-bearing coarse regions; choosing one global level can underrepresent queries that need multiple extents. These are mechanism hypotheses/limitations, not per-error causal proof.

Total slot files in GiB: figurines2.359, ramen3.110, teatime11.569, waldo8.240. Feature storage is roughly three times the single native field; no equal-capacity or efficiency claim is justified.

Before source extent reconstruction, the new maximum readout has IoU29.916/AP70.611 versus the original native mixture36.052/73.550. It therefore does not establish an upstream semantic gain later lost by the scope decoder. The corresponding localization is75.924 versus79.456. Do not compare this intermediate stage's inherited selection metric to the final temporal selected candidate.

## Final four-scene development results

Percent, macro average; selected projection remains next63's fixed48.518 for every final arm, by construction.

| Method | 2D IoU | AP | Localization | Selected projection |
| --- | ---: | ---: | ---: | ---: |
| Retained next63 selected candidate | 58.087 | 74.965 | 82.245 | 48.518 |
| Temporal conflict abstention | 58.085 | 74.854 | 81.397 | 48.518 |
| Temporal positive admission | 56.518 | 73.635 | 78.904 | 48.518 |
| Three slots, elementwise max | 57.433 | 74.544 | 83.404 | 48.518 |
| Three slots, source-selected global level | 57.247 | 74.287 | 81.924 | 48.518 |

The maximum arm improves localization to83.404 but lowers IoU and AP. None is a joint replacement. No per-query winner stitching or threshold tuning on annotations.

## Revised conclusion and next research priority

Do not call early averaging a proven implementation bug or the established dominant bottleneck. It discards information, but retaining three relative area ranks did not improve retrieval. Likewise, typing temporal evidence more cautiously prevents part of a regression; it does not create reliable identity or requested scope supervision. The requested large performance gain has not been obtained.

The recent crop scoring, scope graph, temporal and scale controls do not justify continuing small variations of the same weak labels as the primary research path. Independent referential extent evidence remains missing. Another unresolved system-level comparison is full source-only RGB+MoGe reconstruction versus the current fixed-count carrier refinement. Historical RGB-carrier exchanges showed a substantial difference but used unmatched reconstruction budgets/splits, so they cannot establish a fair geometry gain. Prioritize a matched source/input-budget reconstruction comparison with separate native-semantic and actual selection readouts before attributing the remaining gap solely to semantics. This full reconstruction comparison was not implemented in this round and is not a claimed result.

## Artifacts and execution

- output/typed_temporal_next64/summary.json: complete abstain/positive controls and retained/union comparators.
- output/scale_field_next65/summary.json: native and final scale controls, comparators and slot bytes.
- scripts/research/refine_typed_temporal_next64.py, align_typed_temporal_next64.py, run_typed_temporal_next64.py, summarize_typed_temporal_next64.py.
- scripts/research/rebuild_scale_field_next65.py, refine_scale_field_next65.py, align_scale_field_next65.py, run_scale_field_next65.py.
- scripts/research/select_coherent_scale_next65.py, refine_coherent_scale_next65.py, align_coherent_scale_next65.py, run_coherent_scale_next65.py, summarize_scale_field_next65.py.

All launched work completed;20 new evaluations in total. Existing source/target permissions, geometry, evaluator and final selected membership preserved. No new tests or audit framework. No target RGB/annotations generate predictions. These are repeatedly used LERF development results, not an independent final test. Selected projection is hard selected Gaussians compared to2D annotations, not true3D-GT IoU. No new ScanNet/NVOS/UQIS results, SOTA equivalence, commit or global promotion.
