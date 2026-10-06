# Contribution-aware evidence and deployment-aligned set decoding — next56

## Result

A retained research candidate improves four-scene selected-projection IoU from
44.037% to47.863% (+3.826 percentage points), with every scene improving.
Semantic2D IoU58.087/AP74.965 remain exactly unchanged by design. This is progress
in actual element-set retrieval, not a new semantic-recognition gain, independent
physical3D benchmark result, or evidence of CVPR readiness.

The large majority of the gain is a stronger optimization control: continuing
from the retained membership rather than restarting from semantic scores gives
47.346%. The new deployment-aligned objective adds0.517pp at matched additional
120 updates/query. Do not attribute the entire3.826pp to a new mechanism.

## Mechanisms and controls

All controls keep next52 RGB+MoGe geometry, source observations, teachers, queries,
rendering and frozen evaluation contracts. No target annotation enters inference.
These are repeatedly used development scenes/queries, not an independent final test.

1. **Actual evidence mass in other-view consensus.** The old solver averaged
   per-view estimates whenever evidence mass was nonzero. The experimental option
   weights those estimates by confidence-weighted contribution mass instead.
   This changes pseudo-label agreement, not identity truth. Native-field raw scope
   IoU47.420→47.958 while AP75.876→75.753. With the same old-confidence transport,
   2D IoU58.087→58.093 and AP74.965→74.945: effectively no useful joint gain.
   The option remains off by default. Strongest-contributor ownership was not run.
2. **Matched set continuation (`agreement`).** Initialize from retained
   `selection_scores`, not `scores`; optimize the same subset-alpha loss for an
   additional120 updates/query, with the old native field qualifying positives
   and negatives exactly as before. This yields47.346% selected IoU.
3. **Other-view qualification (`crossview`).** Same initialization and budget;
   replace old-native-field label qualification with leave-current-view-out
   contribution-weighted source estimates. Positives require other-view positive
   support; negatives require other-view negative support; absent/conflicting
   correspondence abstains. Full4 gives47.485%, only0.140pp over continuation.
   Repeated detection agreement does not establish complete extent; this does
   not solve the independence/identity problem. No default promotion.
4. **Deployment decision (`decision`).** Keep old-field qualification and same
   budget as control2. Anneal membership temperature geometrically0.5→0.1; optimize
   a smooth pixel decision `sigmoid(logit(alpha)-logit(10.5/255))`, rather than alpha
   reconstruction. The cutoff comes directly from the frozen evaluator's rounding
   to byte and `byte > 10` rule; it is not fitted to target GT. Final evaluation
   still thresholds element membership at0.5 and renders the actual hard subset.
   The relaxation is not an exact hard-set solver. Full4 gives47.863%.
5. **Annealing alone (`anneal`).** Same schedule and source labels, but original
   alpha reconstruction objective. This separates membership sharpening from
   deployment coverage alignment. Full4 gives47.603%; the deployment coverage
   objective adds0.260pp beyond this matched annealing control.

The optimized response is distinct from fixed semantic scores, consistent with
ADR0011. Full-scene transmittance and selected-subset transmittance differ; no
semantic-probability metric is substituted for actual subset performance.

## Four-scene results

| Scene | Retained next54 | Continue old objective | Other-view qualification | Deployment decision |
|---|---:|---:|---:|---:|
| figurines |43.181|47.436|47.882|47.633|
| ramen |30.271|34.202|34.090|35.063|
| teatime |55.430|58.841|58.617|58.820|
| waldo_kitchen |47.265|48.903|49.351|49.935|
| Macro |44.037|47.346|47.485|47.863|

Decision vs matched continuation improves three scenes, with teatime−0.021pp.
Do not assemble per-query or per-scene winners. The retained candidate is one
uniform `decision` recipe. Each scene runs an additional120updates per eligible
query;18/9/12/14 queries are eligible. Measured decoder script times are approximately
14.7/12.3/20.3/17.7 seconds for figurines/ramen/teatime/waldo respectively, excluding
process startup, original map/teacher preparation and evaluator. These are not
end-to-end query latency figures.

## What improves and what remains broken

Query-wise examples (average over that query's annotated observations) relative
to next54: chopsticks47.21→60.30; bowl49.47→62.95; three cookies55.52→78.89;
plate in waldo57.52→73.45. These gains indicate better actual selected coverage.
They do not establish improved source geometry or new teacher knowledge.

There are important regressions: coffee55.31→46.77, refrigerator70.38→65.15,
and spoon45.67→42.87. Corn remains0. Extra source optimization cannot create
missing identity evidence, and partially correct masks can still drive the set
toward an incomplete/incorrect target. All-scene macro gains do not mean all
queries are fixed. No GT-dependent exception rules were introduced.

The old-confidence prior, next38 qualification, source-time detectors and1536D
native feature storage remain. This is not a completed compact four-teacher field,
independent object-completeness model, four-query deployment or general benchmark
replacement. No new ScanNet/NVOS/UQIS result is claimed.

## Research consequence

Retain the decision candidate as a useful set-decoding improvement, not a central
novelty claim. The big unresolved research issue remains determining whether an
observation identifies the right target AND covers its requested whole/part/union.
Contribution weighting and other-view voting are insufficient here. A subsequent
identity/completeness intervention needs a fixed-input comparison against this
stronger continuation baseline, held-out validation, and source evidence that
can distinguish coffee/mug contents, parts and full objects. Do not return to
source reconstruction loss as the sole quality proxy.

Artifacts:
* `output/corroborated_set_next56/decision/lerf_<scene>/typed_readout.pt`
* `output/evidence_mass_next56/summary.json`
* `scripts/research/decode_corroborated_set_next56.py`
* `scripts/research/run_corroborated_set_next56.py`
* `scripts/research/refine_evidence_mass_next56.py`
* `scripts/research/align_evidence_mass_next56.py`
* `scripts/research/run_evidence_mass_next56.py`
* `scripts/research/summarize_evidence_mass_next56.py`

No tests or audit framework were added. Geometry and input caches are unchanged.

All started fitting and evaluation jobs completed. The summary contains six full4
controls (two semantic variants and four selected-set variants). No global mainline
default is changed; use the explicit decision artifact for the retained research candidate.
