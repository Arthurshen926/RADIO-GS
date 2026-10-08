# Native identity transfer on frozen Gaussian fields

Frozen before new predictions/scoring, 2026-10-08. Full nine-room UQIS image
cohort: all67 queries. No fitting, new source views, extra teachers or labels.
This follows the completed five-task baseline; it is not a replacement for it.

One implementation and the original field/checkpoint/threshold/scale/observed
support per scene. Encode only the authorized current image, with the original
SigLIP/DINO preprocessing. New code reads no source RGB/masks/tracks/patch bank.
Shared member logits and exact selected-set transport remain unchanged.

Four fixed controls, not a search:

1. `parent`: exact current image prediction, replayed against saved Gaussian
   scores, selections and candidates before accepting any comparison.
2. `rank_same_candidates`: same original eight candidates and their semantic
   and maximum local cosine, fused by equal within-candidate percentile ranks.
   Isolates ordinal calibration on the fixed supported candidate set.
3. `bidirectional_same_candidates`: same eight candidates, anchor semantic
   cosine, and unchanged members. Each candidate's selected observed Gaussian
   local vectors form a set; uniform Gaussian mass and uniform query-patch mass
   feed v10's bidirectional matching formula, then equal candidate rank fusion.
   All selected rows and query patches participate. Empty scopes are unavailable,
   not supplemented by source patches or forced positive anchors. Uniform
   Gaussian mass is explicit and is not a claim of source-patch-area equivalence.
4. `rank_global_candidates`: fuse ordinal native channels over all eligible
   Gaussian anchors, then the same original spatial NMS/count. It tests candidate
   proposal separately from the two fixed-candidate controls.

These are deliberately bounded query-side tests. The set control changes the
local support from a single fused element to its learned extent as well as using
bidirectional matching; its effect cannot be attributed to the formula alone.
The same-eight-candidate oracle already bounds achievable IoU at22.13%, below
v10 image39.90%. Even success cannot establish full recovery or equivalent
source-patch information. A failure motivates a construction/representation
investigation; it cannot by itself distinguish bad scope from bad local vectors.

Persist all67 x4 Gaussian membership arrays and receipts before opening evaluator
labels. Then report actual-set IoU, AP, precision, recall and anchor correctness
with scene-macro aggregation; unsupported/empty outputs remain in the denominator.
Report paired scene differences and bootstrap uncertainty, with opened-development
status explicit. Do not promote an arm from one metric or choose per-query winners.
No threshold, scope or rank-weight sweep follows. Other modalities/defaults remain
the previously measured method; this study is not a new five-task system result.

Validate pure operators against v10 numerics, affine codec equivalence, missing
support, tie ranks, full patch coverage and duplicate-mass invariance. Run a
copied-export replay with source opens denied. Receipt binds code, protocol,
fields, query encodings, predictions and evaluator results.
