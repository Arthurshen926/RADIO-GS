# Thin-mask positive cores and extent leakage (next92)

## Method change

Source-only counting showed fixed3x3erosion at96x128removed91.2%of the red-spatula
secondary's positive pixels. Add an optional positive-core construction to the
existing decoder: keep the eroded interior and add positive distance-transform
ridges (positive pixels whose distance to the mask complement is locally maximal
in a3x3neighborhood). The teacher probabilities, negative/unknown masks, source
identities, geometry, optimizer budget and evaluator remain fixed. No source pixels
outside the binary teacher foreground are directly added as positives. Default
positive-core remains `eroded`; the medial rule is an experimental flag.

The rule is NOT a higher-resolution teacher, a full medial-axis reconstruction or
verified thin-object supervision. Local ridges can retain narrow noise/incorrect
track fragments as well as true thin structure. It cannot recover tracker-missing
views or subpixel content already lost when saving the source probability grid.

Compare all four scenes using next88primary tracks and next91description-admitted
tracks. The13regional queries are recomputed; other54query columns are retained.
This is not a query-specific parameter change.

## Completed results

| Input evidence | Eroded positive core | Interior plus ridges |
|---|---:|---:|
| primary | 59.226 | 59.292 |
| description | 59.425 | 59.248 |

All eight scene evaluations completed. Values are macro selected Gaussian2D
silhouette projection IoU percentages. Primary-only gain is small; composition
with the stronger next91candidate regresses. Do NOT adopt the positive-core
replacement or assemble per-query wins. Retain next91 research candidate59.425
and next80 stable reference. Semantic66.930/AP84.146remain unchanged.

## Actual failure mode, measured after predictions were fixed

For the scored spatula image, the frozen evaluator's threshold10 masks give:

| Quantity | next91 eroded | Medial positives |
|---|---:|---:|
| GT pixels |2468|2468|
| Predicted pixels |7084|31072|
| Intersection |2130|2468|
| Recall |86.30%|100.00%|
| Precision |30.07%|7.94%|
| IoU |28.70%|7.94%|

More retained source positives do increase recall, but the reconstructed set
expands far outside the annotated target. The already accepted candidate's main
remaining error on THIS query is poor precision, not mostly missing the object.
This does not generalize to onion/ladle or prove whether the excess originates
in source tracking drift, geometry footprints, or the solver's conflicting-view
handling. The measurement uses GT only for post-evaluation error decomposition,
never for hypothesis admission, masks, thresholds or optimizer decisions.

## Decision and next focused question

The next mechanism investigation should separate inaccurate tracked positives
from Gaussians that explain both foreground and background across views, and
measure how these errors enter the actual subset-rendered selection. Added
positives require evidence of compatible extent, not merely candidate identity.
Do not continue expanding positive supervision based solely on its pixel count.
Identity remains open for other failed queries; next91's single admitted example
is not general validation of its caption-based gate.

Across next90–92,20scene evaluations completed. Native/categorical ranking
replacement failed; query-blind descriptions yield a small source-driven expansion
gain; medial-positive supervision fails composition. No new other-benchmark,
four-query, compact-student, independent3DGT or SOTA claim.

Artifacts: output/next92/summary.json and spatula_precision_recall.json;
output/next91/source_raster_diagnostic.json. Reproduction:
run_medial_scope_next92.py --scene SCENE --gpu GPU; the matched description arm
uses --tracks-base output/description_tracks_next91/union,
--retained-base output/description_scope_next91,
--output-base output/medial_description_scope_next92.
