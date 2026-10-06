# Corroborated source exclusion and remaining boundary leakage (next93)

## Core question and method

The source subset decoder qualified negatives with the old native response:
source complement pixels with an old response above0.5became unknown. This
preserves useful native recall, but can also protect false-positive scope. Test
whether observed source exclusion can correct that prior, keeping native features,
geometry, positive labels, source frames, hypotheses, optimization budget and
scoring unchanged. Only13regional queries are recomputed;54others are retained.

Three complete four-scene variants:

- Local: allow observed source-complement negatives in a9x9dilation neighborhood,
  retaining the existing3x3unknown boundary band. Radius4is a fixed exploratory
  design, not an evaluation-selected parameter or physical-distance guarantee.
- Complement: allow the full visible complement beyond that boundary, including
  pixels vetoed by old native positivity. This is a mechanism control, not a claim
  that every uncovered target pixel is negative.
- Corroborated: accumulate confidence-weighted positive/negative known-pixel
  evidence on visible Gaussian support. For each view, subtract its own evidence,
  project the other-view negative fraction back, and admit additional negatives
  only with coverage>0.5and negative fraction>0.5. The same thresholds used for
  existing source positive corroboration are used without tuning. Only active
  positive-containing views contribute, and explicit track negative permissions
  remain respected. Camera corroboration is NOT independent semantic ground truth.

These are optional experimental overrides of ADR0011's prior-positive veto,
not changes to its retained default. The semantic response and query encoders
remain untouched. Missing/occluded Gaussian support is not made negative simply
because it has no source observations. Source masks may still be incomplete,
and agreement between correlated wrong masks can still exclude true targets.

## Completed results

All12scene evaluations completed. Values are selected Gaussian2Dsilhouette
projection IoU percentages, not independently annotated3DGT metrics.

| Scene | next91 | Local | Complement | Corroborated |
|---|---:|---:|---:|---:|
| figurines | 67.225 | 67.075 | 67.104 | 67.065 |
| ramen | 49.552 | 52.431 | 54.238 | 54.234 |
| teatime | 62.034 | 62.098 | 62.098 | 62.098 |
| waldo_kitchen | 58.890 | 58.890 | 59.054 | 59.054 |
| Macro | 59.425 | 60.123 | 60.623 | 60.613 |

Preserve corroborated exclusion as a LERF research candidate60.613, up1.188points
from59.425. Its mean is essentially equal to the broader complement control60.623;
prefer the evidence-qualified rule, while explicitly retaining its limitations.
Stable next80/default remains unchanged. No per-query winner assembly.

Kamaboko44.23→72.80 and sake cup38.82→60.89 are the main repairs. Post-evaluation
per-object mean precision/recall distinguishes actual error removal:

| Query | Precision before→after | Recall before→after |
|---|---:|---:|
| kamaboko |44.5→74.7%|98.6→96.8%|
| sake cup |39.7→66.6%|83.6→74.7%|
| pour-over vessel |73.0→77.5%|86.8→86.8%|
| waldo |92.3→94.2%|89.2→82.8%|

Waldo IoU82.47→78.22 is a real regression; figurines falls67.225→67.065.
Cross-view exclusion still can penalize incomplete positive scope. Onion remains0,
ladle0.02, and spatula28.71is effectively unchanged. The improved macro does not
mean all identities or object extents are fixed. Semantic66.930/AP84.146and the
untouched54query columns are bit-identical to next91.

## Tracking versus Gaussian footprint: focused attribution

Review all11nonseed present views of the recovered red-spatula track, using only
source RGB. Most retained masks lie on the object, but there are rasterized breaks,
small fragments and a partial/misaligned early view. This review does not quantify
teacher accuracy or exclude tracking drift as a cause.

Independently replay the actual selected subsets in the scored spatula camera;
both next91and the failed next92medial branch match the saved predictions byte
for byte. Partition their visible Gaussian contribution using GT only for this
post-hoc diagnosis. Among background alpha mass in predicted-positive pixels:

| Fixed prediction | Primitives with no foreground contribution | Primitives contributing to both foreground and background |
|---|---:|---:|
| next91 |36.80%|63.20%|
| next92medial |81.24%|18.76%|

For next91,182visible primitives contribute only outside the target,217are mixed,
and126contribute only inside it. These are single-view contribution classes,
NOT intrinsic3Dobject labels, an unavoidable-error bound, or percentages of IoU
loss. Mixed contributions do show why binary whole-Gaussian selection can face
boundary conflicts; the medial failure also adds many fully extraneous primitives.
No GT-informed set is used to generate any benchmark prediction.

The additional negative policies barely change spatula, so the old-native veto
is NOT its dominant remaining failure. Source-track quality, geometric footprint
mixing and underconstrained selection remain distinguishable follow-up questions.
The current experiment repairs a confirmed exclusion bottleneck on other queries
without claiming that one mechanism explains every failed object.

## Remaining focused work

1. Preserve correct target support when source scope is incomplete (waldo regression)
   rather than interpreting repeated partial masks as complete object exclusion.
2. For thin boundaries, distinguish removable extraneous primitives from mixed
   foreground/background footprints using SOURCE-view constraints. A mixed
   footprint does not by itself justify changing frozen geometry or using target GT.
3. Keep identity failures (onion/ladle) separate from extent failures. Neither
   longer fitting nor the new negative rule supplies missing semantic identity.

No new NVOS/ScanNet/UQIS, four-query compact-field, independent-test or SOTA result.
This remains the optional source-assisted transient inference track, with the
same frozen geometry/native map and unchanged target-input permissions.

## Reproduction and artifacts

run_scope_exclusion_next93.py --scene SCENE --gpu GPU --policies local complement corroborated.
Summarize with summarize_exclusion_next93.py. The decoder default remains
--negative-admission prior and --positive-core eroded.

- output/next93/summary.json: all12scene outcomes.
- output/next93/error_decomposition.json: post-evaluation precision/recall.
- output/next93/secondary_track.jpg: all present source-view scope crops.
- output/next93/extent_attribution.json: subset replay and alpha contributions.
- output/next93/retained_outputs.json: retained semantic/other-query values.
- output/scope_exclusion_next93/corroborated/lerf_*/fit.json: admitted evidence counts.
