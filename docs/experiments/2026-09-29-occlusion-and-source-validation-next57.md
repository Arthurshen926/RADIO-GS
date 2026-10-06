# Occlusion-conditioned updates and held-source validation — next57

## Decision

Five full-four-scene controls completed. None replaces next56's47.863% selected
projection IoU. Semantic2D IoU58.087/AP74.965 are unchanged by construction.
No large gain was achieved this round. The useful finding is a concrete source
identity/extent failure that cross-view validation with the same teacher accepts.

| Control | Selected projection IoU, % |
|---|---:|
| Retained next56 decision |47.863|
| Full-scene exposure limits all updates |44.707|
| Full-scene exposure limits deletions |44.717|
| Full/subset relative exposure limits deletions |45.201|
| Half-source fit, no acceptance gate |47.498|
| Same fit, held-source acceptance gate |47.496|

All are fixed recipes, with no target-GT per-query winner selection. The metric
renders selected Gaussian subsets and compares to2D GT, not physical3D GT.

## Mechanisms

### Occlusion hypothesis

Subset rendering removes nonmembers that can occlude a target in source RGB.
A full-subset mask loss can therefore propose removal of a globally valid member
because it is hidden in one view. This is a plausible mechanism, not established
as the dominant measured error by these experiments.

Start from the SAME next54 retained membership as next56, using the same120updates,
old-field-qualified labels, annealing and deployment coverage objective. Compute
full-scene accepted alphaT and alpha contributions per Gaussian per source view.
Exposure is sum(alphaT)/sum(alpha). After each Adam step, scale its proposed logit
change by exposure, either for both signs or only deletion. These are update
preconditioners; they do not change forward rendering or the frozen evaluator.
Adam moments are retained, so this is not exact optimization of a newly specified
likelihood. It is not a calibrated visibility probability.

This direct exposure includes self-occlusion and suppresses roughly87–94% of
proposed update magnitude in the all-update control. Do not interpret that
fraction as the proportion of wrong updates. Reducing useful optimization explains
why these results alone cannot establish an occlusion error fraction.

The relative control renders each initial hard query subset, compares its per-
Gaussian contributions to those of the full scene, and limits deletion by their
ratio. This distinguishes target self-occlusion from removal of nonmember occluders.
New elements have unrestricted updates. Ratios are fixed at the initial set,
not recomputed as membership evolves; contributions use accepted raster hits.
The added per-query/view rendering cost is substantial relative to the simple
control. It remains worse than next56 and is rejected.

### Held-source update acceptance

Use alternating eligible source views for fitting and the remaining views for
validation. Each fitted query still gets120updates. Validation compares initial
and candidate HARD selected-subset renderings at the exact frozen alpha cutoff,
using confidence-weighted IoU on known positive/negative pseudo-label regions.
Accept only strictly improving updates. Save both the unfiltered fitted artifact
and accepted artifact before target evaluation.

The held views were not used for this refit, but the parent map/membership can
already depend on them. These are same-teacher pseudo-labels and nearby views,
not independent human labels, unseen scenes or final-test evidence. All target
GT is used only by the frozen evaluator and subsequent visual diagnosis.

Acceptance counts: figurines15/16, ramen9/9, teatime9/11, waldo10/14 eligible queries.
The gate is essentially neutral on the macro and does not catch key regressions.

## Concrete failure: coffee

After predictions were generated, inspected source images with cached detector
contours and a separate GT diagnostic. Source views8/58 detect the glass with a
tea bag for `coffee`; view103 detects only the top of the white mug; view174 can
include both regions. The inspected benchmark annotation for `coffee` denotes
the white mug extent, close to `coffee mug`, whereas `tea in a glass` denotes the
other glass. This is both mistaken entity evidence and contents/container scope
mismatch. Visual review is assistant interpretation, not new human GT.

The held-source coffee pseudo-IoU improves15.916→20.773%, so the gate accepts the
update. Yet query-averaged annotated selected IoU falls55.308→47.780% relative to
its next54 initialization. Refrigerator likewise improves held pseudo-IoU
96.112→97.130% while annotated IoU falls70.381→65.149%.

Coffee supplies visual evidence of a teacher-semantic mismatch. We did NOT inspect
refrigerator's source masks here and must not assert the same cause for that query.
The examples show why same-teacher held-view agreement cannot certify true entity
or full intended extent. They do not prove this is every query's dominant failure.

No query-specific `coffee -> coffee mug` rule, target-image feature, target mask,
or GT-dependent prompt modification was inserted into inference.

Figures:
* `output/visibility_set_next57/coffee_source_masks.png`
* `output/visibility_set_next57/coffee_gt_diagnostic.png` (GT diagnostic ONLY)

## Consequence

Keep next56 decision. Further scalar weighting, visibility damping and same-teacher
cross-view voting do not currently address the concrete contents/container and
identity errors. The next substantive intervention needs separately represented
entity identity and requested extent, preserving per-instance source evidence
instead of treating a same-name union as verified identity. Any stronger source
semantic model or scope interpretation must be compared at matched input budgets
and without benchmark-specific GT aliases. A different teacher is not automatically
independent truth either.

The source validation experiment also provides a stronger negative result than
ordinary training loss: the wrong supervision can transfer across views. A CVPR
claim requires showing that an identity/extent intervention corrects this under
held-out task definitions and beats the stronger next56 decoder. No new semantic,
ScanNet, NVOS or UQIS gain or global default promotion is claimed.

## Artifacts

* `scripts/research/decode_visibility_set_next57.py`
* `scripts/research/run_visibility_set_next57.py`
* `scripts/research/decode_source_validation_next57.py`
* `scripts/research/run_source_validation_next57.py`
* `scripts/research/summarize_visibility_next57.py`
* `output/visibility_set_next57/summary.json`

All started jobs completed. No defensive tests or audit framework were added;
geometry, source caches, annotation files and frozen evaluator were unchanged.
