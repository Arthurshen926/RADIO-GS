# Metrics and reporting

## Per-query target mask

The evaluator constructs a binary vertex mask by comparing the private official
instance-ID array with the target instance. Predictions are continuous
`float32[V]` values; no method-specific mesh conversion occurs in the evaluator.

## Ranking metrics

- **Average precision (AP):** non-interpolated, tie-aware binary AP over all
  official mesh vertices. This is the primary calibration-free query metric.
- **Oracle IoU:** maximum IoU over complete score-tie thresholds. It diagnoses
  localization/shape potential but is not a deployable operating point.

## Fixed-boundary metrics

- fixed IoU at probability 0.5;
- accuracy at IoU 0.25 and 0.50;
- selected purity and positive coverage;
- target-centroid error in metres;
- maximum IoU with declared same-class distractor instances.

The same-class diagnostic always reports its denominator. Queries without a
declared same-class distractor are not silently counted as zero.

## Aggregation

For every metric:

1. average queries within each scene and modality;
2. average scenes equally, preventing large scenes from dominating;
3. obtain 95% confidence intervals with 2,000 scene-clustered bootstrap samples
   and frozen seed 20260813.

## Primary v0.2 summaries

`UQ-Rank` is the equal-modality mean of scene-macro AP over the 31-target Core
cohort. It is the primary candidate metric because it is independent of score
calibration.

`UQ-Mask` is the equal-modality mean of scene-macro fixed-boundary IoU over the
same cohort. It becomes valid only after probability mapping and threshold are
fit on an independent dev cohort and bound by a calibration receipt. Without
that receipt it must be labeled `diagnostic_unverified`.

The Relational Text Challenge reports text-only AP and mask diagnostics over 36
targets. It is never averaged into UQ-Rank or UQ-Mask.

## Required report metadata

Every result should state benchmark version, split/tier, formal eligibility,
method identity, field-storage class and bytes, prediction seal hash, calibration
status, query/target/scene counts, bootstrap settings, and any evaluator recovery
or post-hoc correction.
