# Attribute the completed architecture baseline

Parent: c6a7730. Analyze the fixed five-task outputs; do not fit fields, change
thresholds, choose checkpoints or publish diagnostic oracles as method scores.
The user requests architectural attribution and synthesis after full coverage.
This supersedes engineering optimization as the next activity.

1. Inventory all 30 distinct fields: training episode exposure, fitted scale,
   observed and anchor-eligible fractions. Re-read the compiler, query and loss
   equations to distinguish implemented mechanisms from intended capabilities.
2. ScanNet, all ten rooms: replay native/member category metrics. Decompose
   correct-to-wrong transitions on identical vertices; measure uncovered target
   classes, the perfect-observed-label ceiling, score ties and low member support.
   These distinguish missing observation from observed-point readout errors.
3. UQIS, all 67 targets and 268 queries: partition false negatives into observed
   and unobserved elements. Measure target-specific coverage, actual-set purity
   and recall, and support-to-target correspondence independently of output
   dominance. For text/image only, replay the top candidate exactly and evaluate
   all eight saved candidate extents at the unchanged default scale. The best
   candidate under GT is a privileged finite-candidate diagnostic, never a new
   prediction. It bounds this candidate-selection change only, not all identity
   models. No GT anchor or scale optimization.
4. NVOS, all eight cases: replay signed memberships, persist positive and
   negative component diagnostics, and measure how removing negative evidence
   changes the retained response at the unchanged cutoff. This is a diagnostic
   intervention, not a positive-only promotion. Prompt coverage and target
   precision/recall remain separate; all cases stay in the report.
5. LERF, all four scenes: replay the existing actual-subset masks and decompose
   precision/recall using the retained evaluator. Combine with earlier measured
   source identity/extent and readout controls without calling different runs
   a matched causal comparison.
6. Native fusion: on each field's two fixed development views, measure source
   teacher direction retention and partial-region query ranking before/after
   Gaussian fusion/projection. Use stored source teachers, not new encoders or
   public queries. This diagnoses the whole fusion/projection path, not fusion
   alone or cross-view complete-instance correctness.

Target labels are permitted only in these explicitly privileged diagnostics.
Baseline predictions and model payloads remain immutable. Report exact
decompositions, controlled intervention effects, associations and untested
mechanistic hypotheses separately. Do not add their metric deltas together:
coverage, identity, scope and readout errors interact. Unequal teachers, source
budgets and label supervision still prevent causal attribution of the full
v10-to-current gap. Outputs live in
`output/query_field_20261008/architecture_attribution_v1`.
