# Joint partial-element and actual-subset relation training

Follow-up to the completed element-relation study at `f364d25`. That study
improves source BCE and NVOS response but seriously damages LERF selections;
correct-support UQIS recall increases while precision and IoU decrease. This
protocol is a response to those measured failures, not an independent final test.
It is fixed before any new model or benchmark prediction in this follow-up.

## Intervention

One new arm, `joint_conditional_relation`, starts from the exact same all30
original parents. Retain the native/geometry residual, one confidence scalar,
protected native/geometry/support and original acceptance parameters, the v2
query program, thresholds, candidates and all source data. No v10 labelled head,
extra labels, runtime source bank or per-task model routing.

Use the same source pair sequence, element samples, target confidences, scales,
two epochs, seed20261008,128 samples per stratum, Adam .001 and clipping1 as
`conditional_relation`. Add the matched protected control's two-view relaxed
subset-alpha loss, using the SAME primary support anchor for both cameras.
The total objective is fixed:

`0.5 * element_BCE + 0.25 * primary_rendered_BCE + 0.25 * secondary_rendered_BCE`.

Both loss types retain known positive/negative mass and leave unknown evidence
unknown. The view-balanced rendered term constrains the transient selected
opacity of the full observed carrier, including elements hidden in the original
full-scene projection. It is a differentiable surrogate; hard actual selected
sets still require retained benchmark verification. Loss weights are not swept.

Compute each branch's gradients before one shared clipping/optimizer step.
No extra regions, source pairs, group epochs, native updates or output capacity
are introduced. More forward/backward computation is explicitly part of the
intervention; do not call this a compute-matched comparison or pure rendering
causal proof. Compare with the completed conditional-only and element-only
arms, matched protected rendered control and previous common program.

## Accounting and source audit

Record sampled row coverage as before. Also accumulate nonzero counts and
absolute magnitudes of the ACTUAL weighted rendered-loss gradient at each
full member logit before parameter clipping. Report observed-row coverage and
quantiles. These are loss-gradient measurements, not optimizer parameter-update
coverage, not equal to source projection mass, and not an inference artifact.

After final training, replay the same held-out source pairs/samples using
seeds20261011/20261012. Report parent/final partial element BCE and parent/final
paired rendered BCE on all valid regions. No early stopping, checkpoint choice
or loss-weight change. The same metric/prototype parent formula is already
checked against deployment. Rendering comparison uses frozen original grouping
and protected base parameters, without the new residual, for the parent.

## Full completion and interpretation

Build all30 exports before benchmark prediction; verify complete source traces
and element sample digests against the conditional-only control. Export capacity
must match that control exactly. Complete predictions before scoring all four
LERF scenes, eight NVOS inputs, ten ScanNet rooms and all268 UQIS inputs.
Keep both LERF readouts, NVOS response and actual subset, ScanNet19/15/10,
UQIS four-input IoU/AP, paired scene intervals and precision/recall.

Repeat all67 point-query observation diagnostics, retaining the59 correct
supports, and native-output invariance across all268 UQIS inputs. Copy one real
export for fresh-process source-denied replay of four inputs and a category menu.
Report whether LERF overselection and point precision recover, and how much NVOS
benefit survives. Source loss improvements cannot substitute for these results.
No per-task/scene winner selection or automatic default promotion.

The scalar exposure summary still does not encode pair co-visibility or mask
disagreement. Native mean descriptors and true text/crop-query function retention
are unchanged. Joint training is not presumed to repair those separate gaps.
