# Native prior contribution and source-evidence conflicts

Status: complete30 scene-condition runs (three conditions x10 scenes). Retain
system_v2 unchanged. No new teacher, geometry, source image, class-specific
parameter or evaluator was introduced. These are development results, not an
independent test or a SOTA claim.

## Questions and fixed conditions

Does native map information help after source grounding, or are scores explained
by source tracking alone? Does assigning the strongest proposal's identity
confidence to an entire category union hide a correctable mismatch? Does a
partial categorical observation better express the point-label task than19
independent binary source observations?

All experiments reuse the same1158 authorized source frames, frozen unlabelled
mesh geometry, native1536 map and literal SAM3 detections. Queries are freshly
encoded. Predictions are written before the unchanged point evaluator reads
labels. No per-scene winner selection or parameter sweep is used.

## Full10 results

| Condition |19-class mIoU|15-class mIoU|10-class mIoU|
|---|---:|---:|---:|
|Frozenv2 native prior + visible category vote|57.921506|58.972766|70.905986|
|Spatial proposal-owner identity confidence|57.671561|58.732932|70.839214|
|Uniform1/19 prior, identical0.1 regularization|55.830779|56.187885|68.723240|
|Partial categorical observation vote|56.800609|57.427235|68.499532|

The native-prior information contributes2.090728/2.784880/2.182746 percentage
points relative to the uniform prior. This changes semantic prior content,
not geometry, map support, source identity teacher or regularization strength.
It is **not** an ablation removing every VFM or every map dependency. It supports
the narrower claim that native field information helps this fixed source-assisted
category system. It does not prove compact multi-teacher learning, reference-query
benefit, or generalization to new scenes.

Native information helps19-class IoU in7/10 scenes; it hurts scene0200 by1.07pp,
scene0590 by2.81pp and scene0645 by3.05pp.15class helps7/10 and10class8/10.
A uniform globally active semantic prior is therefore useful but not uniformly
reliable. Per-scene switches based on these evaluation results are prohibited.

## The observed confidence mismatch is not an error-rate estimate

Existing source evidence takes the maximum mask probability over instances,
then applies the maximum identity confidence over those instances to the whole
view/category. Across raw positive source pixels (mask probability>.5),31.6075%
have a mask-probability owner with lower confidence than the view/category maximum.
This counts potential confidence borrowing, including overlap and duplicate
proposals; it is not31.6% false labels or3D errors.

The first candidate binds positive eroded pixels to the confidence of the proposal
that supplies their maximum mask probability. It leaves the negative rule and
native prior unchanged. Full10 changes by-0.249945/-0.239834/-0.066772pp, so it is
not promoted. This asymmetric change does not establish a correct treatment of
negative evidence or completeness, and ties/overlapping detections also affect
proposal ownership. Do not call it a proven pipeline bug fixed by this candidate.

## Partial categorical observations: a bounded failed method candidate

The second candidate keeps the complete vocabulary together before lifting.
For each proposal it erodes the mask at the existing radius, multiplies its
probability by its own native identity confidence, and takes the maximum
within each category. It normalizes the resulting category strengths at covered
pixels. Maximum individual strength weights the observation; overlap cannot
increase confidence above that maximum. Uncovered source pixels are unknown,
not negative. The categorical vector is lifted by the same projection adjoint,
with one common observation mass and the same0.1 native visibility prior.

This avoids borrowing identity from spatially unrelated proposals and represents
partial coverage explicitly. It nevertheless loses1.120898/1.545531/2.406454pp
againstv2. Changes in negative evidence, categorical normalization, boundaries
and confidence are coupled; the result does not isolate one of them as the cause.
The implementation is opt-in research only. Do not replace the default or
continue an unrestricted threshold/normalization sweep.

## Evaluation-only localization of harmful native corrections

`prior_conflicts.json` compares the saved uniform-prior andv2 point predictions
against labels only after inference. On1,242,774 valid19-class points, native
information changes130,871 predictions:52,853 are corrected,36,455 change from
correct to wrong, and41,563 remain wrong under a different class. These pooled
counts are not scene-macro IoU and do not define a deployable selector.

Largest harmful correct-to-wrong transitions include:

| Ground truth / uniform-prior correct label | Native-prior result | Points |
|---|---|---:|
|floor|chair|4216|
|cabinet|counter|3915|
|shower curtain|curtain|3237|
|wall|floor|2873|
|wall|picture|1890|

The extra errors are caused by adding the native prior under this controlled
comparison, not by changed geometry or new source images. Their origin within
the native path—teacher recognition, region pooling, cross-view writes, or
uncalibrated class competition—is not uniquely separated here. The examples
show both spatial/class contamination and fine-to-generic substitution; treating
all remaining errors as only scope completion would be misleading.

## Decision and next mechanism requirement

Retainv2 and its22-job validated system. Neither method candidate improves the
complete cohort. This round adds causal component evidence and failure
localization, **not a headline accuracy increase**. LERF/NVOS are unchanged;
no new LERF, reference-feature or four-query claim follows from ScanNet results.

A useful next mechanism must distinguish when source identity is sufficiently
reliable to preserve and when an uncertain source observation needs the map.
Do not blindly strengthen, remove or veto the native prior. First compare the
native observations before pooling with the written element features for the
harmful corrections, using the same input permissions. Field-coordinate
preservation alone does not certify element-level semantic correctness.

## Artifacts and reproduction

- `output/native_evidence_contribution/summary.json`: two full10 controls.
- `output/native_evidence_contribution/identity_assignment_counts.json`: raw
  positive-pixel confidence ownership counts, without GT.
- `output/native_evidence_contribution/prior_conflicts.json`: evaluation-only
  correction/spoilage counts and transitions.
- `output/categorical_observation/summary.json`: full10 categorical candidate.

```bash
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
 scripts/research/probe_native_evidence_contribution.py
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
 scripts/research/run_categorical_observation_cohort.py
/root/miniconda3/envs/cybersim_agent/bin/python \
 scripts/research/diagnose_native_prior_conflicts.py
/root/miniconda3/envs/cybersim_agent/bin/python \
 scripts/research/measure_source_identity_ownership.py
```

All jobs completed; GPUs released. The uniform control changes only prior
information and retains the native identity teacher. Research switches
`source_identity_assignment=spatial_proposal` and
`source_category_fusion=categorical_observation_vote` are absent from activev2
Scene configurations. The candidate is specific to mutually exclusive class
menus and is not silently applied to LERF whole/part or reference tasks.
