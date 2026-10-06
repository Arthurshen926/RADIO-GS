# Source admission runtime and adaptive source coverage

This round continues the fixed provided-unlabelled-mesh ScanNet development
track. UQIS remains paused. Existing sealed evaluators and source files are not
overwritten. No independent-test or same-input SOTA claim is made.

## Correctness repair: source rejection must affect actual inference

The previous source-gated 52.05/52.94/66.44% ScanNet result was assembled from
saved arm metrics. The underlying evaluation still called the rejected scope
model. A real runtime now validates the scope and admission source receipts,
their model ancestry, explicit boolean decisions and recorded validation
comparisons. Invalid receipts fail. A valid source rejection returns calibrated
native logits without invoking scope inference.

All eight scenes were replayed from the full field and the fixed query encodings.
Ground-truth label files and training arrays were forbidden. Prior evaluation
reports were forbidden until predictions and source decisions were saved.
All predictions match the previous source-only decision exactly; scene0590_00
has zero scope forward calls. The other seven scenes execute bounded scope.
The macro remains 52.0479/52.9352/66.4351%: this is a runtime repair, **not a new
performance gain**. The source views were seen by the parent field, so passing
these source checks does not prove independent generalization or absence of
native-semantic harm. All four current LERF fields pass these two source checks;
the rejection bypass alone will not improve their scores.

Twenty-six focused gate/bounded-readout/category-protocol tests pass, including
rejected-branch nonexecution, forged ancestry, contradictory acceptance,
nonfinite evidence and preservation of the original admission tolerance.
A further 30 focused regression tests covering registered geometry/projection,
visibility, query identity, scope relations and evidence bindings pass (56 total).

Evidence: `output/source_accepted_runtime_next29/summary.json`; runtime:
`radio_gs/querying/source_accepted_category.py`; replay:
`scripts/audits/replay_source_accepted_category_next29.py`.

## Method control: expand coverage only where source geometry justifies it

Reuse the sealed mesh visibility matrix, exclude the existing validation views,
and extend greedy source coverage to 32 views. The new rule is fixed before new
semantic reconstruction/evaluation: expand to 32 only if weighted visibility
increases by more than five percentage points; otherwise retain the same16-view
field. This selects 0000,0140,0590 from source geometry alone. All eight scenes
remain in the final denominator. This is an increased-input/training-budget
experiment, not an equal-cost comparison.

| Scene | 16-view visibility | 32-view visibility | Rebuild |
|---|---:|---:|---|
| 0000 |75.62%|89.01%|yes|
| 0062 |94.53%|97.08%|no|
| 0070 |88.73%|91.79%|no|
| 0097 |90.59%|92.18%|no|
| 0140 |86.63%|93.86%|yes|
| 0347 |94.56%|96.63%|no|
| 0400 |92.18%|94.40%|no|
| 0590 |86.65%|92.51%|yes|

All old selected16 frames are retained within the new32 sets. The three scenes
rebuild their complete source teachers, shared field, range/threshold/admission
dependencies. Evaluation uses the runtime source gate, including rejection
bypass; diagnostic source-rejected models are never silently promoted.

Registration: `output/scannet_expanded_budget_next29/adaptive_plan.json` and
`evaluation_registration.json`. **All three complete reconstructions, cold checks, final evaluations and the
eight-scene summary are now complete.**

## Additional source-side checks

The expanded source initialization supports 80,174 / 360,783 / 219,809
elements in 0000 / 0140 / 0590, versus 76,120 / 341,098 / 210,672
previously: 32,876 additional supported elements. These are source-support
counts, not target recall. Source basis energy retention is 94.67 / 95.51 /
94.99 percent; this does not certify unseen-query semantic retention.

The inherited frozen base driver contains a descriptive `train16` inventory
string. Its actual frame lists are dynamic. The current source plans, teacher
manifests and evaluator enforce 32 training frames and the unchanged original
two validation frames. The frozen driver is not rewritten merely to change this
metadata string.

Assistant visual inspection of source scene0000 frame3060 found cushions on
the sofa, while the caption teacher proposed a dog. This is recorded in
`caption_source_visual_diagnostic.json`, with no training use or target GT.
A same-image, same-checkpoint SAM replay at the original0.7 confidence threshold
returns **zero dog masks**, recorded in `caption_dog_grounding_diagnostic.json`.
Thus this particular caption hallucination is rejected by grounding; it is not
evidence of a false dog written into the field. No corrective patch or training
label is introduced for it. This one example does not certify all source phrases. The completed inventory
also confirms zero dog regions in reviewed frame3060, but retains one region
in frame1940 at confidence0.707; its physical identity is not independently
verified, so neither rejection nor correctness is claimed for the whole scene.

The completed `native_protection_audit.json` compares actual tensors in all three
base fields: supported64D latent prefixes and decoder bases are exact, decoder
biases are zero, and observation support equals the source initialization. All
training teacher pixels were visited; validation teacher pixels had zero visits.
This excludes a repeat of training-induced native-prefix drift in these builds,
but does not certify compression, fusion, calibration or final range semantics.

## Remaining acceptance boundaries

- Source scope acceptance is enforced at category inference, but its source
  examples were visible to the parent map; it is not an unseen-family guarantee.
- Overlap-normalized pixel evidence still averages whole/part and potentially
  incompatible region semantics. More source views alone do not resolve this.
- Some legacy two-view pseudo-validation cohorts have only two tasks per query
  type. Their high IoU cannot support a broad query-generalization claim.
- This control changes source coverage and budget with fixed geometry, fixed
  native dimensionality and the existing complete dependent-head recipe.
- There is no new LERF2D/selected-render LERF3D or NVOS score in this round; UQIS
  remains paused. No shared four-query or SOTA acceptance is asserted.

## Completed fixed-protocol results

All values below are eight-scene macro mIoU in the frozen VALA Gaussian-domain
19/15/10-class ScanNet protocol, on the same provided-unlabelled-mesh input
track. The three rebuilt scenes use32 source training views; the five others
retain exactly their16-view fields. This increases total training views from128
to176; it is not an equal-input or equal-compute architectural comparison.

| Split | Previous source-accepted | Adaptive source budget | Delta (pp) |
|---|---:|---:|---:|
|19|52.0479%|54.5190%|+2.4711|
|15|52.9352%|55.6667%|+2.7315|
|10|66.4351%|69.8166%|+3.3815|

| Rebuilt scene | 19-class before → after | 15-class before → after | 10-class before → after | Scope admitted |
|---|---:|---:|---:|---|
|scene0000_00|39.72 → 47.43|36.68 → 44.86|51.51 → 61.54|True|
|scene0140_00|42.29 → 48.61|48.32 → 55.26|64.20 → 72.49|True|
|scene0590_00|36.17 → 41.91|42.54 → 49.27|62.95 → 71.69|False|

All nine final scene/split comparisons improve. No per-scene target winner
selection was performed. The source-rejected0590 scope is bypassed, including
its forward call, exactly as predeclared. Its new scope source validation is
47.2447%, below its retained parent47.6277%; more views did not make that
correction reliable. The other two source scope checks pass.

## Where the gain appears

The same fixed predictions are decomposed without changing the deployment rule.
Both native and accepted predictions replay exactly in all three decompositions.

| Readout | Recipe | 19 | 15 | 10 |
|---|---|---:|---:|---:|
|native|baseline|45.0897%|46.0969%|58.2484%|
|native|candidate|47.0795%|48.3839%|60.2158%|
|calibrated_native|baseline|46.6024%|47.6618%|59.9470%|
|calibrated_native|candidate|48.9405%|50.3264%|63.0893%|
|source_accepted|baseline|52.0479%|52.9352%|66.4351%|
|source_accepted|candidate|54.5190%|55.6667%|69.8166%|

For19 classes, about1.99pp of improvement is already present in native
matching,2.34pp after calibration, and2.47pp in the final readout. This locates
most of this measured gain upstream of scope correction. The intervention
changes source coverage, fusion evidence, the source-fitted basis and complete
training/head fitting together; it does not isolate view count from all of
these effects or prove that every query improved.

There is an important counterexample:0140 native10-class mIoU regresses from
56.9875% to53.8167%, while the new calibration reaches65.9411% and final scope
72.4947%. The final gain must not hide that native regression. Whole/part
mixture and compression/aggregation remain unresolved; a separate, explicitly
post-hoc single-scene basis diagnostic is excluded from the eight-scene result.

Primary evidence: `summary.json`, `calibration_ladder_summary.json` and all
three bound evaluation reports under `output/scannet_expanded_budget_next29/`.
All1773 frozen source hashes remain unchanged (`frozen_source_audit.json`).

## Completed localization of the native regression

This is a **post-hoc single-scene diagnostic**, selected because0140 regressed
in native10-class matching. It is not an extra benchmark candidate and is not
spliced into any macro result. Reconstruct the exact same query-independent
32-view pixel-weighted SigLIP evidence before reading the saved query encodings.
Keep geometry, observed support, masks, weights and evaluation fixed; compare
the full1536D fused vectors with projections onto the new and previous64D bases.

| Native representation on the same new evidence | 19 | 15 | 10 |
|---|---:|---:|---:|
|uncompressed1536|41.6516%|47.8316%|62.9234%|
|new_basis64|38.9073%|44.6936%|53.8167%|
|old_basis64|36.8700%|42.3691%|58.1024%|

The replay of current64D predictions is exact for every Gaussian in every
split (maximum score roundoff1.05e-7). Full1536D native10-class matching is
62.9234%, versus53.8167% with the current64D projection: a9.1067pp loss at
compression, with aggregation and geometry held fixed. The previous64D basis
recovers58.1024% for10 classes, but worsens19/15 relative to the new basis.
Thus merely restoring the previous matrix is not a shared repair. This
diagnosis implicates the current projection strategy, not a proof that every
possible64D representation is incapable. Source energy retention near95%
does not guarantee preservation of query-class distinctions.

These are native-only results. The actual final0140 result remains
48.6140/55.2571/72.4947%. The full1536D diagnostic has **not** refitted the
calibration or range heads, and neither diagnostic basis is a drop-in
replacement for the current trained field. No incompatible basis substitution
or target-selected per-scene promotion is made. Evidence:
`scene0140_00/native_basis_diagnostic/report.json`.

## Next focused method work

1. Retain the verified source-coverage intervention as a versioned development
   candidate, with its larger input/training cost explicit.
2. Repair compression against **source-only native query-response/rank
   preservation**, not reconstruction energy alone. Compare a query-aware
   basis, a source-selected larger basis or a protected native residual using
   independently held-out source views; keep benchmark queries out of fitting.
   Rebuild affected structural/calibration/range dependencies before complete
   eight-scene comparison. Do not insert the post-hoc diagnostic scores above.
3. Continue separating incompatible whole/part evidence and strengthen
   independent range admission.0590 remains a concrete source rejection,
   despite greater coverage; the deployment bypass remains necessary.
