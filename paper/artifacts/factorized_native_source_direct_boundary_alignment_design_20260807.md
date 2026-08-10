# Factorized-native source direct-boundary alignment: implementation proposal

Status: source-only engineering proposal prepared after the already-recorded
Figurines unary diagnostic. It is **not** represented as a pre-target
preregistration. Before an execution, the exact trainer, weights, schedule,
inputs, and gates below must be SHA-bound in a separate execution authority.

## Audit result

The promoted contrast V2.1 model is a visual-geometry model, not an absolute
text-boundary model. Its five losses are raw visual cosine, common-centred
visual residual cosine, residual Gram geometry, visual-prototype response, and
spread. The item named `absolute_visual_probe_calibration` compares
`student_i dot teacher_j` with `teacher_i dot teacher_j`; it does not contain a
text embedding or the four canonical negatives. Consequently it cannot
identify the inference decision plane

`descriptor dot positive_text - max_k descriptor dot canonical_negative_k = 0`.

The existing generic response loss is not a drop-in fix for the factorized
candidate:

1. it is not wired into factorized-native V2.1 training;
2. its absolute-relevance Smooth-L1 is averaged over all region/query units,
   while source teacher-positive rates are only 0.1334% and 0.3130% on the two
   held-out scenes, so its gradient is negative-dominated;
3. it constructs the teacher target by separately log-mean-exp reducing
   positive and negative responses across views, then subtracting and applying
   sigmoid. The exact teacher used by the current frozen audit instead computes
   per-view `max-negative` and sigmoid first, then averages valid-view
   probabilities. These operations do not commute.

There is also a smaller deterministic coverage defect: V2.1 runs 60 steps with
64 rows/scene/step over 4096 rows. The cyclic iterator therefore trains on only
3840/4096 rows (93.75%) in every source-train scene. Any follow-up should use
exactly 64 steps for one complete source epoch, or 128 for two; 64 is the
minimal experiment.

## Minimal candidate (DBA-v1)

- Initialization: the promoted contrast V2.1 direction-only checkpoint.
- Parameters: the same global factorized-native readout; no new inference
  parameter, query head, scene embedding, or calibration parameter.
- Text supervision: the already frozen target-blind 806-row fit bank and the
  exact frozen four-row canonical negative bank. No benchmark query string.
- Teacher target for source region `i`, generic text `q`, valid view `v`:

  `p_T(i,q) = mean_v sigmoid(10 * (t_iv dot q - max_k t_iv dot n_k))`.

- Student inference margin:

  `m_S(i,q) = s_i dot q - max_k s_i dot n_k`.

- Direct primary risk:

  `0.5 * mean_{p_T>=.5} softplus(-10*m_S)
   + 0.5 * mean_{p_T<.5} softplus(10*m_S)`.

  Equal class mass prevents the 0.1--0.3% teacher-positive class from being
  erased. Unlike a positive-slope temperature, gradients pass through the
  descriptor and can rotate it across the zero-margin boundary.
- Confidence preservation: add `0.25` times class-balanced Smooth-L1 between
  `sigmoid(10*m_S)` and `p_T` (`beta=0.05`).
- Visual preservation: retain the complete contrast-V2.1 visual objective.
  Use the direct risk as an auxiliary, not a replacement. The initial
  engineering run should freeze a single global auxiliary coefficient before
  validation is evaluated. A defensible starting coefficient is `0.25`,
  matching the earlier source-response auxiliary and visual-probe weight.
- Schedule: 64 steps, 64 rows per source-train scene per step, four scenes with
  equal gradient accumulation, AdamW at the inherited `2e-4`, inherited weight
  decay and gradient clipping. Evaluate steps 0, 8, 16, ..., 64.
- Batch validity: each source batch must contain both teacher boundary classes.
  Fail closed otherwise. The implemented source audit confirmed that no row
  interleaving is needed: all 64 contiguous batches in all four train scenes
  contain both classes and the 64-step cycle covers all 4096 rows exactly.

The implemented loss primitive is
`radio_gs/losses/factorized_native_source_boundary_alignment.py`.
The executable support audit is
`radio_gs/scripts/audit_factorized_native_source_boundary_batch_support.py`.

Observed positive pair counts per 64 x 806 batch were: scene0001_00 min/mean/max
`64/134.31/253`, scene0002_00 `90/193.66/313`, scene0003_00
`29/65.63/110`, and scene0005_00 `24/52.55/114`. Every scene had zero
empty-positive batches and zero empty-negative batches. This is source-only
evidence; no target or benchmark artifact was opened.

## Frozen source-validation promotion gate

All metrics use all 4096 canonical rows and all 806 generic text rows on each
of `scene0004_00` and `scene0008_00`. Step 0 is the exact promoted V2.1
checkpoint. Promote only if every check passes:

### Boundary capability (both scenes independently)

- class-balanced hard BCE strictly improves by at least `1e-4`;
- teacher-positive recall improves by at least `0.02` absolute;
- F1 improves by at least `0.01` absolute;
- precision is at least `0.25`;
- class-balanced Brier score strictly improves by at least `1e-4`;
- sampled teacher/student margin rank correlation is no worse than step 0 by
  more than `0.005`;
- predicted-positive rate is at most `max(0.02, 8 * teacher-positive-rate)`,
  excluding a trivial all-positive solution.

### Visual capability preservation (both scenes independently)

- mean all-view cosine drop from step 0 is at most `0.002`;
- p05 all-view cosine drop is at most `0.005`;
- centred-residual mean and p05 drops are each at most `0.01`;
- student/teacher spread ratio remains at least `0.75`;
- centred pair-Gram correlation remains at least `0.20`, with MAE increase at
  most `0.01`;
- absolute visual-probe correlation remains at least `0.20`, response MAE
  increase is at most `0.01`, and response std ratio stays in `[0.75, 1.25]`.

Eligible checkpoints rank by: macro F1 improvement, macro recall improvement,
macro balanced-BCE improvement, macro centred-residual cosine, then earliest
step. Validation contributes neither gradients nor hyperparameter selection.

No target descriptor, exact LERF query, target mask, target label, or benchmark
metric may be opened until this source gate is frozen and passed.

## Compute estimate

At the fixed 64-row batch, the direct score tensor is only `64 x 806`; at four
teacher views the temporary teacher score tensor is at most `64 x 4 x 806`.
The text-side work is roughly 0.4 billion multiply-adds (about 0.8 GFLOP) per
source scene per step before kernel fusion, but it is regular matrix multiplication and the
peak extra activation footprint is well below 100 MB. Teacher probabilities
can be precomputed once into six source-only `4096 x 806` float16 caches
(about 40 MB total) because they never depend on model parameters. With that
cache, a 64-step four-scene fine-tune should be in the same order as the prior
60-step contrast run, not a benchmark-length experiment. Full two-scene
validation adds about 6.6 million student margin values per checkpoint; an
8-step interval gives nine evaluations including step 0.

This experiment has a clear falsification outcome: if the source boundary
metrics cannot improve without violating visual-preservation gates, the
remaining limitation is the direction-only readout capacity or the target-
blind text-bank coverage, not graph propagation or scalar calibration.
