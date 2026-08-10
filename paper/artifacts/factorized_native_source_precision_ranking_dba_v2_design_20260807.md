# Factorized-native DBA-v2: source-global precision-constrained ranking

Status: frozen after the source-only DBA-v1 rejection and before DBA-v2 code,
authority, training, or any target/query execution.

## Source-only failure evidence

DBA-v1 starts from macro precision/recall/F1
`0.37377 / 0.14378 / 0.20764`.  At step 8 it reaches
`0.21026 / 0.37591 / 0.26925`: recall and F1 improve, but precision falls
below the fixed `0.25` floor and sampled margin rank correlation drops from
`0.69390` to `0.68571`.  By step 64 the predicted-positive rate has expanded
from `0.000865` to `0.008400`, precision is `0.14187`, and F1 has fallen to
`0.22309` despite recall `0.52295`.

The failure is structural.  Equal total positive/negative mass makes every
negative unit carry about 1/500 of a positive unit's gradient.  The many easy
negative units dominate the negative class mean numerically but the few
highest-margin false positives that determine zero-boundary precision are not
given special status.  The loss therefore buys recall by translating a broad
response tail across zero.  Its confidence term does not preserve the global
ordering strongly enough to prevent the observed rank regression.

## Frozen candidate

DBA-v2 keeps the exact promoted contrast-V2.1 direction-only checkpoint, full
V2.1 visual objective, four train/two held-out validation scenes, 806-row
target-blind fit bank, four canonical negatives, exact per-view teacher, and
64-step complete-coverage schedule from DBA-v1.  It introduces no inference,
scene, query, or calibration parameter.

For each `64 x 806` source-train batch:

1. Compute the exact teacher probability and exact student inference margin.
2. Keep every teacher-positive unit.  From teacher-negative units select the
   highest student-margin units, with count
   `min(N_negative, 3 * N_positive)`.  The ratio three is not tuned: precision
   `>= 0.25` is equivalent to allowing at most three false positives per true
   positive.  Selection is deterministic and only chooses which negative
   margins receive gradient; it introduces no parameter.
3. Boundary loss is half the mean positive `softplus(-10 m)` plus half the
   mean selected-hard-negative `softplus(10 m)`.  This directly pushes the
   zero-boundary tail that controls false positives rather than averaging it
   with millions of easy negatives.
4. Confidence preservation is class-balanced Smooth-L1 between
   `sigmoid(10 m)` and exact teacher probability over the same positive and
   selected-hard-negative set, beta `0.05`, weight `0.25`.
5. Boundary ranking pairs each positive margin with deterministic quantiles of
   the selected hard-negative margins and applies
   `softplus(10 * (m_negative - m_positive))`.  This has weight `0.25` and
   directly raises positive units above the false-positive tail.
6. Global-order preservation sorts all batch teacher probabilities, pairs
   evenly spaced lower/upper quantiles, and applies
   `softplus(-10 * (m_upper - m_lower))` whenever the frozen teacher
   probability gap is at least `0.05`.  It uses at most 4096 deterministic
   pairs and weight `0.25`.  This targets the sampled held-out rank metric
   without validation fitting.

The complete DBA-v2 auxiliary remains weighted `0.25` relative to the full
visual objective.  All scalar weights reuse the existing `0.25` source
auxiliary convention; there is no sweep.

## Schedule and promotion

- 64 AdamW steps, learning rate `2e-4`, inherited weight decay and gradient
  clipping.
- 64 contiguous cyclic rows from each of four train scenes per step, covering
  every one of 4096 rows exactly once.
- Evaluate steps `0, 8, ..., 64` on all `4096 x 806` pairs for each held-out
  source scene.
- Reuse every DBA-v1 boundary and visual promotion gate unchanged, including
  precision `>= 0.25`, recall `+0.02`, F1 `+0.01`, maximum rank drop `0.005`,
  positive-rate cap, and all visual-preservation checks.
- Reuse DBA-v1 checkpoint ranking unchanged.

No target descriptor, benchmark query, target mask/label, or benchmark metric
may be opened before this source-only gate passes and is frozen.
