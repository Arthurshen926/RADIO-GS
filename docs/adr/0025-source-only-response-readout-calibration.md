# ADR0025: Calibrate response decisions from held-out reconstruction views

Date: 2026-10-17. Status: completed; source-estimated readout not promoted.
Extends ADR0024. Does not alter any historical N01 or v10 score.

## Question

N01 improves NVOS IoU while leaving imperfect foreground extent; its mean AP
exceeds 98%. A19 shows that fitted values and readouts cannot be interchanged.
Test whether a source-estimated decision threshold can recover foreground from
the existing continuous response. This changes the readout, not the teacher,
geometry, feature representation or final response field.

## Fixed N02 design

Reproduce N01 coverage/high on all eight tasks (512x384 source grid, 120 Huber
updates). Three arms use this identical final all-source field: fixed 0.5,
source-fit calibration, and source-holdout calibration. For each query, every
fourth sorted eligible source view is a calibration view. The source-fit arm
predicts these views using the full field. The source-holdout arm fits a second
field excluding their reconstruction constraints. Both calibrators use the same
calibration views, teacher targets, confidence weights and common supported
pixels. The second field is never substituted into the target prediction.

For threshold t, maximize the ratio of soft source TP to total positive mass
plus soft FP. The candidates are k/512, k=1,...,511; ties favor the value closest
to 0.5, then the smaller k. No positive mass falls back to 0.5. The rule is fixed
for every query; the learned thresholds may differ. Zero is excluded to keep
unsupported zero responses negative. This is a teacher-based estimate, not
target-GT threshold tuning or an assertion that teacher scores are calibrated.

This explicitly relaxes N01's fixed 0.5 threshold ONLY for the two newly declared
calibrated arms. All eight raw response images and all 24 decisions are sealed
before target labels are read. Score both GT boundary conventions, IoU/P/R, and
the unchanged continuous-response AP. No per-task choice among arms is allowed.

## Interpretation and limits

Calibration views are held out from response reconstruction constraints only.
Geometry and teacher tracking remain shared across source views; this is not
independent teacher inference or independent scene validation. The final field
uses all source constraints, so calibration-to-refit transfer is also tested.
The known benchmark cohort remains development data. A source-score increase
does not guarantee target-score improvement. Neither fitting a threshold nor
ordinary histogram optimization is claimed as a new mathematical primitive.

Other benchmark outputs remain unchanged hash-verified references, not fresh
inference or evidence for universal threshold calibration. Keep small reports,
the full source histograms, code/config snapshots and hashes. Use bounded tmpfs
for the eight response images and release it after scoring. No weight copies.

## Outcome

Fixed / fit-source / reconstruction-holdout IoU is 87.7776 / 87.5151 / 87.5836.
The held-source arm improves six tasks on retained GT but loses 0.1940 points
overall. In trex it gains 30,527 TP but adds 118,640 FP. Source objectives improve
for every task; the reconstruction holdout is insufficient to transfer that
criterion into a better aggregate. Preserve the fixed N01 readout and the full
negative control, without selecting task-wise winners.

A20's exact-response audit attributes orchids' AP change to foreground +0.2212
and background -2.2373 points in score space. GT hybrids are diagnostic only.
The next local-correction experiment should preserve background rejection while
recovering supported foreground and inspect complete reference scope using legal
source evidence. This is a new hypothesis to test, not a claimed solved mechanism
or permission to gate predictions with GT.
