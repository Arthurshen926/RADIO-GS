# ADR0028: Keep positive identity guidance without treating every low score as exclusion

Date: 2026-10-17. Status: registered before N05 inference.

The complete eight-task N04 binary comparison identifies competing failures.
Dense response guidance preserves identity but locks incomplete extent (flower).
Box-only guidance can lose the query referent/extent (horns-center, trex).
N04's formal exact-source replay and AP checks are still running; no N05 score
has been observed. This is a new developmental intervention, not a final test.

N05 changes the authority of the old response at the frozen image decoder.
All three variants receive the same box and eight deterministic inferred positive
points. Point selection maximizes response confidence times interior distance;
successive points also multiply distance to the closest selected point. These
are estimated identity hints, not newly annotated user points or a proof of
membership. A single-pixel region receives one point; an empty prior falls back.

Compare: signed dense log-odds; nonnegative dense log-odds (old negative values
become neutral zero); sparse positives without a dense prior. The second is the
primary mechanism. It withholds old exclusion evidence while keeping exactly
the same positive log-odds and points. A neutral dense embedding is an operational
SAM prompt, not a calibrated representation of epistemic uncertainty.

Use the same observed target RGB, frozen SAM3, geometry, 0.5 threshold, 512x384
confidence pooling and 120-step 3D writeback as N04. Keep both direct image and
actual re-rendered 3D outputs for each branch, with v10 as the seventh arm.
Primary arm: positive-core 3D writeback. No per-task selection, GT-derived input,
threshold changes, morphology on target labels or field/teacher retraining.

Seal all 56 responses before opening GT. Both GT conventions, exact integer
confusion counts, independent confusion checks, IoU/P/R/pixel accuracy and all
source teacher agreement are reported. AP is omitted in this IoU-focused N05
experiment, not replaced by pixel accuracy or an invented proxy. Other four
benchmarks are unchanged references. No persistent dense tensors or new weights.
