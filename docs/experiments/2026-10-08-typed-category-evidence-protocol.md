# Typed category evidence, fixed before new cohort scores

This orthogonal readout control follows ADR0014 and accompanies the full-cohort
relation-training experiment. Its original member-only category output remains
reported unchanged. No new relation-training or prediction recipe is modified.

Known problem: independent instance-member scores are not semantic class scores.
The original field loses14.66pp against its own native19 control. Keep native
category evidence and consume structural support in its proper role:

`p = softmax(100 * native_cosines)` over the complete19-category menu;
`score = p + p * (1 - p) * member_support`;
choose the category with the largest score, preserving observed/void handling.

Temperature100 comes from the frozen v10 ScanNet native prior declaration.
The bounded positive-support operator is an existing v10 primitive. Their use
with Gaussian-field member support is a new hypothesis, not the old v10 solver
or an already validated classification probability. `p <= score <= 2p`; zero
member support keeps native evidence exactly. No confidence threshold or weight
search, no labels for constructing scores, no per-scene choice. Restrict19/15/10
columns only after computing complete-menu evidence; do not renormalize subsets.

Apply this exact readout to all10 rooms of each of three field conditions: the
saved parent, matched local-anchor continuation, and cross-view continuation.
Persist all30 score payloads before new scoring. Reuse the completed prediction
native/member arrays with hashes; this is equivalent to consuming those same
field/query outputs, not reading source observations or evaluator answers.

Report native, member-only, and typed evidence mIoU19/15/10, plus changed decision
counts relative to native. The original four other benchmark families are not
changed by a category-only readout. The common candidate recipe is the cross-view
field plus this predeclared typed category reader, with its full ablations visible;
it is not promoted merely because the native prior is stronger than member-only.
Keep the old default if joint evidence does not support replacement.
