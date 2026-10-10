# Current-reference evidence within the composed Gaussian set

Fixed all eight NVOS requests, seven existing fields/120 source frames. Same
retained v10 positive-derived SAM reference masks, inherited RGB/SAM production
cost (not rerun). No new source data, training, scene state, target labels in
prediction, target-camera conditioning, per-case routing, or threshold search.
Sequential development study, not a blind test or unified five-task result.

Two arms, all16 predictions and eight baseline payload replays before scoring:

- visible_reference reproduces reference_visibility_v1 visible_reference.
- Primary conditioned_members keeps exactly the same full-scene reference
  observation, greedy region choices and candidate union. It additionally assigns
  reference foreground/background evidence to Gaussian rows inside that union.

Let A be the normalized full-scene contribution matrix, o the observed-row mask,
q the fractional reference foreground raster after the existing area resampling,
and c the binary composed set. For pixels with (A o)>0.5 only, W=A/(A o); other
pixels abstain. Each Gaussian has visible evidence mass d=W^T 1 and positive mass
p=W^T q. Its score is c*p/d where d>0; where d=0 retain c. The selected set is c
intersect {d=0 or p>=d/2}; exact ties preserve prior selection. It cannot add rows
outside c. Both response and actual-subset metrics are measured for this SAME
output; response is continuous, actual selection is discrete. Scores are not
calibrated probabilities. Tiny nonzero evidence remains a limitation; this is
not a solution to hidden-part identity, as a unit counterexample demonstrates.

The operation minimizes a separable visibility-weighted foreground/background
assignment disagreement on the candidate rows. It does NOT minimize the coupled
rendered-mask residual, inherit v10's multiview tracking, protect boundaries with
v10's eroded unknown band, or infer instance identity behind an occluder. This
explicitly isolates whether missing within-region reference exclusion is useful.
No optimization iterations or new trainable parameters are introduced.

Record visible/unseen/rejected/tied member counts and verify unseen rows unchanged.
All candidate region IDs and greedy traces must equal the control. Report both
formal metrics, precision, recall, AP, all case deltas and scene-cluster intervals.
If precision improves but recall falls, do not call it restored functionality.
Do not compare these mask-input results as raw-scribble gains. No default promotion.
Cold replay covers one copied-field case with Python source-access denial; do not
claim full OS isolation. Inspect full-cohort actual-set error figures after scores.
