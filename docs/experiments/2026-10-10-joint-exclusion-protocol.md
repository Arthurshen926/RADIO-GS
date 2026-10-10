# One-pass shared member exclusion with joint positive coverage protection

Freeze before real-scene independent-condition validation and before any new
benchmark predictions/scores. Same4LERF/68texts, same72source views and compiled
16x16/view factors. No new source inputs, labels, query-fitted answers, training,
parameter search, per-query optimizer or per-case arm choice.

Fixed policy: start from native confidence_scope membership. Visit selected
Gaussians once, ascending initial score (row ID breaks ties). Only a Gaussian
incident to a CURRENTLY violated known negative factor may be removed. Require
that every initially covered positive factor remain covered after removal.
Unknown factors generate no exclusion request. No members can be added.

Joint binary opacity is equivalent at cutoff10.5/255 to sum of optical thickness
-log(1-alpha) above -log(1-cutoff). Compiled alpha cutoff/ceiling is retained;
exclusive termination near opacity1 does not alter this low binary threshold.
Use conservative positive margin1e-4 inherited from opacity compiler tolerance,
capped at initial coverage so a barely covered factor cannot be damaged. All
updates are sequential; simultaneous removal of redundant alternatives is unsafe.
This is a fixed greedy set inference rule, not a learned or globally optimal
solver, and not the complete v10 objective/footprint penalty.

Three matched arms: finite (previous confidence_scope), frozen_visibility
(same pass/order/labels, but freeze initial selected-set compositing weights),
joint_coverage (PRIMARY). Semantic response is SAME measure_prior for all arms;
actual sets drive both2D/3D. Entire4scene predictions precede target scoring.
No automatic default promotion. This is sequential development on known scenes,
not an independent final test and not a whole-five result.

Before benchmark prediction, all4 fields must pass16 seeded construction
conditions: empty/all accepted regions, one random region/view, and random region
subsets with probabilities.005/.02/.1; independently random initial member scores.
Validate both candidate arms against the actual compiled nonlinear opacity,
not only the internal linear inequality: no initially covered positive factor
may become uncovered, no new negative factor may become covered, no addition,
no change to initial scores except deleted rows becoming zero. Check every factor.
Do not read benchmark query files or cached predictions in this stage.

Additional fresh-process field+text-vector API check should reproduce benchmark
predictions without reopening observations or cached response inputs. Fixtures
are comparator only. Report sampled coverage/negative changes, deletion volume,
full actual-set IoU/P/R, semantics, failures and confidence intervals. A protected
sample is NOT a guarantee for unsampled image locations or full3D object extent.
