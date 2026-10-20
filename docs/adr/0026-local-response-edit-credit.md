# ADR0026: Verify local response edits using source reconstruction holdouts

Date: 2026-10-17. Status: completed; selective benefit versus matched attenuation, not promoted.
Extends ADR0024/0025. Does not change historical outputs or scientific defaults.

N02 rejects its source-estimated global threshold as an aggregate improvement.
A20 locates an orchids ranking loss in changed background scores, without proving
an upstream cause. N03 tests selective 3D edits to the original v10 response.

Use the same N01 coverage/high candidate and the same every-fourth-source split
as N02. No new teachers, features, geometry, target RGB, or threshold search.
Let b be the v10 response, c a candidate, A a shared-support normalized source
operator, and rho the Huber loss. Attribute each pixel's loss improvement along
the straight response path b+t(c-b) to its contributing elements. For element i,
credit is (c_i-b_i) times the adjoint of the weighted Huber loss secant. The
signed credits sum to the complete source loss improvement. Accumulate absolute
pixel contributions separately as gross credit. Gate = max(net,0)/gross, or
zero for no evidence; apply the gate to the all-source edit. A split-derived
gate also requires the split and final edit to have the same direction.

This exact accounting identity is not a target-accuracy guarantee, nor a claim
that individually gated edits retain the original loss-decrease identity.
Credits are source-teacher based; target-GT foreground/background groups from
A20 never enter inference. Holdout excludes reconstruction terms only: teacher
tracking and geometry remain shared. v10's anchor already used all source views;
therefore this is candidate-reconstruction holdout, NOT independent validation.

All eight tasks use six fixed arms: v10, original coverage, coverage on the
v10 observed domain, fit-source local, held-source local, and held-source uniform
control. The last control matches positive and negative element-space L1 edit
mass separately. Fit/held credits use the SAME held-view list, teacher targets,
weights and common support; only the candidate used for credit changes.
Anchored outputs use the original v10 observed support. The coverage-domain
control isolates the support/readout change from locality. Full coverage uses
its original support to reproduce N01. Primary candidate is held-source local,
fixed before any scoring; no per-task arm choice.

Seal all 48 prediction images before opening target GT. Report both GT protocols,
IoU/P/R/AP, all tasks, edit-mass controls and source loss accounting. Other four
benchmarks remain hash-bound references: this is a targeted NVOS mechanism test,
not proof of all-five transfer. If locality cannot beat matched attenuation,
report that result instead of claiming local verification is effective.

Persist <=2 MiB of reports/source snapshots and no dense predictions or weights.
Transient predictions are private tmpfs with a 256 MiB cap and explicit cleanup.

## Outcome and convergence decision

Held-local gives 87.5538 IoU / 98.2704 AP. Relative to v10 it improves IoU in
all eight tasks (+1.0403 points). Relative to matched signed attenuation it
improves mean IoU by 0.3209 points (7/8 tasks; descriptive seven-scene interval
[0.0750,0.6771]). Thus locality has measured benefit beyond the update budget.
However, against coverage it loses 0.2238 IoU points (7/8 tasks harmed) while
raising mean AP 0.1177 points, largely through orchids. Its local policy does
not preserve all beneficial coverage edits. The support-domain control changes
IoU by only 0.000065 points; reconstruction holdout does not improve mean IoU
over fit-local. No default change, per-task stitching, or universal gain claim.

A21 confirms net lost foreground and added false positives relative to coverage
in orchids and trex despite lower mean background scores. A22 finds high
reference seed--track agreement (96.49--99.82%); neither mask is GT and both
may share incomplete extent. Keep N01 as the principal best-IoU candidate. Stop
threshold/constant searches around this credit gate. Prioritize matched tests
of source extent, cross-view transport, geometry, and explicitly declared target
RGB access, without changing the feature representation or treating source
loss improvements as completed capability transfer.
