# Separate semantic response from selected-set decoding

Status: LERF development candidate, 2026-09-29; not a default across benchmarks.

This refines ADR0001 and the query-transient extension in ADR0010. A semantic
response rendered with full-scene transmittance and the silhouette of a
selected Gaussian subset are different quantities. Removing nonmembers changes
transmittance. One scalar need not correctly parameterize both outputs.

Preserve next44's semantic response `p` for 2D ranking/localization. Initialize
transient membership `m` with `p`, then fit the actual subset-alpha renderer to
source-only partial masks. Freeze geometry, opacity, native query encoders and
map features. Use one fixed 120-update/query recipe with masked BCE plus overlap;
select `m > 0.5` and retain the existing alpha-byte threshold10 scorer.

The retained research candidate uses the frozen next38 field to qualify source
supervision: a detector-positive pixel is positive only when its projected field
response exceeds the existing0.5 retrieval threshold; a detector-negative pixel
is negative only when the field response does not exceed that threshold. A
conflict becomes unknown, not a new positive or negative label. Missing
observations and boundary bands remain unknown. Queries without eligible positive
views retain their parent membership. This agreement is not a calibrated truth
or completeness probability; it cannot recover concepts absent from both inputs.

The next44 posterior is not used as the independent conflict reference because
it already incorporates the runtime detections under examination. Even next38
is not independent ground truth: source images/teacher families can be shared.

Artifact contract: `scores` is semantic response, `selection_scores` is decoded
membership. The evaluator renders these predeclared outputs separately; it does
not choose branches per scene/query using GT. Old artifacts without a selection
field retain their original path and scoring. The candidate preserves2D metrics
by construction and improves selected projection on all four development scenes;
that does not claim improved2D identity or physical unseen-surface reconstruction.

Geometry preprocessing trials are separate. The source-RGB scale/opacity fit
keeps MoGe centres/normals fixed and freezes a new carrier before recomputing
semantic features, following ADR0008. It has mixed results and is not retained.
Historical RGB geometry remains a diagnostic exchange, not a fair-input mainline
replacement with an established training split.

Scope: no new NVOS/ScanNet/UQIS claim. A held-out transfer and reliable handling of
remaining coffee/refrigerator regressions are required before broader promotion.
See docs/experiments/2026-09-29-carrier-and-set-decoding-next48.md.

2026-09-30 development clarification (next67): a subset decoder must not freeze
membership solely because full-scene transmittance hides an element in the
active source views. Removing nonmembers can expose that element in the
consumed subset render. The expanded-RGB LERF candidate permits all semantic-
observed memberships to receive subset-render gradients while preserving the
existing known-pixel masks and frozen geometry/features. This does not turn
occluded/unknown pixels into negative labels. Matched240-step selected
45.862→51.414; all4 scenes improve versus next66, teatime still regresses versus
the old-carrier candidate. No global cross-benchmark promotion. See
docs/experiments/2026-09-30-subset-visibility-support-next67.md.


2026-10-01 experimental clarification (next93): source-complement exclusion may
correct old native positivity when supported by other source views. This optional
`corroborated` negative-admission policy explicitly relaxes the prior-positive
veto above; it does not change the retained default (`prior`). It uses leave-one-
view-out visible Gaussian evidence, preserves track-level negative permissions,
and keeps semantic responses/geometry fixed. Four-scene selected projection
59.425→60.613, but waldo recall regresses: repeated partial masks do not certify
complete object exclusion. Preserve as a development candidate, not a global
text-union/four-query guarantee. See the next93experiment report.


2026-10-02 experimental clarification (next96): optional selected-membership
compactness may penalize unoccluded source-camera footprints in addition to known
pixel reconstruction. This is explicitly a prior over unknown scope, not observed
negative evidence. Preserve source masks, frozen geometry/native semantic scores
and the default `support-prior none`. The high-resolution source+footprint candidate
reaches61.478 selected projection versus60.613, but Waldo recall regresses severely;
no completeness or universal four-query claim. Keep optional research status and
the prior candidate for rollback. See the next96experiment report.

2026-10-02 reference-instance integration: the same type distinction is necessary
for NVOS. Shared actual-subset fitting produces memberships whose normalized
full-scene response scores only6.8182%IoU, while their actual subset silhouettes
reach85.0794%under a separate diagnostic cutoff. Preserve an independently
fitted continuous response (frozen NVOS86.4891%) and selected membership in one
typed readout. No target metric chooses the head. A common map/evidence/CLI does
not imply interchangeable output operators or a single shared scalar objective.
See docs/experiments/2026-10-02-reference-instance-integration.md.
