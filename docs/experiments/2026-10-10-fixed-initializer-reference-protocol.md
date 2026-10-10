# Fixed-energy initializer intervention

Sequential development after completed joint_subset_reference_v1. Same4scenes,
68texts/72sourceviews/5137regions, same cached teacher acceptance and partial
positive/negative labels, geometry and observed domain. Three arms: finite_parts
(cached atom_parts), joint_source_initializer (cached joint_footprint),
joint_finite_initializer (PRIMARY). No independent test or deployment claim.

New solver uses finite atom_parts initialization instead of source_parts, but
KEEPS footprint normalization by the original source_parts initializer. Thus
cameras, labels, active view order, footprint measure and weight0.1, 240 steps,
Adam0.05, temperature0.5->0.1 and actual-alpha objective all remain fixed.
Only the transient starting logits change. No source confidence/mass cutoff or
weight sweep. No-positive-source queries keep their initializer; verify the two
initializers agree there. All predictions sealed before target scoring.

This is explicitly source-assisted, query-transient fitting, NOT field-owned
standard inference or benchmark-answer distillation. Do not update the deployed
field, promote per-case winners, or interpret finite-step fitting as a global
capacity bound. Its purpose is to separate initialization from the complete
member-solver function, following the repository's next100 fixed-measure lesson.
