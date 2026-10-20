# ADR0037: Do not repair an already feasible reference response unconditionally

2026-10-18. N14 fixed AFTER complete N12 scores and source diagnostics, before
N14 prediction. This is adaptive development on the opened cohort, not an
independent final test. N13 remains frozen and running; its partial scores are
not used for this design.

N12 restores missing scope but its signed-point repair can add unwanted regions
when the semantic reference mask already satisfies every original annotation.
Test two fixed source-only policies on the EXACT two N12 reference candidates:

1. Primary feasibility policy: retain semantic completion iff all original
   positive pixels are foreground and all original negative pixels background;
   otherwise use signed-constrained completion. No tolerance is tuned.
2. Source-quality control: choose the higher SAM-predicted reference mask IoU,
   with semantic-first ties. If quality is absent, fall back to feasibility.

The decision reads only reference-query agreement and source predicted quality,
never target response/GT, scene names or past per-task scores. It selects one
complete seed before identical two-frame transport. Record the full source
decision, rerun target predictions, verify their hashes against the corresponding
N12 candidates, then perform actual native-ray 3D fitting and re-rendering.
Do not present cached row selection as fresh inference or silently stitch winners.

Seven arms: v10, static_image/native, feasible_image/native (primary),
quality_image/native (secondary). All56 before GT. Fixed 0.5, two GT protocols,
IoU/AP, independent counts, source residuals, complete eight tasks. Other four
benchmarks remain unchanged references. No universal promotion or SOTA claim.
Same explicitly bounded 8MiB continuation study, <=1MiB new run receipts,
256MiB transient prediction bank and 128MiB spatial links. No new teacher,
features, training, weights or durable dense predictions. Preserve failures.
