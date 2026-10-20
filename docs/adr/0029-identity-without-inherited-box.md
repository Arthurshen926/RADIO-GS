# ADR0029: Separate query identity from the extent inherited from an old response

Date: 2026-10-17. Status: registered before N06 inference.

Complete N05 shows that removing negative dense logits is harmful on average.
Its positive-only 3D arm is 85.3829 versus signed 88.0159 IoU. Sparse guidance
recovers flower but includes a second object in horns-center. The full spatial
audit also shows missing orchids branches outside the old response-derived box.
Those GT diagnostics motivate this experiment, but never become inference inputs.

N06 compares three fixed prompts on all eight tasks: the N05 box plus inferred
positive points; the identical points without a box; and the same unboxed points
with signed dense log-odds. Primary arm is unboxed sparse guidance plus actual
3D writeback. No new point labels, point rules, thresholds, target-specific
switches or checkpoint training. This tests the authority of inherited extent;
it does not presume that unboxed prompts will preserve identity or improve IoU.

Use frozen SAM3 and observed target RGB, the original Gaussian domain, 512x384
confidence pooling and 120 Huber steps. Each image output is also written to 3D
and rendered, yielding seven arms including freshly replayed v10. All 56 outputs
precede GT scoring. Report both GT conventions and independent confusion counts;
AP is not measured. Reproduce the boxed N05 control and check all source tracks.
Other benchmarks remain hash-bound references, not new inference. This remains
view-conditioned response refinement on an opened development cohort.

Preserve source/config/input receipts and small diagnostics; no new weights or
persistent dense predictions. The complete N04 binary comparison and N05 scores
are already known; N04's independent AP scoring is still in progress.
