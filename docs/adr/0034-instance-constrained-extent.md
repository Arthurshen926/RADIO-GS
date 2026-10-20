# ADR0034: Instance-constrained additive extent recovery

2026-10-17. N11 registered after N09/N10 analysis, before N11 predictions/GT.

N09 recovers missing flowers but confuses other instances; N10 shows that
removing SAM alone does not solve this. Keep the exact static and signed SAM
image proposals from N09 (replayed with array-hash checks). The intervention
admits connected proposal regions using query identity, before actual 3D fit.

The base is the static image response. Partition signed SAM >=0.5 into
8-connected components. For each component sum positive and negative identity
evidence. Admit iff positive mass > negative mass and positive mass >1e-8.
Accepted pixels take max(base, proposal); all others retain base. Unknown/tied
evidence cannot authorize an edit. This is additive image refinement; subsequent
3D optimization is not claimed monotone. No scene-name rules or area thresholds.

Three matched policies: unrestricted (positive=1, negative=0); v10 response
anchor (positive=old response, negative=1-old response); original signed-scribble
geometry (primary). For the last, rasterize the reference view on native rays.
Each Gaussian receives alpha-weighted positive/negative scribble coverage divided
by its total reference visible contribution, including unmarked pixels. Thus
unknown coverage contributes to neither sign. Restrict to the original observed
Gaussian domain. Render both signed fields to target with the same occlusion and
observed-mass normalization. This provides instance evidence, not a complete mask.

Nine arms: v10, static_image/static_native, union_image/union_native,
prior_anchor_image/prior_anchor_native, scribble_anchor_image/scribble_anchor_native.
Primary scribble_anchor_native. All72 predictions before GT; both GT tracks,
IoU/AP and independent confusion counts. Same native 1008x756 rays, 120 Huber
updates, abs(2p-1) confidence, old geometry/observed domain/initialization, 0.5
readout. Source teacher residuals and GT-conditioned error transitions are
diagnostics only. No target-driven threshold or per-task winner selection.

Both DINOv2 and SAM3 are frozen external teachers; original negative scribbles
and target RGB remain explicitly added inputs relative to v10. Other four
benchmarks stay hash-bound references, not new inference. View-conditioned
development evidence, not an independent final test or query-global field.

Inherit N09 source/input archives by hash, snapshot new sources only. Compact
receipts within 270KiB new durable output and the unchanged original 2GiB study
budget. Transient predictions <=256MiB, no weights/dense durable export. Preserve
failed attempts. Fail before scoring if control replay or numerical checks fail.

Execution correction before predictions/GT: v1 passed a CPU tensor to the
NumPy-only array digest routine and failed during its first proposal check.
Archive v1; v2 explicitly converts to contiguous float32 exactly as the original
prediction bank does. No method or output changes. Lower the v2 reservation to
240KiB to account for the retained failed attempt inside the original budget.
