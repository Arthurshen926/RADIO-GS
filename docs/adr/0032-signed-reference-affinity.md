# ADR0032: Transfer signed query evidence without a reference extent gate

2026-10-17. Registered after N08, before any N09 target prediction/scoring.

N08 recovers most static image-to-3D loss. N07 memory and historical positive
SAM ensembles did not solve incomplete query extent. N09 tests whether treating
unlabelled reference pixels as unknown can recover scope across disconnected
parts, without assuming the incomplete reference SAM mask is complete.

Use existing frozen DINOv2 ViT-B/14 at 1008x756 (ImageNet normalization) on the
original reference RGB and authorized target RGB. No feature-field changes,
training, new checkpoint, target annotation or task-specific policy. Use the
official positive AND negative reference scribbles, explicitly extending the
positive-only v10 input. Negative scribbles are legal query input, not target GT.

Area-pool scribbles to the 54x72 patch grid. Any covered patch gets its sign;
patches touched by both signs stay unknown. Compare nearest positive and
negative normalized descriptors by their cosine-margin difference. Normalize
with the median signed leave-one-out reference margin / log(9), so the median
correct source margin corresponds to 0.9. Missing support/nonpositive median
falls back to a neutral field. No target-selected scale or threshold.
Neutral-affinity failure disables the edit and reuses the static image response;
it must not prompt SAM with an all-foreground 0.5-threshold box.

Primary evidence uses only signed scribbles. Matched pseudo-mask control uses
the retained reference SAM mask/complement as labels, overridden by the same
scribbles; all other operations and input access agree. Static N08 is replayed.
Decode each transferred probability through the same response_prompt and frozen
SAM3 single-mask decoder; the bounding box derives from that probability, not
the old v10 extent. No maximum-area mask selection or per-task winner choices.
Fit each decoded response into original observed Gaussians on native rays with
120 Huber steps and corrected gsplat footprint, then actually re-render it.

Nine arms: v10; static_image/static_native; signed_affinity/pseudo_affinity;
signed_image/pseudo_image; signed_native/pseudo_native. Primary signed_native.
All 72 responses must precede GT. Fixed 0.5, both GT boundary conventions,
independent integer confusion counts and AP, source-teacher residuals, complete
eight-task error figure. No source-teacher agreement equals GT claim.
Other four benchmarks remain unchanged references. This is view-conditioned
development evidence, not a query-global field or independent SOTA certification.

Use bounded private RAM (256MiB prediction bank) and existing immutable 2GiB
study budget; at most 1.25MiB new durable receipts/figure, no dense exports.
Preserve failures; no partial target scores or runtime edits after source sealing.
