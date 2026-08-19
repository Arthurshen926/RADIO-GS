# Region/instance prior restoration

The 2026-08-17 experiments test a bounded proxy for the SAM+text decomposition
without adding a second persistent semantic field. The canonical RADIO/SigLIP
score is the identity authority. The current weak SAM proxies are restricted to
bounded extent, boundary, or low-margin repair; they do not yet instantiate the
proposed query-independent co-membership hierarchy.

## Audit correction: this is a bounded proxy, not the full object topology

The experiments below validate only conservative proxies for the proposed
SAM-distilled hierarchy.  The LERF source cache was produced by prompting the
official SAM3 model with the benchmark query strings; it is not a
query-independent automatic multiscale mask hierarchy.  Its exact-MPR geometry
lifting is query independent, but proposal generation is query conditioned.
The frozen authorities are retained verbatim and the corrected scope is bound
in
`paper/artifacts/region_instance_prior_query_independence_erratum_20260817.json`.

The full4 LERF3D coverage audit explains the small aggregate gain.  Only 42 of
67 categories receive an extent update; 25 are exact fallbacks.  Among the 42
updated categories, 26 improve and 16 regress.  The current operator therefore
does not yet implement cross-view instance association, part/object hierarchy,
mask-aligned region descriptors, jointly-visible negative evidence, or a
low-dimensional same-instance head.  Results in this document must be read as
evidence for the identity/extent decomposition, not as validation of the full
hierarchical object-topology method.

The first-principles failure mode is now measured: averaging sparse text scores
over a complete proposal reduces the proposal mean to roughly 0.5--13% of the
identity peak.  The original v1 replacement-collapse measurement was additionally
confounded by applying sigmoid twice to SAM3 probability-valued mask tensors and
must not be used as quantitative evidence.  After correcting the tensor contract,
the strong-replacement Figurines diagnostic recovers to 0.47111 mIoU, but remains
below the paired primitive baseline.  Absence from a sparse prompted proposal is
unknown rather than reliable negative evidence.  The bounded variants therefore
preserve the original peak and add only a conservative extent residual.

Full-cohort development results after the SAM probability-contract audit are:

- LERF2D full4 sample-micro mIoU: 0.34975 -> 0.36152.
- LERF3D full4 same-pipeline sample-micro mIoU: 0.48451 -> 0.49244;
  Acc@0.25: 0.77404 -> 0.79808; Acc@0.50: 0.52885 -> 0.54327;
  Boundary-F: 0.71201 -> 0.70536; trimap IoU: 0.37674 -> 0.36908.
- ScanNet OVS paper8 19/15/10 mIoU: 0.33535/0.33370/0.42213 ->
  0.33810/0.33617/0.42548.
- NVOS full8 foreground IoU: 0.81762 -> 0.81776.

The previous LERF3D 0.48451 -> 0.49513 row in
`paper/artifacts/region_instance_prior_method_improvements_20260817.json` is a
historical v1 result and is superseded by the probability-contract erratum and
the v2 result above.  V2 improves pooled mIoU, Acc@0.25, and Acc@0.50, but it is
not promotable: Boundary-F and trimap IoU regress, Ramen and Waldo regress in
mIoU, and the source proposals were generated with benchmark query strings.
LERF2D improves in every scene.
ScanNet improves mIoU and mAcc on all three official class splits, but only as a
scene0000-selected weak graph residual rather than the proposed SAM hierarchy.
NVOS improves only marginally, consistent with its official-SAM box baseline
already being near the useful ceiling of this particular consensus operation.

The fixed LERF3D settings selected on Figurines are not stable under transfer:
Figurines and Teatime improve by 0.03026 and 0.03109 mIoU, while Ramen and Waldo
regress by 0.02405 and 0.00779.  This is also not a clean query-independent
transfer result because each scene's source proposals use its benchmark query
names.  The ScanNet residual was selected on scene0000; on the other seven
scenes its aggregate mIoU and mAcc deltas remain positive for all 19/15/10-class
splits.

The follow-up quality-aware full-posterior sentinel is also rejected.  Despite
using proposal confidence only for within-view ranking/noisy-or observation
confidence and preserving the text prior under missing evidence, Figurines
falls from 0.57658 to 0.52282 and Ramen from 0.40458 to 0.24364 mIoU.  Positive-
only proposal expansion can still move background primitives across the fixed
0.6 selection threshold; `promotion=false`.

The actual query-free P0 path is now separate from those query-conditioned
diagnostics.  The reviewed Figurines sparse pilot decodes official SAM3 packed-
boolean proposals from eight legal source views and lifts them with exact MPR.
Its producer, preflight, source RGB, construction frame manifest, checkpoint,
primitive rows, exact-MPR formula and every responsibility shard are SHA-bound.
The identity seeds come from a dedicated field-only score cache that never
opens the proposal/membership cache, removing the previous circular audit.

This grid-12 pilot produces only 67 proposals over eight views and 44,538
thresholded exact-MPR memberships.  Of 21 field-only identity seeds, 7/21
(33.33%) are covered by any proposal in at least one view and only 1/21 (4.76%)
has a covering proposal in at least two views.  Therefore no benchmark
prediction is constructed and `promotion=false`.  The first lift is retained
as superseded because float32 accumulation produced a maximum membership of
1.000000715; the reviewed v2 clamps the theoretical probability to `[0,1]`
without changing thresholded support (SHA
`6781633ac2c14a9ef7787153921a4a7502c23c2b9befa90f9debe787aef1cc57`).

Crucially, grid-12 point prompting at a single full-image scale is not the
attachment's Stage-A multiscale AMG hierarchy: it has no crop pyramid, dense
AMG coverage, or materialized containment-parent graph.  The measured failure
is undercoverage of this sparse eight-view pilot, not evidence against a SAM
region prior.  The sealed receipt is
`paper/artifacts/lerf_query_free_sparse_p0_grid12_coverage_20260817.json`.

These are development results, not prospectively blind paper claims.  None of
these weak proxies is currently suitable for promotion as the complete
region/instance component.  They do not use target masks to construct
predictions, but the missing query-independent hierarchy and the failed metric
gates must be resolved before a clean transfer/main-table rerun.

## What the weak proxies test from SAM+CLIP-style methods

The tested operators approximate a division of labour rather than adding a
second semantic vote. RADIO/SigLIP supplies the identity unary and its maximum
is immutable. Query-prompted official source-SAM masks (LERF), local topology
from the official RADIO-to-SAM adaptor (ScanNet), or query-transient official SAM
masks (NVOS/SPIn) supply bounded extent, boundary, or low-margin consistency.
They probe, but do not fully implement, the latent-region factorization

`P(g belongs to q) = sum_R P(R matches q) P(g belongs to R)`.

The corrected strong-replacement diagnostic remains below the primitive
baseline: a full object mask contains many weakly semantic pixels, so mean
pooling dilutes identity, while sparse proposal absence cannot be treated as a
negative.  The bounded residual therefore preserves the identity seed, requires
source-view support, and limits the extent contribution.  This is also why the
old fixed 1:1 primitive/region average should not be restored.

## SPIn Available-Nine completion runtime

All nine new `all_view_features` bundles now exist. The remaining work is field
materialization and the frozen full-nine readout. Two runtime defects found by
the real million-Gaussian scenes were fixed without changing the mathematical
method:

- sharded exact-MPR resume now compares support weights numerically while the
  first-pass denominator and immutable cache hash remain authoritative;
- the semantic rendering loss uses exact global two-pass pixel chunking, which
  preserves the global centered-cosine objective while fitting the 24 GiB
  devices.

Whole-file implementation hashes are treated as lineage fields when an
immutable historical cache SHA is already explicit. All formula, compositor,
geometry, camera, view-set, support, and safety fields still compare exactly.
This prevents audit-only source edits from invalidating frozen caches without
weakening their scientific identity.

The active completion queue uses physical GPUs 0, 1, 2, and 4; GPUs 3 and 5
are occupied by external jobs. Exact-MPR aggregation is currently host-memory
scatter/IO bound and consumes the available CPU capacity, so low instantaneous
GPU utilization in that stage is expected. A follow-up process waits for all
nine `method_v1_gate.json` files, then seals all signed-field predictions before
opening target RGB and opens evaluation masks only after the full-nine
prediction barrier is complete.

The final focused regression set for the SAM extent/readout operators, exact
MPR resume and lineage checks, factorized cohort loader, and chunked semantic
loss contains 89 passing tests. The only warning is PyTorch's existing
`TypedStorage` deprecation notice.
