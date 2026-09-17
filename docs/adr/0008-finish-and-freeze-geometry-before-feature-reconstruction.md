# Finish and freeze geometry before feature reconstruction

Status: Accepted for implementation by user direction, 2026-09-14.

The geometry stage is independent of semantic reconstruction. This supersedes
the sequencing of ADR 0007's first post-compositor depth-weighting pilot:
semantic losses must not be expected to repair invalid geometry, opacity, or
visibility. MoGe-3 depth detail is retained as source evidence; independent-view
scale requires robust registration and validation, not an assumed unit weight.

The geometry stage audits calibrated pinhole rays, pixel-centre conventions,
world units, source depth/normal consistency, covariance extent, visibility,
free space, surface support and heldout coverage. It reads neither semantic
features nor benchmark queries/labels. Fixed heldout source frames must remain
excluded from both scale recovery and surface fusion. Their MoGe predictions
measure cross-view agreement, not ground-truth accuracy. Quality-filtered,
strictly source-track COLMAP points supply an additional sparse reference;
coverage and reference quality must accompany all error metrics.

Rejected single-view affine calibration does not justify blindly admitting a
global-scale fallback. Recover its multiplicative scale from at least three
trusted mapping neighbors with bounded within-neighbor and between-neighbor
log dispersion; otherwise isolate the view. A bounded robust joint scale fit
is evaluated separately. Lower correspondence loss does not authorize a
candidate whose heldout geometry gets worse.

A voxel is an aggregation/indexing cell, not the semantic map. Same-cell view
count and independent source-depth support are separate. A cell observed in
one view may be retained only when independent source-depth checks corroborate
it. Gaussian geometry uses surface normals to orient a thin covariance; it has
no trainable parameters. The resulting Gaussian domain is available to the
Single Compact Feature Field, but no old row-indexed feature or mask cache may
be reused after a geometry change.

Bind centres, rotations, scales and opacities, together with camera/raster
contracts and file hashes. Frame names alone cannot identify projection caches.
Cold loads verify complete geometry state. Feature training cannot update
geometry; changing features must leave alpha and depth unchanged. Query-time
selected-only rendering remains a different evaluation contract and must not
be used to conceal geometric foreground contamination in visible-scene tests.

New MoGe cache writes default to FP32 point maps and normals, and bind precision,
inference resolution, refinement steps and inference precision in resume
identity. Old FP16 predictions remain immutable lower-precision controls.
Their quantization cannot be undone by converting them to FP32 after loading.

Geometry experiments are development evidence. No complete geometry or
semantic baseline is promoted solely from source-depth agreement. Coverage,
boundary fidelity, sparse-reference consistency and shared 2D/3D transport must
be checked, and incomplete scenes remain explicitly unapproved for semantic
reconstruction as a claimed repaired mainline.

Implementation clarification, 2026-09-15: the independent-scale bundles are
available as development carriers with the scene limitations reported above.
This closes the geometry implementation iteration, not the accuracy promotion
gate. Their `post_visibility_normalized_v1` feature-confidence contract keeps
alpha/depth independent of feature evidence; unsupported affine-decoded
features are zeroed. A teacher build must bind the sealed geometry report and
use its calibrated cameras, clipping planes and mapping/heldout split.

2026-09-16 input-domain and acceptance clarification: released undistorted
NVOS images require their matching dense/sparse pinhole model, including the
transformed sparse 2D observations, bound to hwf_cxcy.npy. All geometry stages
must share explicit complete-filename identities where numeric suffixes collide.
Source-pair residual improvement alone cannot accept joint scales: rendered
heldout-source depth error, coverage and free-space consistency must not degrade.
These MoGe-reference checks do not certify independent ground-truth accuracy.
See `docs/experiments/2026-09-16-signed-support-and-nvos-camera-repair.md`.

2026-09-17 bounded M4 experiment, authorized by the next-stage checklist:
an isolated geometry preprocessing control may use a pinned, frozen DINOv2
descriptor from mapping RGBs to reject ambiguous source correspondences.
It must not consume a reconstructed semantic field, benchmark queries/masks,
or heldout RGB descriptors; no feature gradients update geometry. Admission
also requires source-depth agreement and reprojection cycle consistency.
Keep a geometry-only correspondence control with the same sampling and solver.
Source-feature similarity is a correspondence cue, not an independent geometric
confidence vote or an instruction to merge semantically similar objects.
The old unconditional feature prohibition remains the deployment default;
this exception is limited to the explicitly identified preprocessing experiment.
Only independent geometry acceptance followed by fresh geometry-bound asset
reconstruction can promote its output.

2026-09-17 normal transport correction: after intrinsic-ray alignment, a
nonzero affine depth offset must transport normals through the inverse-transpose
Jacobian of X'=(a+b/Z)X. Correcting centres without correcting tangent planes is
not a consistent surface warp. Existing frozen bundles remain immutable; the
loader's explicit legacy option exists only to reproduce historical controls.

2026-09-17 source local-scale candidate: an explicit materialized depth/normal
prior may carry bounded source-track corrections after separate point-ID fit,
admission and audit partitions. Both geometry construction and feature-stage
depth gating consume that identical hash-bound prior in world-depth units;
scalar scale/offset and normal alignment must not be applied twice. Verify the
raw prediction, camera and raster identities. Locally unsupported pixels remain
invalid. Audit rejected source views as well as admitted fusion-heldout views;
whole-view admission must not hide scene coverage loss. This does not promote
the candidate from sparse geometry diagnostics alone.
