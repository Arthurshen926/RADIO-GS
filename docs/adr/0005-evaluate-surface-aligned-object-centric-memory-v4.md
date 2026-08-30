# Evaluate Surface-Aligned Object-Centric Universal Memory v4

Status: Proposed

## Context

SUGM-v3 established useful isolation and diagnostic contracts, but its
functional blocks did not close the visual/semantic write path or the
text-to-3D query path.  Exact marginal responsibility faithfully transports
the current Gaussian renderer's contribution relation, but does not establish
that overlapping Gaussians are the correct, unique physical surface.  Further
changes to the v3 `320/128/48/16` layout would therefore confound carrier,
registration, object organization, query, and compression failures.

## Proposed decision

Evaluate **Surface-Aligned Object-Centric Universal Memory v4** in causal
stages:

1. compare Gaussian exact-MPR transport with sparse-surface and mesh carriers;
2. retain confidence-weighted positive, negative, and unknown evidence on the
   visible surface;
3. only after the geometry-registration gate passes, organize elements into a
   soft object codebook with top-2 sparse memberships;
4. connect image and prompt selectors before text selection;
5. render the same element posterior for 2D and evaluate it directly for 3D;
6. compress local features and memberships only after the exact method closes.

All v4 method code depends only on the `SurfaceCarrier` contract.  It cannot
import v3 method modules, contain scene identities, or introduce
benchmark-specific thresholds.  Geometry, camera, teacher, mask, and query
inputs are hash-bound in receipts.  Target RGB is forbidden in strict paths,
and benchmark/source audit inputs are unreadable unless an explicit diagnostic
authorization is recorded.

The first gate compares same-view mask round-trip, cross-view transfer,
boundary leakage, element purity, registration entropy, surface coverage, and
available depth/normal consistency.  At least three preregistered primary
metrics must clearly improve without a purity regression, with all selected
scenes moving in the same direction.  A ScanNet mesh oracle that does not beat
Gaussian transport stops the carrier route.  A failed monocular sparse surface
first triggers calibration and pose audits, not SVRaster, 2DGS, or MAtCha.

## Consequences

ADRs 0001--0004 and their artifacts remain immutable retained baselines.
Exact-MPR is a Gaussian carrier baseline, not physical-registration authority.
The v3 feature partition, D48 expansion, persistent D16 boundary block, graph
propagation, and connected components are not imported.  Historical
checkpoints may be read only by explicit baseline adapters outside v4 method
code.  Object codebook, query adapters, benchmark runs, and compact distillation
remain blocked until their preceding gates pass.
