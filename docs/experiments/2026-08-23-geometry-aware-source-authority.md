# Geometry-aware source authority continuation

## Decision

The new evidence does not justify global 3DGS unfreezing.  The supported part
of the frozen ScanNet source geometry is already locally semantically pure.
The earlier proposal to repair ScanNet by adding scene-redundant categorical
calibration authority is now closed: category-specific scale/bias fitting is
not the benchmark-independent object mechanism sought by the method, and its
three-scene LOSO experiment already failed decisively.  No additional source
scene is required for this branch.

SPIn is outside this continuation's critical path.  LERF2D, LERF3D, ScanNet
OVS, and NVOS retain their existing evaluation contracts.

## Implemented controls

- Added a deterministic official-annotation-to-canonical-field ScanNet source
  cohort audit.  It reads no benchmark masks, predictions, or metrics and
  binds source assets by SHA-256.
- Added an ellipsoid-aware semantic-support geometry audit using official mesh
  labels, anisotropic Gaussian covariance, opacity, and a fixed 5 cm surface
  support radius.
- Added source-only frozen h128 physical-region scoring with the official
  RADIO SigLIP2 summary head and exact split19 text bank.
- Added a 38-parameter positive diagonal scale plus bias categorical
  calibrator with fail-closed scene LOSO.
- Added a separate paper8 development evaluator which cannot run unless the
  source checkpoint proves every LOSO fold passed.  It was not run in this
  continuation because the source gate failed.

## Geometry audit

| Source scene | supported Gaussian fraction | mean purity on supported rows | joint risk > 0.05 |
|---|---:|---:|---:|
| scene0046_00 | 0.81749 | 0.98776 | 0.01289 |
| scene0231_01 | 0.56849 | 0.98622 | 0.01378 |
| scene0695_03 | 0.63116 | 0.99117 | 0.00674 |

An unsupported Gaussian is explicitly unknown rather than a semantic
boundary.  This correction matters particularly for scene0231_01, whose
surface-distance tail contains many unsupported rows.  Once validity is
separated, the observed boundary-mixing risk is sparse.  Together with the
existing LERF exact-adjoint carrier ceiling, this rejects global geometry
optimization as the next mainline intervention.  A future split-only oracle
must remain confined to the roughly one-percent high-risk tail and must clear
a 1--2 point paired LERF2D/LERF3D gate.

## Class-complete authority and failed minimal-cohort LOSO

The deterministic minimum set cover is:

1. scene0046_00
2. scene0231_01
3. scene0695_03

It covers all 19 NYU40 evaluation classes and all three scenes have compatible
canonical fields.  The source-only frozen region score materialization took
about two minutes per scene after increasing the semantic batch from 64 to
256.

The fixed categorical calibrator failed every held-out source scene:

| Held out | raw source mIoU | calibrated source mIoU | delta |
|---|---:|---:|---:|
| scene0046_00 | 0.41401 | 0.16825 | -0.24576 |
| scene0231_01 | 0.43404 | 0.16674 | -0.26730 |
| scene0695_03 | 0.50973 | 0.27241 | -0.23732 |

No checkpoint was promoted and paper8 was not opened.  The failure is
structurally informative: union coverage is not LOSO identifiability.  Across
the 11 currently compatible independent source fields, classes 10, 28, 33,
and 36 occur in only one scene each.

## Closed categorical-calibration construction

The proposed scene0203_00 and scene0153_00 continuation is no longer part of
the method or critical path.  Existing partial assets are diagnostic only and
receive no further GPU allocation.  The ScanNet continuation instead uses a
query-independent region/instance authority: source-view regions are
associated across views, lifted by exact responsibilities, and represented by
the same Canonical Capability Feature.  Evaluation remains normalized text
similarity followed by categorical competition; region evidence may improve
membership or low-margin consistency but may not learn class-specific
temperature, bias, or a hidden background classifier.

## Benchmark implications

- **LERF2D/LERF3D:** local semantic support errors exist, but present evidence
  bounds them to a tail correction.  The main unresolved variable remains
  physical-track precision at Gaussian selection, not renderer geometry.
- **ScanNet OVS:** the source-category calibration branch is rejected.  The
  retained bounded low-margin SAM residual is positive but small; the next
  high-value gate is query-independent cross-view region membership with
  normalized text prototypes and no per-class calibration.
- **NVOS:** retain the existing reliability-gated carrier-native all-view
  development result (macro IoU 0.91623).  The next meaningful experiment is
  robust multi-registered-view exact adjoints; the required sealed per-view
  candidate inventory has not yet been materialized.
- **SPIn:** postponed by the explicit user instruction and does not block this
  four-benchmark continuation.
