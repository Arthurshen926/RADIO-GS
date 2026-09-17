# Frozen geometry closure and feature reconstruction audit

The independent-scale geometry bundles in
`/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914`
are the uniform development carrier for the next feature stage. They include
trusted-neighbor scale recovery, but exclude the last joint residual scale fit.
Geometry construction is closed for this iteration; it is not an all-scene
accuracy acceptance or semantic baseline promotion. Figurines still has about
80% coverage at the geometry audit resolution, and ramen regresses against the
filtered sparse COLMAP reference. The retained predictions are FP16.
Historical working semantic controls remain the benchmark comparison.

## Repaired geometry–feature coupling

1. **Semantic reliability changed visibility.** Previously, both ordinary
   canonical rendering and column-staged forward/adjoint paths supplied
   primitive confidence as an opacity multiplier. Frozen tensors alone do not
   prevent this from changing occlusion, alpha, or depth. FrozenGaussianGeometry
   now declares `post_visibility_normalized_v1`. Its feature observation is
   `sum(w_i c_i f_i) / sum(w_i c_i)`, with the original front-to-back geometric
   contributions `w_i`. Confidence is an extra rendered channel, never an
   opacity change. The result exposes feature support separately from geometric
   alpha. The direct exact compositor rejects opacity scaling on this geometry.
   Legacy geometry keeps its original replay contract.

2. **Affine bias produced features without evidence.** A zero coefficient map
   does not decode to zero when the decoder has a bias. The frozen canonical
   and column-staged paths now zero decoded pixels without geometric or feature
   support. A visible region with missing semantic evidence remains visible in
   alpha/depth, but has no fabricated semantic feature.

3. **Teacher construction did not inherit the full audited camera contract.**
   The repaired-geometry builder now requires the sealed geometry report,
   verifies that it names the loaded geometry hash, uses its COLMAP intrinsics
   and metric clipping planes, restricts selection to mapping frames, and
   verifies every selected feature pose and intrinsic matrix. It rejects
   heterogeneous intrinsics that the single-camera renderer cannot represent.
   This also prevents accidental use of geometry-heldout frames as feature
   training views. New responsibilities are generated against the new geometry.

Four-scene actual GPU rendering verified unchanged alpha and depth when
confidence varies, unchanged complete geometry fingerprints, and weighted
linear 2D/3D commutation error below 9e-6. This proves an observation invariant,
not feature fidelity. See
`paper/artifacts/frozen_geometry_confidence_transport_20260915.json`.

## Real full-dimensional feature reconstruction

`scripts/audits/audit_frozen_radio_reconstruction.py` independently reconstructs
all 1280 backbone channels from four deterministic source mapping neighbors,
then evaluates the first geometry-heldout source view. It verifies each loaded
feature tensor against the resealed manifest, uses calibrated cameras, builds
fresh exact contribution matrices, and checks that the sparse observation
equals actual gsplat rendering. It never loads old per-Gaussian features.

The diagnostic uses per-pixel normalized raw RADIO, an adjoint-mean
initialization, and 20 diagonal-preconditioned least-squares iterations.
It is an **uncompressed feature-field diagnostic**, not D512 training, not the
MPR teacher's purity-weighted initialization, and not a native adaptor metric.

| Scene, heldout frame 1 | Supported coverage | Initial cosine | Reconstructed cosine | Initial MSE | Reconstructed MSE |
|---|---:|---:|---:|---:|---:|
| figurines | 85.34% | .75745 | .76022 | .00033830 | .00036982 |
| waldo_kitchen | 73.41% | .73518 | .73364 | .00037615 | .00040028 |

Training cosine improves from roughly .86–.92 to .89–.95 in figurines, and
.92–.93 to .96–.97 in waldo. Heldout MSE gets worse in both. All-pixel heldout
cosine also falls slightly (.67043→.67019 and .56504→.55988). Conditional metrics
therefore must not be presented alone. Supported coverage requires geometric
alpha > .1 and more than half the visible contribution assigned to rows
observed during training. Low-resolution feature-grid coverage is not the
480-pixel geometry audit coverage.

This is direct evidence that reducing reconstruction loss on the fitted views
does not establish a reusable 3D representation. The experiment is consistent
with an underconstrained, overfitting per-primitive field; it does not isolate
how much remaining error comes from geometry, feature receptive-field context,
view dependence, or limited source coverage.

## Consequences for the next reconstruction implementation

- Keep the independent geometry frozen. Do not respond to feature residuals by
  adjusting position, covariance, opacity, or semantic-dependent visibility.
- Treat visibility-purity-weighted MPR as a teacher initialization/reliability
  heuristic, not the inverse of the actual renderer. Train and validate through
  the explicitly recorded observation operator.
- Preserve the raw feature normalization contract. The normalized-backbone
  diagnostic does not validate applying official adaptor heads to normalized
  input or applying a global summary head pointwise. Native adaptor/crop
  targets require their own matching extraction and supervision contracts.
- Bound movement from reliable multiview teacher initialization and use
  source-heldout selection. Any regularization coefficient must be fixed on a
  separate development split, not tuned against the final heldout examples
  reported above.
- Separate zero evidence, cross-view disagreement, semantic confidence, and
  geometric support. None alone is an opacity or free-space observation.
- Test compact-field compression and DINO/SAM/SigLIP capability retention
  separately before claiming semantic recovery. This iteration establishes
  working real-source transport and exposes the reconstruction generalization
  failure; it does not establish those capability results.

## Production teacher pilot and validation

The actual `build_gaussian_multiview_teacher_cache` entry point completed on
figurines with source frames 3, 6, 8, 11. It generated a new exact contribution
authority and a 1280-dimensional RADIO teacher on 274,805 frozen Gaussian rows;
46,322 rows have valid feature evidence (16.86%, from only four views).
The independent reconstruction diagnostic above does not substitute for this
production-path check.

The artifacts are under
`/mnt/pool/sqy/results/RADIO-GS/output/frozen_radio_teacher_pilot_20260915/figurines`.
The command is retained in `scripts/audits/build_frozen_radio_pilot.py`.
The companion inventory is
`paper/artifacts/frozen_radio_teacher_pilot_20260915.json`.

The teacher reports mean visibility purity .02384 and median .00420. At a
46×62 RADIO grid, many surface primitives share a feature pixel. Purity is a
compositor mixing statistic, not a calibrated probability that a surface or
semantic observation is wrong. Multiplying opacity by this value would
aggressively change visibility, exactly the coupling repaired in this iteration.

Validation: 453 tests passed, including geometry, exact contribution authority,
feature projection/fidelity contracts, canonical finetuner paths and v4
regressions; four existing dependency/sparse-tensor warnings. Actual four-scene
GPU transport and two-scene real RADIO reconstruction ran separately.
No compact D512 field or new semantic benchmark baseline is claimed.
