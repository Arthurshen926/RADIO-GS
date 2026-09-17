# Geometry first: repair and feature-coupling audit

## Scope and decision

The primary map remains a Gaussian Single Compact Feature Field. Geometry is
constructed and validated first, then frozen. No semantic features or labels
were used to construct the new geometry; no semantic field was trained in this
step. Sparse voxels are indexing/fusion cells, not the semantic representation.
ADR 0008 supersedes the earlier post-compositor-weighting-first sequence.

## Geometry audit, findings and corrections

| Layer | Finding | Implemented handling | Remaining limit |
| --- | --- | --- | --- |
| Cameras | Four scenes use COLMAP PINHOLE cameras; frame-1 poses agree with stored verified poses. A nonrigid matrix could previously be accepted as a camera rotation. | Require valid focal lengths and a rigid positive-determinant camera rotation. Scale remains separate. | COLMAP poses themselves are an input authority, not independently surveyed ground truth. |
| Calibration pixels | COLMAP image-edge coordinates received an extra half pixel during depth sampling. | Versioned image_edge_v2 calibration; legacy_index_v1 remains explicit replay. Recalibrated all four scenes. | The retained initial depth predictions were already cached in FP16. |
| Native rays/normals | MoGe forced intrinsics differ slightly from calibrated COLMAP intrinsics; native ray mismatch reaches about 2.16 pixels at p95 in figurines. Re-unprojecting depth without transforming normals is inconsistent. | Re-unproject onto calibrated rays and transform normals by the inverse transpose of the intrinsic correction. | Normal-vs-point-map angles are diagnostics, not normal ground truth. |
| Scale | Independent bounded affine fitting rejects many views (ramen: 71/120). Blind global fallback severely contaminates fusion. | Recover rejected-view multiplicative scales from at least three trusted mapping neighbors, with within/between-neighbor dispersion checks. Isolate failures. | The subsequent small bounded joint fit is not consistently better after rendering and is not promoted. |
| Historical Gaussian geometry | Near-camera rows can occlude real surfaces before exact-MPR ever sees them. Waldo frame 1: 19 rows cover 2790 pixels. | Rebuild geometry from calibrated, corroborated MoGe depth/normal observations, rather than attempting semantic repairs of those rows. | Camera-clearance and centre-only opacity culling were separately tested and were insufficient. |
| Voxel support | Requiring two views to hit the same small voxel removes valid surfaces even when independent depth projections corroborate them. | Separate minimum voxel contributors from minimum source-depth agreement; new build requires independent depth support while allowing a one-view cell. | Within-cell averaging still limits detail; it is not TSDF integration or an exact surface intersection. |
| Gaussian footprint | Old covariance/opacity is not physical surface evidence. | New thin, normal-aligned Gaussian covariances; 1 cm cells, 0.75 cm tangent sigma, 0.1 cm normal sigma in the evaluated dense candidate. | A 1 cm sampled carrier does not establish preservation of all sub-centimetre details. |
| Projection cache | Gaussian and mesh caches keyed only by camera name could silently reuse a changed projection. | Mesh keys include camera/raster state; precomputed Gaussian transport rejects a changed camera and validates an embedded binding when present. | Old Gaussian tables without embedded camera binding still need their external authority; they are marked as such. |
| Geometry cache | Resume identity omitted resolution/refinement/inference precision. Geometry was always stored in FP16. | New writes default FP32, bind all relevant settings, reject incompatible existing files, and prevent mixed-precision shard merging. | No FP32 re-inference is claimed for the older prediction cohort used here. |

## Fixed source evaluation

Experiments construct geometry from mapping source views and hold out a fixed
list of other source frames. Eight heldout views per scene are rendered at
480-pixel width. The final recalibration and scale-recovery run uses the same
heldout frame IDs as the dense control. Scale recovery never reads heldout
views. Query RGB, query text and labels are absent.

The initial all-source experiment changed the heldout list and included
uncalibrated reference depths. It is not a matched comparator and is retained
only as a failed development diagnostic. The fixed-split experiment isolates
that issue; blindly adding global-scale fallbacks still fails, especially in
ramen. Neighbor-supported recovery is evaluated separately.

Primary reporting uses conditional depth error plus coverage and the fraction
of all valid reference pixels with agreeing rendered depth (holes remain in
the denominator). MoGe depth agreement is not ground-truth geometric accuracy.
An independent cold process additionally compares strictly source-track COLMAP
points, retaining tracks with at least three source views and mean reprojection
error at most two pixels. Sparse reference coverage and quality are reported;
unfiltered sparse points were demonstrably unreliable, particularly in
figurines. They must not be called dense ground truth.

Final artifacts live under:

`/mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914/<scene>/`

Each scene has an independent-scale/recovered-initialization geometry bundle,
a joint-scale comparison bundle and an experiment report. `independent_scale`
in the filename means *without the final joint residual fit*; accepted
neighbor-based scale recovery is already included. This naming is retained in
the receipts to avoid silently changing asset identities.

Final cold evaluations are recorded in
`paper/artifacts/geometry_final_cold_<scene>_20260914.json`.
The aggregated inventory is `paper/artifacts/geometry_repair_inventory_20260914.json`.

## Feature representations and coupling

| Representation/use | What was checked | Conclusion |
| --- | --- | --- |
| Raw RADIO and factorized direction/amplitude | Feature signatures, normalization, capability source contracts, factorized training integration. | The declared representation can be checked; reconstruction quality on a new geometry requires rebuilding its teacher. A test pass is not semantic accuracy. |
| Dense local Gaussian latent | Canonical primitive-local code/decoder path and geometry buffering. | Geometry is not trainable in the new geometry module; canonical field training has no geometry optimizer. |
| DINO appearance and SAM feature views | Native-before-lifting versus decoder-after-lifting contracts and source/adapter signatures. | Nonlinear adaptation and aggregation are not interchangeable. Equal dimensionality alone does not prove semantic compatibility. |
| SAM mask/candidate support | Source provenance, independent support versus categorical membership, native lift and relation contracts. | Overlapping parts/objects must remain evidence until selection; no automatic mutually exclusive normalization or top-2 compression. |
| SigLIP text/crop identity | Historical crop-summary identity route versus spatial/summary distribution assumptions. | Retain the matched identity path. Applying a summary head to arbitrary averaged spatial tokens is still an unvalidated alternative, not an automatic identity replacement. |
| Multiview aggregation | Exact compositor responsibility, contribution-mean accumulation and missing-observation handling. | Exact arithmetic on a polluted geometry is still the wrong observation assignment. Rebuild every assignment after geometry changes. |
| Reliability | Distinguish alpha contribution purity from physical depth correctness. | Purity can be high for a single incorrect foreground Gaussian. Semantic confidence must not silently masquerade as a geometry repair. |
| 2D and 3D transport | Actual GPU tests on four frozen geometry bundles: constant-field preservation, linear query/render commutation, invariant alpha/depth, unchanged full geometry fingerprint. | Transport errors are about 1e-6 to 1e-5. These establish operator consistency, not nonlinear text posterior equivalence or semantic fidelity. |
| Supervision | Existing factorized trainer CPU integration exposed a missing optional Namespace default for sparse gradients. | Fixed the default handling and reran the integration test. No geometry training was added to semantic supervision. |

`FrozenGaussianGeometry` contains buffers only and fingerprints centres,
normals, rotations, scales and opacity. Its cold loader validates the entire
file and tensor state. The teacher builder accepts an explicit frozen geometry
checkpoint, creates fresh exact-MPR authority, and records the full binding.
It does not load historical semantic weights in that path. Old row-indexed
features and masks cannot be treated as corresponding to new Gaussian rows.

The renderer now rejects nonfinite confidence before it can poison alpha/depth.
The existing `row_confidence` API still changes effective opacity by definition;
its use is therefore a rendering-contract change, not a harmless feature-only
weight. Selected-only and full-scene rendering remain separate protocols.

## What is not complete

The geometry pollution is isolated from semantic reconstruction, and concrete
replacement geometries have been built and cold-tested. This does not certify
all scenes as finished: figurines still has coverage/detail limitations, ramen
can trade better dense MoGe agreement for worse sparse COLMAP agreement, and
joint residual-scale optimization fails a uniform improvement criterion.
The final cohort still starts from retained FP16 MoGe predictions, not a new
native-precision inference pass. Mesh behavior has implementation/tests, not
a new measured real-mesh oracle result in this experiment.

Consequently no joint method is promoted, and no new semantic-field training
or benchmark improvement is claimed. The next acceptance work is geometric
coverage/boundary preservation and independent source geometry consistency;
actual new-field capability fidelity follows only after geometry is accepted.

## Final measured results

Uniform recovered-initialization arm, without the final joint residual fit.
Depth errors are conditional on rendered support; agreement coverage penalizes holes.

| Scene | Old MoGe AbsRel | New MoGe AbsRel | New coverage | Old/new agreement coverage | Old/new filtered COLMAP AbsRel |
| --- | ---: | ---: | ---: | ---: | ---: |
| figurines | 0.3354 | 0.1159 | 0.7990 | 0.2564 / 0.4126 | 0.3113 / 0.0461 |
| ramen | 0.0996 | 0.0915 | 0.9820 | 0.4575 / 0.5819 | 0.0560 / 0.0668 |
| teatime | 0.1544 | 0.0876 | 0.9678 | 0.3151 / 0.5950 | 0.1191 / 0.0569 |
| waldo_kitchen | 0.2971 | 0.0413 | 0.9743 | 0.0778 / 0.7196 | 0.3353 / 0.0292 |

421 related tests passed. Final cold geometry files and full-state bindings verified; synthetic feature transport leaves geometry, depth and alpha unchanged. `git diff --check` passed. No semantic model training or benchmark promotion occurred.
