# AGILE3D Full-Observation and PFPR Benchmark Design

## Goal

Replace two benchmark-specific shortcuts with auditable interfaces for one
canonical RADIO Gaussian field:

1. AGILE3D reads world clicks and official 5 cm points directly through the
   canonical field, without the 10 cm observed-row lift.
2. Pose-free image evaluation measures held-out RGB patch-to-3D-location
   retrieval rather than instance-label aggregation.

## Non-negotiable protocol constraints

- AGILE3D retains its released PLY domain, object list, click simulator,
  cumulative positive/negative clicks, forced clicked labels, and metrics.
- The method never opens object IDs, object labels, click trajectories, masks,
  or metrics while building a field or predicting a mask.
- A full 312-scene AGILE3D result is publishable only when every field records
  a full registered ScanNet observation source and passes a query-free support
  gate. Sparse `scannet_frames_25k` fields must fail closed rather than be
  coverage-stratified after evaluation.
- Existing 20-scene dense PFIR/AGILE overlap fields are a fixed development
  promotion set, not a replacement for the 312-scene table.
- PFPR reveals only the RGB patch to the method. Held-out pose/depth establish
  the evaluator-private 3-D anchor and never enter the query encoder.

## AGILE3D canonical-field interface

Each scene supplies a canonical capability bank, its typed support graph, and
fixed Gaussian geometry. A click at official coordinate `x` compiles through
`compile_world_3d_query`: covariance-aware Gaussian seeds build official DINO
appearance and SAM3 boundary prototypes. The shared confidence random walker
then operates over the full valid canonical primitive graph.

The final probability at an official 5 cm point is a continuous Gaussian
readout convolved with that fixed evaluator cell. A world-click seed remains
native-covariance by default because the released callback coordinate is an
exact retained source point, rather than a cell average. Both operations are
normalized opacity-weighted sums over fixed Gaussian candidates. A
selected-support readout is intersected with the probability readout,
preserving the shared `SEEDED_COMPONENT` instance policy. No 3NN, 10 cm
cutoff, or observed-domain graph is used by this route.

For every scene, the evaluator writes the fraction of official points with
valid continuous support before opening labels. `--require-support-gate`
rejects a field whose fraction is below the frozen threshold. The local
full-ScanNet source now contains a `.sens` stream for every one of the
released 312 AGILE validation scene IDs. This makes a 312-scene
full-observation rebuild possible, but **not yet complete**: a publishable
aggregate still requires that every reconstructed field records its full
source contract and independently passes the frozen support gate. Until then,
the fixed 20-scene PFPR/AGILE overlap remains a development promotion set, and
the code must fail closed rather than silently falling back to
`scannet_frames_25k`.

### Coverage-preserving geometry bootstrap

The full-.sens materializer records two distinct frame lists: a
numeric-sorted list used by ordinary RGB-D consumers, and the label-free
greedy depth-coverage order that selected those views. A geometry bootstrap
must not silently initialize from the first temporal frames of the sorted
directory. When `coverage_prefix` is enabled, `train_scannet_gs.py` reads only
the source contract and initializes its depth Gaussians from the prefix of
that greedy coverage order; it records the selected IDs and source digest in
`train_metadata.json`. RGB reconstruction still trains on all selected views.
This is a generic canonical-field construction choice, contains no object
labels/clicks/masks, and is required to pass the support gate rather than
relaxing the evaluator threshold.

The same distinction applies to MPR. Legacy fields retain the immutable
`canonical-mpr-v1` temporal-120 policy. The first full-observation control,
`canonical-full-observation-mpr-v1`, retains the same frozen raster,
normalization, and feature-projection rules, but selects its at-most-240 MPR
views from the source manifest's coverage-ranked order after removing held-out
validation/query frames. The cache builder fails if that manifest is absent,
incomplete, or declares labels/private anchors.

The v1 control exposed a construction bottleneck: a field source can contain
480 coverage-selected RGB-D frames while its MPR validity is still limited by
a 240-view semantic prefix. `canonical-full-observation-mpr-v2` is the
explicit repair. It is valid only for an independently materialized 480-view
source and permits up to 480 coverage-ranked non-held-out MPR views; all
other lifting rules and source digest checks are identical to v1. It is a
versioned observation-fidelity contract, not a query-conditioned selection
rule, support-threshold relaxation, or post-hoc coverage split. v1 and v2
must be reported separately.

If a v2 field still fails the label-free, fixed `C_support >= 0.95` gate, the
next permitted escalation is `canonical-full-observation-mpr-v3`: an
independently materialized 960-view prefix followed by a full MPR/canonical
field rebuild. The promotion queue reads only the v2 support audit and runs
only when that audit failed; it never opens AGILE objects, clicks, masks, or
labels. A passing v2 field is not rebuilt as v3. This makes the source-budget
ladder a construction acceptance procedure rather than a score-time knob.

The v3 admission audit has two query-free quantities: canonical-valid support
and the all-Gaussian geometry ceiling on the official 5 cm domain. When the
ceiling itself is below the frozen gate, the v3 branch must rebuild RGB
Gaussian geometry from its 960-view source; merely adding MPR observations
would be a provably insufficient semantic-only repair. Geometry reuse is
allowed only for a scene whose all-Gaussian ceiling already passes, and is
reported as a separate fixed-geometry control.

## PFPR interface

`ScanNet-PFPR-Small v1` samples valid center pixels from a held-out RGB-D
frame, back-projects the center through evaluator-private depth/pose, and
exports only a fixed RGB patch. Canonical DINO local features score the
official 5 cm candidate domain. Fixed-radius spatial NMS produces independent
top-K locations. Evaluation reports R@1/5/10 at frozen 5/10/20 cm tolerances,
top-1 Euclidean error, MRR, and a scene-macro aggregate. No instance ID,
instance ranking, graph solver, or mask IoU appears in the primary PFPR
metric.

## Validation

Unit tests prove that continuous readout is normalized, respects candidate
support, and that all clicked labels still follow the released simulator.
Integration promotion uses the frozen 20-scene dense overlap only to compare
the legacy lift with the direct canonical interface under identical official
click trajectories. A full result requires 312/312 source and support-gate
records before metric aggregation.
