# AGILE3D / Easy3D ScanNet40 3-D Point Prompt

> **Scope note.** This document also covers the project's broader interactive
> prompt protocol. The frozen Easy3D external-baseline reproduction is capped
> at 10 clicks and is defined by
> `paper/artifacts/evaluation_protocol_freeze_20260801.yaml`; generic 20-click,
> IoU@15, or NoC settings here must not be applied to that baseline row.

This evaluation is the released AGILE3D single-object protocol, not the older
RADIO-GS one-random-positive-point diagnostic.

## Frozen data

- Root: `/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet`
- 1,513 official preprocessed ScanNet PLY scenes
- 312 validation scenes and 10,357 target objects
- 40 semantic classes
- `single/object_ids.npy` and `single/object_classes.txt` select the exact
  evaluation instances

Each PLY exposes `x,y,z,R,G,B,label`. The method receives geometry/color and
the simulated clicks. `label` is evaluator-only.

## Frozen interaction

The full scene is shifted by its coordinate minimum and quantized at 5 cm.
The first input click is the target point farthest from any non-target point.
After every prediction, false positives and false negatives form the two
AGILE3D error types. The next click is the point farthest from the complement
of the error type with the larger maximum distance. A false negative yields a
positive click; a false positive yields a negative click.

Clicked labels are overwritten exactly at the clicked voxel, matching the
official evaluator. Interaction continues to 20 clicks. Metrics are:

- IoU@1, IoU@2, IoU@3, IoU@5, IoU@10, IoU@15;
- NoC@50, NoC@65, NoC@80, NoC@85, NoC@90, capped at 20.

## Direct canonical-field method path (current promotion)

The current canonical route removes the old evaluator-specific 3-NN/10 cm
observation bridge. It uses the same typed world-query compiler as the other
RADIO-GS interfaces:

```text
official 5 cm click coordinate
  -> native covariance-aware Gaussian soft seed
  -> one local DINO/SAM prototype per accumulated positive/negative click
  -> typed full canonical primitive graph + hard-seeded confidence random walker
  -> continuous opacity-weighted Gaussian probability at official 5 cm points
```

The descriptor around a click is pooled only from that click's local Gaussian
responsibility; it is not compressed with unrelated earlier clicks. This
preserves multimodal/corrective interaction evidence while keeping the field
and query compiler shared across 3-D point-prompt tasks.

Before any 312-scene promotion, field-side changes are compared on the same
predeclared full-.sens pilot scenes with the same RGB Gaussian geometry,
selected query-free views, released clicks, 5 cm domain, and evaluator:

| Arm | C-RADIO teacher route | What it isolates |
| --- | --- | --- |
| full-source project-raw control | 0.5-scale backbone, frozen adaptor after raw resampling | observation-domain/direct-readout change |
| native-official spatial field | 1.0-scale official DINOv3/SAM3 maps before registration | teacher spatial-fidelity change |
| native-official + generic graph/solver variant | identical field; only a declared query-independent support relation or seed rule changes | method-side discrimination change |

Every arm must first pass the same label-free continuous-support gate.  A
native-official arm is promoted only if it improves the fixed pilot under
the released IoU/NoC evaluator; support coverage is never used to remove
objects or form a favorable reported subgroup.

### Official spatial-teacher fidelity variant

The original compact field queue used a `0.5` C-RADIO input scale, producing a
`15 x 20` raw spatial grid that was bilinearly resampled onto the fixed
`60 x 80` Gaussian-registration raster.  That resampling does not create
boundary evidence that was absent from the teacher grid.  The named
`official_native_adaptor_mpr_v1` field variant instead uses full ScanNet RGB
resolution (`30 x 40` C-RADIO tokens) and obtains DINOv3/SAM3 maps from the
official C-RADIO runtime before any registration:

```text
registered RGB -> official C-RADIO backbone + official DINO/SAM adaptor map
               -> map resampling only to the fixed renderer raster
               -> shared query-free raster responsibility -> MPR
```

The resulting MPR metadata records the native grid, exact feature-manifest
digest, and `official_c_radio_runtime_adaptor_output`; it fails closed if the
extractor did not save both official adaptor maps. The same maps supervise the
v2 rendered capability loss and its held-out capability-fidelity selection:
the predicted canonical raw render is passed through the frozen official
projection, while the teacher is **not** projected a second time after raw-map
interpolation. The resulting capability artifact carries both MPR and render
teacher provenance; a named high-fidelity evaluation can require this audit
before labels are opened. This changes only the query-independent
teacher/field reconstruction. Geometry, full-`.sens` view selection, the
native world-query compiler, released clicks, 5 cm evaluation domain,
thresholds, and IoU/NoC evaluator remain identical. It is a predeclared
field-fidelity promotion candidate, not a silently substituted benchmark
result.

### Capability-retained canonical-field candidate

The DINO/SAM prompt interfaces read the official capability views, whereas
the raw RADIO MPR probe is a retention constraint on the shared canonical
field.  Requiring that latter cosine to improve by exactly zero can turn a
valid capability-fidelity update into a no-op: it rejects a checkpoint even
when both held-out official DINO and SAM maps improve.  The named
`canonical_mpr_v2_capability_retained_v1` candidate therefore uses the same
v1 initialization, geometry, coverage-ranked MPR cache, held-out frames,
official teachers, losses, and seed as the strict variant, but selects the
best held-out DINO/SAM fidelity checkpoint subject to two fixed **field-side**
retention constraints:

```text
each official capability cosine >= its v1 value - 0.002
raw MPR probe cosine          >= its v1 value - 0.020
```

Those constants belong to the named reconstruction variant and are shared by
all DINO/SAM query interfaces; they are not fitted per scene, object, click,
class, or benchmark result.  Selection sees no AGILE data.  It is evaluated
only by rebuilding the derived capability bank and the same query-independent
graph, then running the unchanged continuous support gate and released
IoU/NoC protocol.  Thus it tests whether better spatial capability fidelity
survives into interactive segmentation without turning field checkpoint
selection into test-set calibration.

The frozen promotion default weights the resulting local prototypes by their
continuous Gaussian support mass.  Equal mass per click remains an explicit
ablation only: on the predeclared dense-view pilot it degraded the high- and
middle-support scenes, so it is not silently selected by the command-line
default.

### Opposite-click Gaussian overlap ablation

One compact primitive can have non-negligible native Gaussian responsibility
for both a positive and a later negative click.  The historical solver keeps a
positive-first hard-constraint tie break for reproducibility.  The named
`exclusive_relative` ablation instead hard-clamps a primitive only when one
click sign has strictly larger local responsibility; an exact overlap remains
soft and is resolved by the same official-DINO/SAM unary and seeded
Laplacian.  It uses only accumulated method-visible click coordinates and
fixed Gaussian geometry, not an object label, target mask, click outcome, or
metric.  It is evaluated under the identical released interaction trajectory
and is not a change to the AGILE3D protocol.

### Exact official-capability graph ablation

The compact default support graph uses a deterministic 256-dimensional signed
hash of the official DINOv3 and SAM3 capability banks when calculating local
appearance/boundary edge affinities. This keeps CPU graph construction small,
but it can blur precisely the close cosine distinctions that matter at a
semantic boundary. `exact_official_capability_graph_v1` is therefore a
separate, query-independent graph variant:

```text
frozen canonical official DINO/SAM capability banks
  -> native-dimensional local cosine affinity in fixed GPU chunks
  -> identical 16-NN geometry topology and typed graph mixture
```

It changes neither the canonical field, field selection, source frames,
Gaussian seed, released clicks, 5 cm output points, threshold, nor evaluator.
The graph reads only the field's already frozen official capability tensors;
it never reads an object ID, semantic label, mask, click result, or metric.
Its artifact records `capability_affinity.mode=exact_official_capability` and
is compared as an explicit graph row rather than silently replacing the
hashed-baseline result. Before this row opens any AGILE label, a paired
label-free audit verifies that both graphs have exactly the same valid
primitive rows and 16-NN topology, then records per-channel affinity deltas,
edge correlation, and sampled outgoing-neighbor rank agreement. This makes a
final metric change attributable to a measured removal of hash distortion
rather than an undocumented graph change.

### Canonical surface-relation graph ablation

The base typed graph already combines local geometry, DINO appearance, and
SAM boundary affinity. `surface_relation_pca_v1` adds one optional,
query-independent relation channel estimated directly from the canonical
primitive centres: a local-PCA **unoriented** normal and a planarity
confidence. The edge uses `|n_i^T n_j|`, so arbitrary PCA sign does not change
a relation; low-planarity corners/lines are blended back to neutral affinity.
This makes nearby but geometrically distinct surfaces less likely to exchange
support while preserving uncertain regions, and it applies to every canonical
region/prompt query rather than an AGILE object class or evaluator label.

It is built from the frozen field after capability reconstruction, opens no
benchmark labels/images/clicks, and is reported as a distinct graph variant.
The default graph and all released AGILE clicks, 5 cm points, solver settings,
and metrics remain unchanged.

### Tangent-continuity and co-visibility graph variants

`surface_manifold_v1` is a stricter, still query-independent extension of the
normal-only graph.  A local normal agreement cannot distinguish two nearby
parallel surfaces: their normals match even though a Euclidean kNN edge would
leak support across the physical gap.  For each candidate edge it therefore
also measures the displacement's component along both local normals.  A
within-surface displacement is tangent; a cross-layer shortcut is not.  Low
local-PCA planarity reverts the relation to neutral, so corners and sparse
geometry are not assigned an invented boundary.

`surface_manifold_covisibility_v1` additionally uses the already frozen MPR
top-1 raster-responsibility sidecar to record whether two primitives were
registered in common RGB-D views.  The sidecar is digest-bound to the raw MPR
cache and capability field; it contains no object ID, click, label, mask, or
metric.  This relation is deliberately an explicit graph ablation rather than
a replacement for the official point domain or an observation-domain lift.
Both variants retain the same full-`.sens` source, native Gaussian seed,
continuous 5 cm field readout, threshold, clicked-point overwrite, and
released IoU/NoC evaluator.  They are promoted only if they improve the fixed
pilot under that unchanged evaluator.

### Predeclared compositional canonical-support candidate

The individual graph/scorer rows isolate a failure mode, but they need not be
mutually exclusive.  `canonical_support_v2` is therefore a separately named
composition tested only after those rows complete on the same frozen pilot:

```text
native official DINO/SAM primitive rows (no signed feature hash)
  + local-PCA tangent continuity
  + MPR responsibility co-visibility
  + four fixed query-free scene background modes
```

Every term is built once from the canonical field and its query-free training
views.  It has no instance/object list, semantic label, target mask, click
outcome, threshold fitting, or metric feedback.  In particular, the four
background modes are not fitted per object or per click.  The composition
still uses the released click sequence, native covariance seed, hard seeded
confidence random walker, fixed `0.5` support threshold, continuous official
5 cm readout, and unchanged IoU/NoC evaluator.  It is an explicit candidate
row (`eval_exact_capability_manifold_covisibility`), never a silent replacement
for the compact hashed graph.

### Query-free scene-mode background bank

`scene_mode_background_v1` is a named canonical-field scorer, not an
evaluator-side correction. Before the first click is interpreted, it fits a
deterministic set of spherical modes to the scene's already frozen primitive
DINO and SAM capability banks. The modes are then used together with the
existing scene-mean negative as a fixed background reference for every click
round. Consequently a local positive prototype must be more specific than
the scene's repeated appearance modes; later negative clicks remain hard
constraints exactly as in the released interaction protocol.

The bank opens only the canonical primitive descriptors. It never reads an
AGILE object ID, semantic label, target mask, click trajectory, prediction, or
metric, and it is fit once per scene rather than per object/query. Both four
and eight modes are fixed capacity ablations. The four-mode row is the one
carried into the predeclared compositional candidate because it was selected
before the new full-`.sens` result exists; it will be re-evaluated, not
silently promoted, on the same new field:

```bash
  --feature-calibration none --background-centroids 4 \
  --calibration-sample-size 8192 --centroid-iterations 4 \
  --score-calibration none
```

This is deliberately recorded as an explicit method variant in every report;
it does not replace the official 5 cm domain, click policy, hard clicked-point
overwrites, or IoU/NoC evaluator. It may be promoted to a formal
full-observation run only after the same source-support gate as the canonical
baseline is passed.

When comparing field reconstructions for DINO/SAM prompt interfaces, a
separate `capability_pareto` checkpoint-selection policy is permitted.  It
selects only on query-free held-out official-DINO/SAM render fidelity and the
raw-MPR constraint; it does not open AGILE labels, objects, clicks, masks, or
metrics.  It is recorded in the field checkpoint and is never mixed with the
historic raw-RADIO validation selection in one reported row.

For the official 5 cm evaluator domain, continuous **output** reading uses a
Gaussian convolved with a uniform 5 cm voxel cell (per-axis variance `h²/12`).
This is an evaluator-domain geometric readout, not a 10 cm radius cut-off. A
released callback click is an exact retained PLY point, so its Gaussian seed
uses the native covariance by default; the convolved seed is retained only as
a named geometric ablation. Its absolute output support is audited before
labels are opened. A future
`scannet_full_observation_v1` report requires an explicit full-source
declaration in each render contract and a pre-label continuous-support gate;
it fails closed for `scannet_frames_25k` inputs. For the full-`.sens` contract
the gate uses support mass `>= 0.01`: this is approximately a unit-opacity
Gaussian at three standard deviations (`exp(-9/2)`), so a point is not counted
as covered solely by a long Gaussian tail. The same fixed cutoff masks
unsupported official output points; it is geometry-only and does not depend
on class, object, click, or evaluator label.

The support preflight parses only `x,y,z,R,G,B` from the binary official PLY
with a selective record-stride reader. Although the release stores `label` in
the same vertex record, that property is not materialized until every selected
field has passed preflight and the released evaluator starts click simulation.
The baseline queue first persists this audit as `support_preflight.json` with
`labels_opened: false`; a failed audit stops before object loading and is the
trigger for increasing the query-free full-`.sens` view budget, rather than a
coverage-based result split.

### Full-observation MPR v2: preserve the 480-view source prefix

The first full-source implementation, `canonical-full-observation-mpr-v1`,
is frozen as a **240-view MPR control**. The 480-view source audit showed
that merely giving geometry 480 RGB-D views is insufficient when raw RADIO,
DINO, and SAM MPR silently truncate the same source back to its first 240
coverage-ranked non-held-out views: primitive validity then remains the
dominant support-gate bottleneck. This is a field-construction mismatch, not
an AGILE object-coverage subgroup or a threshold to relax.

`canonical-full-observation-mpr-v2` is therefore a separate audited lifting
contract. It preserves the same raster responsibility, normalization,
feature-projection, held-out-frame exclusion, and label-free source ordering
as v1, but permits up to **480** coverage-ranked non-held-out observations
when (and only when) the source manifest itself materialized a 480-view
prefix. Its cache declaration/digest, declared view count, and source-manifest
digest are all checked before AGILE labels are opened. A 240-view source may
not be relabelled as v2, and v1/v2 trajectories are never mixed in one result
row or merge.

The filename `canonical_mpr_v2.pt` denotes the second canonical field-training
stage; it is not evidence that the cache used the 480-view MPR v2 contract.
Every formal report records `canonical_mpr_contract` explicitly to avoid that
ambiguity.

### Full-observation MPR v3: fixed 960-view promotion after a failed v2 gate

`canonical-full-observation-mpr-v3` is a separately auditable source-fidelity
rung. It is permitted only after the same scene's **label-free** v2 support
record is below the already frozen `0.95` gate, and only with an independently
materialized **960-view** prefix of the same RGB-D-only greedy coverage order.
It does not change the official 5 cm domain, click sequence, continuous
readout cutoff, graph, solver, query features, or evaluator.

This is not post-hoc score selection: a v2 field that passes preflight exits
the v3 queue without reconstruction. If v2 fails, the larger prefix is rebuilt
under a new source root and MPR digest, then must pass the same support
preflight before any AGILE object or label is opened. The 240/480/960-view
rows remain distinct controls.

Before that queue is chosen, a second label-free diagnostic reads the fixed
official geometry against **all** reconstructed Gaussians. If this
geometry-only ceiling is already below `0.95`, MPR alone cannot repair the
scene: v3 retrains RGB Gaussian geometry from the 960-view source and then
rebuilds render contracts, MPR, and the canonical field. If the ceiling
already passes, v3 may reuse the frozen query-free geometry as an explicit
``fixed geometry + semantic MPR`` control. This branch is decided before
labels and never from AGILE IoU.

The geometry-rebuild branch uses the same fixed construction budget for every
such scene: the first 240 coverage-ranked RGB-D frames initialize up to
300,000 Gaussians, while RGB training still samples the complete 960-view
source. This is deliberately larger than the historic 50-frame/200k bootstrap
because the all-Gaussian diagnostic has already proved that its spatial support
is insufficient; it is not an object-specific reconstruction setting.

After the rebuilt geometry has produced its query-free raw MPR contract, the
queue repeats the same all-Gaussian 5 cm support audit **before** constructing
the expensive DINO/SAM MPR caches. A rebuilt scene that still fails the fixed
gate stops at `geometry_support_gate/<scene>.json`; it cannot consume semantic
reconstruction time and later be presented as a low-coverage AGILE result.
This second gate has the same geometry/RGB-only reader and explicit
`labels_opened: false` / `object_list_opened: false` audit as the admission
diagnostic above.

The auditable diagnostic is available as:

```bash
bash radio_gs/scripts/run_repo_python.sh -m \
  radio_gs.benchmarks.agile3d_scannet40.audit_geometry_support \
  --benchmark-root /mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet \
  --field-root /path/to/reconstruction_v1 \
  --scene-names sceneXXXX_YY \
  --output /path/to/geometry_support_ceiling.json \
  --device cpu
```

Its report explicitly records `labels_opened: false`,
`object_list_opened: false`, and whether geometry rebuild is required; it is
not a score report.

```bash
CUDA_VISIBLE_DEVICES=5 bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.agile3d_scannet40.evaluate_canonical_field \
  --benchmark-root /mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet \
  --field-root /path/to/reconstruction \
  --geometry-cache-root /path/to/geometry_cache \
  --output /path/to/result.json \
  --observation-contract dense_overlap_pilot \
  --evaluation-voxel-size-m 0.05 \
  --world-point-prototype-mode per_click_local \
  --device cuda:0
```

The legacy cache-export evaluator below remains an ablation/control rather
than the canonical promotion route.

## Legacy observation-domain control

Every scene is reconstructed independently of clicks and labels into the same
compact canonical RADIO primitive field used by the other query interfaces.
The official frozen DINOv3 and SAM3 adaptors provide appearance and boundary
views. A world click compiles positive/negative evidence; the shared
hard-seeded confidence random walker produces primitive support. In the
registered multiview setting, the solver runs only on primitives with an
actual canonical observation. A fixed, query-free 3-nearest-observed geometric
map (at most 10 cm, two official voxels) lifts a click into that domain and
projects support back to official points. Thus an unobserved zero row never
becomes a click prototype or a graph node. The final instance mask retains
only thresholded support components touched by a positive registered click;
this is the same `SEEDED_COMPONENT` rule used by the project's other
world-3D query route, and prevents an unconnected but visually similar object
from being emitted as the clicked instance. Both raw feature coverage and the
geometrically projectable fraction are recorded per scene.

For v2 render-fidelity selection, four evenly spaced interior registered RGB-D
frames are deterministically held out from all raw/DINO/SAM MPR construction.
They are selected from frame IDs alone and used only for the unlabeled
canonical-field fidelity gate.  They do not use AGILE3D object labels, click
trajectories, object lists, or metrics; the official PLY `label` remains
evaluator-only.  This small observation split is part of our multiview-field
training contract, not a modification of the released AGILE3D interaction
protocol.

For the current cache-based evaluator, canonical features are exported into
the official PLY row order before 5 cm quantization. The export opens no
instance labels, queries, predictions, or metrics.

## Commands

Export one finished scene field:

```bash
CUDA_VISIBLE_DEVICES=5 bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.scripts.export_canonical_field_to_agile3d_mesh \
  --field-checkpoint /path/to/scene/canonical_field.pt \
  --capability-cache /path/to/scene/capability_views.pt \
  --mesh-ply /mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet/scans/sceneXXXX_YY.ply \
  --output output/agile3d_scannet40/features/sceneXXXX_YY.npz \
  --device cuda:0
```

Evaluate a pilot or the full official list:

```bash
CUDA_VISIBLE_DEVICES=5 bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.agile3d_scannet40.evaluate_feature_cache \
  --benchmark-root /mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet \
  --feature-root output/agile3d_scannet40/features \
  --output output/agile3d_scannet40/results.json \
  --device cuda:0
```

`--scene-names` is only for a labeled pilot result. A directly comparable
ScanNet40 number must retain all 312 scenes and all 10,357 selected objects.

## Observation-data boundary

The released AGILE3D package contains the official point clouds and evaluator
labels, but not the registered RGB-D streams required to build a multiview
feature field. The convenience
`/mnt/pool/Datasets/ScanNet/data/tasks/scannet_frames_25k` package supplies a
scene-dependent subset of registered views (3--55 views across the 312
official validation scenes; median 15 at the current dataset revision). A
full-312 run constructed from those views is a **registered multiview-field
setting**, not a directly comparable result against PLY-only AGILE3D/Easy3D
methods: the methods receive different observation modalities and some
canonical rows remain unobserved. The evaluator records per-scene canonical
feature coverage to make this visible.

The direct canonical evaluator no longer uses the old 3NN/10 cm
observation-domain lift. Each released click is compiled at its world
coordinate through covariance-aware Gaussian responsibilities, and every
official 5 cm point receives a continuous opacity-weighted field readout.
Sparse source views nevertheless remain a reconstruction failure when that
continuous support is absent; the evaluator never converts a missing field
into a zero-valued feature or a synthetic completion.

`run_agile3d_pfir_dense20_overlap_gpu5_queue.sh` is the valid interim
multiview-field evaluation: it reuses the one PFIR field for the 20 official
AGILE3D validation scenes that have 117--240 dense registered RGB-D views and
runs the unchanged released interaction simulator on their official PLYs. It
must be reported as a 20-scene dense-view overlap subset, never as the full
312-scene AGILE3D table. A full directly comparable multiview-field result
requires acquiring registered ScanNet `.sens`/RGB-D data for all 312
validation scenes under the ScanNet terms of use.

For an AGILE-only dense-view promotion, the source may instead materialize all
valid RGB-D frames in this fixed 20-scene pilot (with `instance/` and `label/`
omitted) under `dense_agile_all_observations_pilot`.  This uses no query image
or evaluator target, and is intentionally kept distinct from PFPR's
query-held-out source contract.  It is still a 20-scene pilot rather than a
full-312 result.

This workspace now contains one complete source `.sens` file for each of the
312 released AGILE3D validation scene IDs. They are materialized under the
query-free `scannet_full_observation_v1` source contract by deterministic
greedy depth voxel coverage (an initial 240-frame budget, escalated to
480/960 or all valid views when the pre-label continuous-support gate fails),
with the sensor-file digest and every selected frame recorded. Evaluation of
those fields uses a fail-closed support gate. The 20-scene overlap remains the
development promotion set; a 312-scene result cannot be reported until every
one of the 312 reconstructed fields has independently passed that gate and
the released evaluator has processed all 10,357 objects.

The view budget is not a score-time knob. A failed gate triggers a new,
versioned field source with a larger prefix of the same query-free greedy
coverage order, followed by geometry, MPR, and canonical-field reconstruction
from that source. The gate is then re-run before any object list or label is
opened. We never lower the support cutoff, increase a point readout radius, or
remove unsupported objects to turn a failed source into a reportable row.

For this source contract, field construction retains the same query-free
coverage ranking at both places where it otherwise could be lost to a
numeric-sorted frame directory: the Gaussian geometry bootstrap uses a prefix
of the ranked depth frames, and MPR uses the versioned
`canonical-full-observation-mpr-v1` (240-source control),
`canonical-full-observation-mpr-v2` (independently materialized 480-source
promotion), or `canonical-full-observation-mpr-v3` (independently materialized
960-source promotion after a v2 support-gate failure) to select the available
coverage-ranked views after the fixed render-fidelity held-out frames are
removed from the MPR input. These rungs keep the raster aggregation and
official per-view capability projection unchanged; they change only a
source-provenance-backed view budget. The cache builder
rejects an absent/incomplete manifest or one declaring private anchors, masks,
instances, or semantic labels. The direct AGILE evaluator independently
checks this MPR declaration and its source-manifest digest before opening
object labels, so a dense geometry rebuild cannot silently reuse a legacy
temporal-120 feature cache.

## Full-source materialization (formal run prerequisite)

When the registered ScanNet `.sens` streams are available, the field source is
decoded with the same RGB-D-only contract rather than copied from a labeled
projection package.  The formal default greedily covers the scene's own
5 cm RGB-D surface voxels from registered depth and poses. It opens neither
AGILE3D labels nor object IDs, clicks, masks, or query images. It records the
full sensor-file digest, every selected frame index, greedy selection order,
and the exact source policy. Pose-diverse FPS remains a named query-free
ablation only.

```bash
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.prepare_field_contract \
  --full-scannet-observations \
  --sens-root /path/to/ScanNet/scans \
  --output-root /path/to/scannet_full_observation_v1 \
  --scenes-file /path/to/agile3d_val_scene_ids.txt \
  --max-frames 240 \
  --candidate-stride 1 \
  --frame-selection-policy depth_voxel_coverage \
  --coverage-voxel-size-m 0.05 \
  --coverage-depth-stride 8
```

The resulting field must then pass `--require-support-gate` at the frozen
continuous-support threshold before the 312-scene AGILE result is merged.  A
failed scene is a reconstruction/data-quality failure to repair by increasing
query-free field views; it is never removed or explained away by a coverage
stratum. The evaluator runs this support preflight for every selected scene
before opening the released object list or per-vertex target labels.
