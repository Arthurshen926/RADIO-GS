# AGILE3D / Easy3D ScanNet40 3-D Point Prompt

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

## RADIO-GS method path

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

The observation-domain lift is not a completion oracle: an unobserved point
farther than 10 cm from every observed canonical primitive receives no semantic
support. The evaluator will still apply the released AGILE3D forced-click rule
at that point, but the method does not invent a remote instance from the click.
This makes sparse scenes conservative rather than silently treating their zero
feature rows as meaningful evidence.

`run_agile3d_pfir_dense20_overlap_gpu5_queue.sh` is the valid interim
multiview-field evaluation: it reuses the one PFIR field for the 20 official
AGILE3D validation scenes that have 117--240 dense registered RGB-D views and
runs the unchanged released interaction simulator on their official PLYs. It
must be reported as a 20-scene dense-view overlap subset, never as the full
312-scene AGILE3D table. A full directly comparable multiview-field result
requires acquiring registered ScanNet `.sens`/RGB-D data for all 312
validation scenes under the ScanNet terms of use.
