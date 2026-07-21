# ScanNet-PFIR-Small v1

ScanNet-PFIR-Small evaluates pose-free real-image exemplar queries in a known
static ScanNet scene. A method receives only:

- the scene identifier, which selects one already reconstructed scene field;
- one real RGB crop produced from a held-out ScanNet frame.

It does **not** receive the query camera pose, depth, 2-D mask, class name,
instance ID, or 3-D position. “Pose-free” does not mean cross-scene retrieval.

## Frozen construction contract

The test split is selected from official ScanNet v2 validation scenes. The
development split uses separate official-train spaces and must not overlap any
scene used to train or validate a global region-summary readout. Scene
selection is deterministic and keeps at most one scan per physical
`sceneXXXX` space. Before hash selection, a scan must expose at least 24
complete finite-pose RGB/depth/instance/label frames; otherwise the required
two held-out queries, temporal-neighbor removal, and five remaining field
views are not jointly feasible.

The ordered official-val 20→30 pool is frozen in
`split/scannet_pfir_small_v1_test_candidates.txt`. The ten official-train
development candidates are frozen separately in
`split/scannet_pfir_small_v1_dev_candidates.txt`; those physical spaces must
be removed before retraining the global region-summary readout. Test scenes
must never be used for readout training, validation, or query calibration.

Each target must satisfy:

- at least 500 vertices in the official annotation mesh;
- at least 1000 projected pixels **or** 1% of the query image;
- no wall, floor, or ceiling instances;
- independently confirmed 2-D-to-3-D identity purity greater than 90%;
- at least five remaining field views;
- at least 70% official-mesh surface coverage from remaining field views.

For each instance, one complete/easy-medium view and one fixed hard view are
selected without looking at method output. Query frames are shared across
instances where possible so the held-out union does not erase a short scan.
All query frames, ±5 extracted temporal neighbors, and frames within both
0.10 m translation and 8 degrees rotation are removed from:

- RGB geometry optimization;
- MPR construction;
- canonical fitting and render fitting;
- every capability-teacher extraction.

One scene is reconstructed once after taking the union of exclusions. It is
never reconstructed per query.

### Query-free feature-fidelity validation

The canonical feature field has a separate, deterministic four-view fidelity
gate.  After query/near-pose exclusions, four evenly spaced interior frame IDs
are selected only from the remaining registered RGB-D frame manifest.  They
are excluded identically from raw RADIO, DINOv3, and SAM3 MPR construction and
are used only to select the query-independent v2 render checkpoint.  This is
not a PFIR query, crop, mask, target identity, or test-label calibration; the
selection manifest records that no benchmark label, mask, or query was opened.
All other allowed field frames remain available for geometry fitting and MPR.

The main input is a tight GT-derived bbox with 10% padding, but only RGB
inside that rectangle is exposed. A GT-masked crop is released as a separate
oracle variant and must not be mixed into the main table.

## Ground-truth resolution

ScanNet projected instance values such as `3001` are not 3-D object IDs. The
builder samples valid depth, projects depth-camera samples into the color
image, lifts them with the private query pose, maps them to the official
annotation mesh, and accepts the dominant `objectId + 1` only above 90%
purity. Pose, depth, projection masks, and resolved IDs stay evaluator-only.

## Tracks

Track A is the primary image-to-3-D instance ranking task. For every candidate
instance, the evaluator takes the mean of its top 20% mesh-vertex scores and
reports Recall@1, Recall@5, MRR, same-category Recall@1, category-macro
Recall@1, and scene-macro Recall@1. It has no threshold or graph component.

Track B converts scores into a full 3-D mask using threshold and support-solver
settings frozen on PFIR-dev. It reports query-micro and scene-macro 3-D mIoU,
Acc@IoU 0.15/0.25/0.50, centroid error, and same-category distractor success.
Empty predictions receive a deterministic scene-scale centroid penalty.

All predictions are evaluated on official ScanNet annotation-mesh vertices.
Primitive predictions are mapped to the mesh without opening instance GT.
Confidence intervals use a scene-clustered bootstrap.

## Release layout

- `manifest.method.json`: the only manifest passed to a method;
- `manifest.masked_oracle.method.json`: separate masked-crop upper bound;
- `manifest.public.json`: construction details and reproducibility hashes;
- `manifest.evaluator.json`: hidden target and candidate instance IDs;
- `manifest.internal.json`: local auditable union of the above;
- `release.json`: hashes of every frozen manifest.

The method-input manifest contains only `scene_id` and `crop_rgb` as available
inputs. The public/internal records retain exact field-frame lists and hashes
so the no-leakage contract can be audited.

## Build and evaluate

The formal release must be built from `.sens`,
`_2d-instance-filt.zip`, and `_2d-label-filt.zip`, not the sparse all-scene
`scannet_frames_25k` convenience package. The reference preparation exports
every 20th raw frame while retaining original frame IDs:

```bash
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfir.preparation.prepare_scene_set \
  --scene-names /path/to/frozen_scene_list.txt \
  --raw-root /path/to/downloaded/scans \
  --output-root /path/to/pfir_dense20_frames \
  --label-map /path/to/scannetv2-labels.combined.tsv \
  --frame-skip 20 --workers 2 \
  --report output/scannet_pfir_small_v1/dense20_preparation.json
```

```bash
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfir.build_benchmark \
  --frames-root /mnt/pool/Datasets/ScanNet/data/tasks/scannet_frames_25k \
  --annotations-root /path/to/scannet-v2/scans \
  --split-file /path/to/scannetv2_val.txt \
  --split-role test \
  --output-dir output/scannet_pfir_small_v1/test
```

The formal builder starts with 20 valid scenes and extends up to 30 only if
fewer than 200 queries survive. It refuses to freeze a formal test release
unless it contains at least 200 queries, 20 scenes, 50 same-category queries,
and 10 non-structural categories. `--allow-incomplete` is only for explicitly
named pilots.

Track predictions use one `{query_id}.npy` file per query in official-mesh row
order:

```bash
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfir.evaluate_predictions \
  --benchmark-dir output/scannet_pfir_small_v1/test \
  --prediction-dir /path/to/mesh_scores \
  --annotations-root /path/to/scannet-v2/scans \
  --track ranking \
  --output output/scannet_pfir_small_v1/ranking.json
```

Required uniformly run baselines are random instance/category prior,
LUDVIG-style direct DINO aggregation, direct full-1280 RADIO MPR, compact
canonical RADIO-GS, independent DINO/SigLIP/SAM fields, and the old HCD field.
All share geometry, held-out frames, crops, mesh mapping, and evaluator.
SigLIP-only, DINO-only, SAM-only, SigLIP+DINO, DINO+SAM, all-three unary, and
SigLIP+DINO with DINO+SAM boundary support are declared method ablations; their
weights and Track-B threshold must be frozen on PFIR-dev.
