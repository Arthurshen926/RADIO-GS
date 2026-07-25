# ScanNet-PFPR-Small v1 (legacy native-color crop release)

This frozen release remains readable for reproducibility, but its query crop
was constructed on the original ScanNet color raster (`1296 x 968` in the
common scans) while the canonical RGB-D field is built and rendered on the
depth-aligned raster (`640 x 480`).  A 128 px crop consequently represented a
different physical field of view from the observation that supplied the field
descriptor.  It is not used as the next promotion target.  The corrected,
strictly versioned protocol is [ScanNet-PFPR-Small v2](scannet_pfpr_small_v2.md).

ScanNet-PFPR-Small v1 evaluates a single pose-free RGB patch as a 3-D location
retrieval query inside one known reconstructed ScanNet scene:

```text
held-out RGB patch + canonical field -> ranked public 3-D point hypotheses
```

It is intentionally separate from ScanNet-PFIR instance ranking. PFPR does
not use an instance ID, category, mask, connected component, or mIoU as its
primary target.

## Frozen construction

- 20 dense-view ScanNet scenes reused from the frozen PFIR field split.
- 10 queries per scene, for 200 queries total.
- Every query is a fixed `128 x 128` RGB patch centered on a valid depth
  sample from a frame excluded from that scene's canonical-field construction.
- The evaluator privately backprojects the center depth sample with its camera
  pose to obtain the 3-D anchor. Pose, depth, source frame ID, anchor, masks,
  instance IDs, and semantic labels are absent from `manifest.method.json`.
- The public candidate domain is annotation-mesh geometry only, stably
  quantized at 5 cm. Anchors must lie within 5 cm of this public domain at
  construction time; no label information is involved.

The method-facing manifest exposes only `scene_id` and `crop_rgb`. The
candidate geometry is shared public scene geometry, not a query registration
hint.

## Full-`.sens` field-source promotion

The initial 20-scene fields use the frozen dense-view source. A stricter
source contract is now available for the same fixed 20 scenes: decode each
complete ScanNet `.sens` sequence, exclude only the source frame of every
evaluator-private RGB crop, then choose up to 240 remaining frames by
query-free 5 cm depth-voxel coverage. The field input records the complete
sensor-file digest, the selected frame digest, and only a hash of the
withheld source-frame set. It does not store anchor coordinates, depth pixels,
instance IDs, masks, or labels.

```bash
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.prepare_field_contract \
  --full-scannet-pfpr-queryheldout-observations \
  --method-manifest output/scannet_pfpr_small_v1/test_v1/manifest.method.json \
  --evaluator-manifest output/scannet_pfpr_small_v1/test_v1/manifest.evaluator.json \
  --sens-root /path/to/ScanNet/raw \
  --output-root /path/to/scannet_pfpr_full_sens_queryheldout_v1 \
  --max-frames 240 --candidate-stride 1 \
  --frame-selection-policy depth_voxel_coverage \
  --coverage-voxel-size-m 0.05 --coverage-depth-stride 8
```

The corresponding render contract must declare
`scannet_full_observation_pfpr_queryheldout_v1`; it fails closed unless the
field source has both complete `.sens` provenance and an anchor-free withheld
frame-set digest. This is a source-fidelity promotion, not a change to the
fixed patch, candidate domain, NMS, or evaluator. The local source collection
currently covers the fixed 20 PFPR scenes only.

For the full-observation contract, a public candidate must also have
continuous opacity-weighted support mass at least `0.01` (approximately a
unit-opacity Gaussian at three standard deviations). Candidates below that
fixed geometry-only cutoff receive a finite floor score and cannot be selected
as a retrieval hypothesis. The support fraction is checked before private
anchors are opened; this prevents a remote Gaussian tail from appearing as a
valid 3-D correspondence.

For this full-observation source only, reconstruction preserves the
materializer's query-free coverage order during Gaussian depth bootstrap and
uses `canonical-full-observation-mpr-v1` for the bounded 240-view MPR subset.
The held-out crop source frames remain excluded before either operation. No
private anchor, depth pixel, pose, mask, instance, or semantic value enters
the ordering; it is recorded in the field and MPR provenance so it can be
audited independently of PFPR retrieval metrics. The PFPR scorer repeats the
same fail-closed MPR/source-digest check before scoring a crop, preventing a
dense geometry field from silently retaining an old temporal-120 DINO cache.

The same field-side `official_native_adaptor_mpr_v1` fidelity variant used by
AGILE3D is also permitted here: full-resolution C-RADIO produces its native
official DINOv3/SAM3 spatial maps before map resampling and MPR. Its manifest
must declare the native grid and official extractor outputs. The same direct
official maps are used consistently in v2 render-time DINO/SAM fidelity loss
and held-out capability selection, rather than reapplying an adaptor after a
raw-map interpolation. This changes no PFPR method input (the held-out RGB
crop remains pose-free), private anchor, public 5 cm candidates, fixed 10 cm
NMS, or retrieval metric; it is solely a shared canonical-field reconstruction
variant and must be reported as such.

For that named variant, the scorer can require
`--require-official-extracted-capability-teachers`. This provenance gate
checks that both DINO/SAM MPR targets and the v2 render-fidelity targets were
native official runtime outputs. It does not change the crop input, public
candidates, NMS, scores, or private-anchor evaluator.

## Evaluation

A method writes one finite score vector per query, aligned with the public
scene candidate array. The evaluator applies fixed 10 cm Euclidean NMS before
taking ranked hypotheses. It reports:

- R@1/R@5/R@10 at 10 cm;
- R@1 at 5/10/20 cm;
- top-1 mean and median 3-D error;
- first-correct MRR at 5/10/20 cm;
- both query-micro and scene-macro averages.

Correctness is Euclidean distance to the private 3-D anchor, never instance
identity. The initial release uses Euclidean surface-proxy distance; a future
geodesic supplement must retain this frozen main table.

## Build and run the official-DINO baseline

```bash
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.build_benchmark \
  --frames-root /mnt/pool/sqy/3d_understanding/segmentation_benchmarks/ScanNet-PFIR-Small/frames_dense20 \
  --annotations-root /mnt/pool/sqy/3d_understanding/segmentation_benchmarks/ScanNet-PFIR-Small/annotations \
  --source-pfir-public-manifest output/scannet_pfir_small_v1/test_v1_final/manifest.public.json \
  --output-dir output/scannet_pfpr_small_v1/test_v1
```

The first baseline uses the official C-RADIO DINOv3 spatial adaptor's center
3x3 descriptor and scores it against the canonical DINO primitive field.  A
single center-token variant is a named query-only ablation: it changes neither
the held-out patch, public candidate domain, NMS, nor evaluator anchor. The
primitive scores are read continuously onto the same public 5 cm candidate
domain using opacity-weighted Gaussian kernels convolved with the evaluator
voxel cell. It reads `manifest.method.json` and `manifest.public.json` only.

## Optional global crop-context bridge

The held-out 128 px patch is intentionally pose-free, so its local DINO
descriptor can differ from the token that the same official adaptor would
produce in the full source image. `global_crop_context_adapter_v1` addresses
only this query-side context shift. It receives the method-visible crop's
official DINO center-3x3 descriptor and spatial global mean, then adds a
low-rank residual in the same frozen DINO feature space. The residual is
zero-initialized, so an untrained checkpoint is exactly the center-3x3
baseline.

The adapter is trained once from RGB-only crop/full-image pairs from
`scannet_frames_25k`; all PFPR and AGILE evaluation physical spaces are
excluded before sampling. It reads no PFPR crop, private anchor, depth, pose,
field, instance, label, mask, text, click, target, or metric. Its capacity is
selected solely on a separate, physically disjoint RGB-only validation set,
then the selected checkpoint is frozen for the entire PFPR evaluation. It
never changes candidate geometry, 10 cm NMS, scoring readout, or the private
evaluator.

This bridge is retained as a documented query-compiler ablation, not the
current PFPR default. On its fully scene-disjoint RGB-only validation it raises
crop-to-full-token cosine, but on the frozen 200-query PFPR result it lowers
`R@5_10cm` from `0.160` to `0.150`. This distinction is useful: matching a
full-image teacher token alone is not sufficient to preserve globally
discriminative point retrieval. The checkpoint must therefore never replace
the base compiler merely because its 2-D teacher-alignment number is higher.

```bash
CUDA_VISIBLE_DEVICES=5 bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.score_dino_center \
  --benchmark-dir output/scannet_pfpr_small_v1/test_v1 \
  --field-root output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1 \
  --geometry-cache-root output/scannet_pfpr_small_v1/crop_context_adapter_v1/adapter_geometry \
  --prediction-dir output/scannet_pfpr_small_v1/crop_context_adapter_v1/adapter_predictions \
  --crop-context-adapter-checkpoint /path/to/frozen_context_bridge.pt \
  --device cuda:0
```

The report stores the checkpoint digest and its fail-closed provenance
manifest. `--crop-context-adapter-checkpoint` is mutually exclusive with the
older crop-spatial bridge so that a row always names one well-defined query
compiler.

## Query-independent canonical-DINO calibration

`scene_diagonal_robust_normalized_cosine` is a training-free alternative that
fits a robust diagonal centre and scale from the *frozen canonical DINO field*
in each scene, then applies exactly that same transform to every crop query.
It uses neither a query anchor nor any evaluator-only input. It addresses the
otherwise arbitrary per-dimension scene bias introduced by compact field
reconstruction while retaining a direct local DINO query and the identical
continuous 5 cm readout/NMS contract.

On the frozen 200-query PFPR v1 evaluator, this changes `R@5_10cm` from
`0.160` to `0.210`, `R@10_10cm` from `0.200` to `0.235`, and `MRR_10cm` from
`0.105` to `0.134`. Its top-1 median error also improves (`1.712 m` to
`1.597 m`), while the mean top-1 error is slightly higher (`2.065 m` to
`2.092 m`); formal tables must report both rather than selecting one metric.

```bash
CUDA_VISIBLE_DEVICES=5 bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.score_dino_center \
  --benchmark-dir output/scannet_pfpr_small_v1/test_v1 \
  --field-root output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1 \
  --geometry-cache-root output/scannet_pfpr_small_v1/dino_center_v1/geometry \
  --prediction-dir output/scannet_pfpr_small_v1/field_calibration_v1/predictions \
  --feature-calibration diagonal_robust \
  --calibration-sample-size 8192 \
  --device cuda:0
```

```bash
CUDA_VISIBLE_DEVICES=5 bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.score_dino_center \
  --benchmark-dir output/scannet_pfpr_small_v1/test_v1 \
  --field-root output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1 \
  --geometry-cache-root output/scannet_pfpr_small_v1/dino_center_v1/geometry \
  --prediction-dir output/scannet_pfpr_small_v1/dino_center_v1/predictions \
  --device cuda:0
```

```bash
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.evaluate_predictions \
  --benchmark-dir output/scannet_pfpr_small_v1/test_v1 \
  --prediction-dir output/scannet_pfpr_small_v1/dino_center_v1/predictions \
  --output output/scannet_pfpr_small_v1/dino_center_v1/results.json
```
