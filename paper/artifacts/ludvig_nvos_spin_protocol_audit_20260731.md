# LUDVIG NVOS / SPIn-NeRF protocol audit (2026-07-31)

## Scope and pinned implementation

- Upstream: `naver/ludvig`
- Audited commit: `4461fc515439bb498a75d71738a1e73cf7a452ed`
- Method variant reproduced: LUDVIG-SAM (ViT-H SAM)
- Paper protocol: mean of three independent runs
- Local patch: exposes the seed and writes exact per-run JSON results; it does
  not change masks, thresholds, aggregation, or uplifting.

The released LUDVIG protocol is an **online multiview propagation** protocol,
not a strict-unseen reusable-field protocol. Original 3DGS is trained for 30k
iterations on every registered RGB view. Query-specific uplifting then
iterates over every COLMAP camera and calls the 2D foundation model on every
RGB, including the NVOS target frame. Target ground-truth masks remain
scoring-only.

## Exact original-3DGS source provenance

The LUDVIG checkout does not record a gaussian-splatting gitlink:
`gaussiansplatting/` is a regular `040000 tree`. Treating it as a locked
submodule would therefore invent provenance. We instead compared its vendored
files against the official graphdeco-inria history. Official commit
`f7a116fb1397d9842239127d39dc212f93171f70` has:

- the exact LUDVIG `train.py` git blob `36faf0de...` and SHA-256
  `c5a61947e2abcf56bf83451ae9633799d96894910ea2982a01f209c47cec462d`;
- 20 byte-identical files among 30 common tracked paths;
- ten differing paths that contain LUDVIG's camera, renderer, Gaussian-model,
  and argument extensions.

The isolated official checkout pins rasterizer
`8064f52ca233942bdec2d1a1451c026deedd320b`, simple-knn
`44f764299fa305faf6ec5ebd99939e0508331503`, and rasterizer GLM
`5c46b9c07008ae65cb81ab79cd677ecc1934b903`. All tracked and untracked source
status checks are clean. The two CUDA extensions were compiled without source
patches from those exact directories; the dry run resolves PyTorch
`2.0.1+cu118`.

The original defaults were parsed from the locked source and asserted before
launch: 30k iterations; position LR `1.6e-4 -> 1.6e-6`, delay multiplier
`0.01`, max steps 30k; feature/opacity/scaling/rotation LR
`0.0025/0.05/0.005/0.001`; densification from 500 through 15k every 100 steps
at threshold `2e-4`; opacity reset every 3000; DSSIM weight `0.2`; SH degree
3; automatic resolution `-1`; and `eval=false`.

The CPU-only immutable fern all-view dry run passed:

- 20 registered training views and no holdout;
- staged PINHOLE 3985x2988 RGBs, automatically resized to 1600x1199;
- converted/raw pose deltas q=`1.1102230246251565e-16`, t=`0.0`;
- no algorithm or environment-compatibility source patch.

Manifest:
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/fern/training/attempts/exact_f7a_allview_dryrun_v2/training_manifest.json`
(SHA-256
`c81670c3310147a7ce204d7ea424a2fe8e855f994b14b0a73b16d30e6a3f6ef2`).

### Hardware blocker and exact resume boundary

The hardware record was last good at 2026-07-31 12:49:00.349 and first
reported `[GPU is lost]` at 12:49:00.600. No LUDVIG training, evaluation, or
blocked flock process is queued, so a lock release cannot auto-launch work
onto an unhealthy device. Resume only after explicit hardware-recovery
confirmation, in this order:

1. train one fern 20/20 all-view original-3DGS checkpoint for 30k iterations;
2. validate and hash its `iteration_30000/point_cloud.ply`;
3. run LUDVIG-SAM seeds 0, 1, and 2 on that same checkpoint and aggregate only
   `selected_iou` against the paper fern mean 85.5;
4. resume the prepared remaining 23 hybrid NVOS runs and aggregate the
   explicitly non-paper 8-task x 3-seed diagnostic;
5. reuse the exact fern checkpoint for the SPIn-NeRF fern three-seed
   diagnostic described below.

## Paper values and released evaluation details

The paper's aggregate foreground-IoU values are:

| Benchmark | DINOv2 | SAM | SAM2 |
|---|---:|---:|---:|
| NVOS (8 tasks) | 92.4 | 91.3 | 91.3 |
| SPIn-NeRF (10 scenes) | 93.8 | 93.8 | 93.8 |

For LUDVIG-SAM, the paper reports:

- NVOS: fern 85.5, flower 97.6, fortress 98.1, horns_center 97.9,
  horns_left 94.1, leaves 96.4, orchids 73.1, trex 88.0.
- SPIn-NeRF: orchids 92.2, leaves 96.3, fern 97.0, room 96.5, horns 94.5,
  fortress 98.3, fork 86.8, pinecone 88.8, truck 94.9, lego 92.7.

Released NVOS LUDVIG-SAM:

- uses only the official positive scribble (the official negative scribble is
  ignored);
- samples three positive points per SAM call and makes ten calls per view;
- averages the masks, min-max normalizes each prediction, and uses
  `score > 0.25`;
- bilinearly resizes continuous predictions to the ground-truth resolution;
- computes foreground IoU and takes an equal-weight macro over eight
  one-target tasks.

Released SPIn-NeRF LUDVIG-SAM:

- uses the complete first annotated mask as the reference prompt;
- chooses the SAM candidate and threshold separately per scene by reference
  ground-truth-mask IoU;
- excludes the reference frame, averages target frames inside the scene, then
  takes an equal-weight scene macro.

These choices are encoded fail-closed in
`paper/artifacts/promptable_nvs_protocol_registry.yaml`.

## Dataset and camera audit

NVOS has all eight RGB cohorts, COLMAP cameras, official scribbles, target
masks, and local 30k strict-unseen 3DGS point clouds. Those point clouds omit
the target RGB during geometry training; they are not the paper's all-view
geometry.

The downloaded fern `sparse/0/cameras.bin` is `SIMPLE_RADIAL`, 4032x3024, but
the RGBs in `images/` are already undistorted 3985x2988 images. The downloaded
COLMAP undistortion output in `dense/sparse` is the matching `PINHOLE` model:

- source fx=fy=3260.5263328805895, cx=1992.5, cy=1494.0;
- 20 registered image names exactly match 20 RGB filenames and dimensions;
- converted versus original pose maximum deltas:
  qvec `1.1102230246251565e-16`, tvec `0.0`;
- at LUDVIG's 1600x1199 evaluation resolution, the audited effective
  fx=fy=1309.11973214779, cx=800.0, cy=599.5.

The wrapper stages `dense/sparse/{cameras,images,points3D}.bin` and a symlink to
the original undistorted `images/` under the immutable attempt output. The raw
dataset is never modified. Source SHA-256 values and all checks above are in
the attempt manifest.

The first fern attempt failed before evaluation because it directly exposed
the `SIMPLE_RADIAL` model to LUDVIG's PINHOLE-only loader. That failed manifest
and log remain at the seed root. A separate dry-run manifest demonstrates the
validated staging conversion.

## Local fern pilot

The serialized GPU0 run `pinhole_stage_retry_1` completed successfully:

| Quantity | Value |
|---|---:|
| Official fixed-threshold foreground IoU | 84.0153532982 |
| Paper fern IoU (three-run all-view mean) | 85.5 |
| Local minus paper | -1.4846467018 |
| Diagnostic target-oracle IoU | 85.9793292663 |
| Fixed / oracle threshold parameter | 75 / 54 |
| Locked GPU wall time | 94.2800 s |
| Queue wait / total launcher wall time | 974.8389 / 1069.1191 s |
| Uplifting / upstream preprocessing+uplifting log time | 26.4 / 39 s |
| Observed GPU0 memory | about 8076 MiB |

The official selected value uses parameter 75, equivalent to normalized
`score > 0.25`; the oracle row is recorded only to diagnose threshold
sensitivity and is not a valid reported metric. The selected-to-oracle gap is
1.9640 IoU points. The paper's fern result lies between those two local
values, so the released frame pairing and evaluator produce a normal,
explainable score. The remaining -1.48-point selected gap cannot be assigned
to the evaluator: this is one seed and uses a different geometry-training
visibility protocol.

This run is intentionally labeled `strict_geometry_hybrid_diagnostic`:
strict-unseen 3DGS geometry is combined with released LUDVIG uplifting that
sees the target RGB and invokes SAM on it. Consequently:

- it is not an exact paper reproduction, because the paper uses all-view 30k
  geometry and averages three runs;
- it is not a strict-unseen exact match, because target RGB is visible during
  query-time uplifting and a 2D model runs on the target;
- it is valid only as evidence that the released LUDVIG evaluator, fixed
  threshold, frame mapping, and local data are operating in the expected
  range.

Exact result and timing:

- `output/protocol_audit_20260731/ludvig/nvos/strict_geometry_hybrid_diagnostic/fern/seed_0/attempts/pinhole_stage_retry_1/fern/sam/protocol_result.json`
- `output/protocol_audit_20260731/ludvig/nvos/strict_geometry_hybrid_diagnostic/fern/seed_0/attempts/pinhole_stage_retry_1/run_manifest.json`
- `output/protocol_audit_20260731/ludvig/nvos/fern_seed0_hybrid_pilot_summary.json`

### Prepared full hybrid cohort

After the successful pilot, an audited 8-task x 3-seed hybrid cohort was
prepared:

- plan:
  `output/protocol_audit_20260731/ludvig/nvos/nvos_hybrid_8scene_3seed_v1_plan.json`;
- expected unique runs: 24;
- completed/reused: 1 (`fern`, seed 0, `pinhole_stage_retry_1`);
- pending: 23; failed/running: 0;
- camera/RGB/pose and strict-geometry point-cloud preflight: passed for all
  eight NVOS tasks.

Execution is opt-in and serialized. The coordinator calls the per-run wrapper
one at a time, and every child acquires `/tmp/radio-gs-gpu0.lock` before its
GPU section. Failed attempts are immutable and visible; later retries get a
new attempt id.

The formal aggregate requires exactly eight tasks with unique seeds 0, 1, and
2. It seed-averages within each task and then takes the equal-weight task
macro. It reads only the official fixed-threshold `selected_iou`; the
target-oracle diagnostic is never admitted. Even after all 24 runs complete,
this cohort remains a three-seed **hybrid** result, not an all-view paper
geometry reproduction and not a strict-unseen exact match.

## SPIn-NeRF availability

Nine local scenes are complete; Fork is absent. The validated diagnostic
manifest is:

`output/protocol_audit_20260731/manifests/spin_nerf_full_reference_mask_9scene_diagnostic_v1.json`

It contains 423 annotated frames across orchids, leaves, fern, room, horns,
fortress, pinecone, truck, and lego. No local SPIn-NeRF 30k 3DGS checkpoints
currently exist, so this audit does not report a nine-scene local metric. The
only future like-for-like nine-scene paper context is the LUDVIG-SAM macro
over those same scenes, 94.5777777778; it must not be compared to the paper's
full ten-scene 93.8 row.

### Fern checkpoint reuse and distinct SPIn evaluator path

The released `script/seg.sh` routes both NVOS fern and SPIn-NeRF fern to the
same `llff_data/fern` 30k geometry. The local SPIn fern source and the source
behind the audited NVOS conversion have byte-identical:

- raw `cameras.bin`: `ca0f39ee49fd61d9c66bd56c39f4ec89bf57b251e286f28cb43620c84f1355bc`;
- raw `images.bin`: `55440bb428f6be4f57200bd46677648989914d7e2315fe5d30454d0aa9fb86b6`;
- raw `points3D.bin`: `e27cbffde2922f437243bb25bb2fd4c9de8067c0805c1888540f3b0c37baad19`;
- all 20 raw RGB filenames and file hashes.

Consequently, the exact NVOS fern all-view checkpoint is also the correct
SPIn fern checkpoint; a second 30k training run would duplicate the same
scene geometry. The CPU preflight verified PINHOLE 20/20 and the released
sorted mapping from `image000.png -> IMG_4026` through
`image019.png -> IMG_4045`. All 20 main masks are binary 1008x756 masks.
`image000` is the full-mask reference and is not scored; 19 target frames
remain.

This single-scene run is useful because SPIn follows a distinct evaluation
path: it chooses one of three SAM candidates and its threshold separately for
fern using reference-mask IoU, then averages only the 19 target IoUs. After
GPU recovery and exact checkpoint validation, seeds 0/1/2 are estimated to
take roughly 6-15 minutes total and will be compared with the paper fern value
97.0. This result will remain ineligible for both the local nine-scene macro
and paper ten-scene 93.8 row.

CPU-only preflight:
`output/protocol_audit_20260731/ludvig/spin/preflight/fern_checkpoint_reuse_v1/preflight_manifest.json`
(SHA-256
`ecf1df13466f32f4b16babbcb33b5206fcb613cea0579e5a94abe9f1e77db9dd`).

## Reproduction artifacts

- `reproductions/ludvig/upstream.lock.json`
- `reproductions/ludvig/official_3dgs.lock.json`
- `reproductions/ludvig/patches/0001-reproduction-seeds-and-json-results.patch`
- `reproductions/ludvig/train_nvos_all_view_3dgs.py`
- `reproductions/ludvig/preflight_spin_fern.py`
- `reproductions/ludvig/run_ludvig_sam.py`
- `reproductions/ludvig/aggregate_results.py`
- `reproductions/ludvig/spin_nerf_9scene_rgb_dir_map.json`
- `reproductions/ludvig/README.md`
