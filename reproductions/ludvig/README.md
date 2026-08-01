# LUDVIG NVOS / SPIn-NeRF protocol reproduction

This directory pins and wraps the official ICCV 2025 LUDVIG release without
changing RADIO-GS field, training, or query code.

## Protocol status

- Official LUDVIG is an online multiview propagation method. It trains 3DGS on
  all registered RGB views and calls SAM on every view during query-specific
  uplifting. Its NVOS number is therefore **not** a strict-unseen exact match.
- NVOS LUDVIG-SAM uses the positive official scribble, 3 positive points per
  SAM call, 10 calls per view, and a fixed 0.25 threshold after per-frame
  min/max normalization. The released code does not use the negative scribble.
- SPIn-NeRF uses the complete first-frame mask and selects the SAM candidate
  and threshold separately for each scene by reference-mask IoU. The reference
  frame is excluded; target frames are averaged inside each scene, followed by
  a scene macro.
- The paper averages three independent runs, while the release hard-codes seed
  0. The local patch exposes `--seed` and emits exact JSON results.
- Local SPIn-NeRF lacks Fork. The only valid comparison is the available
  9-scene macro against the paper's same-scene LUDVIG-SAM macro, 94.5778 IoU;
  it is not the paper's 10-scene 93.8 row.

## Checkout and patch

```bash
git clone https://github.com/naver/ludvig.git /root/baselines/LUDVIG
git -C /root/baselines/LUDVIG checkout 4461fc515439bb498a75d71738a1e73cf7a452ed
git -C /root/baselines/LUDVIG apply \
  /root/RADIO-GS/reproductions/ludvig/patches/0001-reproduction-seeds-and-json-results.patch
```

Install the upstream dependencies in an isolated environment. The launcher
requires the patched rasterizer on `PYTHONPATH` and locks GPU 0 itself. On this
machine it also prepends `/root/baselines/LUDVIG/.driver535` so PyTorch loads
the `libcuda.so.1` matching the 535 kernel driver; the resolved library and
SHA-256 are recorded in every non-dry-run manifest.
LUDVIG-SAM requires the audited ViT-H SAM checkpoint SHA-256
`a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`;
an arbitrary file passed through `--sam-checkpoint` is rejected before GPU
work.
The only permitted tracked checkout diff is the two-file reproduction patch
with SHA-256
`7c86d5883058fb9e529608ba6cf856c04e66d37d338a69a8ca6863725292e9ac`;
the launcher also verifies both patched file hashes and rejects staged or
additional tracked changes.

## Exact original-3DGS geometry source

LUDVIG does not contain a gaussian-splatting gitlink: `gaussiansplatting/` is
a regular `040000 tree`. We therefore compared its vendored files against the
official graphdeco-inria history. Official commit
`f7a116fb1397d9842239127d39dc212f93171f70` has the exact LUDVIG `train.py`
blob (`36faf0de...`, SHA-256 `c5a61947...`) and 20 of 30 common tracked files
are byte-identical; the remaining ten are LUDVIG-specific changes. The
auditable reconstruction and original defaults are pinned in
`official_3dgs.lock.json`.

The isolated checkout also pins the exact gitlinks:

- diff-gaussian-rasterization `8064f52ca233942bdec2d1a1451c026deedd320b`;
- simple-knn `44f764299fa305faf6ec5ebd99939e0508331503`;
- rasterizer GLM `5c46b9c07008ae65cb81ab79cd677ecc1934b903`.

`train_nvos_all_view_3dgs.py` refuses tracked or untracked source changes,
checks the source hashes and literal defaults, validates that the compiled
extensions were installed from those exact submodule directories, and stages
all 20 fern views. Its GPU section is inside the shared GPU0 flock. The
CPU-only immutable dry run is:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  reproductions/ludvig/train_nvos_all_view_3dgs.py \
  --attempt-id exact_f7a_allview_dryrun_v2 --dry-run
```

It passed with PINHOLE 20/20, 3985x2988 RGBs, effective 1600x1199 training
resolution, and original 30k hyperparameters. The manifest is
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/fern/training/attempts/exact_f7a_allview_dryrun_v2/training_manifest.json`
(SHA-256 `c81670c3310147a7ce204d7ea424a2fe8e855f994b14b0a73b16d30e6a3f6ef2`).

The 2026-07-31 driver/UVM interruption is retained only as historical failure
evidence. After the host recovered, all released all-view NVOS checkpoints and
the exact seeds `[0,1,2]` completed on physical GPU0. The immutable aggregate is
`output/protocol_audit_20260731/ludvig/nvos/released_all_view_full8_3seed_summary.json`
(SHA-256 `65e1f8e5c1f17083f66e5b7d4f6f03687f6806394c78a5cfce25546ca42e3546`).
Its eight-scene macro is 91.257685 IoU versus the paper's same-scene recompute
of 91.3375, a -0.079815 point difference.

## Fail-closed dry run

The existing NVOS checkpoints were trained under strict-unseen geometry. They
can diagnose LUDVIG's target-visible uplifting, but cannot be labeled as the
released all-view geometry protocol. The downloaded NVOS root has distorted
`SIMPLE_RADIAL` cameras in `sparse/0`, while `images/` are already the
undistorted 3985x2988 RGBs. `--stage-nvos-pinhole` creates an isolated,
validated view of the existing `dense/sparse` PINHOLE model under the attempt
output. It checks RGB names/count/dimensions, pose equivalence, centered
principal point, target-resolution intrinsics, and hashes without modifying
the downloaded data:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  reproductions/ludvig/run_ludvig_sam.py \
  --benchmark nvos --scene fern --seed 0 \
  --geometry-protocol strict_geometry_hybrid_diagnostic \
  --attempt-id pinhole_stage_dryrun \
  --stage-nvos-pinhole \
  --upstream /root/baselines/LUDVIG \
  --python /root/miniconda3/envs/cybersim_agent/bin/python \
  --pythonpath /path/to/isolated/ludvig/dependencies \
  --dry-run
```

For an official all-view run, pass a 30k all-view `point_cloud.ply`:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  reproductions/ludvig/run_ludvig_sam.py \
  --benchmark nvos --scene fern --seed 0 \
  --geometry-protocol released_all_view \
  --attempt-id all_view_seed0 \
  --stage-nvos-pinhole \
  --gs-source /path/to/all_view/point_cloud/iteration_30000/point_cloud.ply \
  --gs-training-manifest /path/to/all_view/training_manifest.json \
  --upstream /root/baselines/LUDVIG \
  --python /root/miniconda3/envs/cybersim_agent/bin/python \
  --pythonpath /path/to/isolated/ludvig/dependencies
```

Attempt directories are immutable: the launcher refuses to overwrite an
existing manifest. Use a new `--attempt-id` after a failed or interrupted run.
The manifest reports queue wait, locked GPU wall time, and total wall time.

Repeat seeds 0, 1, and 2, then aggregate:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  reproductions/ludvig/aggregate_results.py \
  --benchmark nvos \
  --input-root \
    output/protocol_audit_20260731/ludvig/nvos/released_all_view \
  --output output/protocol_audit_20260731/ludvig/nvos_summary.json
```

Geometry protocols must be aggregated from separate roots. The aggregator
rejects mixed geometry, an unpinned LUDVIG commit or SAM checkpoint,
mismatched target-view visibility/calibration/aggregation, invalid IoUs,
duplicate scene/seed pairs, or seeds of one scene that reference different
checkpoint hashes. It binds NVOS results to the fixed threshold parameter 75
and recomputes SPIn scene IoU from target-only frame rows. A released all-view
fern result with exact seeds `[0,1,2]`, embedded patch provenance, and one
shared checkpoint is eligible for a per-scene paper-protocol check against
85.5, but remains ineligible for the full eight-task paper row.

## Audited 8-scene x 3-seed hybrid cohort

`run_nvos_hybrid_cohort.py` is prepare-only unless `--execute` is passed. Its
CPU preflight validates all eight task camera/RGB/pose mappings and all
strict-geometry point clouds, discovers completed scene/seed pairs, and writes
an immutable 24-task plan:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  reproductions/ludvig/run_nvos_hybrid_cohort.py
```

The original plan reused the completed fern seed 0 pilot and scheduled the
remaining 23 runs. The complete exact 24-run cohort is now recorded by the
aggregate above; the command remains useful only for immutable discovery and
verification:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  reproductions/ludvig/run_nvos_hybrid_cohort.py --execute
```

The coordinator never takes an outer GPU lock or launches runs in parallel.
Each child launcher independently acquires `/tmp/radio-gs-gpu0.lock`, so
another queued experiment runs first. A failed child remains visible and the
next coordinator invocation allocates a new attempt id.

The aggregator rejects duplicate completed scene/seed pairs and requires the
exact seeds `[0, 1, 2]`. NVOS aggregation reads only `selected_iou`, i.e. the
official fixed-threshold result; target-oracle values are never aggregated.
After all 24 unique runs complete, the coordinator writes the full cohort
summary automatically.

All outputs are isolated under `output/protocol_audit_20260731/ludvig/`.

## SPIn-NeRF fern CPU preflight

The released `script/seg.sh` routes both NVOS fern and SPIn-NeRF fern through
the same `llff_data/fern` 30k checkpoint. Locally, all three raw COLMAP files
and all 20 raw SPIn RGBs are byte-identical to the raw source behind the
audited NVOS PINHOLE conversion. The exact NVOS all-view fern checkpoint can
therefore be reused for SPIn fern; a second 30k training run is neither needed
nor justified.

`preflight_spin_fern.py` verifies that equivalence, stages the undistorted
PINHOLE model, and reproduces the released sorted mapping:
`image000.png -> IMG_4026` through `image019.png -> IMG_4045`. `image000` is
the full-mask reference; 19 later frames are scored. The distinct SPIn path
selects one of three SAM candidates and its threshold from reference-mask IoU
and excludes that reference from the scene mean.

The CPU-only preflight manifest is
`output/protocol_audit_20260731/ludvig/spin/preflight/fern_checkpoint_reuse_v1/preflight_manifest.json`
(SHA-256 `ecf1df13466f32f4b16babbcb33b5206fcb613cea0579e5a94abe9f1e77db9dd`).
That preflight was followed by the exact three-seed evaluation. The same
contract was then completed for all locally available SPIn-NeRF scenes.

## Completed SPIn-NeRF nine-scene cohort

The local source lacks `fork`, so the strict local endpoint is the nine-scene
cohort `fern`, `fortress`, `horns`, `leaves`, `lego`, `orchids`, `pinecone`,
`room`, and `truck`, each with exact seeds `[0,1,2]`. The immutable aggregate
is
`output/protocol_audit_20260731/ludvig/spin/released_all_view_fern_leaves_orchids_fortress_horns_room_pinecone_truck_lego_3seed_summary.json`
(SHA-256 `ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17`).
It reports 93.720045 IoU versus 94.577778 for the paper's matching nine scenes,
a -0.857733 point difference. The paper's 93.8 ten-scene row is context only
and is not a strict comparison because it includes `fork`.

Pinecone required an explicit official-COLMAP undistortion and a fresh exact
original-3DGS 30k all-view model. The staging and training entry points are:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python \
  reproductions/ludvig/stage_spin_pinecone_official_undistortion.py \
  --attempt-id official_colmap_3p6_v2

/root/miniconda3/envs/cybersim_agent/bin/python \
  reproductions/ludvig/train_spin_pinecone_all_view_3dgs.py \
  --attempt-id exact_f7a_allview_30k_v1
```

The official COLMAP 3.6 output is PINHOLE 4015x3011 for all 99 views; original
3DGS trains at effective 1600x1199. The resulting 2,598,612-Gaussian PLY has
SHA-256 `27d5670cd642542cdba671a7a5718ae463b1097c437d2f4e232999090aef451e`.
The three Pinecone IoUs are 81.401456, 86.281102, and 86.136876 percent, for
84.606478 versus the paper's 88.8. The reference frame is excluded and all 98
targets are scored in every seed.
