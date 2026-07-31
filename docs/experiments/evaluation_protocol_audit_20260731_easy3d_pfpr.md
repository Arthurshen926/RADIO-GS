# Evaluation protocol audit: Easy3D / AGILE3D and PFPR

Date: 2026-07-31

This artifact separates benchmark-protocol effects from method changes. It
does not modify RADIO-GS training, field construction, or query engines.

## Easy3D on AGILE3D ScanNet40

### Fixed assets

- Easy3D repository: <https://github.com/facebookresearch/easy3d>
- audited Easy3D commit:
  `b3f5bd70defaa9a601edb0975802775b056c784a`
- official v1.0 checkpoint SHA256:
  `4a13d16ba2f2470031287812dbbdf1ec6aa14097cb3738e0fe596bb708dc475f`
- AGILE3D repository: <https://github.com/ywyue/AGILE3D>
- audited AGILE3D commit:
  `b73638da41edbabe52a1b578d52ddeb8fa552173`
- released ScanNet selection: 312 scenes, 10,357 objects, 40 classes
- `object_ids.npy` SHA256:
  `d734230755fdde72ee04f8ca199b15c19f330233588eba1e24021fe36459a037`
- `object_classes.txt` SHA256:
  `f2bf7241e0cbad22056c9fe9b029818ac3b7d1d9846a96399eee68bac2fef537`
- voxel size: 5 cm
- checkpoint only; no retraining

The Easy3D paper reports IoU@1/2/3/5/10 of
`68.2 / 74.6 / 77.3 / 79.6 / 81.7`.

The public Easy3D repository has a trainer and interactive visualization but
no complete quantitative evaluator over the released 10,357-object list.
`evaluate_easy3d.py` therefore loads the untouched public checkpoint and
model, encodes each scene once, and runs a resumable object-batched adapter.
The primary adapter uses the released AGILE3D single-object simulator:

1. first positive click at the center of the false-negative region;
2. subsequent click at the larger false-positive/false-negative error center;
3. false positive wins an exact radius tie;
4. clicked voxel is overwritten before the point-IoU measurement and the next
   interaction;
5. point IoU is computed after mapping voxel predictions to all input points.

Easy3D preprocessing is intentionally preserved for checkpoint compatibility.
Coordinates are shifted, truncated onto the 5 cm grid, and lexicographically
sorted by `torch.unique`. The public code then uses duplicate-index writes such
as `voxel_colors[point_voxel_id] = point_colors`. A multithreaded main-process
reimplementation is not a valid substitute: its duplicate write winner varied
and failed an exact feature smoke test.

The released trainer uses a four-worker `DataLoader`. Each worker has
`torch.get_num_threads() == 1`; two rounds over eight worker samples produced
16/16 identical coordinate, feature, inverse-map, valid-mask, and label
hashes. In this real worker path, the result is exactly the deterministic last
input row for each duplicate voxel. The formal evaluator therefore consumes
immutable arrays emitted by the untouched public `VoxelDataset` inside the
real four-worker path:

- cache schema: `official-easy3d-worker-preprocessing-v1`;
- formal manifest: 312 scenes, SHA256
  `39035ec87a3ff73bd9cfd6eec9a93182b7ebd7d9b2e84515b1c0e51cad453d23`;
- cache size: 936 MiB;
- full cache audit: 49,540,568 points, 8,146,089 voxels, and all
  10,357/10,357 objects present before and after voxelization.

This exposes a paper-prose/code mismatch: Sec. 3.2.1 says that RGB features
inside one voxel are averaged, while the released execution path is stable
last-write. Across all 312 scenes, 90.82% of voxels contain duplicate points,
and 85.08% of voxels differ from the prose-style RGB mean by more than one
8-bit step. Every result binds both the cache manifest and the per-scene NPZ
and array-content hashes.

The paper says that it uses AGILE3D's official repository, while the released
Easy3D forward path differs from `eval_single_obj.py` in several details.
Therefore a paired GPU pilot evaluated the exact same 113 objects from three
scenes spanning the 10th, 50th, and 90th voxel-count percentiles under two
named contracts:

- `easy3d_released_code`: Easy3D's `voxel_valid` false-positive policy, no
  clicked-point overwrite, invalid interaction tokens after convergence, and
  its `1e-6` point-IoU denominator epsilon;
- `agile3d_release`: AGILE3D's all-voxel error policy and clicked-point
  overwrite, applied on the Easy3D preprocessing needed by the checkpoint.

Both use Easy3D's integer voxel coordinates for error-center selection. The
second row is consequently an explicit interaction/metric adapter, not a
claim that the checkpoint was fed through MinkowskiEngine's different
first-row quantization.

### Paired protocol pilot and decision

The paired scenes were:

| Scene | Voxel percentile | Voxels | Objects | Classes |
| --- | ---: | ---: | ---: | ---: |
| `scene0494_00` | 10th | 10,002 | 13 | 8 |
| `scene0700_02` | 50th | 24,080 | 37 | 14 |
| `scene0435_01` | 90th | 44,217 | 63 | 23 |

Both contracts used the official checkpoint, the same hashed preprocessing
arrays, batch size 4, BF16, and 10 clicks. All 113 objects evaluated with zero
failures:

| Contract | IoU@1 | IoU@2 | IoU@3 | IoU@5 | IoU@10 | Mean absolute paper gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paper | 0.6820 | 0.7460 | 0.7730 | 0.7960 | 0.8170 | — |
| `agile3d_release` | 0.6800 | 0.7368 | 0.7657 | 0.7889 | 0.8102 | 0.650 pp |
| `easy3d_released_code` | 0.6799 | 0.7338 | 0.7600 | 0.7800 | 0.7970 | 1.266 pp |

The `agile3d_release` paper gaps are
`-0.20 / -0.92 / -0.73 / -0.71 / -0.68` percentage points. The released-code
contract falls increasingly behind after the first click, reaching -2.00
points at click 10. The frozen decision criterion is the smaller mean absolute
gap across the five paper IoUs, so `agile3d_release` is the sole primary
contract for the formal run. Scene-evaluation time summed to 23.03 seconds for
that contract and 21.11 seconds for the released-code diagnostic.

The machine-wide GPU trace from the subsequent formal attempt observed a
maximum total GPU0 memory usage of 4,756 MiB at batch size 4. This is a total
device reading, not PyTorch allocated memory.

Decision artifact:
`output/protocol_audit_20260731/easy3d_agile3d_pilot3_protocol_decision.json`.

The paper compares at most 10 clicks, so the formal Easy3D comparison uses
IoU@1/2/3/5/10. NoC@50/65/80/85/90 is additionally emitted as a diagnostic
with an explicit cap of 10; it must not be mixed with AGILE3D's published
20-click NoC.

### Official AGILE3D result-ID mismatch

AGILE3D's bundled `our_single_scannet.csv` has 10,357 recorded object
trajectories, and the released object list also has 10,357 rows. Their keys
are not identical:

- exact key intersection: 10,016 objects;
- released keys silently unmatched by the official evaluator: 341;
- CSV keys not present in the released list: 341.

The mismatch is usually an object-ID offset, for example the CSV contains
`scene0011_00_obj_0` while the release starts at object 1. Running the
official evaluator behavior on only the implicit intersection reproduces its
published AGILE3D row:

| Cohort | Objects | IoU@1 | IoU@2 | IoU@3 | IoU@5 | IoU@10 | IoU@15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| official legacy key intersection | 10,016 | 0.630600 | 0.706378 | 0.751389 | 0.797688 | 0.835440 | 0.848500 |
| all bundled CSV trajectories | 10,357 | 0.632707 | 0.708913 | 0.753608 | 0.799406 | 0.837011 | 0.849965 |

The difference is about 0.2--0.3 IoU percentage points. It is a reference
cohort bug, not a model or interaction change. Every new Easy3D trajectory is
therefore aggregated twice without rerunning inference:

- `complete_release_selection`: all 10,357 released objects, the primary row;
- `agile3d_legacy_paper_script_intersection`: the same predictions restricted
  to the 10,016 reference-compatible keys, shown only to diagnose paper-row
  differences.

A separate CPU presence audit opened all 312 official worker-cache scenes. All
10,357 released objects exist in the raw point labels and all 10,357 still
exist after the released 5 cm worker voxelization. This includes 10,016/10,016
legacy-match objects and 341/341 release-only objects. Thus the key mismatch
cannot be explained by absent instances or voxel collision; no object needs
to be silently skipped. The evaluator nevertheless emits an explicit failure
row if a future asset does lose an object.

The worker cache is constructed and audited on CPU before inference:

```bash
/root/miniconda3/envs/easy3d/bin/python \
  -m radio_gs.benchmarks.agile3d_scannet40.cache_easy3d_preprocessing \
  --data-root /mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet \
  --easy3d-repo /mnt/pool/sqy/3d_understanding/segmentation_benchmarks/reproductions/easy3d \
  --output-root output/protocol_audit_20260731/easy3d_official_worker_preprocessing \
  --num-workers 4

/root/miniconda3/envs/easy3d/bin/python \
  -m radio_gs.benchmarks.agile3d_scannet40.audit_easy3d_preprocessing_cache \
  --data-root /mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet \
  --easy3d-repo /mnt/pool/sqy/3d_understanding/segmentation_benchmarks/reproductions/easy3d \
  --cache-root output/protocol_audit_20260731/easy3d_official_worker_preprocessing \
  --output output/protocol_audit_20260731/easy3d_agile3d_scannet40/official_worker_cache_audit.json
```

Formal command (GPU 0 is additionally serialized by the project lock). The
`LD_PRELOAD` is a machine-specific repair for this host's incorrect
`libcuda.so.1` symlink from version 580 to a version-535 kernel driver; it is
not part of the benchmark protocol:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.535.288.01 \
flock /tmp/radio-gs-gpu0.lock -c '
/root/miniconda3/envs/easy3d/bin/python \
  -m radio_gs.benchmarks.agile3d_scannet40.evaluate_easy3d \
  --data-root /mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet \
  --easy3d-repo /mnt/pool/sqy/3d_understanding/segmentation_benchmarks/reproductions/easy3d \
  --checkpoint /mnt/pool/sqy/3d_understanding/segmentation_benchmarks/reproductions/easy3d/weights/pretrained_easy3d.pth \
  --preprocessing-cache-root /root/RADIO-GS/output/protocol_audit_20260731/easy3d_official_worker_preprocessing \
  --agile-reference-csv /mnt/pool/sqy/3d_understanding/segmentation_benchmarks/reproductions/AGILE3D/results/our_single_scannet.csv \
  --output-dir /root/RADIO-GS/output/protocol_audit_20260731/easy3d_agile3d_formal_agile3d_release_v1 \
  --interaction-contract agile3d_release \
  --object-batch-size 4 --max-clicks 10 --device cuda'
```

Scene JSON shards are provenance-checked and resumed rather than overwritten.

### Formal run state

Formal attempt 001 completed 68/312 scene shards and 2,676 objects with zero
object failures. GPU0 was then lost while `scene0207_02` was in progress. The
last good trace sample was `2026/07/31 12:49:00.349`; the first
`[GPU is lost]` sample was `12:49:00.600`. No shard was written for the
in-progress scene, and no formal aggregate is reported from this incomplete
prefix.

For debugging only, the unchanged query-micro aggregator was applied to the
68 completed shards:

| Hardware-interrupted prefix | Objects | IoU@1 | IoU@2 | IoU@3 | IoU@5 | IoU@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| complete release keys | 2,676 | 0.70164 | 0.75922 | 0.78339 | 0.80626 | 0.82681 |
| legacy-key intersection | 2,608 | 0.70159 | 0.75901 | 0.78336 | 0.80644 | 0.82686 |

This is a lexicographic scene prefix covering 21.8% of scenes and 25.8% of
objects, not a representative sample. Its positive paper gaps must not be
reported as reproduction accuracy or used to replace the pending 312-scene
aggregate. The diagnostic is explicitly marked `not_formal_result: true` and
`paper_metric_comparable: false` in
`output/protocol_audit_20260731/easy3d_agile3d_formal_agile3d_release_v1/partial68_hardware_interrupted_diagnostic.json`.

The evaluator, monitor, and lock shell were terminated by exact PID, the lock
was verified released, and the 68 complete shards were preserved. Attempt 002
is prepared with a new immutable trace name and the identical evaluator
arguments, but must not start until GPU0 recovers. Exact failure provenance,
the completed-shard set hash, and the resume argv are in
`output/protocol_audit_20260731/easy3d_agile3d_formal_agile3d_release_v1/formal_attempt_001_failure.json`.

## PFPR: LUDVIG-style DINO uplift sanity

This result is deliberately **not** named “LUDVIG reproduction.”

Official LUDVIG commit
`4461fc515439bb498a75d71738a1e73cf7a452ed` supports segmentation, not PFPR
patch-to-3D point retrieval. Its reference path uses DINOv2 ViT-g with
registers, multi-view inverse-rendering uplift into Gaussians, and, for NVOS,
a 200-neighbor / 100-iteration graph diffusion driven by scribbles.

The audited historical PFPR cache instead uses:

- method-visible input: only `scene_id` and the depth-aligned v2 RGB crop;
- C-RADIO's official DINOv3-7B spatial adaptor, center 3x3 L2 mean;
- cosine similarity to the RADIO-GS canonical DINO field;
- continuous opacity-weighted Gaussian / 5 cm cell readout;
- deterministic 10 cm point NMS;
- no pose, depth, mask, class, instance, or private 3-D anchor in scoring;
- private anchor opened by the evaluator only for distance/rank metrics.

It does not use the official LUDVIG wrapper, DINOv2 model, inverse-rendering
implementation, graph diffusion, or scribble regularizer. The historical
cache also did not record its generator commit. The audit consequently binds
the existing score vectors by their combined SHA256, records the scorer file
SHA and repository commit *at audit time*, and explicitly declines to infer a
historical generator commit.

The available corrected-v2 cache is a six-scene, 60-query diagnostic with its
formal continuous-support gate disabled. Re-evaluation from the immutable
score vectors exactly reproduces the saved micro and macro summaries:

| Metric | Query-micro |
| --- | ---: |
| top-1 mean / median error | 1.2669 m / 0.6899 m |
| R@1 / R@5 / R@10 at 5 cm | 0.0833 / 0.1333 / 0.1833 |
| R@1 / R@5 / R@10 at 10 cm | 0.1667 / 0.3500 / 0.4167 |
| R@1 / R@5 / R@10 at 20 cm | 0.2333 / 0.4500 / 0.4667 |
| MRR at 5 / 10 / 20 cm | 0.1112 / 0.2463 / 0.3290 |

The monotonic recall values and the lower median than mean error are
internally coherent: some crops retrieve the local neighborhood while a long
tail of repeated indoor appearances fails by several metres. This is enough
for a sanity check on the self-built PFPR benchmark, but it is neither a
formal 20-scene PFPR result nor comparable to a LUDVIG paper metric.

Audited outputs:

- `output/protocol_audit_20260731/pfpr_ludvig_style_v2_partial6/audit.json`
- `output/protocol_audit_20260731/pfpr_ludvig_style_v2_partial6/evaluation.json`

Re-evaluation entry point:

```bash
python -m radio_gs.benchmarks.scannet_pfpr.audit_ludvig_style_uplift \
  --benchmark-dir /mnt/pool/sqy/results/RADIO-GS/output/scannet_pfpr_small_v2/test_v2_r1 \
  --prediction-dir /mnt/pool/sqy/results/RADIO-GS/output/scannet_pfpr_small_v2/full_sens_official_v3_960_r1/eval_fast_no_support_gate_partial6_20260728/predictions \
  --prediction-report /mnt/pool/sqy/results/RADIO-GS/output/scannet_pfpr_small_v2/full_sens_official_v3_960_r1/eval_fast_no_support_gate_partial6_20260728/predictions/prediction_report.json \
  --source-results /mnt/pool/sqy/results/RADIO-GS/output/scannet_pfpr_small_v2/full_sens_official_v3_960_r1/eval_fast_no_support_gate_partial6_20260728/results.json \
  --output-dir output/protocol_audit_20260731/pfpr_ludvig_style_v2_partial6 \
  --repository-root /root/RADIO-GS
```

## Verification

The dedicated CPU tests cover Easy3D's sorted/last-write equivalence,
fail-closed shard resume, paired-pilot protocol selection, KD-tree error-center
equivalence, FP/FN tie behavior, explicit NoC cap, AGILE3D key-intersection
reporting, PFPR cache identity, and rejection of private-anchor leakage:

```text
tests/test_easy3d_pfpr_protocol_audit.py: 9 passed
```
