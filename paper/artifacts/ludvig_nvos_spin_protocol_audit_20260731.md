# LUDVIG NVOS / SPIn-NeRF protocol audit (updated 2026-08-01)

## Scope and pinned implementation

- Upstream: `naver/ludvig`
- Audited commit: `4461fc515439bb498a75d71738a1e73cf7a452ed`
- Method variant reproduced: LUDVIG-SAM (ViT-H SAM)
- SAM checkpoint SHA256:
  `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`
- Paper protocol: mean of three independent runs
- Local patch: exposes the seed, writes exact per-run JSON results, and makes
  image/camera/SAM/feature shapes fail closed. The final v3 also computes the
  1600-pixel long edge exactly instead of allowing floating-point floor to
  produce 1599. It does not change the released threshold, negative-scribble
  behavior, metric, seed aggregation, or uplifting algorithm.

The wrapper rejects any SAM checkpoint with a different hash before GPU work,
and the aggregator rechecks that hash together with the pinned LUDVIG commit,
target-view visibility, foundation-model calls, benchmark-specific
calibration, cohort, and aggregation. It also validates the evaluator result
schema: NVOS must use fixed threshold parameter 75, while SPIn target-frame
IoUs are recomputed into the recorded scene mean.

The 36 earlier exact run manifests embed reproduction patch/diff SHA256
`7c86d5883058fb9e529608ba6cf856c04e66d37d338a69a8ca6863725292e9ac`.
The three final Trex manifests embed resolution-safe v3 SHA256
`2c21257316c6f65d25eea2bbd98481bd3e42f0d84df23a13c1bd1cb645e7d602`
and its patched-file hashes. Those original 39 evaluations plus the later
Room, Truck, Lego, and Pinecone checks give 51 current exact runs; all have
approved, verified upstream-patch provenance. The earlier hybrid fern manifest predates
the field; its post-hoc source audit matches v1, but it remains a separate,
paper-ineligible diagnostic.

The final local source SHA256 values recorded by this audit are:

- launcher `reproductions/ludvig/run_ludvig_sam.py`:
  `1f1bc95d70f22fcfb1cfe2df1bc3f2416ebb667c2faeddb7dc24bcd9bceed68f`;
- aggregator `reproductions/ludvig/aggregate_results.py`:
  `20fdda242368540aba731afe4d477635030a25e90d0718e6b41231d64f812842`;
- NVOS/all-view trainer `reproductions/ludvig/train_nvos_all_view_3dgs.py`:
  `7f1d73e2f0bf38c8e09466f78b7954a36c1c97e1a72569828d8226953fe9a632`;
- fixed SPIn-NeRF LLFF room trainer
  `reproductions/ludvig/train_spin_llff_room_all_view_3dgs.py`:
  `561980d9bea84e171a86a978c83529dea99a294adddb93aa4194108d16ebc0e6`;
- fixed SPIn-NeRF Truck trainer
  `reproductions/ludvig/train_spin_truck_all_view_3dgs.py`:
  `83f0ab28c169a4ac7aa39ca3c33b889c5b2d91ca632172b717c0ca93725c5e78`;
- fixed SPIn-NeRF Pinecone trainer
  `reproductions/ludvig/train_spin_pinecone_all_view_3dgs.py`:
  `6f54b377edec7d2febccf9043b99b150376887c10d7e38df2f40cff66e6edc2d`;
- Pinecone undistortion contract
  `reproductions/ludvig/stage_spin_pinecone_official_undistortion.py`:
  `195728f4707c84308ece90f29d9dbd2ea6b082c5f9494d2cccc903f355a679d7`;
- GPU0 thermal guard `reproductions/ludvig/run_gpu0_thermal_guard.py`:
  `4daa492d2026d58f5d83184dc12dbf4f87747bf1d8630c0e70bf563c5acfed79`.
  Its current defaults are a 3 s polling interval, 2 s query timeout,
  78/81/70 C warning/pause/resume thresholds, and two stable cool samples
  before resume. This guard is execution infrastructure only and is not part
  of the evaluation protocol.

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

The CPU-only immutable fern all-view dry run passed before launch:

- 20 registered training views and no holdout;
- staged PINHOLE 3985x2988 RGBs, automatically resized to 1600x1199;
- converted/raw pose deltas q=`1.1102230246251565e-16`, t=`0.0`;
- no algorithm or environment-compatibility source patch.

Manifest:
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/fern/training/attempts/exact_f7a_allview_dryrun_v2/training_manifest.json`
(SHA-256
`c81670c3310147a7ce204d7ea424a2fe8e855f994b14b0a73b16d30e6a3f6ef2`).

### Completed exact NVOS eight-task checkpoints and evaluation boundary

The full 20/20-view fern original-3DGS training completed at 30k iterations.
Its immutable training manifest is
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/fern/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
(SHA-256
`8b7ebe7ff6d946c4ae03b1943835a4d73ee94b0e53d663a5933de03ea947909e`).
The validated `iteration_30000/point_cloud.ply` contains 1,121,272 vertices,
is 278,076,988 bytes, and has SHA-256
`4eaac89e70e00e4776e1bd9505b7560c483edd5df8ee1958c84b91091e49134b`.
The manifest predates the generalized multi-scene trainer's explicit
`geometry_scene` field, so the consumer's legacy fallback is restricted to
this existing `benchmark=NVOS, scene=fern` artifact.

The full 34/34-view flower training completed at 30k iterations with no
holdout. Its immutable manifest is
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/flower/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
(SHA-256
`945e3fe89dbf451a74dfea4d8edfb406fef4fc1045c9719b73debce0f456b4e6`).
It records `benchmark=NVOS`, `scene=geometry_scene=flower`, PINHOLE 3982x2986
input, automatic 1600x1199 training/evaluation resolution, zero held-out
views, and no source patch. Training completed with return code 0 in
1586.195 GPU-wall seconds. Its 646,853-vertex, 160,421,075-byte PLY has
SHA-256
`eb9658266c5dc10d639fbe9371858edcb537f0ad2e4a8f0d8e817b55e2c567ec`.

The full 26/26-view leaves training also completed at 30k iterations. Its
manifest is
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/leaves/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
(SHA-256
`2a610c0c356ae05f9a7d712c4352e032b137fa0794df3d701ec550759b67ba69`).
It explicitly records `benchmark=NVOS`, `scene=geometry_scene=leaves`, PINHOLE
3954x2965 input, automatic 1600x1199 training resolution, zero held-out views,
no legacy dataset fallback, and no source patch. Its 1,697,651-vertex,
421,018,980-byte PLY has SHA-256
`da74e7c81debf212d0b3c6f7278a62259dfcc045f97401cb9370b1e62d340378`.

The full 25/25-view orchids training completed at 30k iterations. Its manifest
is
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/orchids/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
(SHA-256
`8fc085e1c6ad817772898fd5e24f120b72ec78d3c0efa793a7ccbd761935d3d7`).
It explicitly records `benchmark=NVOS`, `scene=geometry_scene=orchids`, PINHOLE
3949x2961 input, automatic 1600x1199 training resolution, zero held-out views,
no legacy dataset fallback, and no source patch. Its 1,509,479-vertex,
374,352,324-byte PLY has SHA-256
`e04a38f5aa22aaa716ec1c741222e1d1c94fb3cf0305aa4cc06a50a823ae5ef5`.

The full 42/42-view fortress training completed at 30k iterations. Its
manifest is
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/fortress/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
(SHA-256
`e01d76a52f0e0df0176e48301f9753ec610b475114f0449ab4cb634c6938aab3`).
It explicitly records `benchmark=NVOS`, `scene=geometry_scene=fortress`,
PINHOLE 4014x3010 input, automatic 1600x1199 training resolution, zero
held-out views, no legacy dataset fallback, and no source patch. Its
904,176-vertex, 224,237,179-byte PLY has SHA-256
`723bfaa66c2e2da9c5e76662dc317d86fafe0dfed5c109b5e9fd6c11e8626c45`.

The full 62/62-view horns training completed at 30k iterations with no
holdout. Its manifest is
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/horns/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
(SHA-256
`77154d2f5e761ba34a605677be8143214246ff8fbbf310b3400b55e4332488bc`).
It records `benchmark=NVOS`, `scene=geometry_scene=horns`, PINHOLE 4003x3002
input, automatic 1600x1199 training/evaluation resolution, zero held-out
views, and no source patch. Its 962,085-vertex, 238,598,611-byte PLY has
SHA-256
`1f3084daba5e9a70263152f73f704815f452a2f37e81f56cf3a3eb96c3e49803`.
The same checkpoint and manifest are bound fail-closed to both NVOS horns
tasks and to SPIn-NeRF horns.

The full 55/55-view Trex training completed at 30k iterations with no
holdout. Its immutable manifest is
`output/protocol_audit_20260731/ludvig/nvos/released_all_view/trex/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
(SHA-256
`e8341f2ed5f77a48189790b3d61bdab0f1bea7efb45e41134403996bd4e10035`).
It records native NVOS PINHOLE undistortion input at 4002x3001, 1599x1199
effective original-3DGS training images, frozen 1600x1199 evaluation renders,
zero held-out views, and no raw-dataset mutation. Training completed in
2351.850 GPU-wall seconds. Its 703,207-vertex, 174,396,867-byte PLY has
SHA-256
`444c364891ff637a6f82b85fc1236f04a89f3ce7bd8b18bcf4806f88b5e0a7b0`.
The staged COLMAP cameras/images/points3D hashes are respectively
`1b711b32b1ca110eaa979f5f5b9ff02f248ccc066c7f9bf89c34f17493c425fe`,
`e83956f2665b6041bb295f637802630aea9f0cd8d08a3b4f390e7a8f54af9075`,
and `dcbae55ccf2a7e1eb4ac02673fc090c1aa8d4a241ed6194eb1401e63ddd9c32a`;
pose deltas remain q=`1.1102230246251565e-16`, t=`0.0`.

The SPIn-NeRF Room all-view training also completed at 30k iterations over
41/41 registered views with no holdout. Its immutable manifest is
`output/protocol_audit_20260731/ludvig/spin/released_all_view/room/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
(SHA-256
`275c72ced3689cbca6b51cca8cc7830130fa6a1ac2d4eed3d471a346a7345dd4`).
Its 291,619-vertex, 72,323,043-byte PLY has SHA-256
`c12a46a46ab3550905cfe48152916bd75834f5f36485ec4cf1cae03eaef294fd`.
Room is SPIn-NeRF-only in this audit and does not reuse an NVOS result row.

The SPIn-NeRF Truck all-view training completed at 30k iterations over all
251 registered views with no holdout. Its immutable manifest is
`output/protocol_audit_20260731/ludvig/spin/released_all_view/truck/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
(SHA-256
`858ac2c6b6438ef609ac7bc82c701067a0117dc470159179326ed26929de2cd6`).
Its 2,583,730-vertex, 640,766,572-byte PLY has SHA-256
`aa5b6d166277cde9d49632c9ffa427c392a4e52b7c68719c3d6ca312bedc3492`.
Training consumed 1396.462 GPU-wall seconds. A CPU-only postflight recovery
corrected the wrapper's validation of Graphdeco's 1957x1091 camera metadata
against the released 979x546 RGBs; it performed no GPU work and modified no
model file. The recovered manifest records status `complete`, effective
979x546 training/evaluation resolution, and the same PLY identity.

The completed exact geometry checkpoints were evaluated with seeds 0, 1, and 2.
Horns contributes two NVOS task checks and one SPIn-NeRF scene check:

| Per-scene exact check | Seed values (%) | Local three-seed mean (%) | Paper scene (%) | Local - paper (points) |
|---|---|---:|---:|---:|
| NVOS fern, fixed threshold parameter 75 | 84.479523, 84.555207, 84.487121 | 84.507284 | 85.5 | -0.992716 |
| NVOS flower, fixed threshold parameter 75 | 97.548945, 97.580827, 97.591021 | 97.573598 | 97.6 | -0.026402 |
| NVOS fortress, fixed threshold parameter 75 | 98.484158, 98.486720, 98.487367 | 98.486082 | 98.1 | +0.386082 |
| NVOS horns_center, fixed threshold parameter 75 | 97.969493, 97.965612, 97.972776 | 97.969294 | 97.9 | +0.069294 |
| NVOS horns_left, fixed threshold parameter 75 | 92.646144, 92.652081, 92.650710 | 92.649645 | 94.1 | -1.450355 |
| NVOS leaves, fixed threshold parameter 75 | 96.650159, 96.644425, 96.653476 | 96.649353 | 96.4 | +0.249353 |
| NVOS orchids, fixed threshold parameter 75 | 74.560060, 74.574407, 74.554683 | 74.563050 | 73.1 | +1.463050 |
| NVOS trex, fixed threshold parameter 75 | 87.529623, 87.671122, 87.788780 | 87.663175 | 88.0 | -0.336825 |
| SPIn-NeRF fern, reference-calibrated target-frame mean | 97.036403, 97.045689, 97.051700 | 97.044597 | 97.0 | +0.044597 |
| SPIn-NeRF fortress, reference-calibrated target-frame mean | 98.357992, 98.356858, 98.353425 | 98.356092 | 98.3 | +0.056092 |
| SPIn-NeRF horns, reference-calibrated target-frame mean | 90.007836, 89.995405, 89.973144 | 89.992128 | 94.5 | -4.507872 |
| SPIn-NeRF leaves, reference-calibrated target-frame mean | 96.371017, 96.368338, 96.366386 | 96.368580 | 96.3 | +0.068580 |
| SPIn-NeRF orchids, reference-calibrated target-frame mean | 90.044849, 90.217455, 89.933898 | 90.065401 | 92.2 | -2.134599 |
| SPIn-NeRF pinecone, reference-calibrated target-frame mean | 81.401456, 86.281102, 86.136876 | 84.606478 | 88.8 | -4.193522 |
| SPIn-NeRF room, reference-calibrated target-frame mean | 97.126922, 97.091829, 97.343748 | 97.187499 | 96.5 | +0.687499 |
| SPIn-NeRF truck, reference-calibrated target-frame mean | 96.749968, 96.571124, 96.811787 | 96.710960 | 94.9 | +1.810960 |
| SPIn-NeRF lego, reference-calibrated target-frame mean | 93.296140, 92.968601, 93.181267 | 93.148669 | 92.7 | +0.448669 |

Within each shared geometry scene, NVOS and SPIn-NeRF bind the same PLY and
training-manifest hashes shown above. All 51 runs (24 NVOS and 27 SPIn-NeRF)
have verified upstream-patch provenance. The current machine summaries are:

- NVOS:
  `output/protocol_audit_20260731/ludvig/nvos/released_all_view_full8_3seed_summary.json`
  (SHA-256
  `65e1f8e5c1f17083f66e5b7d4f6f03687f6806394c78a5cfce25546ca42e3546`);
- SPIn-NeRF:
  `output/protocol_audit_20260731/ludvig/spin/released_all_view_fern_leaves_orchids_fortress_horns_room_pinecone_truck_lego_3seed_summary.json`
  (SHA-256
  `ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17`).

The final local nine-scene SPIn artifact identities are frozen together. The
previous eight-scene aggregate remains only as a historical intermediate:

| Scene | Training manifest SHA-256 | PLY SHA-256 |
|---|---|---|
| fern | `8b7ebe7ff6d946c4ae03b1943835a4d73ee94b0e53d663a5933de03ea947909e` | `4eaac89e70e00e4776e1bd9505b7560c483edd5df8ee1958c84b91091e49134b` |
| fortress | `e01d76a52f0e0df0176e48301f9753ec610b475114f0449ab4cb634c6938aab3` | `723bfaa66c2e2da9c5e76662dc317d86fafe0dfed5c109b5e9fd6c11e8626c45` |
| horns | `77154d2f5e761ba34a605677be8143214246ff8fbbf310b3400b55e4332488bc` | `1f3084daba5e9a70263152f73f704815f452a2f37e81f56cf3a3eb96c3e49803` |
| leaves | `2a610c0c356ae05f9a7d712c4352e032b137fa0794df3d701ec550759b67ba69` | `da74e7c81debf212d0b3c6f7278a62259dfcc045f97401cb9370b1e62d340378` |
| orchids | `8fc085e1c6ad817772898fd5e24f120b72ec78d3c0efa793a7ccbd761935d3d7` | `e04a38f5aa22aaa716ec1c741222e1d1c94fb3cf0305aa4cc06a50a823ae5ef5` |
| pinecone | `2d1b5d8b14765f03dd0b84a3ed64f41d7dffcf2d776835a2d6270cf328eb816b` | `27d5670cd642542cdba671a7a5718ae463b1097c437d2f4e232999090aef451e` |
| room | `275c72ced3689cbca6b51cca8cc7830130fa6a1ac2d4eed3d471a346a7345dd4` | `c12a46a46ab3550905cfe48152916bd75834f5f36485ec4cf1cae03eaef294fd` |
| truck | `858ac2c6b6438ef609ac7bc82c701067a0117dc470159179326ed26929de2cd6` | `aa5b6d166277cde9d49632c9ffa427c392a4e52b7c68719c3d6ca312bedc3492` |
| lego | `145f2e8e3b10c36f085ffb6c389ec6e91f744310f88345e7da5add058593b5a4` | `544437bb194995502d3a8cfe8a74935944adb6b90b22de2a3d243656d9cc2183` |

The exact NVOS result is complete: seed means are taken within all eight
tasks, followed by an equal-weight task macro of **91.257685%**. This is
**-0.079815 points** versus the 91.3375% macro independently recomputed from
the paper's eight displayed per-task values, and **-0.042315 points** versus
the paper's rounded 91.3% headline. It is eligible for a strict comparison to
the paper's released online-multiview protocol. It is not eligible for a
strict-unseen claim. The local nine-scene SPIn-NeRF macro is
**93.720045%** versus **94.577778%** on the matching paper scenes, a
**-0.857733 point** delta. Every included scene is eligible for a per-scene
three-seed check, and the matching nine-scene aggregate is eligible as a
released online-multiview partial-cohort diagnostic: all registered RGBs train
the original-3DGS field, and query-specific uplifting sees target RGBs exactly
as in the released protocol. It is not a strict-unseen result. Fork is absent
locally, so this diagnostic is not the paper's full ten-scene table.

Trex seed 0 first failed under the earlier v1 execution path. Floating-point
floor resized the 4002-pixel source long edge to 1599 while the camera,
renderer, and SAM tensors used 1600, corrupting prompt-coordinate decoding
and downstream weight indexing. Its post-hoc fixed-threshold IoU (16.6911%)
and oracle IoU (28.3370%) are failure diagnostics only and are excluded from
every aggregate. Resolution-safe v3 fixes the exact long-edge resize and
adds fail-closed shape assertions. It leaves threshold 75, negative-scribble
handling, metric, and aggregation unchanged; only the three successful v3
Trex attempts enter the full result.

Room seed 0 v1 failed before scoring and is excluded from every metric. Its
prompt and camera tensors were 1200x1600 while the staged image tensor was
1199x1600. The accepted v2 attempts apply an evaluation-only fractional
center crop from 4005x3003 to 4004x3003 with box
`[0.5, 0, 4004.5, 3003]`, then bicubic resize to exact 1600x1200 while
preserving source JPEG tables/sampling. The source dataset is not modified.
This is a shape-consistency repair: reference calibration, SAM-candidate
selection, threshold search, target-only scoring, and aggregation are
unchanged.

The NVOS horns_left target-oracle IoUs are 94.565885%, 94.583144%, and
94.587220%, all at threshold parameter 42. They are recorded only as a
diagnostic: the exact result and full eight-task macro use the released fixed
threshold parameter 75 values above. Oracle values never select a reported
metric or protocol.

All nine Horns evaluation jobs used the audited thermal guard. Maximum
temperatures by seed were 65/67/67 C for horns_center, 67/68/67 C for
horns_left, and 67/68/68 C for SPIn horns. Across the nine logs there were
zero pause events and zero telemetry-query failures. The logs are under
`output/protocol_audit_20260731/ludvig/thermal_logs/horns_exact_f7a_allview_v1/`.
This is runtime evidence only; it does not alter evaluation eligibility.

Flower training and all three Flower evaluation jobs used the same guard.
Training reached 82 C and emitted 43 pause/resume cycles, with zero telemetry
query failures and return code 0. The three evaluation logs reached 69/70/69
C, respectively, with zero pause events, zero telemetry-query failures, and
return code 0. The four logs are under
`output/protocol_audit_20260731/ludvig/thermal_logs/flower_exact_f7a_allview_v1/`.
These observations are runtime evidence only.

Trex training used the guard's 78/81/68 C warning/pause/resume policy with
three stable cool samples. It reached 82 C and completed after 37 pause/resume
cycles with zero telemetry failures. The three accepted v3 evaluation logs
reached 72/68/69 C, with zero pause events, zero telemetry failures, and return
code 0. The training log SHA-256 is
`b7be623b0e0e6bd7b155c0aac44a7af3c0fe1902fbbad580febbb406067a3c1b`;
the evaluation log hashes are
`73e72dc78568fe780178890c9be39ec987d6885f7b5913789a5782e691f7a79f`,
`79ac51b06a2afb85595f1d60e52bbde6a30687fa5bbe5dca8771e3aab0bd939c`,
and `57fb3ef7c0fdf84402756e800b9dead346e54dd5dca0a1da2debb18d0cdf48e9`.
These are runtime records, not evaluation-protocol inputs.

Lego training used the throughput-adjusted 78/81/70 C policy with one stable
cool sample. It completed 30k iterations in 1179.04 GPU-wall seconds, reached
82 C, and had 48 three-second pause/resume cycles with zero telemetry failures
or Xid events. The three accepted evaluations reached 70/71/72 C with zero
pauses and produced 93.296140/92.968601/93.181267% target-only IoU. Training
manifest/PLY SHA-256 values are
`145f2e8e3b10c36f085ffb6c389ec6e91f744310f88345e7da5add058593b5a4` /
`544437bb194995502d3a8cfe8a74935944adb6b90b22de2a3d243656d9cc2183`.

Pinecone first stages the frozen 99-view `SIMPLE_RADIAL` source through the
audited COLMAP 3.6 CPU undistortion contract. The resulting 4015x3011
`PINHOLE` cohort downscales exactly to the released 1600x1199 evaluation
shape; source RGB, sparse-model, pose, and 99-mask stem bijections are checked
fail closed. The undistortion manifest SHA-256 is
`a542a5003028ce3fc3c8639bd3609f39de6fc761d4816d41d0a93800d04ce868`.
Its 99-view original-3DGS training completed in 2425.4 GPU-wall seconds,
reached 82 C, and emitted 121 two-second pause/resume cycles with zero
telemetry errors. Training manifest/PLY SHA-256 values are
`2d1b5d8b14765f03dd0b84a3ed64f41d7dffcf2d776835a2d6270cf328eb816b` /
`27d5670cd642542cdba671a7a5718ae463b1097c437d2f4e232999090aef451e`.
The three accepted evaluations reached at most 68 C with zero pauses and
produced 81.401456/86.281102/86.136876%, mean **84.606478%**, or
**-4.193522 points** versus the paper's 88.8. Seed 0 v1 failed before scoring
because the child lacked `PYTHONPATH`; it is excluded, and the current
launcher dry-run now fails closed on that dependency.

NVOS is now an exact **full-row** 8-task x 3-seed reproduction. SPIn-NeRF
has exact three-seed checks for all nine locally available scenes; Fork data
is absent. No SPIn per-scene or local-cohort mean may be presented as the
paper's 10-scene macro. The released
LUDVIG protocol remains ineligible for a strict-unseen claim because target
RGB is visible during all-view 3DGS training and query-specific uplifting.

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

## Earlier strict-geometry hybrid fern pilot

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

Nine local annotation/RGB scene packages are complete; Fork is absent. The
validated diagnostic manifest is:

`output/protocol_audit_20260731/manifests/spin_nerf_full_reference_mask_9scene_diagnostic_v1.json`

It contains 423 annotated frames across orchids, leaves, fern, room, horns,
fortress, pinecone, truck, and lego. Six project-trained gsplat PLYs exist for
orchids, leaves, fern, fortress, pinecone, and lego, but they were produced by
the project's capped gsplat trainer and cannot be relabelled as LUDVIG's
original-3DGS all-view 30k checkpoints. Fern, fortress, leaves, and orchids
have compatible exact checkpoints and three-seed results, horns reuses its
verified NVOS all-view checkpoint, and Room has its own completed 41-view
checkpoint and resolution-safe three-seed evaluation. Truck now also has a
completed native 251-view original-3DGS checkpoint and three exact seed runs.
Lego now has a completed native 102-view official-undistortion original-3DGS
checkpoint and three exact seed runs. Pinecone now has a completed audited
99-view COLMAP-undistortion/original-3DGS checkpoint and three exact seed
runs. Therefore this audit reports all nine locally available per-scene
checks and a matching nine-scene diagnostic. Its paper context is
94.5777777778 and must not be compared to the
paper's full ten-scene 93.8 row because Fork remains absent.

The earlier Truck CPU dry-run manifest remains an immutable staging preflight:
`output/protocol_audit_20260731/ludvig/spin/released_all_view/truck/training/attempts/exact_f7a_allview_truck_cpu_dryrun_v1/training_manifest.json`
(SHA-256
`e8cff51860a8f82e1b8bcebcdc6710d579ba1468d638c3fd2d088db1f0fffef0`).
It is superseded for result eligibility by the completed training manifest and
PLY recorded above; the dry run itself still carries no metric eligibility.

### Cross-benchmark checkpoint reuse and distinct SPIn evaluator path

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

This single-scene check exercises SPIn's distinct evaluation path: it chooses
one of three SAM candidates and its threshold separately for fern using
reference-mask IoU, then averages only the 19 target IoUs. Seeds 0/1/2
completed at 97.036403%, 97.045689%, and 97.051700%; their 97.044597% mean is
0.044597 points above the paper fern value 97.0. It remains ineligible for both
the local nine-scene macro and paper ten-scene 93.8 row.

CPU-only preflight:
`output/protocol_audit_20260731/ludvig/spin/preflight/fern_checkpoint_reuse_v1/preflight_manifest.json`
(SHA-256
`ecf1df13466f32f4b16babbcb33b5206fcb613cea0579e5a94abe9f1e77db9dd`).

Leaves follows the same fail-closed reuse rule. The SPIn raw scene and the raw
source behind `NVOS/llff_undistorted/leaves_undistort` have byte-identical 26
RGB files and raw COLMAP hashes:

- `cameras.bin`: `73bcea72824cf4cb48a1f58e9d03efd9e4728bb198e4f11e4aba004a454d52cb`;
- `images.bin`: `f4bd3b275b1bce1be544dbe79459a1136c2376c8b8c192857799a143f7724bc8`;
- `points3D.bin`: `e47ad2e4be1aae5d19495f5aecca8a625ce1e0d9b24f4645c64cd05361385cc1`.

Each SPIn leaves run records verified cross-benchmark reuse of the exact NVOS
leaves manifest/PLY. Its evaluator uses `image000` as the unscored reference,
averages 25 target-frame IoUs, and produced the 96.368580% three-seed mean
reported above.

Orchids likewise reuses the exact 25-view NVOS orchids manifest/PLY after the
SPIn source is proven byte-identical to the NVOS conversion source. Its raw
COLMAP hashes are:

- `cameras.bin`: `5797e621b8bf269aeccf8127b5f438199e574a92ef0af8761c776da962b5f102`;
- `images.bin`: `bb07036e7932adeb602d16a7d9a6b82426b07992d9efd9865c4b3bc03da34cc9`;
- `points3D.bin`: `953b37835a1f23ab02d5d54e6f572657ed953c4ab1121e9c92efd3325da53a60`.

The released SPIn orchids frame list contains one unscored reference
(`image000`) and 23 scored targets per seed. Upstream omits `image014`, so it
is absent by protocol rather than treated as a failed frame. The three runs
selected reference-calibrated threshold parameters 89/92/93 with SAM
candidate 0 and produced the 90.065401% mean reported above.

Fortress reuses the exact 42-view NVOS fortress manifest/PLY after the SPIn
source is proven byte-identical to the NVOS conversion source. Its raw COLMAP
hashes are:

- `cameras.bin`: `0780ba1b521452b2be37719d8aefd148723625300c8ec5d99fb300eff65a8f26`;
- `images.bin`: `aed84eea58ac55bae70c191fe71bd7f98adb425ea1c8fe227260bad088da83f0`;
- `points3D.bin`: `70bb9247cc05f882ff22153fea2712ed9bcf77fb384cb7fa80c681d6e4cbd60f`.

The released SPIn fortress frame list contains one unscored reference
(`image000`) and 41 scored targets per seed. All three runs selected
reference-calibrated threshold parameter 69 with SAM candidate 2 and produced
the 98.356092% three-seed mean reported above.

Horns also follows the fail-closed reuse rule. The SPIn raw scene and the raw
source behind the audited NVOS conversion have 62 byte-identical RGB files
and raw COLMAP hashes:

- `cameras.bin`: `178c5ec2b39b04072ed7ed2fa9b9358e2e79a1943b5e9aac12ec8456243455a2`;
- `images.bin`: `4137f33bd74c590b13488ab5cfd4cd1ac2b9bd24be0f43e4787a1a60bc28dcad`;
- `points3D.bin`: `a00d19015e645df54f903efc30278cd265ab531c3e055b09545b6a52826b6908`.

Each horns_center and horns_left NVOS run scores exactly one target and uses
fixed threshold parameter 75. Each SPIn horns result contains 62 frame
records: one unscored reference and 61 scored targets. Seeds 0/1/2 selected
threshold parameters 71/70/69, respectively, with SAM candidate 2 in every
run. Their target-only means are 90.007836%, 89.995405%, and 89.973144%,
giving the 89.992128% three-seed mean reported above. All nine manifests bind
the same 62-view training manifest/PLY identity and exact target counts.

Room contains one unscored reference frame and 40 scored targets per seed.
Seeds 0/1/2 selected threshold parameters 73/74/72, respectively, with SAM
candidate 0 in every run. Their target-only means are 97.126922%, 97.091829%,
and 97.343748%, giving the 97.187499% mean reported above. The successful run
manifest SHA-256 values are
`95fe50d036a218e3d4e0e5b28eccacfe9ebb38ef5d4dfe1d1ce47b469b117557`,
`903afb587c034d65284ff0488821370db09523a73a7d41091ab2a93b106590a2`,
and `77b2f522c1a382fa66f3f6f61aadc875d6008e10843fe98b2633e8fb800105f5`.
The excluded v1 failure manifest SHA-256 is
`be2a512557684c94814551ca12f1589c2767d271e8205f8c24923c7165b04512`.

Truck contains one unscored reference frame and 64 scored targets per seed.
All three runs selected threshold parameter 87 and SAM candidate 2. Their
target-only means are 96.749968%, 96.571124%, and 96.811787%, giving the
96.710960% three-seed mean reported above. The run-manifest SHA-256 values are
`abb735687067fb32ed139fda366150a4bd106c20e9c1374ee44a598ab9ef4eaa`,
`665b0d63255ed445ba26723995344e610ae7ca7ecf0dac49bf904d63e74bc267`,
and `b84c52fa446b8ab55625949d79851d27c594455a0123a70199e623dad66ef096`.
The corresponding protocol-result SHA-256 values are
`5c26a177d2a9929d10b5a8281afc9ab0636de61f8158566b031e2ce74f7d40ce`,
`5874c069c9f68f2c58f73cac7805bc59d3b85ed3a1ea5a474a9a8751c98eeb48`,
and `e7806edc5bfca76ff31b5f6f319206e7023bec852f3fee4e1875e0d5bde2fa24`.
Every run binds the completed 251-view manifest and exact PLY identity.

## Reproduction artifacts

- `reproductions/ludvig/upstream.lock.json`
- `reproductions/ludvig/official_3dgs.lock.json`
- `reproductions/ludvig/patches/0001-reproduction-seeds-and-json-results.patch`
- `reproductions/ludvig/train_nvos_all_view_3dgs.py`
- `reproductions/ludvig/train_spin_llff_room_all_view_3dgs.py`
- `reproductions/ludvig/train_spin_truck_all_view_3dgs.py`
- `reproductions/ludvig/train_spin_pinecone_all_view_3dgs.py`
- `reproductions/ludvig/stage_spin_pinecone_official_undistortion.py`
- `reproductions/ludvig/run_gpu0_thermal_guard.py`
- `reproductions/ludvig/preflight_spin_fern.py`
- `reproductions/ludvig/run_ludvig_sam.py`
- `reproductions/ludvig/aggregate_results.py`
- `reproductions/ludvig/spin_nerf_9scene_rgb_dir_map.json`
- `reproductions/ludvig/README.md`
- `output/protocol_audit_20260731/ludvig/nvos/released_all_view_full8_3seed_summary.json`
- `output/protocol_audit_20260731/ludvig/spin/released_all_view_fern_leaves_orchids_fortress_horns_room_pinecone_truck_lego_3seed_summary.json`
- `output/protocol_audit_20260731/ludvig/spin/released_all_view/pinecone/undistortion/attempts/official_colmap_3p6_v2/undistortion_manifest.json`
- `output/protocol_audit_20260731/ludvig/spin/released_all_view/pinecone/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
- `output/protocol_audit_20260731/ludvig/spin/released_all_view/room/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
- `output/protocol_audit_20260731/ludvig/spin/released_all_view/room/seed_0/attempts/exact_f7a_allview_room_seed0_v1/run_manifest.json`
- `output/protocol_audit_20260731/ludvig/spin/released_all_view/truck/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
- `output/protocol_audit_20260731/ludvig/spin/released_all_view/truck/seed_0/attempts/exact_f7a_allview_truck_seed0_v1/run_manifest.json`
- `output/protocol_audit_20260731/ludvig/nvos/released_all_view/trex/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
- `output/protocol_audit_20260731/ludvig/nvos/released_all_view/horns/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
- `output/protocol_audit_20260731/ludvig/nvos/released_all_view/flower/training/attempts/exact_f7a_allview_30k_v1/training_manifest.json`
- `output/protocol_audit_20260731/ludvig/thermal_logs/horns_exact_f7a_allview_v1/`
- `output/protocol_audit_20260731/ludvig/thermal_logs/flower_exact_f7a_allview_v1/`
- `output/protocol_audit_20260731/ludvig/thermal_logs/trex_exact_f7a_allview_v3/`
- `output/protocol_audit_20260731/ludvig/thermal_logs/truck_exact_f7a_allview_v1/`
