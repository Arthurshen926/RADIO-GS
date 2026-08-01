# Evaluation protocol audit: consolidated status (updated 2026-08-01)

This document consolidates the external-method evaluation audit for concept
queries, spatial prompts, and correspondence queries. The canonical selector
is `evaluation_protocol_freeze_20260801.yaml`; it locks one route for each of
the seven benchmark-method pairs and prevents historical diagnostics from
silently replacing them. The locally available nine-scene SPIn-NeRF cohort,
the exact-LUDVIG PFPR one-scene adaptation, and the NVOS eight-task LUDVIG
reproduction are complete. SPIn-NeRF still lacks Fork, so its local
nine-scene result is not the paper's full ten-scene row. Detailed provenance
remains in `evaluation_protocol_registry_20260731.yaml` and
`promptable_nvs_protocol_registry.yaml`.

## Current conclusions

| Family / benchmark | Reproduced method and local metric | Paper context | Protocol conclusion |
| --- | --- | --- | --- |
| concept / LERF-2D | OccamLGS: 63.6200% mIoU, 82.8487% LocAcc (four-scene macro) | 61.3% / 82.5% | Canonical protocol reproduction. Exact camera names, released all-view RGB geometry intent, test-frame-excluded semantic lifting, raw relevance level selection, fixed 0.5 threshold, and released filtering/smoothing close the loop. Local checkpoint identity remains a disclosed compatibility boundary. |
| concept / LERF direct 3D | VALA on compatible Occam RGB geometry: 54.1249% mIoU, 79.3526% Acc@0.25, 56.6114% Acc@0.5 | 43.29% / 64.30% | Canonical protocol reproduction diagnostic. Extensionless test stems, three-level official significance/robust lifting, raw-peak level selection, threshold 0.6, and official per-object/scene-macro evaluation are frozen. This is not a paper-identical RGB-geometry training claim. |
| concept / ScanNet OVS | VALA `paper8`: split 19 34.5269/51.5906; split 15 37.9606/56.7696; split 10 47.3642/67.4650 mIoU/mAcc | 32.11/50.05; 35.10/54.77; 46.21/65.61 | Canonical paper-cohort protocol reproduction. Full-resolution official significance, robust aggregation, anisotropic Gaussian pseudo-GT, opacity-volume weighting, present-GT classes, and scene-equal aggregation are fixed. `scene0645_00` is code9 sensitivity only. |
| concept / historical ScanNet P0 domain isolation | Two-scene split 10: Gaussian-domain 16.3307/34.3724 versus mesh-domain 14.4578/33.5118 mIoU/mAcc | no paper-comparable row | Superseded diagnostic retained only to show that changing mesh to an incomplete Gaussian proxy was insufficient. It must not replace the exact `paper8` result. |
| spatial / AGILE3D | Easy3D `agile3d_release`: IoU@1/2/3/5/10 = 69.7137/76.0135/78.5942/80.9951/83.0384% | 68.2/74.6/77.3/79.6/81.7% | Full 312 scenes, 10,357 objects, zero failures. Source-grounded paper-facing diagnostic; all five values are 1.29–1.51 points above the paper. |
| spatial / AGILE3D sensitivity | Easy3D released forward: 69.7138/75.9464/78.3255/80.3703/82.1321% | same paper row | Full coextensive sensitivity. The AGILE3D-minus-released-code difference changes from −0.0001 points at click 1 to +0.9063 points at click 10. |
| spatial / NVOS | LUDVIG-SAM exact all-view 8-task x 3-seed macro: 91.2577%; Trex seeds 87.5296/87.6711/87.7888%, mean 87.6632% | full eight-task 91.3%; Trex 88.0% | Complete released online-multiview paper protocol. Delta is −0.0423 points versus the rounded headline and −0.0798 versus the independently recomputed 91.3375% eight-task paper macro. Not strict-unseen. |
| spatial / SPIn-NeRF | LUDVIG-SAM exact all-view local nine-scene x 3-seed macro: 93.7200%; Pinecone seeds 81.4015/86.2811/86.1369%, mean 84.6065% | matching nine-scene paper macro 94.5778%; Pinecone 88.8%; full ten-scene 93.8% | Matching-scene delta is -0.8577 points; Pinecone is -4.1935 points. Every local scene has an exact three-seed check, but Fork is absent, so no strict full-table claim is possible. |
| correspondence / PFPR | Exact-LUDVIG one-scene adapter: Top-1 mean/median 1.9873/1.9546 m; R@10 at 20 cm 10%, all 10 cm recalls 0. Historical C-RADIO/DINOv3 six-scene sanity: 1.2669/0.6899 m and R@1/5/10 at 10 cm 16.67/35.00/41.67% | no corresponding LUDVIG paper task | Exact Phase A-E is complete for `scene0050_02`, 120 query-held-out source views, 300k shared Gaussians, ten public crops, and 31,143 public candidates. The released LUDVIG DINO/PCA/uplift is exact, but center-3x3 crop pooling plus cosine/Gaussian readout is an explicit learning-free custom adapter because LUDVIG publishes no PFPR head. The weak result is interpretable evidence that this direct adapter lacks crop-to-3D correspondence capability, not a paper or protocol comparison. |

## Protocol corrections frozen by this audit

1. OccamLGS LERF-2D resolves annotation frames by exact filename over all
   registered cameras. RGB geometry follows the released all-view intent;
   semantic lifting excludes released test frames. Raw OpenCLIP relevance
   selects one of three levels; threshold 0.5, 30x30 OpenCV filtering, 7x7
   legacy smoothing, and a four-scene equal macro are fixed.
2. VALA LERF-3D `test.txt` contains extensionless stems. A zero-match split is
   a hard error because suffixed names leak annotated frames into semantic
   lifting. Three feature levels, official marginal significance, robust
   lifting (`tau_mass=0.75`, `tau_abs=0.13`), raw-peak level selection,
   KNN10, threshold 0.6, selected-only alpha rendering, per-object metrics,
   and a four-scene equal macro are fixed.
3. VALA ScanNet uses the paper's eight scenes; `scene0645_00` is a separate
   code9 sensitivity. Queries are fixed ScanNet class-name embeddings for the
   19/15/10 vocabularies. The primary evaluator uses full-resolution official
   significance, anisotropic Mahalanobis Gaussian pseudo-GT (radius five
   times maximum scale, top-K 1000, class-balanced vote),
   `sigmoid(opacity)*sx*sy*sz` weights, present-GT classes, and scene-equal
   aggregation.
4. Historical LangSplatV2, Dr. Splat L1/L3, old VALA proxy/resolution-2,
   mesh/Gaussian P0, and code9 results remain provenance/sensitivity rows.
   None may replace the three canonical concept-query rows selected by the
   freeze manifest.
5. Easy3D's paper-facing contract is frozen from the paper's stated AGILE3D
   benchmark semantics. Pilot proximity to the published test row is never a
   protocol selector. The released Easy3D forward is retained as a complete
   sensitivity row.
6. LUDVIG `released_all_view` runs must bind the evaluated PLY to a completed,
   hashed original-3DGS manifest with the audited commits, 30k iterations,
   0 held-out views, `eval=false`, resolution `-1`, and an exact camera count.
   Hybrid geometry is explicitly barred from carrying this provenance label.
7. LUDVIG-SAM runs must also bind the audited ViT-H SAM checkpoint SHA256
   `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`,
   pinned LUDVIG commit, an approved hashed reproduction patch, released
   online target-view calls, and benchmark-specific calibration and
   aggregation. Resolution-safe v3 fixes exact 1600-pixel resizing and adds
   fail-closed tensor-shape checks without changing threshold, metric, or
   aggregation. Room's v2 evaluation staging additionally applies a
   fractional centered crop `[0.5, 0, 4004.5, 3003]` before exact 1600×1200
   bicubic resize, preserving JPEG tables/sampling and leaving source data
   unchanged. NVOS results must carry fixed threshold parameter 75; SPIn scene
   means are recomputed from target-only `frame_results`.
8. LUDVIG eligibility is fail-closed. The completed NVOS 8-task × seeds
   `[0,1,2]` all-view row is a strict comparison to the released paper
   protocol, but target-RGB visibility still forbids a strict-unseen claim.
   SPIn-NeRF nine-scene local completeness never implies ten-scene eligibility.
9. PFPR method-visible inputs remain only `scene_id` and the public 128x128
   depth-aligned RGB crop. Source RGB/depth/poses must come from the frozen
   query-held-out field contract; evaluator-private anchors and manifest are
   prohibited during method execution. Exact LUDVIG checkpoint loading must
   also record that its vendored ViT lacks register-token support and permits
   only the frozen `register_tokens` unexpected key. Phase B freezes the
   scene PCA transform; Phase C uses official `GaussianModel.apply_weights`;
   Phase D scores every public candidate before Phase E is allowed to open the
   evaluator manifest. Continuous readout requires strictly positive support
   but has no tuned support threshold: all 31,143 candidates pass.
10. Target-oracle masks/thresholds remain diagnostic-only and cannot select a
   reported metric, method, or protocol.

## Easy3D full-run integrity

| Contract | Result SHA256 | 312-scene shard-set SHA256 |
| --- | --- | --- |
| `agile3d_release` | `c771f29400e912565ee2ea5a754d0fd80a7fafc1eb91e38b6db1953cdcbbc09d` | `6fc27936ebf1b1e5a14df6c87e81b7699e6a72c8c661550efbdb3fc9ea467f44` |
| `easy3d_released_code` | `c8c0c6820cf88166e77002a20ddf68af44c7be2e58a2e36cd26b147bdbd2f5a1` | `1915fef1216b822abe373b58b7962a4ecac2ddd8452258296cf2cb6f89a55189` |

Both runs bind Easy3D commit
`b3f5bd70defaa9a601edb0975802775b056c784a`, official checkpoint SHA256
`4a13d16ba2f2470031287812dbbdf1ec6aa14097cb3738e0fe596bb708dc475f`,
evaluator SHA256
`a2af43c40df442a4e739ce925a9a1d7de831c6b102173f22e1e09460edc6ae6e`,
and preprocessing-manifest SHA256
`39035ec87a3ff73bd9cfd6eec9a93182b7ebd7d9b2e84515b1c0e51cad453d23`.
The source-grounded pilot policy is
`output/protocol_audit_20260731/easy3d_agile3d_pilot3_source_grounded_policy_v2.json`
(SHA256
`1f065485514b82af0c8afe64b4941b07cc2135f8a3094b92a9ee4f19f954bbcc`).

## Completed NVOS eight-task and local SPIn-NeRF nine-scene boundaries

The audited original-3DGS fern, flower, fortress, horns, leaves, orchids, and
Trex checkpoints are complete. Fern binds
training-manifest/PLY SHA256
`8b7ebe7ff6d946c4ae03b1943835a4d73ee94b0e53d663a5933de03ea947909e` /
`4eaac89e70e00e4776e1bd9505b7560c483edd5df8ee1958c84b91091e49134b`;
flower binds
`945e3fe89dbf451a74dfea4d8edfb406fef4fc1045c9719b73debce0f456b4e6` /
`eb9658266c5dc10d639fbe9371858edcb537f0ad2e4a8f0d8e817b55e2c567ec`;
fortress binds
`e01d76a52f0e0df0176e48301f9753ec610b475114f0449ab4cb634c6938aab3` /
`723bfaa66c2e2da9c5e76662dc317d86fafe0dfed5c109b5e9fd6c11e8626c45`;
leaves binds
`2a610c0c356ae05f9a7d712c4352e032b137fa0794df3d701ec550759b67ba69` /
`da74e7c81debf212d0b3c6f7278a62259dfcc045f97401cb9370b1e62d340378`;
orchids binds
`8fc085e1c6ad817772898fd5e24f120b72ec78d3c0efa793a7ccbd761935d3d7` /
`e04a38f5aa22aaa716ec1c741222e1d1c94fb3cf0305aa4cc06a50a823ae5ef5`;
horns binds
`77154d2f5e761ba34a605677be8143214246ff8fbbf310b3400b55e4332488bc` /
`1f3084daba5e9a70263152f73f704815f452a2f37e81f56cf3a3eb96c3e49803`;
Trex binds
`e8341f2ed5f77a48189790b3d61bdab0f1bea7efb45e41134403996bd4e10035` /
`444c364891ff637a6f82b85fc1236f04a89f3ce7bd8b18bcf4806f88b5e0a7b0`.
For the five NVOS-overlapping scenes shared with SPIn-NeRF, both benchmark
runs bind the same hashes. Room uses its own 41-view SPIn-NeRF all-view
checkpoint, with training-manifest/PLY SHA256
`275c72ced3689cbca6b51cca8cc7830130fa6a1ac2d4eed3d471a346a7345dd4` /
`c12a46a46ab3550905cfe48152916bd75834f5f36485ec4cf1cae03eaef294fd`.
Truck binds training-manifest/PLY SHA256
`858ac2c6b6438ef609ac7bc82c701067a0117dc470159179326ed26929de2cd6` /
`aa5b6d166277cde9d49632c9ffa427c392a4e52b7c68719c3d6ca312bedc3492`.
Pinecone binds audited-undistortion/training-manifest/PLY SHA256
`a542a5003028ce3fc3c8639bd3609f39de6fc761d4816d41d0a93800d04ce868` /
`2d1b5d8b14765f03dd0b84a3ed64f41d7dffcf2d776835a2d6270cf328eb816b` /
`27d5670cd642542cdba671a7a5718ae463b1097c437d2f4e232999090aef451e`.
Flower and Trex are NVOS-only checks here. All exact evaluations use seeds
`[0,1,2]` and verified upstream-patch provenance.

The current summaries are pinned as NVOS SHA256
`65e1f8e5c1f17083f66e5b7d4f6f03687f6806394c78a5cfce25546ca42e3546`
and SPIn-NeRF SHA256
`ee300d2eb805600374461f953eb7a89ad1c890c2f02bbb347957e2f164e75e17`.
NVOS is complete at 91.257685%: -0.042315 points versus the rounded 91.3%
headline and -0.079815 versus the 91.3375% macro recomputed from the paper's
eight displayed task values. The local nine-scene SPIn-NeRF macro is
93.720045% versus 94.577778% on matching paper scenes (-0.857733 points), and
remains a local-cohort diagnostic rather than the paper's full row. Pinecone
contributes 81.401456/86.281102/86.136876%, mean 84.606478%, or -4.193522
points versus its paper scene value 88.8. Lego contributes
93.296140/92.968601/93.181267%,
mean 93.148669%, or +0.448669 points versus its paper scene value 92.7. Room contributes 97.126922/97.091829/97.343748%,
mean 97.187499%, or +0.687499 points versus its paper scene value 96.5. Truck
contributes 96.749968/96.571124/96.811787%, mean 96.710960%, or +1.810960
points versus its paper scene value 94.9. Each NVOS task scores one target per
seed; SPIn horns scores 61 targets, while Room and Truck each score their
target-only cohort after excluding the reference frame.

The excluded Trex v1 attempt diagnosed a real protocol-path bug: a
floating-point floor produced a 1599-wide prompt image against 1600-wide
camera/render/SAM tensors. Resolution-safe v3 fixes the exact long-edge resize
and adds fail-closed shape checks. Its threshold, negative-scribble behavior,
metric, and aggregation are unchanged. The failed v1 score is retained only
as failure evidence and never enters the 24-run NVOS aggregate.

Room seed 0 v1 is likewise excluded, but it produced no metric: the prompt
and camera tensors were 1200×1600 while the staged image tensor was
1199×1600. The accepted Room v2 attempts use an evaluation-only fractional
center crop from 4005×3003 to 4004×3003, then exact 1600×1200 bicubic resize
with preserved JPEG tables/sampling. The source dataset is unchanged, and the
repair changes neither reference calibration nor target-only aggregation.

The remaining boundary is the unavailable SPIn-NeRF Fork scene, not an
unfinished local run or NVOS protocol uncertainty:

1. All nine locally available SPIn-NeRF scenes are complete; Fork remains
   absent, so the local nine-scene macro cannot become the paper's ten-scene
   row.
2. The 23 remaining strict-geometry/released-query hybrid NVOS runs remain an
   optional, explicitly non-paper diagnostic and cannot substitute for the
   already complete all-view cohort.
3. The fixed SPIn-NeRF `room`, `truck`, `lego`, and `pinecone` entries all
   have completed exact three-seed results. Pinecone seed 0 v1, which failed
   before scoring because `PYTHONPATH` was missing, remains excluded.

## Exact-LUDVIG PFPR one-scene closeout

The `scene0050_02` Phase A-E chain is complete and keeps the method/evaluator
privacy boundary explicit. Phase A stages 120 query-held-out source views and
binds the shared 300k-Gaussian geometry. Phase B reproduces the released
vendored ViT-g/14 behavior, including the audited discard of the official
checkpoint's sole `register_tokens` tensor, and fits the frozen scene PCA40
over all 277,440 scene tokens. Phase C reconstructs the official sliding
windows and calls LUDVIG's released inverse-render `apply_weights`; 299,992 of
300,000 Gaussians receive positive observation weight. Phase D sees only ten
public 128x128 crops, reuses the frozen scene transform, pools the center 3x3
of each 9x9 token grid, and produces finite scores for all 31,143 public
candidates. Only Phase E opens the private anchors.

The resulting custom-adapter metrics are Top-1 mean/median error
1.987313/1.954631 m, R@10@20cm 0.10, MRR@20cm 0.02, and zero recall at 5 or
10 cm. On this same scene the older C-RADIO/DINOv3 diagnostic gives
0.266407/0.220645 m and R@1/5/10@10cm 0.20/0.60/0.70, but it uses a different
feature field and is not an exact-LUDVIG result. The contrast is retained only
as an interpretability check: LUDVIG publishes no PFPR crop-retrieval head,
and the frozen learning-free adapter is not selected or tuned from private
metrics.

Phase manifest SHA256 values are:

- Phase A: `de3f0281ae863d9f640eaabce5c805f35a9fca03221ceee0a67756fc9023e22a`;
- Phase B: `1d546a335e2f3ec807c69b23f06a7876d1a325d53e16a94b0d64f8b7556d147b`;
- Phase C: `dcf8d864da50aa455f805d94bce707daf4cfc4ac1f602ea03141b38a57eb13fb`;
- Phase D: `e7f615ed0013cc32b858c92df364c36585dcfa1a10758202bcb361035c42b6ee`;
- Phase E: `d85c59aebb21cb6c6b6c1251c737f82c88c36eea58e4c8175167a71797c32901`.

Phase B/C/D GPU0 peaks were respectively 58/52/49 C with zero thermal pauses.
Phase B reached 282.61 W, while the shorter Phase C/D jobs reached
222.08/164.34 W. These measurements show that the current executed
78/81/70 C policy with one stable cool sample did not reduce PFPR throughput.

The current GPU0 thermal guard SHA256 is
`4daa492d2026d58f5d83184dc12dbf4f87747bf1d8630c0e70bf563c5acfed79`;
its defaults are 3 s polling, 2 s query timeout, 78/81/70 C
warning/pause/resume thresholds, and two stable cool samples. The current
throughput-adjusted launches explicitly pass one stable cool sample. Historical
Horns/Flower/Trex launches intentionally record their executed policy
snapshots rather than inheriting these live defaults. All nine Horns
evaluation logs stayed at or below 68 C with zero pause events and zero query
failures. Flower training completed in 1586.195 GPU-wall seconds after 43
guard pause/resume cycles, reached 82 C, and had zero telemetry failures; its
three evaluation logs reached 69/70/69 C with no pauses or query failures.
Trex training completed in 2351.850 GPU-wall seconds after 37 pause/resume
cycles and reached 82 C; its accepted evaluations reached 72/68/69 C with no
pauses or telemetry failures. All finalized Flower and Trex jobs returned 0.
The guard remains execution infrastructure only: it does not change camera
staging, calibration, cohort, aggregation, or protocol eligibility.

## Detailed reports

- `concept_query_protocol_audit_20260731.md`
- `easy3d_agile3d_protocol_audit_20260731.md`
- `ludvig_nvos_spin_protocol_audit_20260731.md`
- `pfpr_ludvig_style_protocol_card_20260731.md`
- `evaluation_protocol_registry_20260731.yaml`
- `promptable_nvs_protocol_registry.yaml`
