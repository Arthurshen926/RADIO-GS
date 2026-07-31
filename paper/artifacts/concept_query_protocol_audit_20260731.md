# Concept Query evaluation protocol audit — 2026-07-31

## Outcome

This audit separates three evaluation domains that had previously been treated
too similarly:

1. **LERF-2D rendered-view grounding** is sensitive to the exact rendered
   camera, image resolution, feature-level selection, mask readout, and
   cross-scene aggregation. A real camera-indexing defect was found and fixed.
   The corrected four-scene LangSplatV2 diagnostic is close to its published
   row under matched aggregation: **+1.61 pp mIoU / -4.29 pp LocAcc**.
   This delta is a released-code/local-checkpoint diagnostic, not a claim that
   the official pretrained row has been reproduced.
2. **LERF direct 3D selection** evaluates already exported query masks by
   explicit frame/query names. The existing Dr. Splat run is only a local
   compatibility construction and remains far below the reported row. Its gap
   cannot be attributed to the LERF-2D camera-indexing defect.
3. **ScanNet point/Gaussian querying** has no LERF annotation-camera lookup at
   all. The existing VALA run is also only a compatibility construction, with
   a different prediction domain, visibility proxy, view sampling, and GT
   transfer procedure from the reported protocol. Its gap likewise cannot be
   attributed to camera mapping.

OccamLGS was investigated as the preferred common baseline. Its official
repository directly supports LERF-2D and 3D-OVS, but not the LERF direct-3D and
ScanNet adapters used in the paper-facing tables. In particular, the OccamLGS
LERF-3D and ScanNet rows in this project are **OpenGaFF-reported adapter
results**, not results from the official OccamLGS paper or repository. Since
the required OpenGaFF adapter implementation/checkpoints are unavailable
locally, the allowed fallback was used:

- LangSplatV2 for LERF-2D;
- Dr. Splat for LERF direct 3D;
- VALA for ScanNet.

No training, field representation, query compiler, or project-method readout
was changed. The changes are confined to baseline evaluators, reproduction
wrappers, protocol summaries, tests, documentation, and isolated audit
outputs, so they do not interfere with the separate project-method improvement
work.

## Audit decision table

All metrics below are percentages. A gap is local minus reported, in
percentage points.

| Benchmark | Baseline chosen | Local status | Local result | Reported result | Matched gap | Decision |
|---|---|---|---:|---:|---:|---|
| LERF-2D | LangSplatV2 | Complete four-scene released-code-intent/local-checkpoint diagnostic with exact camera-name resolution | 61.51 mIoU scene-macro / 79.81 LocAcc query-micro | 59.90 mIoU scene-macro / 84.10 LocAcc query-micro | +1.61 / -4.29 | Protocol bug fixed; use as an auditable diagnostic, with local checkpoint provenance and mixed camera roles disclosed |
| LERF direct 3D | Dr. Splat | Compatibility only | 17.62 mIoU / 25.61 Acc@0.25 scene-macro | 43.29 / 64.30 | -25.67 / -38.69 | Do not label as an official or strict reproduction |
| ScanNet, 19 classes | VALA | Compatibility only | 11.98 mIoU / 23.74 mAcc | 32.11 / 50.05 | -20.13 / -26.31 | Do not use to validate the published VALA row |
| ScanNet, 15 classes | VALA | Compatibility only | 14.87 mIoU / 28.15 mAcc | 35.10 / 54.77 | -20.23 / -26.62 | Same |
| ScanNet, 10 classes | VALA | Compatibility only | 22.72 mIoU / 41.70 mAcc | 46.21 / 65.61 | -23.49 / -23.91 | Same |

The LERF-2D status is deliberately called “released-code-intent” rather than
“strict held-out.” The reason is documented below.

## LERF-2D: corrected LangSplatV2 diagnostic

### Root cause 1: annotation frame index was used as a train-list index

The released evaluator converted an annotation filename back to its original
frame index and used that integer to index `scene.getTrainCameras()`. With
`eval=True`, LLFF removes every eighth camera from the train list, so its list
indices no longer equal original frame indices. An annotation could therefore
be evaluated against a different RGB pose.

The correction resolves the annotation image stem by exact name over
`train cameras ∪ test cameras`. Missing or ambiguous names fail closed. The
same correction is applied to the normal and quick evaluators. The reusable
minimal patch changes camera resolution only; it does not change checkpoints,
features, similarity scores, thresholding, smoothing, or aggregation.

The impact is material and scene dependent. For example:

| Scene | Old wrong-pose mIoU / LocAcc | Exact-camera mIoU / LocAcc | Change |
|---|---:|---:|---:|
| Teatime | 9.68 / 20.34 | 71.60 / 88.14 | +61.92 / +67.80 pp |
| Waldo Kitchen | 55.58 / 72.73 | 55.61 / 77.27 | +0.03 / +4.54 pp |

The old rows must not be retained as baseline results.

Across the full four-scene cohort, the correction changes the stored summary
as follows:

| Aggregate | Old summary | Exact-camera summary | Change |
|---|---:|---:|---:|
| mIoU, scene-equal macro | 46.01 | 61.51 | +15.50 pp |
| LocAcc, scene-equal macro | 61.76 | 79.85 | +18.09 pp |
| LocAcc, 208-query micro | 60.10 | 79.81 | +19.71 pp |

The unrounded LocAcc values are `0.6176 → 0.7984554` for the scene macro and
`0.6009563 → 0.7980769` for the query micro.

### Root cause 2: an independent feature-level selection defect

The project-side pre-rendered evaluator selected a feature level from the
maximum of each level *after* per-level min-max normalization. Every
non-constant level then has a maximum of one, degenerating selection to the
first level. It now follows the released readout intent and chooses the level
from the peak of the activated, non-min-max-normalized relevance map.

This second defect invalidates the old project-side OccamLGS pre-rendered
numbers. It is independent of the exact-camera LangSplatV2 upstream run and was
fixed to prevent future Occam/LangSplat readouts from silently repeating the
error.

### Exact four-scene result

| Scene | Queries | Local mIoU | Paper mIoU | Gap | Local LocAcc | Paper LocAcc | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Figurines | 56 | 59.66 | 56.40 | +3.26 | 82.14 | 82.10 | +0.04 |
| Teatime | 59 | 71.60 | 72.20 | -0.60 | 88.14 | 93.20 | -5.06 |
| Ramen | 71 | 59.17 | 51.80 | +7.37 | 71.83 | 74.70 | -2.87 |
| Waldo Kitchen | 22 | 55.61 | 59.10 | -3.49 | 77.27 | 95.50 | -18.23 |

Both aggregate families are serialized:

| Aggregation | Local mIoU | Local LocAcc | Notes |
|---|---:|---:|---|
| Four-scene equal macro | 61.51 | 79.85 | Each scene has equal weight |
| 208-query micro | 62.45 | 79.81 | LocAcc is exact: 166/208; mIoU is reconstructed from four-decimal scene logs |

The published LangSplatV2 “overall” row uses **mixed aggregation**:

- published mIoU `0.599` matches the equal mean of the four rounded scene
  mIoUs (`0.59875`);
- published LocAcc `0.841` matches the 208-query weighted mean of the rounded
  scene rows (`0.841399`), not their equal scene mean (`0.86375`).

Consequently, the only matched headline comparison is local scene-macro mIoU
against published mIoU, and local query-micro LocAcc against published
LocAcc: **+1.61 / -4.29 pp**. Both local macro and micro values remain in the
artifact so no alternative aggregation is hidden.

This comparison diagnoses whether the released evaluator behavior and the
local checkpoint cohort are numerically plausible. The three-level
checkpoints were trained from local Occam-compatible RGB starts and
compatibility language features, not loaded from an official LangSplatV2
pretrained release. Therefore the paper delta must not be presented as a
strict official-checkpoint reproduction or as proof that every training-side
detail matches the paper.

### Camera-role manifest and the held-out-view boundary

All twelve level-specific `cfg_args` files were parsed without executing them
and verified to use `eval=True`, the expected source scene and feature level,
and a consistent checkpoint cohort. Nevertheless, exact annotation-name
resolution shows that the released evaluator consumes both LLFF roles:

| Scene | Train-role labeled frames | Test-role labeled frames |
|---|---|---|
| Figurines | `frame_00152`, `frame_00195` | `frame_00041`, `frame_00105` |
| Teatime | `frame_00002`, `frame_00043`, `frame_00107`, `frame_00140` | `frame_00025`, `frame_00129` |
| Ramen | `frame_00006`, `frame_00024`, `frame_00060`, `frame_00119`, `frame_00128` | `frame_00065`, `frame_00081` |
| Waldo Kitchen | `frame_00053`, `frame_00066`, `frame_00140`, `frame_00154` | `frame_00089` |
| **Total** | **15** | **7** |

Thus, this is a **mixed-role released-code/local-checkpoint diagnostic**, not
a pure held-out test-view protocol. This differs from the OccamLGS paper wording about
withheld test views. The official OccamLGS `run_lerf.sh` also trains RGB
geometry without `--eval` and only passes `--eval` during later feature
extraction/rendering, so “held out” cannot be inferred from an evaluation flag
alone.

For future paper-facing runs, choose and name one of two protocols explicitly:

1. `released_mixed_role_exact_name`: reproduce released evaluator semantics
   and disclose 15 train-role / 7 test-role labeled frames; or
2. `strict_test_only`: train geometry with the test split withheld and reject
   every train-role annotation. This is a separate diagnostic protocol and is
   not numerically interchangeable with the released row.

### Readout profiles now frozen in code

| Profile | Feature | Mask threshold | Activation filter | Mask smoothing | Shape mismatch |
|---|---|---:|---|---|---|
| `langsplatv2_released` | normalized | 0.4 | 29×29 Torch average pool | 7×7 strict majority | error |
| `occam_langsplat_paper` | raw | 0.5 | 30×30 OpenCV `filter2D` | exact legacy LangSplat 7×7 edge/tie behavior | error |

Resizing is available only through the explicitly named
`bilinear_compat` policy. Strict profiles never resize silently.

The streamed Occam evaluator additionally:

- resolves every annotation by exact name across the train/test union;
- can enforce `--require-test-only`;
- parses checkpoint `cfg_args` with `ast` and rejects `eval != True` unless
  the caller explicitly permits and labels a training-visible diagnostic;
- holds only one frame/level relevance tensor at a time and never writes a
  full-resolution raw 512-D feature cache.

The available local Occam geometry has `eval=False`, so a strict held-out pilot
correctly stops before GPU evaluation. No new GPU run was queued after this
protocol blocker was established.

## LERF direct 3D: Dr. Splat compatibility result

The local artifact contains 208/208 filename-matched predictions:

| Scene | Queries | mIoU | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|---:|
| Figurines | 56 | 9.08 | 14.29 | 3.57 |
| Teatime | 59 | 19.78 | 28.81 | 5.08 |
| Ramen | 71 | 21.49 | 36.62 | 14.08 |
| Waldo Kitchen | 22 | 20.12 | 22.73 | 22.73 |
| Scene-equal macro | — | 17.62 | 25.61 | 11.37 |
| 208-query micro | 208 | 17.52 | 26.92 | 9.62 |

The paper-facing/OpenGaFF reported Dr. Splat row is `43.29 mIoU /
64.30 Acc@0.25`; the scene-macro gaps are `-25.67 / -38.69 pp`.

This is **compatibility only**, for concrete reasons:

- the official Dr. Splat README still marks evaluation as `TBA`;
- the local model directories are explicitly named
  `*_lerfcompat_topk45_weight_128`;
- Figurines and Waldo configurations use `iterations=0`, restore local Occam
  RGB checkpoints, set `topk=45`, enable a local PQ index, and consume
  compatibility data;
- the Ramen and Teatime saved `cfg_args` are incomplete (`Namespace()`);
- masks are locally exported with threshold 0.4 and evaluated by the common
  nested-mask evaluator;
- these are not official pretrained Dr. Splat checkpoints or an official
  evaluator.

The direct-3D evaluator reads predictions already keyed by
`frame/query` filenames. It never chooses a render camera by an integer list
index. Therefore the LERF-2D camera mapping defect cannot explain this gap.

## ScanNet: VALA compatibility result

| Split | Local mIoU | Reported VALA mIoU | Gap | Local mAcc | Reported VALA mAcc | Gap |
|---|---:|---:|---:|---:|---:|---:|
| 19 classes | 11.98 | 32.11 | -20.13 | 23.74 | 50.05 | -26.31 |
| 15 classes | 14.87 | 35.10 | -20.23 | 28.15 | 54.77 | -26.62 |
| 10 classes | 22.72 | 46.21 | -23.49 | 41.70 | 65.61 | -23.91 |

This run is also **compatibility only**:

- the VALA repository supplies a training/evaluation pipeline but no matching
  official pretrained ScanNet asset for this local run;
- the wrapper loads local RGB Gaussian PLYs rather than an official VALA
  checkpoint;
- because the local gsplat path does not expose VALA's contribution tensor,
  it substitutes `opacity * sqrt(radius)` as a significance proxy;
- it uses all views (`max_views=0`) at resolution 2, whereas the
  OpenGaFF/VALA-facing protocol specifies one keyframe every 20 frames;
- it uses `weight_threshold=1e-5` and four-nearest-neighbor transfer;
- it predicts labels on the ScanNet label PLY by kNN from Gaussian features,
  rather than reproducing the official VALA Gaussian-domain metric exactly.

The project already has a more internally coherent **own-method** evaluator on
optimized Gaussian centers, using anisotropic Mahalanobis-density pseudo
labels, `sigmoid(opacity) * sx * sy * sz` significance, and scene-equal
aggregation. It yields:

| Split | Own-method mIoU | Own-method mAcc |
|---|---:|---:|
| 19 classes | 20.76 | 39.42 |
| 15 classes | 21.07 | 40.29 |
| 10 classes | 26.99 | 50.95 |

These values are useful for defining this project's auditable ScanNet
protocol, but they are not a VALA reproduction and must not be relabeled as
one.

ScanNet evaluation operates on points/Gaussians and semantic labels, not LERF
annotation cameras. Camera-name resolution is therefore categorically unable
to explain the VALA compatibility gap.

## OccamLGS provenance boundary

The official OccamLGS checkout is
`eb98bcbedfdeb8770aae51d62c0263bddbc54329`. It contains:

- an official LERF-2D pipeline that refers to LangSplat evaluation; and
- an official 3D-OVS evaluation path.

It does **not** contain the adapters that generated the paper-facing LERF
direct-3D and ScanNet rows. Those numbers are copied as published context from
OpenGaFF:

- LERF direct 3D: `47.22 mIoU / 74.84 Acc`;
- ScanNet 19/15/10: `31.93/48.93`, `34.25/53.71`,
  `45.16/64.39` mIoU/mAcc.

They must be captioned as “OpenGaFF-reported OccamLGS adapter results,” not
“OccamLGS official reproduction.” Without the OpenGaFF adapter implementation,
checkpoint provenance, and evaluator, an all-three-benchmark Occam
reproduction is not currently auditable.

## Reproduction and evidence

### Checkouts and patches

- LangSplatV2: `1667303d5c111a5b62f69b9b8991d80045e92b5f`
- OccamLGS: `eb98bcbedfdeb8770aae51d62c0263bddbc54329`
- Dr. Splat: `764f608fcdaff213f11c027749eb637cc23aeb8a`
- VALA: `4463b78d450bd6c00df18e1dcbef7e538ac28e55`
- reusable LangSplatV2 exact-camera patch SHA-256:
  `a0ba52f843fdc21a0135f71b2ebe2edb5112c4ef48235a080c3c8828a5b285f3`
- actual audit-checkout `eval_lerf.py` diff SHA-256:
  `c65abd0c79f06ecc56df6e4a8a5093c8203ba785ebd3ff898cb39246221b8994`
- actual audit-checkout full tracked diff SHA-256:
  `31f14de37bb17650526b68f9dc6bd0a904c9322e7be1db2186557e217bbee608`

The reusable package accepts either a clean pinned checkout containing only
the minimal patch, or the exact recorded audit checkout behind an explicit
flag. It hashes staged checkpoints with streaming 1 MiB chunks, checks
source/staged equality, records sizes and hashes, verifies all three
`cfg_args`, and serializes GPU 0 through `/tmp/radio-gs-gpu0.lock`.

### Result artifacts

- LangSplatV2 cohort:
  `output/protocol_audit_20260731/langsplatv2_lerf2d_view_fix/cohort_summary.json`
  (`c8ca2f79e1c39b027d4ab7a114a10bcccf4f07b08517e33251cb072d8ba73da4`)
- camera manifests:
  - Figurines:
    `0469f040e0b54b57246dbadf5837422e9b63ac1570c1a61f190b118c9eef3629`
  - Teatime:
    `a3bc46db0456dc6404579f477630aedd952b43291f1712d329a0c8a94806a9eb`
  - Ramen:
    `cdac103d2b8e279f543c9f62e72e37d233d9ace8ed3f03806cf0cada9911e6b6`
  - Waldo Kitchen:
    `38f05b9e655da911f68ffc1eba04759432f478dbb2f709518f3ebea4f0711e61`
- Dr. Splat:
  `paper/artifacts/drsplat_lerf_summary.json`
  (`9196611f5cdf24edfbd803e74f7a6164338eb4825e849b93ca77ffd15385c8cd`)
- VALA compatibility:
  `output/baselines/vala/scannet_vala8_compat_20260611_res2/vala_scannet_vala8_results.json`
  (`e838635dabf050b528a57c1cc455aa0922041a8e0bc338aad6a9ad42419335b9`)
- own-method Gaussian protocol:
  `output/scannet_pointcloud_eval/vala_gaussian_protocol_20260711/scannet_vala_gaussian_protocol_results.json`
  (`71f81b987769d62d9508b96bbbef41e0b3842796334bbd6eed17b156f2726214`)

### Code and validation

- `radio_gs/scripts/eval_prerendered_lerf_features.py`
- `radio_gs/scripts/eval_occamlgs_lerf_checkpoint.py`
- `radio_gs/scripts/summarize_langsplatv2_lerf_audit.py`
- `reproductions/langsplatv2/run_lerf2d_exact_camera.py`
- `reproductions/langsplatv2/patches/0001-exact-label-camera-resolution.patch`
- `tests/test_eval_prerendered_lerf_features.py`
- `tests/test_langsplatv2_lerf_protocol_audit.py`

Validation on 2026-07-31:

```text
16 passed
py_compile passed for all four evaluation/reproduction scripts
```

## Paper/reporting rules after this audit

1. Never compare a locally aggregated value to a paper overall without naming
   and matching the aggregation.
2. For LangSplatV2, report mIoU scene-macro and LocAcc 208-query micro when
   comparing to the published overall; retain both macro and micro locally.
3. State that the corrected released LERF-2D path contains 15 train-role and 7
   test-role annotation frames. Do not call it pure held-out evaluation.
4. Call the Dr. Splat and VALA local rows “compatibility reproductions,” not
   official reproductions.
5. Label OccamLGS LERF-3D/ScanNet numbers as OpenGaFF-reported adapter rows.
6. Do not use the LERF-2D camera bug to explain direct-3D or ScanNet gaps.
7. Keep strict shape checks, explicit protocol profiles, exact filename
   resolution, checkpoint/config hashes, and fail-closed visibility checks in
   all future baseline evaluations.
