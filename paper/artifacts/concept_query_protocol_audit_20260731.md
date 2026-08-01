# Concept Query evaluation protocol audit — updated 2026-08-01

## Outcome

This audit separates three evaluation domains that had previously been treated
too similarly and now closes all three requested baseline checks:

1. **LERF-2D rendered-view grounding.** The strict streamed OccamLGS readout
   reaches **63.62 mIoU / 82.85 LocAcc** versus the paper's **61.3/82.5**.
   It uses exact annotation camera names, all three iteration-30000 language
   checkpoints, raw-peak level selection, threshold 0.5, the released 30x30
   OpenCV activation filter, 7x7 smoothing, and a scene-equal headline mean.
   The prior LangSplatV2 diagnostic remains useful provenance, but it is no
   longer the preferred closure row.
2. **LERF direct 3D selection.** Running VALA's released three-level semantic
   lifting and direct-3D evaluator on compatible Occam RGB geometry reaches
   **54.12 mIoU / 79.35 Acc@0.25** versus **43.29/64.30**. The old
   **32.53/50.43** Dr. Splat compatibility result used fixed L3, threshold 0.4,
   and compatibility masks, so it was not an implementation of the reported
   VALA protocol. A second silent hazard was found: VALA requires extensionless
   `test.txt` stems; copying Occam's `.jpg` entries leaks the labelled views
   into semantic lifting.
3. **ScanNet OVS.** The exact VALA Gaussian protocol on the paper's eight
   scenes reaches **34.53/51.59**, **37.96/56.77**, and **47.36/67.47** for
   the 19/15/10-class splits, at or above the paper's **32.11/50.05**,
   **35.10/54.77**, and **46.21/65.61**. Full resolution, exact gsplat
   marginal-contribution significance, and robust aggregation explain most of
   the old gap; Gaussian pseudo-GT is mandatory but contributes only about
   0.5--1.8 mIoU on the old cached features. The paper uses eight scenes; the
   current repository's ninth `scene0645_00` is reported only as sensitivity.

OccamLGS is therefore the closed LERF-2D baseline. VALA is the closed semantic
pipeline/protocol baseline for LERF direct 3D and ScanNet. The direct-3D run
reuses Occam RGB geometry and is not a fresh end-to-end VALA geometry
reproduction; the ScanNet run similarly isolates released semantic lifting on
available local RGB Gaussian geometry. Both are sufficient to diagnose the
evaluation-protocol discrepancies because they already reproduce or exceed
the paper numbers.

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
| LERF-2D | OccamLGS | Complete released-code-intent four-scene checkpoint reproduction | 63.62 / 82.85 | 61.30 / 82.50 | +2.32 / +0.35 | Closed; paper-level reproduction |
| LERF direct 3D | VALA semantic pipeline on compatible Occam RGB geometry | Complete four-scene exact semantic/evaluator protocol; RGB provenance compatibility-level | 54.12 / 79.35 | 43.29 / 64.30 | +10.83 / +15.05 | Protocol gap closed; fresh VALA RGB training optional for end-to-end provenance only |
| ScanNet, 19 classes | VALA | Exact released semantic/Gaussian protocol on `paper8`; local RGB geometry | 34.53 / 51.59 | 32.11 / 50.05 | +2.42 / +1.54 | Closed |
| ScanNet, 15 classes | VALA | Same | 37.96 / 56.77 | 35.10 / 54.77 | +2.86 / +2.00 | Closed |
| ScanNet, 10 classes | VALA | Same | 47.36 / 67.47 | 46.21 / 65.61 | +1.15 / +1.86 | Closed |

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

The printed LangSplatV2 “overall” row **numerically matches mixed
aggregation**:

- published mIoU `0.599` matches the equal mean of the four rounded scene
  mIoUs (`0.59875`);
- published LocAcc `0.841` matches the 208-query weighted mean of the rounded
  scene rows (`0.841399`), not their equal scene mean (`0.86375`).

This is a **LangSplatV2-row-specific numerical match**, not a formal
benchmark-wide rule. In the same source-context table, every other listed 2D
baseline Acc mean matches its four-scene macro: LangSplat `84.3000`, GAGS
`81.6575`, OccamLGS `82.5250`, GOI `59.2250`, and GALA `73.4275`. The table is
therefore aggregation-heterogeneous. A cached copy of the source paper's
evaluator prose was not available in this audit, so the arithmetic must not be
upgraded into a universal protocol claim.

Consequently, the only matched headline comparison is local scene-macro mIoU
against published mIoU, and local query-micro LocAcc against published
LocAcc: **+1.61 / -4.29 pp**. Both local macro and micro values remain in the
artifact so no alternative aggregation is hidden.

This comparison diagnoses whether the released evaluator behavior and the
local checkpoint cohort are numerically plausible. The three-level
checkpoints were trained from local Occam-compatible RGB starts and
compatibility language features, not loaded from an official LangSplatV2
pretrained release. More importantly, released LangSplatV2 `train.sh` omits
`--eval` and `ModelParams` defaults to `eval=False`, so the released training
command uses all cameras. All twelve local checkpoint configs instead record
`eval=True`, removing every eighth camera from language-field training. The
audit checkout also contains training-side loss/VQ compatibility edits and
does not establish clean paper-checkpoint identity. Therefore the paper delta
must not be presented as a strict official-checkpoint reproduction; checkpoint,
training split, RGB start, and language-feature provenance are the dominant
remaining mismatch class.

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

Thus, this is a **mixed-role released-code-intent/local-checkpoint diagnostic**,
not a pure held-out test-view protocol. These roles describe the local
`eval=True` checkpoint split; they do not describe released LangSplatV2
training, whose provided command uses `eval=False`. At query granularity, 134
of 208 units are on local train-role labelled views and 74 are on locally
withheld labelled views. The official OccamLGS `run_lerf.sh` likewise trains
RGB geometry without `--eval` and only passes `--eval` during later feature
extraction/rendering, so “held out” cannot be inferred from an evaluation flag
alone.

For future paper-facing runs, choose and name one of three protocols explicitly:

1. `released_train_all_view_exact_name`: follow released LangSplatV2 training
   with `eval=False`, then resolve every label by exact camera name;
2. `local_eval_true_mixed_role_exact_name`: the current compatibility
   diagnostic, disclosing 15/7 labelled frame roles and 134/74 query roles; or
3. `strict_test_only`: train geometry with the test split withheld and reject
   every train-role annotation. This is a separate diagnostic protocol and is
   not numerically interchangeable with the released row.

### Final CPU-only LocAcc gap audit

The final static replay validates 228 raw annotation objects/bboxes, including
20 repeated same-category instances merged by the released loader, yielding
the exact 208 frame-query denominator. Query order matches annotation
first-occurrence order, all bbox coordinates are within the labelled image
dimensions, and all 22 camera stems resolve exactly.

| Scene | Local hits | Nearest hits implied by printed row | Approximate deficit |
|---|---:|---:|---:|
| Figurines | 46/56 | 46/56 | 0 |
| Teatime | 52/59 | 55/59 | 3 |
| Ramen | 51/71 | nearest 53/71 | about 2 |
| Waldo Kitchen | 17/22 | 21/22 | 4 |
| **Total** | **166/208** | **nearest 175/208** | **about 9** |

Ramen `74.7%` cannot be produced exactly by an integer numerator over 71
within half of its printed 0.1-point unit, so nine hits is a nearest-integer
reconstruction rather than an exact paper-side receipt.

Pinned-source checks confirm that localization uses exact annotation prompts,
OpenCLIP ViT-B-16 `laion2b_s34b_b88k`, effective top-k 4, a 29x29 average
pool, peak-based level selection, all maximum-score ties, and inclusive merged
bboxes. The mask threshold is not read by localization and cannot change
LocAcc. The exact-camera run stores no rendered feature/relevance maps,
localization levels, peak coordinates, or per-query hit bits, so no honest
CPU-only top-k/level/tie/bbox/camera-role counterfactual remains.

The machine receipt is
`output/protocol_audit_20260801/langsplatv2_lerf2d_final_gap_audit.json`
(SHA-256
`0bda52d7f8598c731817d2a2e69e62a043e5ab2f16de91840ccecbd11c4543d2`).
Its decision is `strict_paper_reproduction=false`; the most likely remaining
gap class is checkpoint/training/feature provenance, not a known evaluation
protocol error.

### Readout profiles now frozen in code

| Profile | Feature | Mask threshold | Activation filter | Mask smoothing | Shape mismatch |
|---|---|---:|---|---|---|
| `langsplatv2_released` | normalized | 0.4 | 29×29 Torch average pool | 7×7 strict majority | error |
| `occam_langsplat_paper` | raw | 0.5 | 30×30 OpenCV `filter2D` | exact legacy LangSplat 7×7 edge/tie behavior | error |

Resizing is available only through the explicitly named
`bilinear_compat` policy. Strict profiles never resize silently.
The mask threshold and 7x7 smoothing in this table affect segmentation mIoU,
not the independent localization path.

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

The final fail-closed artifact validates 208/208 filename-matched predictions,
zero missing masks, one fixed `feature_level=3` contract across all scenes,
and an equal-scene macro for the paper-context delta. The historical L1 cohort
is retained only as a paired scale diagnostic.

| Scene | Queries | L1 mIoU / A25 | L3 mIoU / A25 | L3 - L1 | Paper context | L3 - paper |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 56 | 9.08 / 14.29 | 32.83 / 48.21 | +23.75 / +33.93 | 54.42 / 80.36 | -21.59 / -32.15 |
| Teatime | 59 | 19.78 / 28.81 | 46.26 / 74.58 | +26.48 / +45.76 | 57.35 / 77.97 | -11.09 / -3.39 |
| Ramen | 71 | 21.49 / 36.62 | 21.84 / 38.03 | +0.36 / +1.41 | 24.33 / 35.21 | -2.49 / +2.82 |
| Waldo Kitchen | 22 | 20.12 / 22.73 | 29.19 / 40.91 | +9.07 / +18.18 | 37.05 / 63.64 | -7.86 / -22.73 |
| **Scene-equal macro** | — | **17.6173 / 25.6116** | **32.5317 / 50.4320** | **+14.9144 / +24.8204** | **43.29 / 64.30** | **-10.7583 / -13.8680** |

For the secondary accuracy threshold, scene-macro Acc@0.50 moves
from `11.3670` at L1 to `30.8340` at L3 (`+19.4670 pp`). The separate L3
208-query micro is `32.5051 mIoU / 51.4423 Acc@0.25 / 29.3269 Acc@0.50`;
it must not be compared to the paper's scene-equal headline.

The paired response is strongly scene dependent. Teatime and Figurines drive
most of the improvement, Waldo Kitchen improves moderately, and Ramen changes
only `+0.36/+1.41 pp` while its Acc@0.50 falls by `4.23 pp`. Feature scale is
therefore a **major corrected protocol factor, but not a sufficient explanation
or an official-paper reproduction**.

The final row remains a scale-paired **compatibility reproduction** with
`strict_checkpoint_reproduction=false` and
`paper_comparison=diagnostic_only`, because:

- the official Dr. Splat README still marks quantitative evaluation as `TBA`;
- all four L3 models start from local OccamLGS-compatible RGB checkpoints, not
  released Dr. Splat pretrained checkpoints;
- all scenes fix `feature_level=3`, `topk=45`, and the same PQ index, but masks
  come from the local VALA single-checkpoint adapter at threshold 0.4;
- the common nested-mask evaluator uses PNG threshold 10 and strict
  `IoU > 0.25` / `IoU > 0.5` comparisons, not an official released evaluator;
- the `43.29/64.30` values are OpenGaFF-reported paper context, not an
  independently verified official Dr. Splat evaluation receipt.

The direct-3D evaluator reads predictions already keyed by
`frame/query` filenames. It never chooses a render camera by an integer list
index. Therefore the LERF-2D camera mapping defect cannot explain this gap.

### Superseding VALA direct-3D protocol reproduction

The Dr. Splat L3 diagnostic above is retained as an ablation, but it is
superseded for protocol diagnosis by VALA's released semantic pipeline:

| Scene | mIoU | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|
| Figurines | 58.35 | 87.50 | 62.50 |
| Ramen | 44.03 | 66.20 | 45.07 |
| Teatime | 69.44 | 86.44 | 77.97 |
| Waldo Kitchen | 44.68 | 77.27 | 40.91 |
| **Scene-equal macro** | **54.12** | **79.35** | **56.61** |

This run uses exact extensionless label-frame holdouts, official
marginal-contribution significance, stochastic robust aggregation at all three
levels, per-query raw-peak level selection, KNN-10 smoothing, threshold 0.6,
and selected-only alpha rendering. It reuses compatible Occam RGB geometry,
so its end-to-end asset provenance is not strict VALA. Nevertheless it exceeds
the reported 43.29/64.30 and establishes that the old 32.53/50.43 result was a
protocol/adapter limitation rather than a VALA method limitation.

The exact view splits are 295/4, 124/7, 171/6, and 182/5 train/test for
Figurines, Ramen, Teatime, and Waldo Kitchen. VALA strips extensions from
COLMAP image names before `test.txt` lookup; suffix-bearing `.jpg` entries
match no camera and are now rejected by
`radio_gs/scripts/audit_vala_lerf3d_split.py`.

## ScanNet: exact VALA Gaussian protocol result

| Split | Local mIoU | Reported VALA mIoU | Gap | Local mAcc | Reported VALA mAcc | Gap |
|---|---:|---:|---:|---:|---:|---:|
| 19 classes | 34.53 | 32.11 | +2.42 | 51.59 | 50.05 | +1.54 |
| 15 classes | 37.96 | 35.10 | +2.86 | 56.77 | 54.77 | +2.00 |
| 10 classes | 47.36 | 46.21 | +1.15 | 67.47 | 65.61 | +1.86 |

The paper states eight scenes. The aligned `paper8` cohort is
`0000,0062,0070,0097,0140,0347,0400,0590`; the current repository's ninth
`0645` scene is later code sensitivity. Adding it changes the means to
34.04/51.45, 37.66/56.77, and 46.22/66.36 and does not alter the conclusion.

The exact run uses full-resolution views, the official gsplat
`alpha * transmittance` marginal contribution at each projected Gaussian
center, released robust aggregation/gating, fixed-vocabulary CLIP text argmax,
anisotropic Gaussian pseudo-GT, opacity-volume weighting, present-GT classes,
and scene-equal aggregation. Re-evaluating the old resolution-2/proxy features
with exact pseudo-GT yields only 12.48/24.07, 16.09/28.87, and 24.51/43.12.
Thus Gaussian pseudo-GT is mandatory but secondary; resolution, exact
significance, and robust lifting are the dominant fixes.

### Historical two-scene P0 metric-domain isolation

This experiment predates the exact eight-scene reproduction above. It is kept
only to quantify why changing the final metric domain alone could not repair
the old proxy-feature cache; its earlier “unresolved” language is superseded
by the full-resolution official-significance result.

A CPU-only paired replay on `scene0000_00` and `scene0400_00` holds the local
checkpoint, cached 512-D Gaussian features, exact upstream text cache, class
splits, and scene aggregation fixed. It changes only the readout domain:
anisotropic pseudo-GT plus `sigmoid(raw_opacity) * product(exp(raw_scale))`
volume weighting on cached pruned Gaussian centers, versus the archived
inverse-distance mesh-kNN4 unweighted result.

| Split | Gaussian-domain mIoU / mAcc | Mesh-domain mIoU / mAcc | Gaussian - mesh |
|---|---:|---:|---:|
| 19 | 12.5795 / 25.1028 | 11.1973 / 24.0377 | +1.3823 / +1.0651 |
| 15 | 13.5702 / 28.8513 | 12.4341 / 28.0875 | +1.1362 / +0.7638 |
| 10 | **16.3307 / 34.3724** | **14.4578 / 33.5118** | **+1.8729 / +0.8606** |

The split-10 improvement is therefore only about **+1.87 mIoU / +0.86 mAcc
points**. On this prespecified two-scene P0, converting the final metric domain
is a secondary effect, not an explanation for the roughly 20--24-point VALA
compatibility mIoU gap. The principal unresolved causes converge to the local
checkpoint and 2D-feature provenance, followed by the compatibility pipeline's
`opacity * sqrt(radius)` render-significance proxy and its downstream robust
aggregation, gating, and pruning behavior.

This result is strictly a **compatibility sensitivity**. It is neither an
eight-scene result nor an official-checkpoint reproduction, and it is forbidden
from carrying a paper delta. The immutable JSON is
`output/protocol_audit_20260801/vala/scannet_cached_vala_gaussian_domain_p0/scannet_cached_vala_gaussian_domain_p0.json`
(SHA-256
`3ea3e2a554e1fe639e04c45f7aafd505ea86dfe95b526a63c487192733ea7c04`).

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

The official LERF-2D path is now reproduced on all four scenes. Directly
streaming the three iteration-30000 language checkpoints gives:

| Scene | mIoU | LocAcc |
|---|---:|---:|
| Figurines | 61.12 | 78.57 |
| Ramen | 59.60 | 73.24 |
| Teatime | 72.92 | 93.22 |
| Waldo Kitchen | 60.84 | 86.36 |
| **Scene-equal macro** | **63.62** | **82.85** |

The paper result is 61.3/82.5. RGB geometry follows the released all-view
LERF intent while semantic feature lifting excludes the released test frames.
The reproduction is therefore paper-level numerically, with local
checkpoint/feature provenance disclosed rather than claimed as an upstream
signed pretrained bundle.

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
- VALA exact-protocol worktree: `48902a541333d65aeb0aebf64ad664777a27c3fc`
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
- LangSplatV2 final CPU-only gap receipt:
  `output/protocol_audit_20260801/langsplatv2_lerf2d_final_gap_audit.json`
  (`0bda52d7f8598c731817d2a2e69e62a043e5ab2f16de91840ccecbd11c4543d2`)
- final gap interpretation:
  `paper/artifacts/langsplatv2_lerf2d_final_gap_audit_20260801.md`
- OccamLGS strict LERF-2D reproduction:
  `paper/artifacts/occamlgs_lerf2d_strict_reproduction_20260801.md`
- VALA LERF direct-3D protocol reproduction:
  `paper/artifacts/vala_lerf3d_protocol_reproduction_20260801.md`
- VALA ScanNet exact-protocol reproduction:
  `paper/artifacts/vala_scannet_exact_protocol_reproduction_20260801.md`
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
  - historical L1 scale diagnostic:
    `paper/artifacts/drsplat_lerf_summary.json`
    (`9196611f5cdf24edfbd803e74f7a6164338eb4825e849b93ca77ffd15385c8cd`)
  - final four-scene L3 scale-paired receipt:
    `output/baselines/dr_splat/lerf_protocol_audit_20260801/remaining_l3_paired_20260801_a1/drsplat_l3_scale_paired_four_scene_summary.json`
    (`39c849e45ee4fcca53fd5977ddfc37d15917ef954e47fe654d012c349cc39aca`)
- VALA compatibility:
  `output/baselines/vala/scannet_vala8_compat_20260611_res2/vala_scannet_vala8_results.json`
  (`e838635dabf050b528a57c1cc455aa0922041a8e0bc338aad6a9ad42419335b9`)
- own-method Gaussian protocol:
  `output/scannet_pointcloud_eval/vala_gaussian_protocol_20260711/scannet_vala_gaussian_protocol_results.json`
  (`71f81b987769d62d9508b96bbbef41e0b3842796334bbd6eed17b156f2726214`)

### Code and validation

- `radio_gs/scripts/eval_prerendered_lerf_features.py`
- `radio_gs/scripts/eval_occamlgs_lerf_checkpoint.py`
- `radio_gs/scripts/audit_vala_lerf3d_split.py`
- `radio_gs/scripts/eval_vala_scannet_checkpoint_gaussian_protocol.py`
- `radio_gs/scripts/run_vala_scannet_baseline.py`
- `radio_gs/scripts/summarize_langsplatv2_lerf_audit.py`
- `radio_gs/scripts/audit_langsplatv2_lerf2d_final_gap.py`
- `reproductions/langsplatv2/run_lerf2d_exact_camera.py`
- `reproductions/langsplatv2/patches/0001-exact-label-camera-resolution.patch`
- `tests/test_eval_prerendered_lerf_features.py`
- `tests/test_audit_vala_lerf3d_split.py`
- `tests/test_eval_vala_scannet_checkpoint_gaussian_protocol.py`
- `tests/test_run_vala_scannet_baseline.py`
- `tests/test_langsplatv2_lerf_protocol_audit.py`

Validation on 2026-07-31:

```text
16 passed
py_compile passed for all four evaluation/reproduction scripts
```

Final CPU-only gap validation on 2026-08-01:

```text
7 passed in tests/test_langsplatv2_lerf_protocol_audit.py
25-row registry validation and targeted concept-query diff-check passed
```

Final Dr. Splat L3 scale-paired closeout validation on 2026-08-01:

```text
12 passed in tests/test_evaluation_protocol_registry.py and
tests/test_aggregate_drsplat_lerf_scale_audit.py
25-row registry validation, receipt SHA-256 check, and targeted diff-check passed
```

## Paper/reporting rules after this audit

1. Never compare a locally aggregated value to a paper overall without naming
   and matching the aggregation.
2. For LangSplatV2 only, report scene-macro mIoU and 208-query-micro LocAcc
   when numerically matching its printed overall. Call this a row-specific
   numerical mixed match and disclose that the source table is aggregation-
   heterogeneous; do not present it as a benchmark-wide formal rule.
3. State that the current local path contains 15 train-role and 7 test-role
   annotation frames (134/74 queries), while released LangSplatV2 training
   defaults to all-view `eval=False`. Keep the local row strict-ineligible and
   name training/checkpoint/feature provenance as the primary remaining gap.
4. Keep the old Dr. Splat 32.53/50.43 row diagnostic-only. For the VALA
   direct-3D protocol comparison, use 54.12/79.35 and disclose that semantic
   lifting/evaluation are exact while RGB geometry is compatible Occam
   geometry rather than a fresh VALA training run.
5. Use the OccamLGS LERF-2D scene macro 63.62/82.85 as the closed reproduction.
   Continue labelling unrelated OccamLGS LERF-3D/ScanNet table values as
   OpenGaFF-reported adapter rows.
6. For ScanNet VALA comparisons, use `paper8` and the exact Gaussian-domain
   values 34.53/51.59, 37.96/56.77, and 47.36/67.47. Keep `code9` separate.
7. Require extensionless VALA LERF `test.txt` stems and verify the exact
   annotation-frame set before any semantic lifting.
8. Do not use the LERF-2D camera bug to explain direct-3D or ScanNet gaps.
9. Keep strict shape checks, explicit protocol profiles, exact filename
   resolution, checkpoint/config hashes, and fail-closed visibility checks in
   all future baseline evaluations.
10. Do not claim that mask-threshold tuning can repair LangSplatV2 LocAcc; the
   released localization function does not read the mask threshold.
