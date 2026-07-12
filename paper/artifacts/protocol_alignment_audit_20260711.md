# Three-track protocol alignment audit (2026-07-11)

This audit separates the restored historical GaussFM numbers from new runs
whose evaluators and readout parameters were frozen before looking at their
test metrics. The historical numbers remain in the paper-facing registry and
tables at the user's request, but they are **legacy post-hoc diagnostics**, not
strict same-protocol main results.

## Executive result

| Track | Restored historical GaussFM result | Frozen protocol-aligned result | Submission status |
|---|---:|---:|---|
| T1 LERF-OVS 2D | 58.89 mIoU / 85.98 LocAcc | **37.61 / 81.27** (`vala_paper_2d`) | Frozen result is valid; external rows remain published context |
| T2 LERF-OVS direct 3D | 50.14 mIoU / 70.44 Acc@0.25 | **0.12 / 0.00** (paper raw-0.6); **5.78 / 6.22** (released-code diagnostic) | Raw 0.6 is paper-level; released-code result is separately labelled diagnostic |
| T3 ScanNet 19/15/10 | 38.06/38.71/47.11 mIoU | **20.76/21.07/26.99** mIoU; **39.42/40.29/50.95** mAcc | VALA Gaussian-domain result is the comparable row |

All displayed values are percentages. T1/T2 are unweighted means of the four
scene-level metrics, not an instance-micro `OVERALL` value. T3 is an unweighted
mean over the same eight scenes used by the local VALA run script.

## Restored historical rows and their status

The following values are real locally generated numbers and have been restored
to `paper/lerf_ovs_main_table.tex`, `paper/lerf_direct_3d_selection_table.tex`,
`paper/scannet_published_context_table.tex`, and `paper/artifacts/final_rows.yaml`.
They are not fabricated, but “locally generated” is not the same as “eligible
for a frozen test-set main result.”

| Track | Figurines | Teatime | Ramen | Waldo Kitchen | Mean |
|---|---:|---:|---:|---:|---:|
| T1 2D mIoU / LocAcc | 52.43 / 82.14 | 65.15 / 89.83 | 63.25 / 90.14 | 54.75 / 81.82 | 58.89 / 85.98 |
| T2 3D mIoU / Acc@0.25 | 51.04 / 67.86 | 56.40 / 76.27 | 59.99 / 83.10 | 33.12 / 54.55 | 50.14 / 70.44 |

| T3 historical mesh-query diagnostic | 19 classes | 15 classes | 10 classes |
|---|---:|---:|---:|
| mIoU / mAcc | 38.06 / 61.29 | 38.71 / 63.15 | 47.11 / 72.00 |

Why these are post-hoc:

- T1 used scene-specific temperature sweeps, a threshold sweep on the same four
  evaluated scenes, peak-component cleanup, a feature-only SAM3 boundary head,
  and several acceptance guards. Its historical localization metric was polygon
  argmax, not the released 30-by-30-smoothed peak-in-bbox definition.
- T2 used a five-template prompt ensemble, an opacity-gated point-summary
  adapter, a globally selected threshold of 0.55 from the same benchmark sweep,
  a 0.5% selection floor, a 3% cap for Figurines and 5% caps elsewhere, RGB
  GrabCut, and a score-component guard. The restored row consistently reads the
  `thr0p55` result for every scene even though each JSON also contains a
  diagnostic `best_by_miou` field.
- T3 selected kNN/candidate-k/calibration/smoothing parameters using aggregate
  metrics on the same eight evaluated scenes, queried annotated mesh vertices,
  and applied scene-mean logit centering plus mesh-kNN propagation.

Inference-time code for these variants may be GT-free, but their reported
hyperparameters were selected using the test annotations. This is test-set
calibration and cannot be relabelled or omitted in a main-result claim. The rows
may only be retained as clearly marked post-hoc/legacy diagnostics unless their
parameters are reselected on disjoint validation data.

## T1: LERF-OVS 2D frozen paper-level protocol

Frozen contract:

- the four official LERF-OVS scenes, labelled views, text strings, polygons, and
  bboxes;
- one raw `{query}` prompt and four fixed generic negatives (`object`, `things`,
  `stuff`, `texture`);
- positive-vs-hardest-negative LERF relevance in FP32;
- binary mask `relevance > 0.5` at image resolution;
- localization from a 30-by-30 box-filtered relevance peak, correct when any
  tied peak is inside any GT instance bbox;
- no threshold sweep, confidence gate, component cleanup, RGB/SAM refinement,
  or test-scene calibration;
- per-instance mean inside each scene, followed by an unweighted four-scene
  macro.

| Scene | Samples | mIoU | LocAcc |
|---|---:|---:|---:|
| Figurines | 56 | 40.18 | 76.79 |
| Teatime | 59 | 47.18 | 83.05 |
| Ramen | 71 | 29.17 | 78.87 |
| Waldo Kitchen | 22 | 33.88 | 86.36 |
| **Scene macro** | 208 total | **37.61** | **81.27** |

The instance-micro localization value is 80.29%; it is not the paper-facing
scene macro. The frozen evaluator source is
`580d3419df13016397cb7d9ec1fc0a4969deca9d447d52f1dcacac0b9160bcb0`.
Result sources and SHA-256 digests:

```text
70b7ac2f3d96e959c641fb82c7f19c5a181cb539d97020936cca1238421188f4  output/radio_gs/lerf2d_vala_paper_frozen_seed42_v3_20260711/figurines/lerf_ovs_results.json
034b40870074f777e3049c12977ca3e93028a6ae75927ff7f83d6e2823efc2ac  output/radio_gs/lerf2d_vala_paper_frozen_seed42_v3_20260711/teatime/lerf_ovs_results.json
a40341c799b54fc825f8a344787c47bb88de99428a223fd4a425a18d053398ec  output/radio_gs/lerf2d_vala_paper_frozen_seed42_v3_20260711/ramen/lerf_ovs_results.json
ee438a76f9e816063d3a376479ff7d0f3052827541055828ef0f7475e630bd59  output/radio_gs/lerf2d_vala_paper_frozen_seed42_v3_20260711/waldo_kitchen/lerf_ovs_results.json
```

### Can CLIP and SigLIP use one 2D evaluator?

Yes at the task/evaluator layer: each method must use the text encoder paired
with its own visual feature space, while images, labels, prompts, negatives,
relevance definition, mask rule, localization rule, and aggregation remain
fixed. At threshold 0.5, positive-vs-hardest-negative relevance is equivalent
to `cos(query) > max cos(negative)`; the temperature cancels from this decision.
This makes the decision rule substantially more portable across CLIP and
SigLIP than a raw cosine threshold.

It is still not a representation-controlled ablation: VALA has three OpenCLIP
language levels and selects a level by a GT-free response, while GaussFM has one
SigLIP2 primitive level. This is a method difference that must be declared. A
strict representation-controlled comparison requires retraining GaussFM with
the same LangSplat CLIP+SAM teacher (supported below).

The frozen preset follows the VALA appendix's paper-level description. VALA's
released 2D code additionally blends a 30-by-30-smoothed relevance map,
per-image min-max remaps the result, applies a `2u-1` transform, and performs a
small majority cleanup; some scripts default to a mask threshold of 0.4. Those
published rows therefore remain context rather than same-evaluator rankings.

## T2: LERF-OVS direct 3D frozen protocols

Shared constraints for both new runs:

- raw `{query}` prompt, fixed four negatives, and direct per-Gaussian feature;
- no point-summary adapter, score cache, prompt ensemble, ratio floor/cap,
  selection sweep, GrabCut/SAM, or component guard;
- physically remove unselected primitives, render selected-only alpha, round to
  PNG uint8, and binarize at `>10` exactly as the released mask evaluator;
- fixed official labelled frames and unweighted four-scene macro.

The paper-level protocol applies raw relevance `>0.6`. The released-code
diagnostic additionally performs kNN-10 smoothing (including self), a 0.5
raw/neighbor blend, per-query scene min-max normalization, `clip(2u-1,0,1)`,
and then threshold 0.6.

| Scene | Paper raw-0.6 mIoU / Acc@0.25 | Released-code diagnostic mIoU / Acc@0.25 |
|---|---:|---:|
| Figurines | 0.00 / 0.00 | 1.25 / 0.00 |
| Teatime | 0.30 / 0.00 | 8.55 / 10.17 |
| Ramen | 0.18 / 0.00 | 5.09 / 5.63 |
| Waldo Kitchen | 0.00 / 0.00 | 8.24 / 9.09 |
| **Scene macro** | **0.12 / 0.00** | **5.78 / 6.22** |

Released-code diagnostic Acc@0.50 is 2.27%. In the paper-level run, every
Figurines and Waldo query selected zero primitives; only one query in each of
Ramen and Teatime selected any. In the released-code run every query selected
something, but the normalization often caused severe over-selection. This is a
representation/readout distribution failure, not a missing-sample evaluator
failure.

Unlike the 2D threshold 0.5 decision, raw 3D relevance `>0.6` requires a
positive-vs-hardest-negative cosine margin greater than
`log(0.6/0.4)/10 = 0.04055`. That numerical margin is not calibrated across
OpenCLIP and SigLIP feature spaces. It must not be tuned on the LERF test masks.
Valid options are to use the same CLIP+SAM representation, select one global
threshold on disjoint validation scenes, or report the fixed released-code
normalization as a separately labelled diagnostic.

Final result digests:

The frozen direct-3D evaluator source is
`c49334631973221b7df3a0fb06cbb8cd82a540f270177a943617309e0f50c2e7`.

```text
bfe63f9b701a40b4e69c02c8ac6bcf8ed95f210dd24fcebbb815d17edba207db  output/radio_gs/lerf3d_vala_paper_frozen_seed42_v2_20260711/figurines/lerf_direct_3d_selection_results.json
e67170d74ef71f57555eb157d4196630edb439acd55f9e1e5191deca2fc7bab0  output/radio_gs/lerf3d_vala_paper_frozen_seed42_v2_20260711/teatime/lerf_direct_3d_selection_results.json
997bb1cc288ea2bc04460e6fab60679ea00bdbe040cfb7b1be7d928fba15cb6c  output/radio_gs/lerf3d_vala_paper_frozen_seed42_v2_20260711/ramen/lerf_direct_3d_selection_results.json
3d6859cc278f271d1c74a165635f196c4e599b8b82c6fcb4957a9db2e9427f85  output/radio_gs/lerf3d_vala_paper_frozen_seed42_v2_20260711/waldo_kitchen/lerf_direct_3d_selection_results.json
d8dc0574c5bcc3589d51df0d1433f4d69a4aec347d31c9a8a1564c5ed7d674f8  output/radio_gs/lerf3d_vala_repo_frozen_seed42_v2_20260711/figurines/lerf_direct_3d_selection_results.json
848421af0d80a69f1ecf63b615fbf6d981bfd52b50311f08446a54803cc8acbf  output/radio_gs/lerf3d_vala_repo_frozen_seed42_v2_20260711/teatime/lerf_direct_3d_selection_results.json
f3bcaff8fc8d69b1e2e200167df99ce4a6e3721809c646f27105c33f46cf0086  output/radio_gs/lerf3d_vala_repo_frozen_seed42_v2_20260711/ramen/lerf_direct_3d_selection_results.json
ba552534b07ba59d9301b0cef1485927beff93b80c80dd4df820d054cf45a6b5  output/radio_gs/lerf3d_vala_repo_frozen_seed42_v2_20260711/waldo_kitchen/lerf_direct_3d_selection_results.json
```

## T3: ScanNet VALA Gaussian-domain protocol

The historical and aligned protocols differ in all of the following, not only
in how ground truth is obtained:

| Component | Historical T3 | Frozen VALA Gaussian-domain run |
|---|---|---|
| Prediction/query support | annotated mesh/label-Ply vertices | optimized Gaussian centers |
| Feature query | kNN-16 from 80 candidate Gaussians | direct feature of each Gaussian |
| Test-scene calibration | scene-mean centering, alpha 0.45 | none |
| Spatial postprocess | mesh kNN-12 propagation, alpha 1.0 | none |
| GT attachment | original annotated point row | anisotropic Mahalanobis-density pseudo label at optimized center; radius factor 5, top-k 1000, class-balanced vote, nearest fallback |
| Opacity/size | hard opacity filter 0.1 | continuous `sigmoid(opacity)*sx*sy*sz` significance |
| Metric measure | unweighted mesh points | significance-weighted Gaussians |
| Class averaging | classes present in each scene | classes present in each scene |
| Scene averaging | unweighted scene macro | unweighted scene macro |

| Evaluation | 19 mIoU / mAcc | 15 mIoU / mAcc | 10 mIoU / mAcc |
|---|---:|---:|---:|
| Historical mesh-query diagnostic | 38.06 / 61.29 | 38.71 / 63.15 | 47.11 / 72.00 |
| Clean Gaussian-center row-aligned diagnostic | 22.63 / 40.72 | 22.91 / 41.50 | 29.06 / 51.90 |
| **VALA Gaussian-domain protocol** | **20.76 / 39.42** | **21.07 / 40.29** | **26.99 / 50.95** |

The complete per-scene record and implementation cross-check are in
`paper/artifacts/protocol_alignment_scannet_20260711.md`.
The frozen ScanNet evaluator source is
`52dce17e1ebff53ce3bf6b3f7e41283a2e094590bb274747485fae30567b0dcd`.

```text
71f81b987769d62d9508b96bbbef41e0b3842796334bbd6eed17b156f2726214  output/scannet_pointcloud_eval/vala_gaussian_protocol_20260711/scannet_vala_gaussian_protocol_results.json
```

The evaluated v67 checkpoints do not have an active semantic-GT label loss:
their direct text/adapter label-loss weights are zero and active pseudo targets
come from the teacher. They do query row-aligned label-Ply coordinates during
training, while the aligned evaluation queries optimized Gaussian centers; this
remaining geometry-domain dependency is disclosed rather than called semantic
label leakage.

## What the historical GrabCut row actually does

The restored T2 row uses `rgb_grabcut_score_component_guard` with two GrabCut
iterations, dilation radius 5, erosion radius 2, score-mass fraction 0.5, at
most two kept components, and a 6000-pixel small-support guard:

1. Start from the projected coarse predicted mask.
2. Dilate it with an elliptical radius-5 kernel to define admissible/probable
   foreground support.
3. Erode it with an elliptical radius-2 kernel to define sure foreground; fall
   back to the coarse mask if erosion is empty.
4. Mark pixels outside the dilated support as definite background and run
   OpenCV GrabCut on the evaluation-view RGB with `GC_INIT_WITH_MASK` for two
   iterations.
5. Keep GrabCut foreground/probable-foreground only inside the dilated support.
6. Rank 8-connected components by normalized compact-score heatmap mass; keep
   components with at least half the top mass, capped at two. If total support
   is under 6000 pixels, keep only the top component.

No GT mask is read during this inference step. Nevertheless, the fixed values
and the surrounding selector were selected on the benchmark metrics. More
fundamentally, GrabCut consumes RGB and produces a separate view-dependent 2D
mask after 3D selection, whereas the direct-3D protocol evaluates the alpha
projection of one selected primitive set. It is valid only as a clearly named
`GaussFM + RGB GrabCut` hybrid/ablation (or if the same refinement is applied to
all methods), not as the sole pure direct-3D main result.

## LangSplat CLIP+SAM feature replacement

The code now supports the replacement mechanically, but it is not a drop-in
swap for an existing 1280-D RADIO/SigLIP checkpoint:

- `radio_gs/scripts/convert_langsplat_language_features.py` materializes
  LangSplat `*_f.npy` prototypes and `*_s.npy` segment maps as finite,
  normalized dense 512-D tensors.
- `radio_gs/scripts/generate_samclip_ablation_configs.py` creates a 512-D
  GaussFM training configuration and disables RADIO/SigLIP-specific helpers.
- `samclip_feature_level` and `samclip_language_feature_dir` are declared in
  `radio_gs/config.py`, so config loading no longer silently drops them.
- both 2D and direct-3D evaluators support native OpenCLIP identity readout with
  paired OpenCLIP text embeddings and frozen cache metadata checks.

The Figurines proof checkpoint has a strict zero-mismatch load contract and a
validation feature cosine of 0.7640. Under the frozen paper-level evaluator,
however, it reaches only **0.066 mIoU / 0 LocAcc**, while the raw LangSplat
level-1 teacher reaches **19.93 mIoU / 58.93 LocAcc** on the same 56 labelled
queries. On the four labelled frames, teacher/rendered feature cosine is still
0.8007, but the fraction of GT pixels with a positive query-minus-hardest-
negative margin falls from 36.13% to 4.70% (margin-sign agreement 67.67%). High
reconstruction cosine therefore did not preserve the small margins required by
open-vocabulary readout. The path is implemented and trainable, but it is not
yet a competitive replacement result. Existing RADIO checkpoints cannot be
reused; the field must be retrained in the 512-D CLIP space.

Final SAM-CLIP diagnostic digests:

```text
7106d890091fa630e706deca18beaecb086afa2a20be442fedba54c97b9d2486  output/radio_gs/lerf2d_samclip_l1_clean_vala_paper_2d_final_v3_20260711/lerf_ovs_results.json
8899eb156a19dcb3cda1731b7bad3f9a61d15a223f73f755827d78da6f1aa78d  output/radio_gs/lerf2d_samclip_l1_teacher_vala_paper_2d_final_v3_diagnostic_20260711/lerf_ovs_results.json
9a093959f1b0a35317fe741618e686d3df0bb3a2164b1f29111caffe7c556092  output/radio_gs/lerf2d_samclip_l1_clean_vala_paper_2d_final_v2_20260711/teacher_rendered_margin_diagnostic.json
```

The original 299 prototype/segment pairs remain available at
`/mnt/pool/sqy/3d_understanding/lerf_ovs/figurines/langsplat/language_features`.
The derived level-1 FP16 cache has been rebuilt at 46-by-62 resolution under
`output/samclip_features_lerf/figurines/l1`: all 299 tensors are finite,
512-dimensional, and nonzero pixels remain unit-normalized. Its manifest hash
is `7e6f663ca8b8194d24b81bda9aef4f21c4783f6e42e771b1c3c3158af941a33b`.

## Why local baseline reproductions can be much worse than copied tables

The evidence indicates both evaluator and reproduction/asset problems:

- OpenGaFF explicitly uses original-paper numbers when available; its table is
  not a claim that every row was retrained with one evaluator. Its LERF table
  also differs from VALA's table for some nominally identical rows.
- several local rows are compatibility adaptations rather than released
  checkpoint reproductions: upstream pretrained/preprocessed assets are absent,
  upstream evaluation is missing for some methods, and local ABI/data/split
  patches were required;
- CAGS has 34 missing rendered masks counted as failures; LangSplatV2's local
  Teatime run collapses while the other scenes are much stronger, pointing to a
  scene asset/training failure rather than one global metric bug;
- the prior GaussFM evaluators could silently reuse a five-template embedding
  cache for a raw-query run and could load incompatible checkpoint components
  with `strict=False`; both now fail closed;
- the 2D OpenCLIP path also had an FP16 zero-vector normalization bug that could
  create NaNs and zero localization scores; it is now FP32, finite-checked, and
  covered by regression tests.

After those evaluator bugs were fixed, the strict GaussFM scores remained low,
especially for direct 3D and SAM-CLIP. The remaining gap is principally a
feature/readout distribution problem exposed when post-hoc support heuristics
are removed, not evidence that the final frozen evaluator is still skipping
data.

## Reporting decision

The restored historical values are preserved, but the current paper is not
submission-safe while they are promoted or stated in the abstract without a
post-hoc qualifier. A defensible submission should use the frozen T1 and T3
rows, treat both T2 frozen variants as negative/diagnostic evidence until a
validation-frozen readout is developed, and retain the historical rows only in
an explicitly labelled legacy/readout-policy ablation.
