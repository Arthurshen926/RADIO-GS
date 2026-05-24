# Direct Field Joint 2D/3D Optimization Audit

Date: 2026-05-24

## Goal

Strengthen the paper claim that one compact foundation-feature Gaussian map can
support both rendered 2D queries and direct 3D primitive/point queries.

The implemented upgrade adds two training-time constraints to the existing
direct-point loss family:

1. **Joint rendered/direct consistency.** Directly queried compact point features
   are aligned with the same compact field sampled from the current rendered
   feature map at visible point projections.
2. **Visibility-weighted text contrast.** The direct text contrast loss can now
   use registration/view-count weights not only as anchor weights but also as
   pair weights, reducing the effect of low-confidence positives and negatives.

These changes keep the existing method framing: VPR remains a label-free
multiview teacher/registration source, while the compact field remains the
student representation used by both rendered and direct readouts.

Follow-up diagnostics showed that v68's render-consistency term was limited by
scene-global cached-teacher sampling. The promoted v70 variant therefore adds
cached-visible direct-point sampling: when a cached teacher and rendered compact
map are both available, a configurable fraction of direct samples is drawn from
primitives visible in the current training views.

## Code Changes

- `radio_gs/training/feature_supervision_mixin.py`
  - Added `rendered_compact` support to `_compute_direct_point_loss`.
  - Added `_compute_direct_point_render_consistency`.
  - Added pair-level `direct_point_text_contrast_pair_weighting=visibility`.
  - Added cached-visible direct-point sampling for teacher-cache supervision.
- `radio_gs/scripts/train_feature_field.py`
  - Passes the current rendered compact map into direct-point supervision.
  - Logs `direct_point_render_consistency` and its valid ratio.
- `radio_gs/config.py`
  - Added config fields:
    - `direct_point_render_consistency_weight`
    - `direct_point_render_consistency_mode`
    - `direct_point_text_contrast_pair_weighting`
    - `direct_point_text_contrast_max_points`
    - `direct_point_text_contrast_center_logits`
    - `direct_point_cached_visible_fraction`
    - `direct_point_cached_visible_candidate_multiplier`
    - `direct_point_cached_visible_balance`
- `radio_gs/scripts/generate_scannet_dino_cv_configs.py`
  - Generated VALA/OpenGaFF-8 configs for the cached-visible joint2D3D
    variant.
  - Exposes visible-candidate oversampling and teacher-balanced visible replay.
- `radio_gs/scripts/build_scannet_vala8_report.py`
  - Adds category-macro stability to VALA/OpenGaFF-8 reports: per-class
    cross-scene mean/std/min/max IoU and accuracy, plus worst/unstable class
    summaries.

## Generated VALA8 Configs

Variant:

`v70_cachedvisible_vc005_rc005_b2_s32768_ft20`

Configs:

- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_scene0000_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_scene0062_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_scene0070_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_scene0097_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_scene0140_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_scene0347_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_scene0400_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_scene0590_00.yaml`

Main new settings:

```yaml
direct_point_view_count_weighting: clipped_log
direct_point_view_count_min_weight: 0.25
direct_point_text_contrast_weight: 0.05
direct_point_text_contrast_pair_weighting: visibility
direct_point_text_contrast_max_points: 4096
direct_point_text_contrast_center_logits: false
direct_point_render_consistency_weight: 0.05
direct_point_render_consistency_mode: cosine
direct_point_cached_visible_fraction: 0.5
```

## Stability Analysis Tool

Added:

`radio_gs/scripts/build_lerf_category_macro_stability.py`

It reports scene macro, sample-weighted macro, scene-category macro, bootstrap
confidence intervals, category/sample gaps, and worst categories. This prevents
overclaiming a boundary or mask-readout improvement that is driven primarily by
frequent/easy objects.

Current activeC LERF2D internal SAM3 readout:

- Scene macro: 0.8598 LocAcc / 0.5511 mIoU
- Sample-weighted: 0.8702 LocAcc / 0.5666 mIoU
- Scene-category macro: 0.8271 LocAcc / 0.5141 mIoU
- Scene macro mIoU bootstrap 95% CI: [0.4793, 0.6229]
- Category minus sample-weighted mIoU: -0.0525

Conclusion: the internal SAM3 boundary readout is a valid sample-weighted
boundary ablation, but it should not replace the main category-macro LERF table.

## Verification

Passed:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m py_compile \
  radio_gs/config.py \
  radio_gs/scripts/train_feature_field.py \
  radio_gs/training/feature_supervision_mixin.py \
  radio_gs/scripts/build_lerf_category_macro_stability.py

CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q \
  tests/test_direct_point_supervision.py \
  tests/test_build_lerf_category_macro_stability.py \
  tests/test_build_train_feature_field_audit.py \
  tests/test_generate_scannet_dino_cv_configs.py
```

Result: 34 direct-point/config tests passed after the cached-visible update.

Latest direct-field/report verification:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest \
  tests/test_build_scannet_vala8_report.py \
  tests/test_direct_point_supervision.py \
  tests/test_generate_scannet_dino_cv_configs.py -q
```

Result: 42 tests passed.

## Training Status

The full VALA/OpenGaFF-8 v70 run is complete. It is not promoted as the main
ScanNet row because the cached-visible joint2D/3D training objective improves
split10 mAcc but weakens split19/15 mIoU. The cleanest positive direct-field
update is instead the DINO-CV compact field evaluated with the same contextual
3D kNN readout that was previously the strongest v67 support row.

### scene0000 training pilots

| Variant | Key change | split19 mIoU / mAcc | split15 mIoU / mAcc | split10 mIoU / mAcc | Conclusion |
| --- | --- | ---: | ---: | ---: | --- |
| v67 | DINO-CV baseline | 0.2953 / 0.5963 | 0.2744 / 0.5863 | 0.3079 / 0.6886 | prior fair mainline |
| v68 | joint render + direct, random cached points | 0.2958 / 0.5994 | 0.2763 / 0.5922 | 0.3068 / 0.6872 | small positive on 19/15 |
| v69 | centered teacher contrast | 0.2951 / 0.5987 | 0.2763 / 0.5919 | 0.3044 / 0.6864 | not promoted |
| v70 | cached-visible fraction 0.5 | **0.2969 / 0.5998** | **0.2777 / 0.5933** | **0.3079 / 0.6876** | current promoted fair label-free variant |
| v71 | cached-visible fraction 1.0 | 0.2968 / 0.5997 | 0.2775 / 0.5933 | 0.3079 / 0.6876 | no gain over v70 |

### VALA/OpenGaFF-8 training variants

| Variant | split19 mIoU / mAcc | split15 mIoU / mAcc | split10 mIoU / mAcc | Conclusion |
| --- | ---: | ---: | ---: | --- |
| v67 DINO-CV Gaussian-index | 0.3704 / 0.6159 | 0.3718 / 0.6268 | 0.4390 / 0.7020 | strongest clean Gaussian-index training row |
| v70 cached-visible joint2D/3D | 0.3571 / 0.5997 | 0.3644 / 0.6178 | 0.4419 / 0.7091 | improves split10 mAcc but hurts 19/15 mIoU; not promoted |
| v72 visible-balanced pilot | 0.2649 / 0.4681 | 0.2875 / 0.5027 | 0.3674 / 0.6544 | 3-scene pilot matches v70 failure mode; stopped |
| v73 view-count-only pilot | 0.2224 / 0.3223 | 0.2411 / 0.3529 | 0.3216 / 0.4917 | first-scene negative; stopped |

### VALA/OpenGaFF-8 contextual direct readout

| Variant | split19 mIoU / mAcc | split15 mIoU / mAcc | split10 mIoU / mAcc | Conclusion |
| --- | ---: | ---: | ---: | --- |
| v67 contextual kNN, alpha=0.5 | 0.3677 / 0.5997 | 0.3748 / 0.6181 | 0.4562 / 0.7008 | previous strongest balanced support row |
| DINO-CV contextual kNN, ens5, alpha=0.5 | 0.3674 / 0.5960 | 0.3716 / 0.6103 | 0.4533 / 0.6973 | ens5 text head hurts contextual readout |
| DINO-CV contextual kNN, `{query}`, alpha=0.5 | **0.3704 / 0.6017** | **0.3771 / 0.6198** | **0.4585 / 0.7032** | promoted balanced direct-field support row |
| DINO-CV contextual kNN, `{query}`, alpha=0.75 | 0.3683 / 0.5957 | 0.3746 / 0.6136 | **0.4612 / 0.7036** | diagnostic: higher split10 mIoU, weaker 19/15 balance |

The paper-facing artifact is
`paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn_scene_mean_a05_results.json`.
Its report includes category stability in
`output/radio_gs/reports/scannet_vala8_dino_cv_contextual_knn_scene_mean_a05_20260524.md`.

Conclusion: the method-level joint 2D/3D losses are implemented and verified,
but the full VALA8 evidence says they should remain an ablation/negative
diagnostic. The stronger paper claim is now supported by a positive pairing of
DINO cross-view compact-field training and contextual direct 3D point readout:
one compact Gaussian feature map supports rendered queries and direct 3D
queries, with the direct readout improving over the previous strongest v67
contextual row under the same VALA/OpenGaFF-8 evaluator.
