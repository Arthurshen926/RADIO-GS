# Unified Multi-Head Feature-Quality Field

Status: implemented as a method-level training upgrade on 2026-05-25.

## What Changed

- Added optional `hybrid_quality_head` and `hybrid_visibility_head` to the
  decoupled hybrid field.
- Both rendered-view decoding and direct Gaussian/point querying now expose
  `quality_logit` and `visibility_logit` when the heads are enabled.
- Added shared targets:
  - feature quality: detached reconstruction agreement in RADIO space.
  - visibility: rendered alpha visibility, optionally binarized.
- Added trainer losses controlled by `quality_loss_weight` and
  `visibility_loss_weight`.
- Added shared preset:
  `radio_gs/configs/unified_feature_quality_base.yaml`.

## Why It Matters

The previous code already had semantic confidence, visibility-weighted direct
point losses, DINO/SAM/SigLIP adaptor supervision, and direct 2D/3D consistency,
but the quality signals were distributed across training and evaluators. This
upgrade makes feature quality and visibility explicit field readouts. It is the
clean method hook for later using the same compact Gaussian map for:

- rendered 2D text grounding;
- direct primitive/query scoring;
- boundary-aware readout confidence;
- diagnostic failure analysis without task-specific retraining.

## Current Evidence Rows

| Track | Current registered result | Source |
|---|---:|---|
| LERF rendered-view OVS | LocAcc 0.8598 / mIoU 0.5707 | `lerf_rendered_grounding_peak_component_20260524.json` |
| LERF direct 3D selection | mIoU 0.5705 / Acc@0.25 0.6835 / Boundary-F 0.6681 | `lerf_sam3_box_global_threshold_sweep_20260517_geometry.json` |
| ScanNet VALA8 point query | 19 mIoU/mAcc 0.3655/0.5057; 15 0.4278/0.7285; 10 0.5785/0.7793 | `scannet_pointcloud_radio_gs_vala8_reproduced_benchmark_20260615.json` |
| 2D Frame-wise RADIO-vs-GaussFM | GaussFM wins 5/6 selected primary metrics | `teacher_vs_ctfgs_2d_usability_20260525.json` |

## Caveat

This patch adds the architecture and training objective needed for the unified
feature-quality claim. The numbers above are the current promoted rows before a
full retraining sweep with the new quality/visibility heads. New long runs
should be reported separately and promoted only if they improve the frozen
protocol metrics.
