# P0-P3 Method Upgrade Status, 2026-05-25

Scope: fixed-protocol verification of the requested method-level upgrades.  The
goal is to prevent unstable branches from leaking into the promoted paper rows.

## P0: Direct-3D feature-only SAM boundary readout

Status: implemented, not promoted.

- Code: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- Evidence: `paper/artifacts/lerf_direct3d_feature_only_sam_boundary_readout_20260525.md`
- Best feature-SAM heatguard macro mIoU: `0.4113`
- Macro delta vs no-RGB direct-field baseline: `+0.0031`
- Scene deltas: Figurines `-0.0061`, Teatime `+0.0271`, Waldo `-0.0116`; Ramen weighted-log branch `-0.0596`

Decision: keep as ablation/failure analysis.  It proves the feature-only SAM
boundary path is wired into direct 3D, but the result is not robust enough for
the main direct-3D row.

## P1: DINO cycle/background/component + feature-boundary readout

Status: implemented, not promoted.

- Code: `radio_gs/scripts/eval_lerf_sam_dino_tasks.py`
- New run: `output/lerf_sam_dino_tasks/formal_v10_dino_feature_boundary_20260525/lerf_sam_dino_task_report.md`
- Baseline promoted run: `output/lerf_sam_dino_tasks/formal_v9_dino_topk_area200_bg110_peak_20260514/lerf_sam_dino_task_report.md`

| Row | Rendered LocAcc | Rendered mIoU | Frame-wise RADIO LocAcc | Frame-wise RADIO mIoU |
|---|---:|---:|---:|---:|
| formal_v9 bg110 peak | 0.7943 | 0.4805 | 0.7660 | 0.5119 |
| formal_v10 feature-boundary | 0.7872 | 0.4823 | 0.7660 | 0.5092 |

Decision: keep the code path, but do not promote.  Rendered mIoU improves only
`+0.0017` while LocAcc drops `-0.0071`; this is not enough to claim that DINO
mask propagation has been fixed.

## P2: ScanNet confidence/proposal readout

Status: implemented, small positive.

- Code: `radio_gs/models/proposal_memory.py`, `radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py`
- Evidence: `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_propconf_vxl005_a02_c060_results.md`

| Split | Baseline mIoU/mAcc | Propconf mIoU/mAcc | Delta |
|---|---:|---:|---:|
| split19 | 0.3806 / 0.6129 | 0.3809 / 0.6134 | +0.0003 / +0.0005 |
| split15 | 0.3871 / 0.6315 | 0.3874 / 0.6321 | +0.0003 / +0.0006 |
| split10 | 0.4711 / 0.7200 | 0.4715 / 0.7207 | +0.0004 / +0.0007 |

Decision: usable as a stability ablation.  It improves split19 without hurting
split15/split10, but the magnitude is too small to frame as a major method
advance.

## P3: LERF rendered GT-free adaptive coarse prompt

Status: implemented, not promoted.

- Code: `radio_gs/scripts/eval_lerf_grounding.py`
- New run: `output/radio_gs/lerf2d_adaptive_prompt_sam3_20260525`
- Promoted heatguard row: `paper/artifacts/lerf_rendered_grounding_feature_sam3_boundary_20260525.md`

| Row | Macro LocAcc | Macro mIoU | Weighted LocAcc | Weighted mIoU |
|---|---:|---:|---:|---:|
| heatguard peak-init | 0.8598 | 0.5889 | 0.8702 | 0.5998 |
| adaptive peak/raw | 0.8598 | 0.5705 | 0.8702 | 0.5835 |

Decision: do not promote.  Adaptive prompt preserves LocAcc and helps Ramen,
but hurts Figurines, Teatime, and Waldo enough to reduce macro mIoU.

## Paper-facing conclusion

The strongest current paper rows remain:

- LERF rendered: feature-only SAM3 boundary heatguard row.
- LERF direct 3D: compact direct field / no-RGB main row, with feature-only SAM
  direct-3D as ablation rather than main result.
- ScanNet VALA8: spatial contextual DINO-CV row, optionally reporting the
  propconf variant as a small stability gain.
- 2D Frame-wise-RADIO-vs-Ours DINO: formal_v9 bg110 peak remains the promoted DINO row;
  it supports rendered LocAcc and narrowed mIoU gap, not DINO mIoU superiority.
