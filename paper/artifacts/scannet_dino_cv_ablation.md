# ScanNet DINO-CV Ablation Report

Date: 2026-05-10

## Protocol

- Variant: `v67_dino_cv001_b2_s32768_ft20`
- Warm start: `v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63`
- Scenes: `scene0000_00`, `scene0062_00`, `scene0070_00`, `scene0097_00`, `scene0140_00`, `scene0200_00`, `scene0347_00`, `scene0400_00`, `scene0590_00`, `scene0645_00`
- Query protocol: `gaussian_index`, `label_point`, `label_index`, `opacity_threshold=0.1`
- Cross-view head: `dino_v3`, `radio_adaptor_cross_view_weight=0.001`
- Batch size: `2`; larger `b4` was rejected earlier because it OOMed on the 24GB GPUs for this DINO cross-view branch.

## Final 10-Scene Macro

| Split | Baseline mIoU | DINO-CV mIoU | Delta | Baseline mAcc | DINO-CV mAcc | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 19 classes | 0.3538 | 0.3640 | +0.0102 | 0.6076 | 0.6205 | +0.0128 |
| 15 classes | 0.3573 | 0.3662 | +0.0089 | 0.6203 | 0.6313 | +0.0110 |
| 10 classes | 0.4293 | 0.4308 | +0.0014 | 0.7051 | 0.7071 | +0.0020 |

## Per-Scene mIoU Delta

| Scene | Delta 19 | Delta 15 | Delta 10 |
|---|---:|---:|---:|
| scene0000_00 | +0.0015 | +0.0006 | +0.0014 |
| scene0062_00 | -0.0002 | -0.0002 | +0.0096 |
| scene0070_00 | -0.0040 | -0.0081 | -0.0008 |
| scene0097_00 | -0.0025 | +0.0046 | +0.0067 |
| scene0140_00 | -0.0049 | -0.0059 | -0.0041 |
| scene0200_00 | +0.0013 | +0.0005 | -0.0008 |
| scene0347_00 | +0.0102 | +0.0114 | -0.0009 |
| scene0400_00 | +0.0772 | +0.0772 | +0.0067 |
| scene0590_00 | +0.0199 | +0.0011 | +0.0008 |
| scene0645_00 | +0.0038 | +0.0080 | -0.0043 |

## Interpretation

DINO-CV is positive at the full-sweep macro level, especially on the 19-class and 15-class direct point-query splits. The gain is not uniform: scene0400_00 and scene0590_00 drive most of the improvement, while scene0070_00 and scene0140_00 regress slightly.

Conclusion for the paper: keep the conservative v67 teacher-balanced result as the main ScanNet row, and report DINO-CV as a full 10-scene compatibility/FAC ablation. The result supports the claim that frozen DINO adaptor structure can preserve or improve transfer, but it should not be framed as a universal accuracy upgrade.

## Artifacts

- DINO-CV result JSON pattern: `output/scannet_pointcloud_eval/*_v67_dino_cv001_b2_s32768_ft20_gidx_labelpoint/scannet_pointcloud_radio_gs_results.json`
- Baseline result JSON pattern: `output/scannet_pointcloud_eval/*_v67_teacherbalanced_fromv63_best_gidx_labelpoint/scannet_pointcloud_radio_gs_results.json`
- Logs:
  - `output/radio_gs/scannet_dino_cv_b2_queue_20260509/gpu4.log`
  - `output/radio_gs/scannet_dino_cv_b2_queue_20260509/gpu5.log`
  - `output/radio_gs/scannet_dino_cv_b2_extra_20260510/gpu0.log`
  - `output/radio_gs/scannet_dino_cv_b2_extra_20260510/gpu1.log`
  - `output/radio_gs/scannet_dino_cv_b2_extra_20260510/gpu1_scene0140.log`
  - `output/radio_gs/scannet_dino_cv_b2_extra_20260510/gpu2.log`
  - `output/radio_gs/scannet_dino_cv_b2_extra_20260510/gpu3.log`
