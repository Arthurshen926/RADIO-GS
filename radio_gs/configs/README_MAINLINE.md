# RADIO-GS Config Mainline

This directory intentionally still contains many historical YAML files. They are
kept in place because scripts, tests, reports, and old reproduction commands
refer to their original paths.

For paper submission work, use only the configs listed below unless a new result
is explicitly promoted into `output/radio_gs/reports/submission_freeze_report.md`.

## LERF-OVS Mainline

| Scene | Active Config | Active Checkpoint |
|---|---|---|
| Figurines | `radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml` | `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/latest.pth` |
| Ramen | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_ramen_fdh_ws240_240ep_seed7.yaml` | `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth` |
| Teatime | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed7.yaml` | `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth` |
| Waldo Kitchen | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed7.yaml` | `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth` |

## ScanNet Mainline

The active ScanNet path is generated and ignored by git:

`radio_gs/configs/generated/scannet_og/scannet_og_hybrid_v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63_{scene}.yaml`

It must be evaluated with:

- `query_mode=gaussian_index`
- `gaussian_index_position_mode=label_point`
- `opacity_filter_mode=label_index`
- `class_splits=19,15,10`

## Supporting Config Families

- `*_nofdh_240ep*`: ablation and robustness companion.
- `*_pure_frozen_depth_only*`: archived support branch.
- `replica_hybrid_v14_room_0_*`: auxiliary Replica utility branch.
- older `replica_explicit_*`, early `replica_hybrid_v*`, and FDH weight sweeps:
  historical development configs, not current paper entry points.
