# LERF Seed Robustness Summary

This report summarizes the conservative `n=3` seed sweep for the key LERF scenes and compares the best rendered-feature LERF-OVS scores selected by each run's own temperature sweep.

- Target scenes: `figurines`, `ramen`, `teatime`, `waldo_kitchen`
- Variants: `nofdh`, `fdh`
- Seeds: `42`, `7`, `123`
- Per-run score: `lerf_eval_best/summary.json -> best.loc_acc` with `best.miou` as the tie-aware supporting metric

## Figurines

| Variant | Seed | Best LocAcc | Best mIoU | Best Temp | Loc Total | Source |
|---|---:|---:|---:|---:|---:|---|
| noFDH | 7 | 0.6964 | 0.4237 | 50.0 | 56 | `output/radio_gs/lerf_figurines_v14_nofdh_240ep_seed7/lerf_eval_best/summary.json` |
| noFDH | 42 | 0.7500 | 0.4133 | 50.0 | 56 | `output/radio_gs/lerf_figurines_v14_nofdh_240ep/lerf_eval_best/summary.json` |
| noFDH | 123 | 0.7679 | 0.4380 | 50.0 | 56 | `output/radio_gs/lerf_figurines_v14_nofdh_240ep_seed123/lerf_eval_best/summary.json` |
| FDH | 7 | 0.7679 | 0.3946 | 50.0 | 56 | `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep_seed7/lerf_eval_best/summary.json` |
| FDH | 42 | 0.8036 | 0.4312 | 50.0 | 56 | `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/lerf_eval_best/summary.json` |
| FDH | 123 | 0.7500 | 0.4068 | 35.0 | 56 | `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep_seed123/lerf_eval_best/summary.json` |

### Mean ± Std

| Variant | Seeds Present | LocAcc Mean | LocAcc Std | mIoU Mean | mIoU Std |
|---|---:|---:|---:|---:|---:|
| noFDH | 3 | 0.7381 | 0.0372 | 0.4250 | 0.0124 |
| FDH | 3 | 0.7738 | 0.0273 | 0.4108 | 0.0187 |

## Ramen

| Variant | Seed | Best LocAcc | Best mIoU | Best Temp | Loc Total | Source |
|---|---:|---:|---:|---:|---:|---|
| noFDH | 7 | 0.8451 | 0.5665 | 40.0 | 71 | `output/radio_gs/lerf_ramen_v14_nofdh_240ep_seed7/lerf_eval_best/summary.json` |
| noFDH | 42 | 0.9014 | 0.5060 | 30.0 | 71 | `output/radio_gs/lerf_ramen_v14_nofdh_240ep/lerf_eval_best/summary.json` |
| noFDH | 123 | 0.8310 | 0.4476 | 25.0 | 71 | `output/radio_gs/lerf_ramen_v14_nofdh_240ep_seed123/lerf_eval_best/summary.json` |
| FDH | 7 | 0.8873 | 0.5856 | 40.0 | 71 | `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/lerf_eval_best/summary.json` |
| FDH | 42 | 0.8732 | 0.5935 | 40.0 | 71 | `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/lerf_eval_best/summary.json` |
| FDH | 123 | 0.8732 | 0.5862 | 40.0 | 71 | `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed123/lerf_eval_best/summary.json` |

### Mean ± Std

| Variant | Seeds Present | LocAcc Mean | LocAcc Std | mIoU Mean | mIoU Std |
|---|---:|---:|---:|---:|---:|
| noFDH | 3 | 0.8592 | 0.0373 | 0.5067 | 0.0595 |
| FDH | 3 | 0.8779 | 0.0081 | 0.5885 | 0.0044 |

## Teatime

| Variant | Seed | Best LocAcc | Best mIoU | Best Temp | Loc Total | Source |
|---|---:|---:|---:|---:|---:|---|
| noFDH | 7 | 0.8475 | 0.5552 | 28.0 | 59 | `output/radio_gs/lerf_teatime_v14_nofdh_240ep_seed7/lerf_eval_best/summary.json` |
| noFDH | 42 | 0.8814 | 0.4263 | 15.0 | 59 | `output/radio_gs/lerf_teatime_v14_nofdh_240ep/lerf_eval_best/summary.json` |
| noFDH | 123 | 0.8814 | 0.5285 | 28.0 | 59 | `output/radio_gs/lerf_teatime_v14_nofdh_240ep_seed123/lerf_eval_best/summary.json` |
| FDH | 7 | 0.8983 | 0.5486 | 25.0 | 59 | `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/lerf_eval_best/summary.json` |
| FDH | 42 | 0.8814 | 0.4206 | 15.0 | 59 | `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep/lerf_eval_best/summary.json` |
| FDH | 123 | 0.8475 | 0.5209 | 25.0 | 59 | `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed123/lerf_eval_best/summary.json` |

### Mean ± Std

| Variant | Seeds Present | LocAcc Mean | LocAcc Std | mIoU Mean | mIoU Std |
|---|---:|---:|---:|---:|---:|
| noFDH | 3 | 0.8701 | 0.0196 | 0.5033 | 0.0681 |
| FDH | 3 | 0.8757 | 0.0259 | 0.4967 | 0.0673 |

## Waldo Kitchen

| Variant | Seed | Best LocAcc | Best mIoU | Best Temp | Loc Total | Source |
|---|---:|---:|---:|---:|---:|---|
| noFDH | 7 | 0.8182 | 0.1485 | 12.0 | 22 | `output/radio_gs/lerf_waldo_kitchen_v14_nofdh_240ep_seed7/lerf_eval_best/summary.json` |
| noFDH | 42 | 0.5909 | 0.3739 | 25.0 | 22 | `output/radio_gs/lerf_waldo_kitchen_v14_nofdh_240ep/lerf_eval_best/summary.json` |
| noFDH | 123 | 0.5455 | 0.2235 | 15.0 | 22 | `output/radio_gs/lerf_waldo_kitchen_v14_nofdh_240ep_seed123/lerf_eval_best/summary.json` |
| FDH | 7 | 0.5909 | 0.3698 | 25.0 | 22 | `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/lerf_eval_best/summary.json` |
| FDH | 42 | 0.7727 | 0.3837 | 25.0 | 22 | `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep/lerf_eval_best/summary.json` |
| FDH | 123 | 0.7727 | 0.3293 | 20.0 | 22 | `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed123/lerf_eval_best/summary.json` |

### Mean ± Std

| Variant | Seeds Present | LocAcc Mean | LocAcc Std | mIoU Mean | mIoU Std |
|---|---:|---:|---:|---:|---:|
| noFDH | 3 | 0.6515 | 0.1461 | 0.2486 | 0.1148 |
| FDH | 3 | 0.7121 | 0.1050 | 0.3609 | 0.0282 |

## Status

All targeted seed runs and eval sweeps are present.

