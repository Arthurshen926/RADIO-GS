# RADIO-GS Training Efficiency Profile
*Generated: 2026-05-02 01:55*

This report separates two evidence types: experiment-log training throughput and explicit profiled workloads with GPU telemetry.

## Training Throughput

| Experiment | Wall Time (h) | Epochs | Ep/hr | Eval Done |
|---|---:|---:|---:|---|
| `lerf_figurines_v14` | 3.0 | 120 | 40.3 | ⏳ |
| `lerf_figurines_v14_fdh_ws240_240ep` | 1.7 | 240 | 138.2 | ✅ |
| `lerf_figurines_v14_fdh_ws240_240ep_seed123` | 2.2 | 240 | 107.1 | ✅ |
| `lerf_figurines_v14_fdh_ws240_240ep_seed7` | 2.1 | 240 | 114.6 | ✅ |
| `lerf_figurines_v14_nofdh_240ep` | 15.8 | 240 | 15.2 | ✅ |
| `lerf_figurines_v14_nofdh_240ep_seed123` | 1.7 | 240 | 144.0 | ✅ |
| `lerf_figurines_v14_nofdh_240ep_seed7` | 1.7 | 240 | 141.7 | ✅ |
| `lerf_figurines_v14_pure_frozen_depth_only` | 18.3 | 120 | 6.5 | ✅ |
| `lerf_prompt_ensemble_overlay_eval_20260501` | — | — | — | ⏳ |
| `lerf_ramen_v14` | 1.2 | 120 | 96.5 | ⏳ |
| `lerf_ramen_v14_fdh_ws240_240ep` | 1.0 | 240 | 231.3 | ✅ |
| `lerf_ramen_v14_fdh_ws240_240ep_seed123` | 1.1 | 240 | 223.1 | ✅ |
| `lerf_ramen_v14_fdh_ws240_240ep_seed7` | 1.1 | 240 | 216.5 | ✅ |
| `lerf_ramen_v14_frozen_dh` | 1.1 | 120 | 106.0 | ⏳ |
| `lerf_ramen_v14_nofdh_240ep` | 15.3 | 240 | 15.7 | ✅ |
| `lerf_ramen_v14_nofdh_240ep_seed123` | 1.2 | 240 | 207.9 | ✅ |
| `lerf_ramen_v14_nofdh_240ep_seed7` | 1.5 | 240 | 157.3 | ✅ |
| `lerf_ramen_v14_pure_frozen_depth_only` | 17.2 | 120 | 7.0 | ✅ |
| `lerf_single_template_overlay_eval_20260501` | — | — | — | ⏳ |
| `lerf_summary_tables` | — | — | — | ⏳ |
| `lerf_teatime_v14` | 1.7 | 120 | 70.0 | ⏳ |
| `lerf_teatime_v14_fdh_ws240_240ep` | 1.6 | 240 | 152.8 | ✅ |
| `lerf_teatime_v14_fdh_ws240_240ep_seed123` | 3.8 | 240 | 63.2 | ✅ |
| `lerf_teatime_v14_fdh_ws240_240ep_seed7` | 4.3 | 240 | 55.6 | ✅ |
| `lerf_teatime_v14_nofdh_240ep` | 16.0 | 240 | 15.0 | ✅ |
| `lerf_teatime_v14_nofdh_240ep_seed123` | 4.3 | 240 | 56.1 | ✅ |
| `lerf_teatime_v14_nofdh_240ep_seed7` | 3.2 | 240 | 75.3 | ✅ |
| `lerf_teatime_v14_pure_frozen_depth_only` | 18.4 | 120 | 6.5 | ✅ |
| `lerf_waldo_kitchen_v14` | 1.4 | 120 | 87.8 | ⏳ |
| `lerf_waldo_kitchen_v14_fdh_ws240_240ep` | 1.4 | 240 | 177.5 | ✅ |
| `lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed123` | 2.1 | 240 | 115.9 | ✅ |
| `lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7` | 1.5 | 240 | 165.0 | ✅ |
| `lerf_waldo_kitchen_v14_nofdh_240ep` | 15.5 | 240 | 15.5 | ✅ |
| `lerf_waldo_kitchen_v14_nofdh_240ep_seed123` | 1.9 | 240 | 128.0 | ✅ |
| `lerf_waldo_kitchen_v14_nofdh_240ep_seed7` | 2.0 | 240 | 119.4 | ✅ |
| `lerf_waldo_kitchen_v14_pure_frozen_depth_only` | 17.6 | 120 | 6.8 | ✅ |

## Profiled Workloads

| Profile | Wall Time | Peak VRAM (MiB) | Mean GPU% | Samples |
|---|---:|---:|---:|---:|
| `figurines_fdh_ws240_sweep_fixed` | 101.636 s | 1568 | 0.43 | 102 |
| `freeze_lerf_figurines_overlay_20260502` | 26.198 s | 1568 | 1.30 | 27 |
| `freeze_lerf_ramen_overlay_20260502` | 40.474 s | 1762 | 1.10 | 41 |
| `freeze_lerf_teatime_overlay_20260502` | 36.997 s | 1850 | 0.65 | 37 |
| `freeze_lerf_waldo_overlay_20260502` | 21.101 s | 2076 | 1.86 | 21 |
| `freeze_scannet_v67_all_eval_20260502` | 150.903 s | 1666 | 1.91 | 151 |
| `ramen_nofdh_240ep_lerf_autoeval` | 173.292 s | 1762 | 1.09 | 173 |
| `room0_pure_frozen_depth_only_autoeval` | 570.981 s | 4071 | 37.63 | 570 |
| `waldo_kitchen_fdh_ws240_240ep_lerf_autoeval` | 215.057 s | 2076 | 1.39 | 215 |

## Notes
- Training wall time is measured from the first to last timestamp in `logs/training.log` when present.
- Time-only training logs are handled as same-day or overnight runs; this is sufficient for the current sub-24h LERF jobs.
- Epochs/hr = max observed epoch / wall hours from the training log.
- GPU telemetry is only available for explicitly profiled workloads under `output/radio_gs/profiles`; most training runs do not have per-run GPU metrics.

