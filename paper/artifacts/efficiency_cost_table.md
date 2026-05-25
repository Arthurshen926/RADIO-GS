# Efficiency / Cost Table

Last updated: 2026-05-10

This report converts the freeze profiles and storage-footprint accounting into a
paper-facing efficiency/cost table. It intentionally separates explicit GPU
telemetry from log-derived training throughput.

## Paper Table

| Evidence | Scope | Wall Time / Cost | Peak VRAM | Source |
|---|---|---:|---:|---|
| LERF overlay evaluation | 4 scenes | 124.770 s total / 31.193 s per scene | 2076 MiB | `output/radio_gs/profiles/freeze_lerf_*_overlay_20260502` |
| ScanNet legacy point-query evaluation profile | 10 scenes | 150.903 s total / 15.090 s per scene | 1666 MiB | `output/radio_gs/profiles/freeze_scannet_v67_all_eval_20260502` |
| Feature-field footprint | 4 LERF scenes | 1.74x-4.04x storage saving | -- | `output/radio_gs/reports/storage_footprint_report.md` |

## Runtime Source Rows

| Profile | Wall Time | Peak VRAM | Notes |
|---|---:|---:|---|
| `freeze_lerf_figurines_overlay_20260502` | 26.198 s | 1568 MiB | frozen overlay evaluation |
| `freeze_lerf_ramen_overlay_20260502` | 40.474 s | 1762 MiB | frozen overlay evaluation |
| `freeze_lerf_teatime_overlay_20260502` | 36.997 s | 1850 MiB | frozen overlay evaluation |
| `freeze_lerf_waldo_overlay_20260502` | 21.101 s | 2076 MiB | frozen overlay evaluation |
| `freeze_scannet_v67_all_eval_20260502` | 150.903 s | 1666 MiB | legacy 10-scene direct point-query evaluation profile |

## Storage Source Rows

| Scene | Direct 1280-D fp16 | Compact total | Saving |
|---|---:|---:|---:|
| Figurines | 412.1 MiB | 237.0 MiB | 1.74x |
| Ramen | 934.3 MiB | 311.2 MiB | 3.00x |
| Teatime | 1123.4 MiB | 338.1 MiB | 3.32x |
| Waldo Kitchen | 1688.8 MiB | 418.5 MiB | 4.04x |

## Training-Cost Note

The training-cost evidence in `output/radio_gs/reports/efficiency_profile.md` is
log-derived wall time, while the runtime rows above are explicit profiled
workloads with GPU telemetry. For the main paper, use the profiled rows and the
storage table. If a supplementary training-cost table is needed, report it as
log-derived throughput and keep it separate from telemetry-based runtime claims.
