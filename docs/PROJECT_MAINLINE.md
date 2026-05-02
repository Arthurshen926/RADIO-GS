# RADIO-GS Project Mainline

Status: 2026-05-02 conservative submission mainline.

This document is the navigation layer for the cleaned project. It separates the
current strongest paper route from historical validation branches.

## One-Sentence Thesis

RADIO-GS reconstructs frozen RADIO foundation features inside 3D Gaussian
scenes so that novel views can render reusable feature maps for open-vocabulary
grounding and cross-domain scene understanding.

## Current Strongest Mainline

The active paper route is:

1. Frozen 3DGS geometry.
2. Hybrid Gaussian feature field with per-Gaussian latent storage and a coarse
   spatial branch.
3. HCD bottleneck codec from compact features back to 1280d RADIO space.
4. Screen-space feature refinement.
5. FDH warm-start training as the main geometry-aware regularizer.
6. LERF-OVS as the primary benchmark.
7. ScanNet v67 direct point query as cross-domain feature-usability evidence.
8. Formal profile runs for evaluation runtime and peak VRAM.

## Source-of-Truth Files

Use these files first:

| Purpose | Path |
|---|---|
| Submission status | `docs/submission_status.md` |
| Current prose draft | `docs/paper_draft_current.md` |
| LaTeX draft | `paper/radio_gs_draft.tex` |
| Freeze report | `output/radio_gs/reports/submission_freeze_report.md` |
| Machine-readable manifest | `output/radio_gs/reports/submission_freeze_manifest.json` |
| Baseline provenance risk | `output/radio_gs/reports/baseline_source_verification.md` |
| Profile summary | `output/radio_gs/reports/submission_freeze_profile_summary.md` |
| Figure shortlist | `output/radio_gs/reports/submission_freeze_figure_shortlist.md` |

## Active Quantitative Claims

### LERF-OVS

Use `output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv`.

| Scene | LocAcc | mIoU | Temperature |
|---|---:|---:|---:|
| Figurines | 0.8214 | 0.4308 | 50 |
| Ramen | 0.9014 | 0.5862 | 40 |
| Teatime | 0.8983 | 0.5486 | 25 |
| Waldo Kitchen | 0.8636 | 0.4106 | 25 |
| Macro | 0.8712 | 0.4941 | - |

### ScanNet v67 Direct Point Query

Use `output/scannet_pointcloud_eval/*_v67_teacherbalanced_fromv63_best_gidx_labelpoint/`.

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3538 | 0.6076 |
| 15 classes | 0.3573 | 0.6203 |
| 10 classes | 0.4293 | 0.7051 |

### Profile Evidence

Use `output/radio_gs/reports/submission_freeze_profile_summary.md`.

| Workload | Wall Time | Peak VRAM |
|---|---:|---:|
| LERF Figurines overlay | 26.198 s | 1568 MiB |
| LERF Ramen overlay | 40.474 s | 1762 MiB |
| LERF Teatime overlay | 36.997 s | 1850 MiB |
| LERF Waldo Kitchen overlay | 21.101 s | 2076 MiB |
| ScanNet v67 10-scene eval | 150.903 s | 1666 MiB |

## Active Configs

Do not treat every YAML under `radio_gs/configs/` as a current paper route. The
active LERF mainline is:

- `radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml`
- `radio_gs/configs/generated/seeds/lerf_hybrid_v14_ramen_fdh_ws240_240ep_seed7.yaml`
- `radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed7.yaml`
- `radio_gs/configs/generated/seeds/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed7.yaml`

The active ScanNet protocol is the generated v67 fair direct point-query config
family under the ignored path:

- `radio_gs/configs/generated/scannet_og/scannet_og_hybrid_v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63_{scene}.yaml`

Replica and room0 configs are supporting evidence only, not the main benchmark
route.

## Historical Branches

| Branch | What It Validated | Current Status |
|---|---|---|
| noFDH seed runs | Robustness and ablation companion for FDH | Supporting table only |
| pure frozen depth-only | Whether frozen depth supervision alone carries useful geometry | Archived support branch |
| room0 Replica depth/segmentation | Downstream utility beyond grounding | Supporting qualitative/auxiliary evidence |
| FDH weight sweeps | Sensitivity of geometry regularization | Archived design evidence |
| Figurines 2x | Small-object feature-resolution hypothesis | Archived unless re-run under frozen protocol |
| ScanNet v43/v62 diagnostics | Early point-query and label-informed debugging | Archived; not fair main table |
| ScanNet v67 | Current fair cross-domain protocol | Active mainline support |
| prompt ensemble / overlay sweeps | Visualization and prompt-sensitivity checks | Supporting only |

## Archive Policy

Archived files are retained, not deleted. They can explain decisions, but they
are not paper-number sources. A result becomes paper-eligible only when it is
listed in `submission_freeze_report.md` or explicitly linked from the current
LaTeX draft.

The main archive locations are:

- `docs/archive/2026-05-02_legacy/`
- `output/radio_gs/reports/archive_legacy_20260502/`
