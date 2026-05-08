# RADIO-GS Project Mainline

Status: 2026-05-03 conservative submission mainline plus adaptor-enhanced candidate.

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
9. Optional RADIO adaptor/cross-view consistency: DINOv3 relation + SAM3
   soft-region alignment for Ramen/Teatime, and DINOv3 cross-view + spatial
   text-heatmap preservation for Figurines.

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

### LERF-OVS Adaptor-Enhanced Candidate

This candidate uses the DINO cross-view + spatial text-heatmap checkpoint for
Figurines, DINO relation + SAM3 region checkpoints for Ramen and Teatime, and
keeps the baseline checkpoint for Waldo Kitchen. It should be reported as an
adaptor/cross-view ablation candidate unless the paper explicitly adopts it as
the main selector.

| Scene | Selected branch | LocAcc | mIoU | Temperature |
|---|---|---:|---:|---:|
| Figurines | DINO cv + spatial text heatmap | 0.8214 | 0.4343 | 50 |
| Ramen | DINO relation + SAM3 region | 0.9014 | 0.5873 | 40 |
| Teatime | DINO relation + SAM3 region | 0.8983 | 0.5592 | 28 |
| Waldo Kitchen | Baseline | 0.8636 | 0.4106 | 25 |
| Macro | - | 0.8712 | 0.4979 | - |

The failed adaptor branches are still informative: Figurines gains mIoU with
strong relation/region and query-distribution heatmap weights but loses
localization, while the conservative spatial heatmap branch is the only
Figurines branch promoted by the preserve-LocAcc rule. Waldo Kitchen does not
benefit from adaptor/cross-view supervision yet.

### DINOv3/SAM3 Downstream Adaptor Probes

Use `output/lerf_adaptor_downstream/mainline/lerf_adaptor_downstream_aggregate.json`.
These results compare original RADIO RGB features (`Teacher`) against
RADIO-GS rendered features (`Rendered`) after frozen DINOv3/SAM3 adaptor
projection. They are diagnostic and should not replace the LERF text-grounding
main result.

| Adaptor | Probe | Teacher LocAcc | Teacher mIoU | Rendered LocAcc | Rendered mIoU |
|---|---|---:|---:|---:|---:|
| DINOv3 | prototype segmentation | 0.6543 | 0.0945 | 0.6277 | 0.0937 |
| DINOv3 | source-target matching | 0.5957 | 0.1032 | 0.5035 | 0.1019 |
| SAM3 | prototype segmentation | 0.8404 | 0.0757 | 0.6649 | 0.0564 |
| SAM3 | source-target matching | 0.7872 | 0.0953 | 0.7092 | 0.0687 |

Qualitative examples are in
`paper/figures/lerf_adaptor_downstream_qualitative.png`. The aggregate takeaway
is mixed: rendered DINOv3 is close in mIoU and has positive Waldo Kitchen
matching cases, while SAM3 still exposes a clear teacher-rendered gap.

### ScanNet v67 Direct Point Query

Use `output/scannet_pointcloud_eval/*_v67_teacherbalanced_fromv63_best_gidx_labelpoint/`.

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3538 | 0.6076 |
| 15 classes | 0.3573 | 0.6203 |
| 10 classes | 0.4293 | 0.7051 |

Targeted DINOv3 cross-view diagnostics are positive but not yet a 10-scene
replacement:

| Scene | Branch | split19 | split15 | split10 |
|---|---|---:|---:|---:|
| scene0070_00 | v67 baseline | 0.2297 | 0.2405 | 0.3238 |
| scene0070_00 | DINO cv weight 0.001 | 0.2437 | 0.2466 | 0.3284 |
| scene0645_00 | v67 baseline | 0.2381 | 0.2458 | 0.2875 |
| scene0645_00 | DINO cv weight 0.003 | 0.2427 | 0.2500 | 0.2833 |

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
| LERF component ablations | Validated FDH, refiner, hybrid, and HCD under controlled seed-7 LERF-OVS | Active paper ablation; see `output/radio_gs/reports/lerf_component_ablation.md` |
| RADIO DINOv3/SAM3 adaptor ablations | Tested adaptor-space consistency, DINO relation, and SAM3 soft-region supervision | Active ablation; only Ramen/Teatime promoted |
| ProFuse-style DINO cross-view branch | Tested cross-view DINO affinity plus text-heatmap peak protection | LERF diagnostic only; improves some overlap scores but does not preserve LocAcc |
| ScanNet DINO cross-view branch | Tested DINO affinity as cross-view context for direct point queries | Positive targeted diagnostic; needs full 10-scene conservative-weight sweep |

## Archive Policy

Archived files are retained, not deleted. They can explain decisions, but they
are not paper-number sources. A result becomes paper-eligible only when it is
listed in `submission_freeze_report.md` or explicitly linked from the current
LaTeX draft.

The main archive locations are:

- `docs/archive/2026-05-02_legacy/`
- `output/radio_gs/reports/archive_legacy_20260502/`
