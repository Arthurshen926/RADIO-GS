# RADIO-GS Submission Status

This document turns the current experiment inventory into a concrete submission
status note. It is meant to answer a narrow question: how close is the current
repository to a top-conference paper package?

## Current level

- **Method maturity**: mature research prototype with paper-facing automation and archive discipline in place
- **Current submission maturity**: conservative paper package with LaTeX draft
- **Estimated top-conference completion**: about 90% for the conservative
  RADIO-GS package, and about 82% for the stricter CTF-GS paper outline until
  external baselines and final venue formatting are closed

The repository now contains a coherent method, a strong LERF-OVS result,
repeatable evaluation code, and a completed 10-scene fair ScanNet v67 aggregate.
What is still missing is mostly paper-freeze discipline: external baseline
provenance, final training-cost measurements, venue-formatted manuscript prose,
and related-work anchoring.

## Current source of truth

The current generated freeze package is:

- [submission_freeze_report.md](../output/radio_gs/reports/submission_freeze_report.md)
- [submission_freeze_manifest.json](../output/radio_gs/reports/submission_freeze_manifest.json)
- [submission_freeze_profile_summary.md](../output/radio_gs/reports/submission_freeze_profile_summary.md)
- [submission_freeze_figure_shortlist.md](../output/radio_gs/reports/submission_freeze_figure_shortlist.md)
- [efficiency_profile.md](../output/radio_gs/reports/efficiency_profile.md)
- [baseline_source_verification.md](../output/radio_gs/reports/baseline_source_verification.md)
- [lerf_component_ablation.md](../output/radio_gs/reports/lerf_component_ablation.md)
- [paper_draft_current.md](paper_draft_current.md)
- [PROJECT_MAINLINE.md](PROJECT_MAINLINE.md)
- [radio_gs_draft.tex](../paper/radio_gs_draft.tex)

These generated files supersede older manually edited status notes when numbers
conflict.

## Best-supported paper framing

The cleanest current framing is now:

> **Compact Teacher Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding.**

The paper-facing method name is **CTF-GS**. `RADIO-GS` remains the repository and
implementation name. The current module names should be mapped as:

| Implementation name | Paper-facing name |
|---|---|
| Hybrid Gaussian feature field | HGCF: Hybrid Gaussian Code Field |
| HCD codec | CTR: Compact-to-Teacher Reconstruction |
| Screen-space refiner | VFA: View-Space Feature Aligner |
| FDH warm-start | FGC: Frozen Geometry-Head Consistency |

Under that framing, the paper solves the following problem:

1. Distill high-dimensional RADIO spatial features into a 3D Gaussian scene representation.
2. Reconstruct usable novel-view features rather than only RGB or low-dimensional task features.
3. Show that one reconstructed feature field can support grounding, depth, and segmentation.

## Strongest current claims

1. The project can already support a strong **LERF-OVS grounding** main result.
2. The project has a technically non-trivial unified pipeline: HGCF compact
   code storage, CTR/HCD reconstruction, VFA/screen refinement, FGC/FDH
   supervision, and ws240 warm-start training.
3. The project now has a paper-facing **ScanNet direct point-query** cross-domain result under the v67 teacher-balanced fair protocol.
4. The project has credible auxiliary evidence that the learned feature field is useful beyond grounding, especially on Replica room_0 depth and segmentation.
5. The freeze package now includes formal LERF overlay/profile runs for all four main scenes plus one full ScanNet v67 evaluation profile.
6. The LERF evaluator already includes a same-protocol feature-source check:
   rendered RADIO-GS features outperform original RADIO RGB features on macro
   localization accuracy (0.8712 vs. 0.7985) and slightly improve macro mIoU
   (0.4941 vs. 0.4922) under the frozen SigLIP2 scoring setup.
7. DINOv3/SAM3 adaptor supervision now has completed full LERF sweeps. The
   promoted adaptor-enhanced candidate keeps macro LocAcc at 0.8712 and improves
   macro mIoU from 0.4941 to 0.4979 by using the Figurines spatial text-heatmap
   cross-view checkpoint plus relation/region checkpoints for Ramen and Teatime.
8. DINOv3/SAM3 downstream adaptor probes are now complete. They are diagnostic
   rather than main-claim improvements: rendered DINOv3 is close to teacher mIoU,
   but macro LocAcc drops; SAM3 still has a larger teacher-rendered gap.
9. The ProFuse-inspired DINO cross-view branch is implemented and evaluated as a
   diagnostic path. It can increase thresholded overlap on some LERF scenes
   (for example Waldo high-temperature mIoU), but it still lowers LocAcc, so it
   is not promoted to the main LERF table.
10. ScanNet DINO cross-view diagnostics are positive on targeted scenes when the
    weight is conservative: scene0070_00 improves split19 from 0.2297 to 0.2437
    at weight 0.001, while scene0645_00 improves split19/15 at weight 0.003 but
    slightly lowers split10.
11. The LERF peak-preservation diagnostic is now positive on Figurines:
    DINO cross-view with `text_heatmap_distill_mode: spatial` keeps LocAcc at
    0.8214 and improves mIoU from 0.4308 to 0.4343.
12. The core LERF component ablation is now closed under the controlled seed-7
    protocol. CTR/HCD is the strongest architectural dependency (`w/o CTR`
    macro LocAcc 0.5306 vs. full 0.8578), FGC/FDH warm-start provides the
    largest training route gain (`w/o FGC` 0.8018), and VFA/HGCF mainly affect
    peak stability rather than raw region coverage.

## What is already paper-grade

### Main result evidence

- [output/radio_gs/reports/submission_freeze_report.md](../output/radio_gs/reports/submission_freeze_report.md)
- [output/radio_gs/reports/submission_freeze_manifest.json](../output/radio_gs/reports/submission_freeze_manifest.json)
- [output/radio_gs/reports/paper_submission_main_table.md](../output/radio_gs/reports/paper_submission_main_table.md)
- [output/radio_gs/reports/paper_submission_result_audit.md](../output/radio_gs/reports/paper_submission_result_audit.md)
- [output/radio_gs/reports/submission_freeze_profile_summary.md](../output/radio_gs/reports/submission_freeze_profile_summary.md)
- [output/radio_gs/reports/efficiency_profile.md](../output/radio_gs/reports/efficiency_profile.md)
- [output/radio_gs/reports/lerf_component_ablation.md](../output/radio_gs/reports/lerf_component_ablation.md)
- [output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv](../output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv)
- [paper/radio_gs_draft.tex](../paper/radio_gs_draft.tex)

### Method explanation assets

- [README.md](../README.md)
- [docs/feature_reconstruction_analysis.md](feature_reconstruction_analysis.md)
- [output/radio_gs/reports/storage_footprint_report.md](../output/radio_gs/reports/storage_footprint_report.md)
- [output/radio_gs/reports/lerf_failure_analysis.md](../output/radio_gs/reports/lerf_failure_analysis.md)

### Qualitative figures

- [output/radio_gs/reports/submission_freeze_figure_shortlist.md](../output/radio_gs/reports/submission_freeze_figure_shortlist.md)
- [output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502](../output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502)
- [output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502](../output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502)
- [output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502](../output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502)
- [paper/figures/lerf_adaptor_downstream_qualitative.png](../paper/figures/lerf_adaptor_downstream_qualitative.png)
- [paper/figures/lerf_sam_dino_tasks_qualitative.png](../paper/figures/lerf_sam_dino_tasks_qualitative.png)
- [output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502](../output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502)
- [paper/figures/lerf_adaptor_downstream_qualitative.png](../paper/figures/lerf_adaptor_downstream_qualitative.png)

## What still blocks a strong main-track submission

### 1. Public baseline provenance is conservatively frozen, but not paper-anchored

The main LERF-OVS comparison has a strong core trio: LERF, LangSplat, LEGaussians.
That is enough to form a serious main table, and the repository now explicitly
freezes those external rows as unresolved draft placeholders rather than claiming
they are exact paper-anchored values. The remaining weakness is not ambiguity,
but that the paper-facing comparison still relies on protocol-misaligned
cross-paper external rows.

### 2. Cross-domain generalization is improved, but needs paper-safe framing

The ScanNet v67 teacher-balanced direct point-query run now provides a 10-scene
cross-domain table. The remaining risk is not lack of ScanNet evidence, but
clean framing: label-supervised and GT-label-balanced ScanNet diagnostics must
stay out of the main fair table.

Targeted DINO cross-view diagnostics:

| Scene | Branch | split19 | split15 | split10 |
|---|---|---:|---:|---:|
| scene0070_00 | v67 baseline | 0.2297 | 0.2405 | 0.3238 |
| scene0070_00 | DINO cv weight 0.001 | 0.2437 | 0.2466 | 0.3284 |
| scene0645_00 | v67 baseline | 0.2381 | 0.2458 | 0.2875 |
| scene0645_00 | DINO cv weight 0.003 | 0.2427 | 0.2500 | 0.2833 |

### 3. Statistical confidence is improved, but still narrow

The conservative route now includes a completed four-scene `n=3` seed summary
for `figurines`, `ramen`, `teatime`, and `waldo_kitchen`. That meaningfully
lowers submission risk and resolves the earlier missing Teatime evidence, but it
is still narrower than a benchmark-wide statistical treatment.

### 4. Small-object failure analysis is now paper-usable, but can be deepened

The generated `lerf_failure_analysis.md` and `lerf_failure_analysis_table.tex`
now provide a paper-facing analysis of fragile categories. A deeper optional
analysis would correlate mask area or feature-cell footprint against LocAcc, but
the current table is enough to support a limitations/discussion section.

### 5. Efficiency and cost have current eval profiles, but need final paper framing

The paper now has formal wall-clock and peak-VRAM profiles for all four frozen
LERF overlay evaluations and the full 10-scene ScanNet v67 point-query
evaluation. The remaining gap is a polished table that separates training cost,
evaluation latency, memory, and feature-resolution trade-offs without comparing
misaligned workloads.

### 6. Main-table provenance is fixed, but future runs still need the same discipline

The LERF-OVS main table is now backed by exact rendered JSON artifacts, and the
repository now freezes `paper_submission_main_table.md` as the canonical paper
main table while `paper_main_table.md` is demoted to supporting seed-robustness
evidence. The remaining requirement is procedural: every new ablation, seed run,
and cross-domain result should be frozen with the same exact-report discipline
instead of relying on informal notes.

## Immediate implementation priorities

1. Close external LERF baseline provenance or run reproduced baselines under one protocol.
2. Convert the refreshed profile evidence and training logs into one polished efficiency/cost table.
3. Decide how much of the fixed-protocol seed-robustness table belongs in the main paper versus appendix.
4. Decide whether the adaptor-enhanced LERF candidate is main-paper or ablation-only.
5. Move `paper/radio_gs_draft.tex` into the target venue template and polish related work.
6. Keep the small-object failure analysis and storage footprint tables in the main paper unless page limits force them to appendix.
7. Promote ScanNet DINO cross-view only after a full 10-scene conservative-weight sweep preserves the main fair protocol.
