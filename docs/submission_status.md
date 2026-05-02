# RADIO-GS Submission Status

This document turns the current experiment inventory into a concrete submission
status note. It is meant to answer a narrow question: how close is the current
repository to a top-conference paper package?

## Current level

- **Method maturity**: mature research prototype with paper-facing automation and archive discipline in place
- **Current submission maturity**: conservative paper package with LaTeX draft
- **Estimated top-conference completion**: about 87% after the 2026-05-02 archive cleanup and LaTeX draft

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
- [paper_draft_current.md](paper_draft_current.md)
- [PROJECT_MAINLINE.md](PROJECT_MAINLINE.md)
- [radio_gs_draft.tex](../paper/radio_gs_draft.tex)

These generated files supersede older manually edited status notes when numbers
conflict.

## Best-supported paper framing

The cleanest current framing is:

> **Foundation feature reconstruction in 3D Gaussian scenes for open-vocabulary scene understanding.**

Under that framing, the paper solves the following problem:

1. Distill high-dimensional RADIO spatial features into a 3D Gaussian scene representation.
2. Reconstruct usable novel-view features rather than only RGB or low-dimensional task features.
3. Show that one reconstructed feature field can support grounding, depth, and segmentation.

## Strongest current claims

1. The project can already support a strong **LERF-OVS grounding** main result.
2. The project has a technically non-trivial unified pipeline: Hybrid Gaussian feature field, HCD codec, screen-space refinement, FDH supervision, and ws240 warm-start training.
3. The project now has a paper-facing **ScanNet direct point-query** cross-domain result under the v67 teacher-balanced fair protocol.
4. The project has credible auxiliary evidence that the learned feature field is useful beyond grounding, especially on Replica room_0 depth and segmentation.
5. The freeze package now includes formal LERF overlay/profile runs for all four main scenes plus one full ScanNet v67 evaluation profile.

## What is already paper-grade

### Main result evidence

- [output/radio_gs/reports/submission_freeze_report.md](../output/radio_gs/reports/submission_freeze_report.md)
- [output/radio_gs/reports/submission_freeze_manifest.json](../output/radio_gs/reports/submission_freeze_manifest.json)
- [output/radio_gs/reports/paper_submission_main_table.md](../output/radio_gs/reports/paper_submission_main_table.md)
- [output/radio_gs/reports/paper_submission_result_audit.md](../output/radio_gs/reports/paper_submission_result_audit.md)
- [output/radio_gs/reports/submission_freeze_profile_summary.md](../output/radio_gs/reports/submission_freeze_profile_summary.md)
- [output/radio_gs/reports/efficiency_profile.md](../output/radio_gs/reports/efficiency_profile.md)
- [output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv](../output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv)
- [paper/radio_gs_draft.tex](../paper/radio_gs_draft.tex)

### Method explanation assets

- [README.md](../README.md)
- [docs/feature_reconstruction_analysis.md](feature_reconstruction_analysis.md)

### Qualitative figures

- [output/radio_gs/reports/submission_freeze_figure_shortlist.md](../output/radio_gs/reports/submission_freeze_figure_shortlist.md)
- [output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502](../output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502)
- [output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502](../output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502)
- [output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502](../output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502)
- [output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502](../output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502)

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

### 3. Statistical confidence is improved, but still narrow

The conservative route now includes a completed four-scene `n=3` seed summary
for `figurines`, `ramen`, `teatime`, and `waldo_kitchen`. That meaningfully
lowers submission risk and resolves the earlier missing Teatime evidence, but it
is still narrower than a benchmark-wide statistical treatment.

### 4. Small-object failure analysis is still incomplete

The figurines scene strongly suggests that feature resolution and object scale are
limiting factors. This should become a targeted analysis section rather than only
an observation spread across reports.

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
4. Assemble the final qualitative figure from `submission_freeze_figure_shortlist.md`.
5. Move `paper/radio_gs_draft.tex` into the target venue template and expand related work.
6. Add a concise small-object failure analysis for Figurines.
