# RADIO-GS Submission Status

This document turns the current experiment inventory into a concrete submission
status note. It is meant to answer a narrow question: how close is the current
repository to a top-conference paper package?

## Current level

- **Method maturity**: strong research prototype
- **Current submission maturity**: partial paper package
- **Estimated top-conference completion**: about 55% to 60%

The repository already contains a coherent method, a strong main task result, and
repeatable evaluation code. What is still missing is the breadth and rigor needed
for a conference main-track review: broader public comparisons, statistical
confidence, cross-domain evidence, and a frozen submission narrative.

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
3. The project has credible auxiliary evidence that the learned feature field is useful beyond grounding, especially on Replica room_0 depth and segmentation.

## What is already paper-grade

### Main result evidence

- [output/radio_gs/reports/sota_comparison_table.md](../output/radio_gs/reports/sota_comparison_table.md)
- [output/radio_gs/reports/comprehensive_results.md](../output/radio_gs/reports/comprehensive_results.md)
- [output/radio_gs/reports/paper_submission_main_table.md](../output/radio_gs/reports/paper_submission_main_table.md)
- [output/radio_gs/reports/paper_submission_result_audit.md](../output/radio_gs/reports/paper_submission_result_audit.md)
- [output/lerf_ovs_eval/figurines_best/lerf_ovs_results.json](../output/lerf_ovs_eval/figurines_best/lerf_ovs_results.json)
- [output/lerf_ovs_eval/ramen_best/lerf_ovs_results.json](../output/lerf_ovs_eval/ramen_best/lerf_ovs_results.json)
- [output/lerf_ovs_eval/teatime_best/lerf_ovs_results.json](../output/lerf_ovs_eval/teatime_best/lerf_ovs_results.json)
- [output/lerf_ovs_eval/waldo_best/lerf_ovs_results.json](../output/lerf_ovs_eval/waldo_best/lerf_ovs_results.json)

### Method explanation assets

- [README.md](../README.md)
- [docs/feature_reconstruction_analysis.md](feature_reconstruction_analysis.md)

### Qualitative figures

- [output/radio_gs/reports/figure_shortlist.md](../output/radio_gs/reports/figure_shortlist.md)
- [output/radio_gs/paper_figures/fig_grounding_comparison.png](../output/radio_gs/paper_figures/fig_grounding_comparison.png)
- [output/radio_gs/paper_figures/fig_radio_flow_comparison.png](../output/radio_gs/paper_figures/fig_radio_flow_comparison.png)
- [output/radio_gs/paper_figures/fig_room0_pipeline_comparison.png](../output/radio_gs/paper_figures/fig_room0_pipeline_comparison.png)

## What still blocks a strong main-track submission

### 1. Public baseline provenance is conservatively frozen, but not paper-anchored

The main LERF-OVS comparison has a strong core trio: LERF, LangSplat, LEGaussians.
That is enough to form a serious main table, and the repository now explicitly
freezes those external rows as unresolved draft placeholders rather than claiming
they are exact paper-anchored values. The remaining weakness is not ambiguity,
but that the paper-facing comparison still relies on protocol-misaligned
cross-paper external rows.

### 2. Cross-domain generalization is still weak

Replica plus LERF-OVS is not enough to support a strong generalization claim for
top-conference review. At least one additional domain, ideally ScanNet, should be
added.

### 3. Statistical confidence is improved, but still narrow

The conservative route now includes a completed four-scene `n=3` seed summary
for `figurines`, `ramen`, `teatime`, and `waldo_kitchen`. That meaningfully
lowers submission risk and resolves the earlier missing Teatime evidence, but it
is still narrower than a benchmark-wide statistical treatment.

### 4. Small-object failure analysis is still incomplete

The figurines scene strongly suggests that feature resolution and object scale are
limiting factors. This should become a targeted analysis section rather than only
an observation spread across reports.

### 5. Efficiency and cost are not yet submission-ready

The paper still needs a dedicated table for training cost, memory, inference
latency, and feature-resolution trade-offs.

### 6. Main-table provenance is fixed, but future runs still need the same discipline

The LERF-OVS main table is now backed by exact rendered JSON artifacts, and the
repository now freezes `paper_submission_main_table.md` as the canonical paper
main table while `paper_main_table.md` is demoted to supporting seed-robustness
evidence. The remaining requirement is procedural: every new ablation, seed run,
and cross-domain result should be frozen with the same exact-report discipline
instead of relying on informal notes.

## Immediate implementation priorities

1. Validate the LERF depth-only pure-frozen recipe and freeze the follow-up four-scene comparison.
2. Add a ScanNet or similarly distinct domain experiment.
3. Add a 2x feature-resolution figurines study and document its effect on small-object grounding.
4. Extend the current key-scene three-seed summary into a broader benchmark-wide statistical section.
5. Add a dedicated efficiency/cost table with runtime and memory profiles.
6. Convert the current report notes into the actual abstract, introduction, contributions, and main experiment sections.
