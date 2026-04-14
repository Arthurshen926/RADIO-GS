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
- [output/lerf_ovs_eval/ramen_v14_fdh_ws240_240ep/lerf_ovs_results.json](../output/lerf_ovs_eval/ramen_v14_fdh_ws240_240ep/lerf_ovs_results.json)
- [output/lerf_ovs_eval/teatime_v14_fdh_ws240_240ep/lerf_ovs_results.json](../output/lerf_ovs_eval/teatime_v14_fdh_ws240_240ep/lerf_ovs_results.json)
- [output/lerf_ovs_eval/waldo_kitchen_v14_full_softmax/lerf_ovs_results.json](../output/lerf_ovs_eval/waldo_kitchen_v14_full_softmax/lerf_ovs_results.json)

### Method explanation assets

- [README.md](../README.md)
- [docs/feature_reconstruction_analysis.md](feature_reconstruction_analysis.md)

### Qualitative figures

- [output/paper_figures/room0_comparison/overview_comparison.png](../output/paper_figures/room0_comparison/overview_comparison.png)
- [output/lerf_ovs_eval/visualisations/figurines/](../output/lerf_ovs_eval/visualisations/figurines/)
- [output/lerf_ovs_eval/ramen_v14_fdh_ws240_240ep/visualisations/ramen/](../output/lerf_ovs_eval/ramen_v14_fdh_ws240_240ep/visualisations/ramen/)
- [output/lerf_ovs_eval/teatime_v14_fdh_ws240_240ep/visualisations/teatime/](../output/lerf_ovs_eval/teatime_v14_fdh_ws240_240ep/visualisations/teatime/)
- [output/lerf_ovs_eval/waldo_kitchen_v14_softmax/visualisations/waldo_kitchen/](../output/lerf_ovs_eval/waldo_kitchen_v14_softmax/visualisations/waldo_kitchen/)

## What still blocks a strong main-track submission

### 1. Public baseline coverage is not frozen

The main LERF-OVS comparison has a strong core trio: LERF, LangSplat, LEGaussians.
That is enough to form a serious main table, but the numbers still need to be
locked to exact paper references before a paper freeze.

### 2. Cross-domain generalization is still weak

Replica plus LERF-OVS is not enough to support a strong generalization claim for
top-conference review. At least one additional domain, ideally ScanNet, should be
added.

### 3. Statistical confidence is missing

Current results are mostly single-run best results. For a strong claim, at least
3 seeds are needed on key settings, especially on high-variance scenes.

### 4. Small-object failure analysis is still incomplete

The figurines scene strongly suggests that feature resolution and object scale are
limiting factors. This should become a targeted analysis section rather than only
an observation spread across reports.

### 5. Efficiency and cost are not yet submission-ready

The paper still needs a dedicated table for training cost, memory, inference
latency, and feature-resolution trade-offs.

### 6. Result provenance is not yet frozen

The internal summary tables already provide a strong paper narrative, but not
every reported best number is currently tied to an exact rendered JSON result in
the repository. Before submission freeze, the main table needs a provenance pass
so every claimed scene score is backed by a concrete run artifact.

## Immediate implementation priorities

1. Freeze the published grounding main table using the generated report files in [output/radio_gs/reports/](../output/radio_gs/reports/).
2. Add a ScanNet or similarly distinct domain experiment.
3. Add a 2x feature-resolution figurines study and document its effect on small-object grounding.
4. Add a three-seed summary for the most important checkpoints.
5. Resolve the generated result audit in output/radio_gs/reports/ so every main-table number has a frozen source.
6. Convert the current report notes into the actual abstract, introduction, contributions, and main experiment sections.