# RADIO-GS Submission Status

This document turns the current experiment inventory into a concrete submission
status note. It is meant to answer a narrow question: how close is the current
repository to a top-conference paper package?

## Current level

- **Method maturity**: mature research prototype with paper-facing automation and archive discipline in place
- **Current submission maturity**: conservative paper package with LaTeX draft
- **Estimated top-conference completion**: about 92% for the conservative
  RADIO-GS package, and about 87% for a stricter VPR-backed primitive-level
  CTF-GS paper after the registered LERF direct-selection upgrade

The repository now contains a coherent method, a strong LERF-OVS result,
repeatable evaluation code, and a completed 10-scene fair ScanNet v67 aggregate.
The remaining conservative-route work is mostly presentation: venue-formatted
manuscript prose, related-work tightening, and deciding which diagnostic tables
belong in the appendix. A stricter 2025-style primitive-level paper is now
credible if framed as VPR-backed direct selection, but still needs careful
discussion around Waldo Kitchen and protocol provenance.

## Current source of truth

The current generated freeze package is:

- [submission_freeze_report.md](../output/radio_gs/reports/submission_freeze_report.md)
- [submission_freeze_manifest.json](../output/radio_gs/reports/submission_freeze_manifest.json)
- [submission_freeze_profile_summary.md](../output/radio_gs/reports/submission_freeze_profile_summary.md)
- [submission_freeze_figure_shortlist.md](../output/radio_gs/reports/submission_freeze_figure_shortlist.md)
- [efficiency_profile.md](../output/radio_gs/reports/efficiency_profile.md)
- [baseline_source_verification.md](../output/radio_gs/reports/baseline_source_verification.md)
- [lerf_component_ablation.md](../output/radio_gs/reports/lerf_component_ablation.md)
- [lerf_direct_3d_debug_audit.md](../output/radio_gs/reports/lerf_direct_3d_debug_audit.md)
- [vpr_protocol_card.md](../output/radio_gs/reports/vpr_protocol_card.md)
- [vpr_contribution_weighting_ablation.md](../output/radio_gs/reports/vpr_contribution_weighting_ablation.md)
- [lerf_direct_3d_published_context.md](../output/radio_gs/reports/lerf_direct_3d_published_context.md)
- [expert4_improvement_completion_audit.md](../output/radio_gs/reports/expert4_improvement_completion_audit.md)
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
   localization accuracy (0.8712 vs. 0.7985) and improve calibrated macro mIoU
   (0.5243 vs. 0.4634) under the frozen SigLIP2 scoring setup.
7. DINOv3/SAM3 adaptor supervision now has completed full LERF sweeps. The
   promoted adaptor-enhanced candidate keeps macro LocAcc at 0.8712 and improves
   macro mIoU from 0.4941 to 0.4979 by using the Figurines spatial text-heatmap
   cross-view checkpoint plus relation/region checkpoints for Ramen and Teatime.
8. DINOv3/SAM3 downstream adaptor probes are now complete. Under the formal
   prompt-constrained task sweep, rendered features beat the frame-wise teacher
   on SAM3-adaptor mask mIoU for point prompts (0.4169 vs. 0.3700), box prompts
   (0.6638 vs. 0.6560), and mask propagation (0.3756 vs. 0.3583). The latest
   DINOv3 robust readout raises rendered mask propagation from the previous
   0.7376 LocAcc / 0.3684 mIoU to 0.7730 LocAcc / 0.4456 mIoU. This exceeds the
   previous fixed-readout teacher mIoU reference of 0.3921, but the same robust
   readout also raises teacher to 0.7801 LocAcc / 0.4806 mIoU, so the correct
   claim is a substantially narrowed DINO propagation gap. A mutual matching +
   homography RANSAC diagnostic further raises rendered dense-match similarity
   to 0.9277 and improves qualitative match cleanliness, but it does not close
   the DINO mask-propagation gap and stays diagnostic.
9. The ProFuse-inspired DINO cross-view branch is implemented and evaluated as a
   diagnostic path. It can increase thresholded overlap on some LERF scenes
   (for example Waldo high-temperature mIoU), but it still lowers LocAcc, so it
   is not promoted to the main LERF table.
10. ScanNet DINO cross-view is now a full 10-scene ablation at conservative
    weight 0.001. It improves macro split19 mIoU from 0.3538 to 0.3640,
    split15 from 0.3573 to 0.3662, and split10 from 0.4293 to 0.4308.
    The strongest balanced ScanNet point-query support row now uses a
    label-free contextual kNN readout (`k=8`, `candidate_k=32`) plus
    scene-mean calibration at alpha 0.5, improving split19/15/10 mIoU to
    0.3637/0.3708/0.4512 with mAcc 0.6033/0.6224/0.7079. Alpha 0.75 further
    raises split10 mIoU to 0.4534 but weakens split19/15 and mAcc, and ScanNet
    alias prompts remain mixed.
11. The LERF peak-preservation diagnostic is now positive on Figurines:
    DINO cross-view with `text_heatmap_distill_mode: spatial` keeps LocAcc at
    0.8214 and improves mIoU from 0.4308 to 0.4343.
12. The core LERF component ablation is now closed under the controlled seed-7
    protocol. CTR/HCD is the strongest architectural dependency (`w/o CTR`
    macro LocAcc 0.5306 vs. full 0.8578), FGC/FDH warm-start provides the
    largest training route gain (`w/o FGC` 0.8018), and VFA/HGCF mainly affect
    peak stability rather than raw region coverage.
13. LERF direct 3D object selection has been upgraded with a no-GT
    rendered-feature-to-primitive registration readout. Under the
    OpenGaussian-style query-select-render protocol, the original Gaussian-center
    result was 0.0804 macro mIoU / 0.0932 macro Acc@0.25 under its earlier
    top10% selector and 0.012 / 0.009 under the same top2% selector used by VPR;
    the registered softmax24 result is 0.3421 / 0.5547, the conservative
    registered+voxel top2% result is 0.3850 / 0.6428, and the current
   paper-facing registered+voxel mean+2.5std selector with a fixed 0.5% floor and 1.8% cap reaches 0.4227 / 0.6906
   after increasing the all-pose VPR registration budget to 128 views, with
   a 1.5% cap accuracy-oriented diagnostic at 0.4184 / 0.7013. Applying the
   optional GT-free RGB snap to the rendered evaluation mask raises the
   paper-facing refined row to 0.4554 / 0.7014.
    This closes most of the primitive-level gap and slightly exceeds
    OpenGaussian's official macro mIoU and Acc@0.25 reference, while still
    requiring a provenance caveat because baselines are not locally rerun and
    Waldo Kitchen remains weak.
    A separate published-context table records newer method references
    (Dr. Splat, CAGS, InstanceGaussian, OpenGaFF) and prevents the paper from
    overclaiming global direct-3D SOTA.
14. The VPR protocol is now explicitly auditable: the paper includes a protocol
    card, separates rendered-view grounding from 3D primitive querying, and
    reports optional VPR cache storage separately from persistent compact
    checkpoint storage.
15. A Dr. Splat-inspired contribution-weighted VPR variant has been implemented
    and tested. It is a negative ablation: alpha weighting reaches 0.2978 macro
    mIoU / 0.5389 Acc@0.25 and alpha-depth weighting reaches 0.2967 / 0.5345,
    below the uniform VPR top2% baseline at 0.3850 / 0.6428 and the adaptive
    128-view meanstd2p5+floor0.005+cap0.018 selector at 0.4227 / 0.6906, so it is not promoted.
16. A GT-free adaptive mean+std rendered-mask threshold was tested for boundary
    refinement. It keeps LERF LocAcc at 0.8712 but lowers macro mIoU to 0.4939,
    so the calibrated fixed global threshold-0.60 row remains the paper-facing
    rendered-grounding metric.

## What is already paper-grade

### Main result evidence

- [output/radio_gs/reports/submission_freeze_report.md](../output/radio_gs/reports/submission_freeze_report.md)
- [output/radio_gs/reports/submission_freeze_manifest.json](../output/radio_gs/reports/submission_freeze_manifest.json)
- [output/radio_gs/reports/paper_submission_main_table.md](../output/radio_gs/reports/paper_submission_main_table.md)
- [output/radio_gs/reports/paper_submission_result_audit.md](../output/radio_gs/reports/paper_submission_result_audit.md)
- [output/radio_gs/reports/submission_freeze_profile_summary.md](../output/radio_gs/reports/submission_freeze_profile_summary.md)
- [output/radio_gs/reports/efficiency_profile.md](../output/radio_gs/reports/efficiency_profile.md)
- [output/radio_gs/reports/efficiency_cost_table.md](../output/radio_gs/reports/efficiency_cost_table.md)
- [output/radio_gs/reports/lerf_component_ablation.md](../output/radio_gs/reports/lerf_component_ablation.md)
- [output/radio_gs/reports/lerf_direct_3d_selection.md](../output/radio_gs/reports/lerf_direct_3d_selection.md)
- [output/radio_gs/reports/lerf_direct_3d_debug_audit.md](../output/radio_gs/reports/lerf_direct_3d_debug_audit.md)
- [output/radio_gs/reports/lerf_direct_3d_query_audit_rgb_snap_sil0p60.md](../output/radio_gs/reports/lerf_direct_3d_query_audit_rgb_snap_sil0p60.md)
- [output/radio_gs/reports/lerf_rendered_grounding_adaptive_threshold_diagnostic.md](../output/radio_gs/reports/lerf_rendered_grounding_adaptive_threshold_diagnostic.md)
- [output/radio_gs/reports/vpr_protocol_card.md](../output/radio_gs/reports/vpr_protocol_card.md)
- [output/radio_gs/reports/vpr_contribution_weighting_ablation.md](../output/radio_gs/reports/vpr_contribution_weighting_ablation.md)
- [output/radio_gs/reports/lerf_direct_3d_published_context.md](../output/radio_gs/reports/lerf_direct_3d_published_context.md)
- [output/radio_gs/reports/expert4_improvement_completion_audit.md](../output/radio_gs/reports/expert4_improvement_completion_audit.md)
- [output/radio_gs/reports/expert5_improvement_update.md](../output/radio_gs/reports/expert5_improvement_update.md)
- [output/radio_gs/reports/scannet_prompt_calibration_ablation.md](../output/radio_gs/reports/scannet_prompt_calibration_ablation.md)
- [output/lerf_sam_dino_tasks/formal_v6_dino_topk_area200_peak/lerf_sam_dino_task_report.md](../output/lerf_sam_dino_tasks/formal_v6_dino_topk_area200_peak/lerf_sam_dino_task_report.md)
- [output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_all_20260514/lerf_sam_dino_task_report.md](../output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_all_20260514/lerf_sam_dino_task_report.md)
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
- [paper/figures/lerf_vpr_direct_3d_qualitative.png](../paper/figures/lerf_vpr_direct_3d_qualitative.png)
- [output/radio_gs/reports/lerf_vpr_direct_3d_qualitative_manifest.json](../output/radio_gs/reports/lerf_vpr_direct_3d_qualitative_manifest.json)
- [output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502](../output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502)
- [paper/figures/lerf_adaptor_downstream_qualitative.png](../paper/figures/lerf_adaptor_downstream_qualitative.png)

## What still blocks a strong main-track submission

### 1. Public baseline provenance is now paper-anchored, with protocol caveats

The main LERF-OVS comparison has a strong core trio: LERF, LangSplat, LEGaussians.
The repository now replaces the old unresolved placeholder rows with exact
official-source values from LERF ICCV 2023 Table 1, LangSplat CVPR 2024 Table 1,
and the LEGaussians CVPR 2024 supplementary Table 5 LA row. The remaining caveat
is protocol scope: these rows are cross-paper context, not reproduced
same-evaluator baselines.

### 2. Cross-domain generalization is improved, but needs paper-safe framing

The ScanNet v67 teacher-balanced direct point-query run now provides a 10-scene
cross-domain table. The remaining risk is not lack of ScanNet evidence, but
clean framing: label-supervised and GT-label-balanced ScanNet diagnostics must
stay out of the main fair table.

The DINO cross-view branch has also been expanded from a two-scene diagnostic to
a full 10-scene ScanNet ablation:

| Split | Base mIoU | DINO-CV mIoU | Delta | Base mAcc | DINO-CV mAcc | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 19 classes | 0.3538 | 0.3640 | +0.0102 | 0.6076 | 0.6205 | +0.0128 |
| 15 classes | 0.3573 | 0.3662 | +0.0089 | 0.6203 | 0.6313 | +0.0110 |
| 10 classes | 0.4293 | 0.4308 | +0.0014 | 0.7051 | 0.7071 | +0.0020 |

The current strongest balanced ScanNet row is not the DINO branch but a
contextual direct point readout on the same v67 checkpoints:

| Split | Gaussian-index mIoU | kNN+calib mIoU | Delta | Gaussian-index mAcc | kNN+calib mAcc | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 19 classes | 0.3538 | 0.3637 | +0.0099 | 0.6076 | 0.6033 | -0.0043 |
| 15 classes | 0.3573 | 0.3708 | +0.0135 | 0.6203 | 0.6224 | +0.0021 |
| 10 classes | 0.4293 | 0.4512 | +0.0219 | 0.7051 | 0.7079 | +0.0028 |

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

### 5. Efficiency and cost now have a paper-facing table

The paper now has formal wall-clock and peak-VRAM profiles for all four frozen
LERF overlay evaluations and the full 10-scene ScanNet v67 point-query
evaluation. `efficiency_cost_table.md` and `paper/efficiency_cost_table.tex`
convert those profiles plus the storage-footprint accounting into a main-paper
efficiency/cost table. Training throughput remains supplementary because it is
log-derived rather than explicit GPU telemetry.

### 6. Main-table provenance is fixed, but future runs still need the same discipline

The LERF-OVS main table is now backed by exact rendered JSON artifacts, and the
repository now freezes `paper_submission_main_table.md` as the canonical paper
main table while `paper_main_table.md` is demoted to supporting seed-robustness
evidence. The remaining requirement is procedural: every new ablation, seed run,
and cross-domain result should be frozen with the same exact-report discipline
instead of relying on informal notes.

### 7. Direct LERF primitive selection is now registration-backed

The new `eval_lerf_direct_3d_selection.py` evaluator follows the
OpenGaussian-style protocol: score 3D primitives with text, render selected
primitives to LERF annotated views, then compute mIoU and Acc@0.25. The first
Gaussian-center implementation was weak, so GPU4/GPU5 were used to add a
VPR registration readout: render VFA-refined RADIO features from
posed views, project them to SigLIP2 space, register visible samples back to
Gaussian centers with depth/alpha checks, and query the registered primitive
embeddings. The current fixed protocol (`registered_view`, `softmax_scene`,
128 all-pose registration views, GT-free voxel-max context aggregation,
meanstd2p5 selector with a fixed 0.5% floor and 1.8% cap) reaches 0.4227 macro mIoU and 0.6906 macro Acc@0.25; the
optional GT-free RGB snap row reaches 0.4554 / 0.7014; the
fixed top0p02 selector remains a conservative audit at 0.3850 / 0.6428. It
should be promoted as a VPR-backed primitive-level result, with the caveat that
Waldo Kitchen remains below OpenGaussian and external baselines are still
official-source rather than locally rerun.

The final implementation pass also added alpha and alpha-depth registration
weighting, which approximates contribution-aware primitive assignment from
registration-style methods. The measured result is negative under the current
center-sampling approximation, so this branch is retained as an audit/appendix
result instead of a method claim.

## Immediate implementation priorities

1. Move `paper/radio_gs_draft.tex` into the target venue template and polish related work.
2. Decide how much of the fixed-protocol seed-robustness table belongs in the main paper versus appendix.
3. Keep the adaptor-enhanced LERF candidate as an ablation/selector result unless the main-row policy changes.
4. Keep the small-object failure analysis, storage footprint table, and efficiency/cost table in the main paper unless page limits force them to appendix.
5. Reproduce external baselines under the local evaluator only if the paper wants a strict SOTA leaderboard claim.
6. If the paper adopts the VPR title, present the registered LERF direct 3D
   result in the main experiments and keep Waldo/threshold/view-count probes in
   the appendix.
7. Do not count optional persisted VPR caches as compact scene storage. The main
   compactness claim should use persistent checkpoint footprint; the optional
   inference cache row is reported for transparency.
