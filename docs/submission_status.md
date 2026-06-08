# RADIO-GS Submission Status

This document turns the current experiment inventory into a concrete submission
status note. It is meant to answer how close the current repository is to a
top-tier journal submission package.

## Current level

- **Method maturity**: mature research prototype with paper-facing automation and archive discipline in place
- **Current submission maturity**: TPAMI-format journal draft and supplementary
  material with compiled PDFs, figure shortlist, paper manifest, claim
  validator, and provenance artifacts
- **Estimated journal-submission completion**: about 96--98% for the current
  CTF-GS package. The main paper and supplementary material now compile cleanly
  and use a journal-style narrative; the remaining work is mostly final human
  proofreading, optional staging of the large-asset release package, and
  optional external-baseline reruns if a strict leaderboard claim is desired.

## 2026-06-01 TPAMI Paper Polish

- Rewrote the TPAMI evaluation/provenance section from freeze-report language
  into formal journal language.
- Removed remaining `current/promoted/paper-facing/diagnostic/artifact appendix`
  wording from the TPAMI main text and table captions.
- Synchronized the claim validator with the compact Direct3D score-component
  main row while still guarding the fixed-threshold VPR row and overclaim risks.
- Added a standalone IEEE-style supplementary PDF at
  [paper/radio_gs_tpami_supplement.tex](../paper/radio_gs_tpami_supplement.tex)
  / [paper/radio_gs_tpami_supplement.pdf](../paper/radio_gs_tpami_supplement.pdf),
  containing rendered-mask calibration, Direct3D controls, feature-memory
  baselines, boundary/geometry controls, training auditability, and additional
  qualitative figures.
- Added a submission-level reproducibility entry point at
  [tpami_reproducibility_package_20260601.md](../paper/artifacts/tpami_reproducibility_package_20260601.md),
  covering integrity checks, canonical `final_rows.yaml` entries, table/figure
  regeneration, and data-dependent rerun templates for all three benchmark
  tracks.
- Added a large-asset release manifest at
  [tpami_large_asset_release_manifest_20260601.md](../paper/artifacts/tpami_large_asset_release_manifest_20260601.md),
  separating required LERF/ScanNet checkpoints, frozen foundation weights,
  evaluation outputs, optional VPR/SAM3 diagnostics, and release checksum
  commands from the small paper-artifact snapshot.
- Verified the TPAMI main/supplement PDF builds, claim validator, checksum
  manifest, and whitespace checks after the paper-polish pass.
- Ran a final figure-quality pass: replaced the weaker LERF 2D/3D `green toy
  chair` example with `green apple`, regenerated
  [lerf_2d3d_ovs_qualitative.png](../paper/figures/lerf_2d3d_ovs_qualitative.png),
  and redrew [radio_gs_framework.pdf](../paper/figures/radio_gs_framework.pdf)
  as a cleaner claim-first TPAMI Figure 1.
- Added upload-facing materials:
  [tpami_cover_letter_draft.md](../paper/tpami_cover_letter_draft.md),
  [tpami_submission_checklist.md](../paper/tpami_submission_checklist.md), and
  [tpami_submission_mode_guide.md](../paper/tpami_submission_mode_guide.md).
  The current PDFs remain anonymous review drafts; the guide records how to
  switch to author-visible manuscripts if the TPAMI portal requests them.

## 2026-05-31 TPAMI Draft Update

- Added the TPAMI/IEEEtran draft at
  [paper/radio_gs_tpami.tex](../paper/radio_gs_tpami.tex), with compiled PDF at
  [paper/radio_gs_tpami.pdf](../paper/radio_gs_tpami.pdf).
- Replaced the framework figure with a journal-style diagram centered on one
  compact foundation-feature Gaussian map and three inference readouts. VPR is
  now shown as a training-only registered teacher/bridge, not as the direct-3D
  inference cache.
- Audited qualitative figures in
  [figure_quality_audit_tpami_20260531.md](../paper/artifacts/figure_quality_audit_tpami_20260531.md):
  keep the LERF 2D/3D OVS figure, ScanNet binary query figure, and direct-3D
  support-policy ablation in the main paper; move DINO/SAM/VPR control figures
  to the supplement.
- Compressed the widest main-paper tables (`lerf_ovs_main_table.tex`,
  `lerf_component_ablation_table.tex`, and `efficiency_cost_table.tex`). The
  TPAMI compile now has no undefined references/citations and no overfull hbox
  warnings after the current pass.
- Polished the TPAMI abstract, introduction, contribution list, method setup,
  inference-readout description, discussion, and limitations to remove
  internal draft wording and make the claim boundaries explicit: VPR is a
  training bridge, RGB/GrabCut is a GT-free classical support policy, and
  official SAM3 box readout is separate from the core direct-field claim.

The repository now contains a coherent TPAMI-format method paper, compiled
supplementary material, a curated main-figure set, repeatable evaluation code,
and a VALA/OpenGaFF-8 ScanNet DINO-CV contextual kNN aggregate. Remaining work
before an actual upload is non-blocking operational work: final human
proofreading, author metadata, and optional external-baseline reruns only if a
strict same-evaluator leaderboard claim is desired.

## Current source of truth

The current generated freeze package is:

- [submission_freeze_report.md](../output/radio_gs/reports/submission_freeze_report.md)
- [submission_freeze_manifest.json](../output/radio_gs/reports/submission_freeze_manifest.json)
- [verify_submission_provenance.py](../radio_gs/scripts/verify_submission_provenance.py)
- [validate_final_rows_registry.py](../radio_gs/scripts/validate_final_rows_registry.py)
- [validate_paper_claims.py](../radio_gs/scripts/validate_paper_claims.py)
- [paper artifact README](../paper/artifacts/README.md)
- [active goal completion audit](../paper/artifacts/active_goal_completion_audit.md)
- [paper artifact checksum snapshot](../paper/artifacts/checksums.txt)
- [submission freeze report public snapshot](../paper/artifacts/submission_freeze_report.md)
- [baseline source verification public snapshot](../paper/artifacts/baseline_source_verification.md)
- [paper main table public snapshot](../paper/artifacts/paper_submission_main_table.md)
- [paper result audit public snapshot](../paper/artifacts/paper_submission_result_audit.md)
- [direct-3D query audit public snapshot](../paper/artifacts/lerf_direct_3d_query_audit.md)
- [direct-3D selection public snapshot](../paper/artifacts/lerf_direct_3d_selection.md)
- [direct-3D Figurines result JSON snapshot](../paper/artifacts/lerf_direct_3d_selection_figurines_results.json)
- [direct-3D Ramen result JSON snapshot](../paper/artifacts/lerf_direct_3d_selection_ramen_results.json)
- [direct-3D Teatime result JSON snapshot](../paper/artifacts/lerf_direct_3d_selection_teatime_results.json)
- [direct-3D Waldo Kitchen result JSON snapshot](../paper/artifacts/lerf_direct_3d_selection_waldo_kitchen_results.json)
- [VPR protocol card public snapshot](../paper/artifacts/vpr_protocol_card.md)
- [published direct-3D context public snapshot](../paper/artifacts/lerf_direct_3d_published_context.md)
- [direct-3D confidence/coverage public snapshot](../paper/artifacts/lerf_direct3d_confidence_coverage_analysis.md)
- [Waldo failure stratification public snapshot](../paper/artifacts/waldo_failure_stratification.md)
- [LERF failure analysis public snapshot](../paper/artifacts/lerf_failure_analysis.md)
- [component ablation public snapshot](../paper/artifacts/lerf_component_ablation.md)
- [alpha/depth boundary alignment public snapshot](../paper/artifacts/alpha_depth_boundary_alignment_report.md)
- [boundary-error readout public snapshot](../paper/artifacts/boundary_error_readout_report.md)
- [controlled baseline gap audit public snapshot](../paper/artifacts/controlled_baseline_gap_audit.md)
- [train feature-field audit public snapshot](../paper/artifacts/train_feature_field_audit.md)
- [efficiency/cost public snapshot](../paper/artifacts/efficiency_cost_table.md)
- [storage footprint public snapshot](../paper/artifacts/storage_footprint_report.md)
- [ScanNet DINO-CV contextual kNN VALA8 snapshot](../paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json)
- [ScanNet category stability / failure analysis](../paper/artifacts/scannet_category_stability_failure_analysis.md)
- [LangSplatV2 LERF summary snapshot](../paper/artifacts/langsplatv2_lerf_summary.md)
- [submission_freeze_profile_summary.md](../output/radio_gs/reports/submission_freeze_profile_summary.md)
- [submission_freeze_figure_shortlist.md](../output/radio_gs/reports/submission_freeze_figure_shortlist.md)
- [efficiency_profile.md](../output/radio_gs/reports/efficiency_profile.md)
- [controlled_evidence_table.md](../output/radio_gs/reports/controlled_evidence_table.md)
- [lerf_nearest_view_cache_baseline.md](../output/radio_gs/reports/lerf_nearest_view_cache_baseline.md)
- [lerf_per_gaussian_1280d_baseline.md](../output/radio_gs/reports/lerf_per_gaussian_1280d_baseline.md)
- [train_feature_field_audit.md](../output/radio_gs/reports/train_feature_field_audit.md)
- [baseline_source_verification.md](../output/radio_gs/reports/baseline_source_verification.md)
- [lerf_component_ablation.md](../output/radio_gs/reports/lerf_component_ablation.md)
- [lerf_direct_3d_debug_audit.md](../output/radio_gs/reports/lerf_direct_3d_debug_audit.md)
- [waldo_failure_stratification.md](../output/radio_gs/reports/waldo_failure_stratification.md)
- [vpr_protocol_card.md](../output/radio_gs/reports/vpr_protocol_card.md)
- [vpr_contribution_weighting_ablation.md](../output/radio_gs/reports/vpr_contribution_weighting_ablation.md)
- [lerf_direct_3d_published_context.md](../output/radio_gs/reports/lerf_direct_3d_published_context.md)
- [expert4_improvement_completion_audit.md](../output/radio_gs/reports/expert4_improvement_completion_audit.md)
- [expert latest follow-up audit](experiments/2026-05-16-expert-latest-followup-audit.md)
- [paper_draft_current.md](paper_draft_current.md)
- [PROJECT_MAINLINE.md](PROJECT_MAINLINE.md)
- [radio_gs_draft.tex](../paper/radio_gs_draft.tex)
- [radio_gs_tpami.tex](../paper/radio_gs_tpami.tex)
- [TPAMI submission strategy](../paper/artifacts/tpami_submission_strategy.md)
- [TPAMI figure quality audit](../paper/artifacts/figure_quality_audit_tpami_20260531.md)
- [TPAMI reproducibility package](../paper/artifacts/tpami_reproducibility_package_20260601.md)
- [TPAMI large-asset release manifest](../paper/artifacts/tpami_large_asset_release_manifest_20260601.md)

These generated files supersede older manually edited status notes when numbers
conflict.

The freeze manifest is now expected to pass:

```bash
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/verify_submission_provenance.py \
  output/radio_gs/reports/submission_freeze_manifest.json --check_paths --root /root/RADIO-GS
```

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
3. The project now has a paper-facing **ScanNet direct point-query** cross-domain result under the VALA/OpenGaFF-8 DINO-CV contextual kNN protocol.
4. The project has credible auxiliary evidence that the learned feature field is useful beyond grounding, especially on Replica room_0 depth and segmentation.
5. The freeze package now includes formal LERF overlay/profile runs for all four main scenes plus one legacy ScanNet evaluation profile.
6. The LERF evaluator already includes a same-protocol feature-source check:
   rendered CTF-GS features outperform original RADIO RGB features on macro
   localization accuracy (0.8598 vs. 0.7985) and improve calibrated macro mIoU
   (0.5707 vs. 0.4634) under the frozen SigLIP2 scoring setup.
7. DINOv3/SAM3 adaptor supervision now has completed full LERF sweeps. The
   promoted adaptor-enhanced candidate keeps macro LocAcc at 0.8712 and improves
   macro mIoU from 0.4941 to 0.4979 by using the Figurines spatial text-heatmap
   cross-view checkpoint plus relation/region checkpoints for Ramen and Teatime.
8. DINOv3/SAM3 downstream adaptor probes are now complete. Under the formal
   prompt-constrained task sweep, rendered features beat the frame-wise teacher
   on SAM3-adaptor mask mIoU for point prompts (0.4173 vs. 0.3700), box prompts
   (0.6638 vs. 0.6560), and mask propagation (0.3756 vs. 0.3583). The promoted
   DINOv3 multi-head readout uses DINO support with SAM-adaptor boundary
   refinement from the same rendered foundation-feature field; it raises
   rendered mask propagation to 0.7872 LocAcc / 0.4677 mIoU, exceeding the
   same-readout frame-wise teacher at 0.7660 / 0.4606. Dense-match mean
   similarity also favors rendered features (0.9048 vs. 0.8547), while dense
   HitRate remains teacher-stronger (0.5396 vs. 0.5723) and is kept as a
   secondary caveat.
9. The ProFuse-inspired DINO cross-view branch is implemented and evaluated as a
   diagnostic path. It can increase thresholded overlap on some LERF scenes
   (for example Waldo high-temperature mIoU), but it still lowers LocAcc, so it
   is not promoted to the main LERF table.
10. ScanNet DINO cross-view is now evaluated on the fixed VALA/OpenGaFF-8
    subset. It improves Gaussian-index split19/15/10 mIoU to
    0.3704/0.3718/0.4390 with mAcc 0.6159/0.6268/0.7020. The strongest balanced
    ScanNet point-query support row uses the DINO-CV compact field with a
    label-free contextual kNN readout (`k=16`, `candidate_k=80`) plus scene-mean
    calibration at alpha 0.45 and one-step spatial logit propagation, reaching
    0.3806/0.3871/0.4711 mIoU and 0.6129/0.6315/0.7200 mAcc. Alpha 0.75 further raises split10 mIoU to
    0.4612 but weakens split19/15 balance, and ScanNet alias prompts remain
    mixed.
11. The LERF peak-preservation diagnostic is now positive on Figurines:
    DINO cross-view with `text_heatmap_distill_mode: spatial` keeps LocAcc at
    0.8214 and improves mIoU from 0.4308 to 0.4343.
12. The core LERF component ablation is now closed under the controlled seed-7
    protocol. CTR/HCD is the strongest architectural dependency (`w/o CTR`
    macro LocAcc 0.5306 vs. full 0.8578), FGC/FDH warm-start provides the
    largest training route gain (`w/o FGC` 0.8018), and VFA/HGCF mainly affect
    peak stability rather than raw region coverage.
13. LERF direct 3D object selection now has a compact-field main row under the
    OpenGaussian-style query-select-render protocol. The strict no-RGB one-map
    row reaches 0.4570 macro mIoU / 0.6851 Acc@0.25; the paper-facing compact
    score-component guard uses the same primitive scores, a frozen SigLIP2
    prompt ensemble, and GT-free RGB/GrabCut boundary snapping plus
    score-heatmap component filtering, reaching 0.5014 / 0.7044. It does not
    read a VPR cache and does not call the official RGB SAM decoder. The
    official-SAM3 box row remains a diagnostic boundary-assisted upper row
    rather than the compact Direct3D main result. Waldo Kitchen remains the
    weakest scene, but the separate low-support heatmap recovery diagnostic
    improves Waldo from 0.3312 / 0.5455 to 0.3414 / 0.5909; it is not promoted
    globally because a fixed recovery floor regresses Ramen.
    This closes most of the primitive-level gap and exceeds the main published
    context references available in the project, while still requiring a
    provenance caveat because baselines are not locally rerun.
    A separate published-context table records newer method references
    (Dr. Splat, CAGS, InstanceGaussian, OpenGaFF) and prevents the paper from
    overclaiming global direct-3D SOTA.
14. The VPR protocol is now explicitly auditable: the paper includes a protocol
    card, separates rendered-view grounding from 3D primitive querying, and
    reports optional VPR cache storage separately from persistent compact
    checkpoint storage.
15. Dr. Splat-inspired VPR variants have been implemented and tested. The
    earlier center-sampled alpha weighting reaches 0.2978 macro mIoU / 0.5389
    Acc@0.25 and alpha-depth reaches 0.2967 / 0.5345, below the uniform VPR
    top2% baseline at 0.3850 / 0.6428 and below the promoted 128-view
    threshold-0.25 + RGB snap selector at 0.4801 / 0.6760. The new
    rasterizer-level paths are also negative on Figurines: all-footprint
    uniform raster hits 0.0002 mIoU, per-pixel dominant alpha-depth hits 0.0178
    mIoU under the 128-view budget, and per-Gaussian top-footprint alpha hits
    0.0004 mIoU under official views. Proposal/OPR component selection on the
    cached strong VPR scores drops Figurines to 0.0430 mIoU. These branches are
    implemented but not promoted.
16. VPR-to-field consistency now has a GT-free registration-confidence weighted
    training variant. It uses normalized `log1p(view_counts)` from the VPR
    feature cache as sample weights and raises the direct-field diagnostic from
    0.4119 mIoU / 0.5876 Acc@0.25 to 0.4363 / 0.6191 under the same threshold
    sweep, floor/cap, and RGB-snap protocol. The improvement is not uniform
    across scenes: Ramen and Waldo improve, Teatime is slightly positive, and
    Figurines drops, so the streamed registered VPR readout remains the main
    direct-3D result.
17. SAM3-adaptor supervision now includes mask-logit and prompt-conditioned
    feature-only boundary heads in addition to the previous soft-region
    prototype loss. Official SAM3 code/weights are available for diagnostics,
    but the promoted paper-facing SAM boundary readouts are driven by the
    reconstructed CTF-GS/RADIO adaptor features and do not call the official RGB
    SAM decoder at evaluation time.
18. A GT-free adaptive mean+std rendered-mask threshold was tested for boundary
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
- [output/radio_gs/reports/compression_downstream_correlation.md](../output/radio_gs/reports/compression_downstream_correlation.md)
- [output/radio_gs/reports/feature_error_text_relevance_report.md](../output/radio_gs/reports/feature_error_text_relevance_report.md)
- [output/radio_gs/reports/boundary_error_readout_report.md](../output/radio_gs/reports/boundary_error_readout_report.md)
- [output/radio_gs/reports/alpha_depth_boundary_alignment_report.md](../output/radio_gs/reports/alpha_depth_boundary_alignment_report.md)
- [output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260517_geometry.md](../output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260517_geometry.md)
- [output/radio_gs/reports/train_feature_field_audit.md](../output/radio_gs/reports/train_feature_field_audit.md)
- [output/radio_gs/reports/lerf_component_ablation.md](../output/radio_gs/reports/lerf_component_ablation.md)
- [output/radio_gs/reports/controlled_evidence_table.md](../output/radio_gs/reports/controlled_evidence_table.md)
- [output/radio_gs/reports/lerf_nearest_view_cache_baseline.md](../output/radio_gs/reports/lerf_nearest_view_cache_baseline.md)
- [output/radio_gs/reports/lerf_per_gaussian_1280d_baseline.md](../output/radio_gs/reports/lerf_per_gaussian_1280d_baseline.md)
- [output/radio_gs/reports/controlled_baseline_gap_audit.md](../output/radio_gs/reports/controlled_baseline_gap_audit.md)
- [output/radio_gs/reports/lerf_direct_3d_selection.md](../output/radio_gs/reports/lerf_direct_3d_selection.md)
- [output/radio_gs/reports/lerf_direct_3d_debug_audit.md](../output/radio_gs/reports/lerf_direct_3d_debug_audit.md)
- [output/radio_gs/reports/waldo_failure_stratification.md](../output/radio_gs/reports/waldo_failure_stratification.md)
- [output/radio_gs/reports/lerf_direct3d_confidence_coverage_analysis.md](../output/radio_gs/reports/lerf_direct3d_confidence_coverage_analysis.md)
- [output/radio_gs/reports/lerf_direct_3d_query_audit_rgb_snap_sil0p60.md](../output/radio_gs/reports/lerf_direct_3d_query_audit_rgb_snap_sil0p60.md)
- [output/radio_gs/reports/lerf_rendered_grounding_adaptive_threshold_diagnostic.md](../output/radio_gs/reports/lerf_rendered_grounding_adaptive_threshold_diagnostic.md)
- [output/radio_gs/reports/vpr_protocol_card.md](../output/radio_gs/reports/vpr_protocol_card.md)
- [output/radio_gs/reports/vpr_contribution_weighting_ablation.md](../output/radio_gs/reports/vpr_contribution_weighting_ablation.md)
- [output/radio_gs/reports/vpr_field_confidence_weighting_20260515.md](../output/radio_gs/reports/vpr_field_confidence_weighting_20260515.md)
- [docs/raster_proposal_audit_20260515.md](raster_proposal_audit_20260515.md)
- [output/radio_gs/reports/lerf_direct_3d_published_context.md](../output/radio_gs/reports/lerf_direct_3d_published_context.md)
- [output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260516.md](../output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260516.md)
- [output/radio_gs/reports/expert4_improvement_completion_audit.md](../output/radio_gs/reports/expert4_improvement_completion_audit.md)
- [output/radio_gs/reports/expert5_improvement_update.md](../output/radio_gs/reports/expert5_improvement_update.md)
- [output/radio_gs/reports/scannet_prompt_calibration_ablation.md](../output/radio_gs/reports/scannet_prompt_calibration_ablation.md)
- [output/lerf_sam_dino_tasks/formal_v12c_dino_sam3_boundary_v9readout_gpu_20260528/lerf_sam_dino_task_report.md](../output/lerf_sam_dino_tasks/formal_v12c_dino_sam3_boundary_v9readout_gpu_20260528/lerf_sam_dino_task_report.md)
- [output/lerf_sam_dino_tasks/formal_v9_dino_readout_sweep_20260514.md](../output/lerf_sam_dino_tasks/formal_v9_dino_readout_sweep_20260514.md)
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
- [paper/figures/lerf_vpr_direct_3d_qualitative.png](../paper/figures/lerf_vpr_direct_3d_qualitative.png) now uses the compact direct-field + official SAM3 box readout masks.
- [paper/figures/lerf_sam3_box_direct_3d_qualitative_pad16.png](../paper/figures/lerf_sam3_box_direct_3d_qualitative_pad16.png) is the boundary-focused pad16 diagnostic.
- [paper/figures/radio_gs_framework.png](../paper/figures/radio_gs_framework.png)
- [output/radio_gs/reports/lerf_sam3_box_direct_3d_qualitative_manifest.json](../output/radio_gs/reports/lerf_sam3_box_direct_3d_qualitative_manifest.json)
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
The local OpenGaussian LERF blocker is now explicitly audited in
`output/baselines/opengaussian/opengaussian_vs_radio_gs_report.md` and
`paper/artifacts/external_baseline_audit.md`. All four local LERF scenes have
images and labels, and `figurines`, `ramen`, `teatime`, and `waldo_kitchen` now
have guarded complete `language_features` symlinks from the LangSplat-format
extraction cache. The generated assets use the documented `--skip_mask_nms`
compatibility patch, so the official OpenGaussian LERF training/evaluation
recipe is not yet a strict full-benchmark rerun.
The 2026-05-17 external-baseline pass extends this audit to the latest P0/P1
methods requested by the expert reply. OpenGaussian, LangSplatV2, OccamLGS,
GAGS, Dr. Splat, LangSplat, LEGaussians, CAGS, Semantic Gaussians, and LaGa are
now tracked in `paper/artifacts/final_rows.yaml`, with local clone paths,
commits, and blocker notes where code is available. CAGS' bundled rasterizer and
PyG compiled ops have been rebuilt into `output/baselines/cags/local_site`,
clearing the import/ABI blockers through `train.py --help` and
`render_lerf_by_text.py --help`. LaGa's local setup now restores the
`third_party/kmeans_pytorch` gitlink through a `.gitmodules` mapping, builds
`simple_knn` plus both diff rasterizers into `output/baselines/laga/local_site`,
and reaches `train_scene.py --help` plus `train_affinity_features.py --help`
with a chunked `torch.cdist` fallback for the incompatible PyTorch3D KNN import.
GAGS and Dr. Splat now also have isolated local sites for their vendored native
and SAM packages, clearing import/ABI blockers through `train.py --help` plus
`render.py --help` for GAGS and `train.py --help` plus
`render_activation.py --help` for Dr. Splat.
LangSplat now has an isolated local site for vendored `simple_knn`,
`langsplat-rasterization`, and `segment-anything-langsplat` under
`output/baselines/langsplat/local_site`, with NumPy pinned to 1.26.4; `train.py`,
`render.py`, and `eval/evaluate_iou_loc.py` reach CLI help. Strict comparison
still needs pretrained checkpoints or a fresh same-protocol run plus
same-evaluator metric export.
LEGaussians also has a local `.gitmodules` repair for both preprocess Segment
Anything gitlinks, clean recursive submodule status, vendored
`simple_knn`/`diff_gaussian_rasterization` builds in
`output/baselines/legaussians/local_site`, and `train.py --help` CLI smoke;
strict reproduction still requires dataset-specific preprocessing, training,
rendering, and same-evaluator metric export.
Semantic Gaussians now has vendored `simple_knn`, `rgbd-rasterization`,
`channel-rasterization`, and `segment-anything` builds in
`output/baselines/semantic_gaussians/local_site`, with compatible
NumPy/scikit-image/viser/TensorFlow imports; `train.py` and `fusion.py` import
successfully. Strict ScanNet export is still blocked by
`MinkowskiEngine==0.5.4` failing to compile against the host PyTorch
2.7.1/CUDA headers at `spmm.cu`, and the LSeg visualizer path still needs
PyTorch-Encoding (`encoding`).
OpenGaFF is tracked as a published context row only: its 2026-05-07 arXiv source
states that code will be publicly released upon acceptance, and no public
implementation was found in arXiv metadata/source or web search on 2026-05-18.
The detailed status is recorded in
`docs/experiments/2026-05-17-three-task-open-baseline-audit.md`. Since that
audit began, OpenGaussian has an all-four-scene LERF compatibility readout
(0.4273 mIoU / 0.5865 Acc@0.25 / 0.4727 Acc@0.5), and OccamLGS has an
all-four-scene normalized pre-rendered compatibility readout (0.8221 LocAcc /
0.4515 mIoU over 208 objects). LangSplatV2 has a completed teatime compatibility
sanity row and now has guarded all-level compatibility extensions in flight:
ramen is training on GPU2, figurines is training on GPU3, and waldo_kitchen is
queued behind the safe-headroom GPU guard. These rows reduce the external-baseline gap,
but they remain compatibility-asset bring-up rows rather than strict
released-checkpoint/official-extraction SOTA rows.
The internal nearest-view cache control is now measured under the same LERF
readout: unwarped closest cached RADIO frames reach 0.2722 macro LocAcc /
0.1545 macro mIoU, far below rendered CTF-GS at 0.8598 / 0.5707. The full
per-Gaussian 1280-D explicit same-evaluator row is also measured: registered
fp16 teacher features attached to Gaussian primitives reach 0.5642 macro LocAcc
/ 0.3182 macro mIoU with 0.2020 mean registered-Gaussian fraction and 1039.7
MiB mean fp16 feature storage. This closes the controlled raw-feature baseline
gap.
Alpha/depth discontinuity instrumentation is implemented in the Direct3D
evaluator, and the strict pad16 geometry-map rerun now gives 208/208 query
records and overlays in `alpha_depth_boundary_alignment_report.md`. The
alpha/depth correlations with boundary error are weak, so this should be used
as mechanism context rather than causal occlusion proof.
The training entry point is now audited in `train_feature_field_audit.md`.
Critical guards are present and feature/text/cache tensor loads now go through
`load_training_tensor_cache`. After modularizing support code under
`radio_gs/training/`, the audit now passes with a 3735-line entry script.

### 2. Cross-domain generalization is improved, but needs paper-safe framing

The current strongest balanced ScanNet row combines the DINO-CV compact field
with contextual direct point readout on the VALA/OpenGaFF-8 subset. Older v67
Gaussian-index and non-DINO contextual rows are now historical diagnostics, not
paper-facing ScanNet numbers.

| Split | DINO-CV kNN+calib mIoU | DINO-CV kNN+calib mAcc |
|---|---:|---:|
| 19 classes | 0.3806 | 0.6129 |
| 15 classes | 0.3871 | 0.6315 |
| 10 classes | 0.4711 | 0.7200 |

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
LERF overlay evaluations and a legacy 10-scene ScanNet point-query
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
128 all-pose registration views, fixed score threshold 0.25, 0.5% floor,
1.8% cap, and GT-free RGB snap) reaches 0.4801 macro mIoU and 0.6760 macro Acc@0.25; the
fixed top0p02 selector remains a conservative audit at 0.3850 / 0.6428. It
should be promoted as a VPR-backed primitive-level result, with the caveat that
Waldo Kitchen remains below OpenGaussian and LERF external baselines are still
official-source or compatibility-asset bring-up rows rather than strict local
official-policy reruns.

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
