# Paper Artifact Snapshot

This directory contains the small, paper-facing result files that make the
current RADIO-GS/GaussFM evidence package auditable without following the local
`output/` symlink. Large checkpoints, rendered images, datasets, and feature
caches remain outside this snapshot.

## Integrity

- `checksums.txt` is the checksum manifest for this directory.
- `final_rows.yaml` is the canonical three-track result registry.
- `final_consistency_audit.md` records the required report/table/figure status.
- `active_goal_completion_audit.md` maps the active expert-request goal to
  current evidence, in-flight rows, and remaining blocking gaps.
- `provenance_report.md`, `paper_assets_manifest.json`, and
  `submission_freeze_manifest.json` preserve the freeze/provenance context.
- `tpami_reproducibility_package_20260601.md` is the submission-level entry
  point for integrity checks, canonical rows, table/figure regeneration, and
  data-dependent rerun templates.
- `tpami_large_asset_release_manifest_20260601.md` lists the large checkpoints,
  datasets, evaluation outputs, and optional diagnostic caches that must be
  staged outside the small artifact snapshot for a public release.
- All small generated `output/radio_gs/reports/*.md` and `*.json` files that
  support the paper package are snapshotted here, excluding launch logs and
  LaTeX duplicates.

Verify the snapshot from the repository root with:

```bash
sha256sum -c paper/artifacts/checksums.txt
```

## T1: LERF Rendered-View OVS

- `lerf_rendered_grounding_paper_ckpt_threshold_sweep.json`: paper-facing
  rendered-view threshold sweep and main GaussFM row source.
- `lerf_rendered_grounding_paper_ckpt_threshold_sweep.md`: markdown summary
  for the same threshold sweep.
- `lerf_rendered_grounding_peak_component_20260524.{md,json}`:
  GT-free peak-connected-component rendered-mask readout promoted for mask
  mIoU/boundary quality.
- `lerf_rendered_grounding_adaptive_threshold_diagnostic.{md,json}`:
  diagnostic adaptive-threshold row.
- `lerf_rendered_grounding_boundary_calibration_report.{md,json}`:
  rendered-view boundary calibration diagnostic.
- `lerf_rendered_grounding_rgb_snap_report.{md,json}`: RGB-snap diagnostic.
- `controlled_evidence_table.json`: frame-wise RADIO reference, nearest-view cache,
  explicit 1280-D memory, and full GaussFM controlled evidence.
- `controlled_evidence_table.md`: markdown summary for the same controlled
  evidence.
- `teacher_vs_ctfgs_2d_usability_20260525.{md,json}`: consolidated 2D
  frame-wise-RADIO-vs-rendered GaussFM feature-usability evidence across SigLIP2 text
  grounding and frozen SAM3/DINOv3 task probes. The current report records
  6/6 selected primary downstream wins for rendered GaussFM features, while
  keeping secondary SAM LocAcc and DINO dense-HitRate caveats separate.
- `dino_sam_boundary_and_waldo_recovery_20260528.{md,json}`: traceable
  record for the DINO multi-head SAM-boundary readout that flips DINO mask
  propagation mIoU in favor of the rendered field, plus the Waldo small-object
  heatmap-recovery diagnostic that improves Waldo Acc but is not promoted
  globally because a fixed pixel floor regresses Ramen.
- `unified_multi_head_feature_quality_field_20260525.{md,json}`: method-level
  implementation note for the explicit quality/visibility field readouts,
  training targets, and current promoted rows before full retraining with the
  new heads.
- `lerf_nearest_view_cache_baseline.json`: nearest-view RADIO cache baseline.
- `lerf_nearest_view_cache_baseline.md`: markdown summary for the cache
  baseline.
- `lerf_per_gaussian_1280d_baseline.json`: explicit per-Gaussian RADIO memory
  baseline.
- `lerf_per_gaussian_1280d_baseline*.{md,json}`: aggregate, per-scene, and
  Waldo-probe explicit-memory baseline support files.
- `lerf_failure_analysis.md`: rendered-view fragile-category analysis.
- `compression_downstream_correlation.{md,json}`: compression/error to
  downstream metric support.
- `feature_error_text_relevance_report.{md,json}`: feature error versus text
  relevance support.

## T2: LERF Direct 3D Selection

- `lerf_direct_3d_selection.md`: VPR fixed-threshold direct-selection table.
- `vpr_protocol_card.md`: promoted direct-3D protocol card.
- `lerf_direct_3d_selection_{figurines,ramen,teatime,waldo_kitchen}_results.json`:
  per-scene VPR source JSONs for the promoted direct-3D run.
- `lerf_direct_3d_query_audit.md`: query-level bootstrap and worst-query audit
  for the `thr0p25` VPR row.
- `lerf_direct_3d_query_audit_rgb_snap_sil0p60.md`: RGB-snap/silhouette
  diagnostic query audit.
- `lerf_direct_3d_silhouette_sweep_report.json`: silhouette-threshold
  diagnostic source JSON.
- `lerf_direct_3d_silhouette_sweep_report.md`: markdown summary for the
  silhouette sweep.
- `lerf_direct_3d_selection_*_20260514.md`,
  `lerf_direct_3d_cache_component_cap_sweep.md`, and
  `lerf_direct_3d_component_cache_ablation.md`: direct-3D diagnostic runs kept
  as appendix/provenance support.
- `lerf_direct3d_confidence_coverage_analysis.{md,json}`: GT-free
  confidence/coverage analysis.
- `waldo_failure_stratification.{md,json}`: Waldo Kitchen failure breakdown.
- `lerf_sam3_box_global_threshold_sweep_20260517_geometry.json`: strict
  pad16 frozen-SAM3 box-readout geometry source.
- `lerf_sam3_box_global_threshold_sweep_20260517_geometry.md`: markdown
  summary for the strict pad16 geometry run.
- `lerf_sam3_box_global_threshold_sweep_20260516.{md,json}`: fixed-threshold
  SAM3-box sweep before the geometry-map rerun.
- `lerf_direct_3d_published_context.md`: published/context rows that are not
  promoted as same-evaluator SOTA claims.
- `lerf_direct3d_proposal_memory_ablation_20260524.{md,json}`:
  proposal-memory score smoothing audit. The branch is retained as an ablation
  because it did not beat the strict pad16 SAM3-box direct-3D row.
- `lerf_direct3d_sam3_training_view_proposal_registration_20260525.{md,json}`:
  official SAM3 training-view object-proposal registration audit. It is
  implemented but not promoted because the four-scene macro mIoU drops.
- `lerf_direct_3d_debug_audit.md`: direct-3D implementation/debug audit.
- `lerf_query_breakdown.{md,json}`: per-query direct-3D diagnostics with
  object-weighted aggregates separated from the paper-facing scene-mean
  aggregate.
- `vpr_contribution_weighting_ablation.md`,
  `vpr_field_confidence_weighting_20260515.md`, and
  `vpr_field_consistency_20260515.md`: VPR diagnostic ablations.
- `lerf_main_qualitative_comparison.{md,json}`: main-paper direct-3D
  qualitative comparison manifest; the promoted panels use compact direct-field
  masks without VPR-cache or official RGB SAM3 readout at evaluation.
- `lerf_direct3d_prompt_ensemble_support_policy_20260528.{md,json}`:
  compact direct-3D support-policy result. This is the promoted no-VPR,
  no-official-RGB-SAM readout for the Direct3D table, but it uses a GT-free
  RGB/GrabCut component-support guard and is not the strict no-RGB one-map row.
- `lerf_direct3d_compact_readout_ablation_20260528.{md,json}`:
  strict compact one-map versus guarded compact readout ablation. The strict
  one-map row disables VPR cache, official RGB SAM, and RGB postprocess.
- `lerf_direct3d_score_component_guard_20260528.{md,json}`:
  promoted compact direct-3D score-component support guard. It ranks RGB/GrabCut
  components by rendered compact score-heatmap mass and gives the current
  overlap-balanced direct-3D row.
- `lerf_vpr_direct_3d_qualitative_manifest.json`,
  `lerf_sam3_box_direct_3d_qualitative_manifest.json`, and
  `lerf_sam3_box_direct_3d_qualitative_pad16_manifest.json`: qualitative
  figure manifests.
- `boundary_error_readout_report.{md,json}`: boundary-readout mechanism report.
- `alpha_depth_boundary_alignment_report.{md,json}`: alpha/depth discontinuity
  alignment report for the strict pad16 SAM3-box row.
- `alpha_depth_boundary_case_figure_manifest.{md,json}`: source manifest for
  the alpha/depth boundary case figure.

## T3: ScanNet Point-Cloud Probe

- `scannet_published_context_table.tex`: paper-facing ScanNet table containing
  published Gaussian-domain context and the local GT-point query result.
- `scannet_pointcloud_radio_gs_vala8_reproduced_benchmark_20260615.json`:
  derivative paper-facing summary that mirrors all eight rows from the raw
  contextual-kNN source below; it is not an independent reproduction artifact.
- `scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.{md,json}`:
  canonical local contextual-kNN source with scene-mean calibration, one-step
  spatial logit propagation, and all eight per-scene rows.
- `scannet_pointcloud_radio_gs_vala8_dino_cv_proposal_memory_vxl005_a04_lm005_results.{md,json}`:
  proposal-memory readout ablation. It improves split19 detail metrics
  (0.3931/0.6255) but is not promoted because split15/split10 decrease.
- `scannet_contextual_knn12_alpha_sweep_20260524.md`: label-free calibration
  sweep that promotes the k16/cand80 alpha=0.45 direct point-query readout.
- `scannet_pointcloud_radio_gs_vala8_direct_point_query_results.{md,json}` and
  older `v67` files: historical diagnostics only; do not use them as
  paper-facing ScanNet numbers.
- `scannet_dino_cv_ablation.md` and `scannet_prompt_calibration_ablation.md`:
  ScanNet ablation records kept outside the compact main table.

## Baseline Reproduction

- `external_baseline_audit.{md,json}`: local clone/build/reproduction status for
  OpenGaussian, LangSplatV2, OccamLGS, GAGS, Dr. Splat, LangSplat,
  LEGaussians, CAGS, Semantic Gaussians, and LaGa.
- `baseline_source_verification.md`: earlier paper baseline source
  verification snapshot.
- `langsplatv2_lerf_summary.{md,json}`: current LangSplatV2 compatibility
  summary. As of this snapshot, all four LERF scenes are complete.
- `langsplat_classic_lerf_summary.{md,json}`: current classic LangSplat
  compatibility summary. All four scenes are evaluated after the split-aware
  train/test feature path fix.
- `legaussians_lerf_readiness_audit.{md,json}`: LEGaussians LERF readiness
  check. The repo/native extensions are ready, but all four LERF scenes still
  lack the official quantized feature files required before training.
- `laga_lerf_readiness_audit.{md,json}`: LaGa LERF readiness check. The
  repo/native extensions and data are ready, but scene checkpoints, affinity
  checkpoints, and notebook descriptor outputs are still absent for all four
  scenes.
- `semantic_gaussians_readiness_audit.{md,json}`: Semantic Gaussians ScanNet
  readiness check. Native extensions and repo entry points are present, but
  MinkowskiEngine/PyTorch-Encoding imports and train/fusion/distill/eval
  outputs are still missing. The four default ScanNet scene zips are extracted,
  and all four scenes have usable raw language features.
- `cags_lerf_summary.{md,json}`: current CAGS local LERF compatibility summary.
  All four scenes are evaluated, with missing rendered masks counted in the
  source JSONs; this is a diagnostic reproduction row rather than a SOTA claim.
- `opengaussian_vs_radio_gs_report.md`: OpenGaussian/RADIO-GS comparison notes.

## Paper Tables And Guards

- `protocol_alignment_audit_20260711.md`: consolidated T1/T2/T3 audit that
  separates restored historical post-hoc numbers from the frozen VALA-aligned
  reruns, records exact result hashes, explains the GrabCut readout, and audits
  the LangSplat CLIP+SAM replacement path.
- `protocol_alignment_scannet_20260711.md`: detailed ScanNet mesh-query versus
  Gaussian-domain protocol and metric audit.

- `paper_submission_main_table.md` and `paper_submission_result_audit.md`:
  paper-facing main table and source audit.
- `submission_freeze_report.md`, `submission_freeze_profile_summary.md`,
  `submission_freeze_figure_shortlist.md`,
  `submission_freeze_qualitative_comparison.md`, and
  `submission_readiness_checklist.md`: freeze/readiness support reports.
- `submission_freeze_gpu_queue.md`: historical GPU queue status from the
  freeze package.
- `efficiency_profile.md` and `efficiency_profile_summary.md`: profiling
  support reports.
- `expert4_improvement_completion_audit.md` and `expert5_improvement_update.md`:
  expert-followup completion/status reports.
- `seed_robustness_summary.md`: seed robustness summary.
- `paper_benchmark_targets.md`: benchmark target planning snapshot.
- `controlled_baseline_gap_audit.md`: controlled-baseline gap summary.
- `lerf_component_ablation.{md,json}`: component ablation source report.
- `train_feature_field_audit.{md,json}`: code-audit source report for the
  training entry point.
- `efficiency_cost_table.md` and `storage_footprint_report.md`: cost/storage
  reporting support.

The repository-level guards are:

```bash
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/validate_final_rows_registry.py
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/validate_paper_claims.py
```
