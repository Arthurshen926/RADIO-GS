# Paper Artifact Snapshot

This directory contains the small, paper-facing result files that make the
current RADIO-GS/CTF-GS evidence package auditable without following the local
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
- All small generated `output/radio_gs/reports/*.md` and `*.json` files that
  support the paper package are snapshotted here, excluding launch logs and
  LaTeX duplicates.

Verify the snapshot from the repository root with:

```bash
sha256sum -c paper/artifacts/checksums.txt
```

## T1: LERF Rendered-View OVS

- `lerf_rendered_grounding_paper_ckpt_threshold_sweep.json`: paper-facing
  rendered-view threshold sweep and main CTF-GS row source.
- `lerf_rendered_grounding_paper_ckpt_threshold_sweep.md`: markdown summary
  for the same threshold sweep.
- `lerf_rendered_grounding_adaptive_threshold_diagnostic.{md,json}`:
  diagnostic adaptive-threshold row.
- `lerf_rendered_grounding_boundary_calibration_report.{md,json}`:
  rendered-view boundary calibration diagnostic.
- `lerf_rendered_grounding_rgb_snap_report.{md,json}`: RGB-snap diagnostic.
- `controlled_evidence_table.json`: frame-wise teacher, nearest-view cache,
  explicit 1280-D memory, and full CTF-GS controlled evidence.
- `controlled_evidence_table.md`: markdown summary for the same controlled
  evidence.
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
- `lerf_direct_3d_debug_audit.md`: direct-3D implementation/debug audit.
- `lerf_query_breakdown.{md,json}`: per-query direct-3D diagnostics with
  object-weighted aggregates separated from the paper-facing scene-mean
  aggregate.
- `vpr_contribution_weighting_ablation.md`,
  `vpr_field_confidence_weighting_20260515.md`, and
  `vpr_field_consistency_20260515.md`: VPR diagnostic ablations.
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

- `opengaussian_scannet_results.json`: local OpenGaussian ScanNet reproduction.
- `scannet_pointcloud_radio_gs_v67_direct_point_query_results.json`: RADIO-GS
  v67 conservative direct point-query row.
- `scannet_pointcloud_radio_gs_v67_contextual_knn_scene_mean_a05_results.json`:
  contextual kNN + scene-mean support row.
- `scannet_dino_cv_ablation.md`: ScanNet DINO cross-view ablation.
- `scannet_prompt_calibration_ablation.md`: ScanNet prompt calibration
  ablation.

## Baseline Reproduction

- `external_baseline_audit.{md,json}`: local clone/build/reproduction status for
  OpenGaussian, LangSplatV2, OccamLGS, GAGS, Dr. Splat, LangSplat,
  LEGaussians, CAGS, Semantic Gaussians, LaGa, and OpenGaFF.
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
