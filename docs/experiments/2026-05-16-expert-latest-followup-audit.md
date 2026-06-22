# Latest Expert Recommendation Follow-Up Audit, 2026-05-16

Source recommendation file: `ChatGPT-RADIO模型多视角重建优化.md`.

## Objective Restatement

The latest recommendation asks RADIO-GS/GaussFM to move from a strong prototype
to a top-journal-ready submission package by locking claims and protocols,
separating rendered-view, primitive-level, and SAM3 boundary-readout evidence,
adding controlled tables and mechanism analyses, and making every paper number
auditable.

## Prompt-to-Artifact Checklist

| Expert requirement | Current evidence | Status | Remaining gap |
|---|---|---|---|
| Lock teacher, scenes, query set, threshold rule, evaluator, seeds, checkpoints, and table provenance | `output/radio_gs/reports/submission_freeze_report.md`, `output/radio_gs/reports/submission_freeze_manifest.json`, `radio_gs/scripts/verify_submission_provenance.py` | Done for current frozen rows | The manifest now carries row-level config, config SHA256, checkpoint, teacher/text head, feature path, selector policy, evaluator, evaluator script SHA256, seed, source JSON, and git metadata. The verifier passes with `--check_paths`. |
| Use conservative claim framing: RADIO-compatible compact 3D scene memory, not broad universal 3D understanding | `docs/paper_draft_current.md`, `paper/radio_gs_draft.tex`, `docs/PROJECT_MAINLINE.md` | Done for current draft | Continue enforcing this wording when adding new abstract/introduction edits. |
| Separate rendered-view grounding, primitive-level direct 3D selection, and SAM3 boundary completion | `output/radio_gs/reports/submission_freeze_report.md`, `paper/lerf_direct_3d_selection_table.tex`, `docs/experiments/2026-05-16-sam3-box-readout-results.md` | Done | Keep SAM3-box scene-locked and pad0 best-by-scene rows diagnostic. |
| Add fixed global threshold for SAM3 box readout before promoting it | `output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260516.md` | Done | Strict row is pad16 `thr0p25`: 0.5705 mIoU / 0.6835 Acc@0.25. |
| Keep post-hoc threshold choices out of the main protocol | `output/radio_gs/reports/final_consistency_audit.md` | Done | Scene-locked pad16 0.5972/0.7009 and pad0 legacy 0.5815/0.7150 remain diagnostics. |
| Build a convincing controlled main table | `output/radio_gs/reports/controlled_evidence_table.md`, `output/radio_gs/reports/controlled_baseline_gap_audit.md`, `output/radio_gs/reports/lerf_nearest_view_cache_baseline.md`, `output/radio_gs/reports/lerf_per_gaussian_1280d_baseline.md`, `output/radio_gs/reports/lerf_component_ablation.md`, `output/radio_gs/reports/paper_submission_main_table.md` | Done for measured controls | The consolidated table now covers teacher, nearest-view RADIO cache, per-Gaussian 1280-D explicit RADIO memory, full GaussFM, core component ablations, storage, runtime, and direct-3D readouts. The raw-feature row reaches 0.5642 macro LocAcc / 0.3182 macro mIoU with 0.2020 mean registered-Gaussian fraction. |
| Prove direct 3D contribution independent of SAM3 | `output/radio_gs/reports/lerf_direct_3d_selection.md`, `output/radio_gs/reports/vpr_protocol_card.md`, `paper/lerf_direct_3d_selection_table.tex` | Done | The strict VPR/RGB-snap primitive-selection row is kept beside the SAM3-box fixed row in the paper table, so the direct-field contribution is visible without relying on SAM3 boundary completion. |
| Turn Waldo Kitchen failure into mechanism analysis | `output/radio_gs/reports/waldo_failure_stratification.md`, `output/radio_gs/reports/lerf_direct3d_confidence_coverage_analysis.md`, `output/radio_gs/reports/boundary_error_readout_report.md`, `output/radio_gs/reports/alpha_depth_boundary_alignment_report.md`, `output/radio_gs/reports/lerf_direct_3d_query_audit_rgb_snap_sil0p60.md`, `output/radio_gs/reports/lerf_failure_analysis.md` | Done for current evidence package | Object-size, zero-prediction, scene-level view coverage, teacher-score confidence, text-margin ambiguity, boundary under/over-selection stratification, and 208/208 alpha/depth geometry-map overlays are generated. The alpha/depth correlations are weak, so frame them as mechanism context rather than causal proof. |
| Add mechanism experiments beyond benchmark numbers | VPR weighting ablation, raster/proposal negative diagnostics, adaptive threshold diagnostic, query-level audit, confidence/coverage analysis, compression/downstream correlation, feature-error/text-relevance audit, boundary-error readout, alpha/depth alignment coverage audit, alpha/depth case figure | Done for current evidence package | The strict pad16 SAM3-box geometry-map rerun is available in `output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260517_geometry.md`, and the compact case figure is generated at `paper/figures/alpha_depth_boundary_cases.png` with manifest provenance. |
| Make reproducible artifact registry paper-grade | `paper_assets_manifest.json`, `submission_freeze_manifest.json`, report generators, `radio_gs/scripts/verify_submission_provenance.py`, and tests | Done for current frozen rows | The verifier now gates required row-level provenance, evaluator script hashes, and referenced-path existence for config, checkpoint, feature/cache, source JSON, and evaluator script paths. |
| Improve code auditability before release | `output/radio_gs/reports/train_feature_field_audit.md`, existing leakage/provenance tests, `radio_gs/scripts/build_train_feature_field_audit.py`, `radio_gs/training/` | Done | The audit now passes: manifest, split, trusted-checkpoint, metrics-history, training-lock guards, wrapped feature/text/cache tensor loading, and the 3735-line modularized entry point are all verified. |

## GPU Experiment State

The strict pad16 SAM3-box geometry-map rerun was completed by using the
available GPU window across GPUs 0/4/5. The new run keeps the strict
`thr0p25` metrics unchanged and populates alpha/depth maps. The full
per-Gaussian 1280-D explicit baseline was then run with 64 registration frames
per scene across GPUs 0/4/5 and summarized from the cached per-Gaussian feature
stores.

Recent completed experiment evidence:

- SAM3-box fixed-global threshold sweep:
  `output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260516.md`.
- SAM3-box strict pad16 geometry-map rerun:
  `output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260517_geometry.md`.
- Nearest-view RADIO cache baseline:
  `output/radio_gs/reports/lerf_nearest_view_cache_baseline.md`.
- Per-Gaussian 1280-D explicit RADIO-memory baseline:
  `output/radio_gs/reports/lerf_per_gaussian_1280d_baseline.md`.
- Low-memory RGB/GrabCut global-selector sweep:
  `output/radio_gs/lerf_direct3d_lowmem_global_selector_20260516_205400_lowmem_gpu0`.
- Waldo SAM3 object-boundary branch diagnostic:
  `output/radio_gs/lerf_direct3d_sam3_object_boundary_gpu0_20260516_171631`.
- Boundary-error readout:
  `output/radio_gs/reports/boundary_error_readout_report.md`.
- Alpha/depth boundary-alignment coverage audit:
  `output/radio_gs/reports/alpha_depth_boundary_alignment_report.md`.
- Alpha/depth boundary case figure:
  `paper/figures/alpha_depth_boundary_cases.png`.

## Next Highest-Impact Tasks

1. If a broader reproducibility package is needed, extend path checks to
   non-paper diagnostic artifacts and add an environment lockfile.
