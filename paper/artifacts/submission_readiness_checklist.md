# Submission Readiness Checklist

Last updated: 2026-05-11

## Required 1-5 Closure

- [x] External LERF baseline provenance: official-source rows are recorded in
  `baseline_source_verification.md`, and the main table generator no longer uses
  unresolved placeholder rows.
- [x] Efficiency/cost table: `efficiency_cost_table.md` and
  `paper/efficiency_cost_table.tex` summarize explicit freeze-profile runtime,
  peak VRAM, and storage footprint evidence.
- [x] Seed robustness: `seed_robustness_summary.md` and `paper_main_table.md`
  contain the completed four-scene n=3 FDH/noFDH comparison.
- [x] ScanNet DINO cross-view ablation: `scannet_dino_cv_ablation.md` reports the
  completed 10-scene conservative-weight DINO-CV sweep.
- [x] Paper draft integration: `paper/radio_gs_draft.tex` now inputs the
  official-source LERF-OVS comparison table and the efficiency/cost table.

## Additional Protocol Closure

- [x] LERF direct 3D object selection: `lerf_direct_3d_selection.md` and
  `paper/lerf_direct_3d_selection_table.tex` report an OpenGaussian-style
  query-select-render experiment. The experiment is complete, but current
  RADIO-GS primitive-level results are weak and should be framed as limitation
  evidence unless direct 3D supervision or instance aggregation is added.

## Remaining Non-Experiment Polish

The remaining conservative-route work is presentation rather than missing
experiment closure: moving the draft into the target venue template, tightening
related work, deciding which robustness/ablation tables go to the appendix under
the page limit, and deciding whether the weak direct-selection table belongs in
the main paper or the limitations/appendix.
