# RADIO-GS Submission Freeze Design

Date: 2026-05-02

## Goal

Move RADIO-GS from a strong research prototype to a conservative, high-completion paper package. The target is not to maximize every branch result, but to make a defensible submission package where the main claims, protocols, numbers, provenance, figures, and remaining risks are internally consistent.

## Scope

This design covers the first implementation wave:

1. Freeze paper-facing protocols for LERF-OVS, ScanNet, Replica, and efficiency.
2. Generate a single submission package report from JSON/CSV artifacts instead of manually syncing multiple notes.
3. Separate paper-safe results from diagnostics and label-leak/oracle experiments.
4. Keep GPU4 and GPU5 busy only with protocol-clean gap jobs that improve the frozen package.
5. Refresh docs that currently contain stale or conflicting numbers.

Out of scope for this wave:

- Rewriting the full paper manuscript.
- Adding a new model family.
- Reporting label-supervised ScanNet diagnostics as main open-vocabulary results.
- Claiming universal SOTA over methods whose protocol is not matched or reproduced.

## Paper-Facing Protocols

### LERF-OVS

Primary evidence remains rendered-feature open-vocabulary grounding. The current best JSON-backed row is usable as a best-achievable table:

- Figurines: 0.821
- Ramen: 0.901
- Teatime: 0.898
- Waldo Kitchen: 0.864
- Macro: 0.871

For the main paper, this must be labeled as a fixed selection rule if used: best rendered LocAcc per scene across discovered variants/checkpoints/temperature sweeps, with mIoU and canonical path tie-breaks. A safer statistical companion is the existing n=3 FDH vs noFDH table. The package should expose both and state which one is main.

### ScanNet

The main fair cross-domain candidate is the v67 teacher-balanced direct point protocol:

- query mode: `gaussian_index`
- Gaussian position mode: `label_point`
- opacity filter: `label_index`
- text prompts: 5-template ScanNet prompt ensemble
- no hard semantic CE as the reported main protocol

The submission package must compute a 10-scene aggregate directly from:

`output/scannet_pointcloud_eval/scene*_v67_teacherbalanced_fromv63_best_gidx_labelpoint/scannet_pointcloud_radio_gs_results.json`

Label-supervised or GT-label-balanced runs such as v43/v62 remain diagnostic or upper-bound evidence only.

### Replica

Replica remains supporting evidence for task utility: depth, segmentation, and feature reconstruction. It should not be framed as the main SOTA benchmark. The package should point to existing room0 reports and summarize the strongest depth/semantic takeaways without overclaiming.

### Efficiency

Existing profiles are sparse. The package should include them as preliminary efficiency evidence and explicitly list missing measurements. Additional GPU jobs should prioritize a room0 or LERF evaluation profile that is directly comparable to the frozen main protocols.

## Implementation Shape

Add a small report-building path rather than editing many markdown files by hand:

- `radio_gs/scripts/build_submission_freeze_report.py`
  - Reads LERF best-scene CSV/JSON.
  - Reads ScanNet v67 per-scene JSON files.
  - Reads existing efficiency summary when present.
  - Writes a markdown report and machine-readable manifest.
  - Emits clear warnings for unresolved baselines, diagnostic-only ScanNet runs, missing figures, and stale report mismatches.

- `tests/test_build_submission_freeze_report.py`
  - Uses temporary synthetic artifacts.
  - Verifies macro computation.
  - Verifies diagnostic warnings.
  - Verifies report content includes paper-safe protocol labels.

- `docs/submission_status.md`
  - Updated to reference the freeze report as the current source of truth.

- `output/radio_gs/reports/submission_freeze_report.md`
  - Generated report, not manually edited.

## GPU Queue Policy

GPU4 and GPU5 are available and can be used. The queue should prefer:

1. Missing fair ScanNet v67 aggregate/visualization jobs if any JSON/figure is missing.
2. Fixed-protocol LERF sweep or visualization jobs that align with the formal evaluator.
3. Efficiency profiles for one LERF scene and one ScanNet/Replica workload.

Do not spend GPU time on:

- New label-supervised ScanNet diagnostics unless a fair result regresses and needs explanation.
- Per-scene oracle temperature chasing for the main paper number.
- Branches that cannot be explained as part of the conservative submission package.

## Success Criteria

The project reaches roughly 90% submission readiness when:

1. One generated report lists all main claims and maps each claim to exact artifacts.
2. LERF, ScanNet, Replica, and efficiency tables have current numbers and provenance.
3. ScanNet fair/diagnostic boundaries are explicit.
4. External baseline status is documented as official, reproduced, or unresolved.
5. A reviewer can reproduce every RADIO-GS number in the paper package from a JSON/CSV source path.
6. The remaining work is mostly manuscript prose and final baseline cleanup, not core evidence generation.

## Risks

- External LERF baseline rows remain unresolved and cannot support a strong SOTA claim until official/reproduced provenance is closed.
- Per-scene best selection can look like oracle selection. The paper needs either a fixed validation selection rule or a statistical companion table.
- ScanNet v67 must be reported as teacher-balanced pseudo-label training, with label-supervised diagnostics kept out of the main table.
- Some reports in `output/radio_gs/reports` are stale; generated reports should supersede them.

