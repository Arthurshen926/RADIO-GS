# Controlled Baseline Gap Audit, 2026-05-17

## Scope

This audit records the current status of two expert-requested controlled
baselines for the LERF paper table:

- nearest-view / cached RADIO baseline under the same LERF evaluator;
- full per-Gaussian 1280-D explicit RADIO-feature baseline under the same
  LERF evaluator.

## Search Result

The nearest-view cached RADIO baseline is now measured in
`output/radio_gs/reports/lerf_nearest_view_cache_baseline.md` and included in
`output/radio_gs/reports/controlled_evidence_table.md`. The measured row is an
unwarped cache-only baseline: each annotated target frame uses the closest
cached RADIO feature frame by camera-center distance, excluding the target
frame itself, then runs the same LERF text scorer and thresholded-mask
evaluator. It reaches 0.2722 macro LocAcc and 0.1545 macro mIoU.

The full per-Gaussian 1280-D explicit RADIO-feature baseline is now measured
in `output/radio_gs/reports/lerf_per_gaussian_1280d_baseline.md` and included
in `output/radio_gs/reports/controlled_evidence_table.md`. The measured row
registers cached frame-wise RADIO features to visible Gaussian centers, stores the
scene memory as fp16 1280-D per-Gaussian vectors, renders those vectors to the
annotated LERF views, and evaluates with the same frozen SigLIP2 scorer. It
reaches 0.5642 macro LocAcc and 0.3182 macro mIoU, with 0.2020 mean registered
Gaussian fraction and 1039.7 MiB mean fp16 feature storage.

Existing artifacts that should not be conflated with these measured rows:

| Artifact | Why it is not the requested measured row |
|---|---|
| `output/radio_gs/reports/controlled_evidence_table.md` | Includes both measured controlled rows; use its source column to distinguish cache-only, explicit 1280-D scene memory, and compact GaussFM. |
| `output/radio_gs/reports/storage_footprint_report.md` | Reports direct 1280-D storage accounting only; performance comes from `lerf_per_gaussian_1280d_baseline.md`. |
| `output/radio_gs/reports/lerf_component_ablation.md` | Includes compact architectural ablations such as w/o hybrid/HCD, but not full raw 1280-D per-Gaussian RADIO reference features. |
| `output/radio_gs/reports/submission_freeze_manifest.json` | Freezes current main paper rows and provenance; it is not the controlled-baseline source for the raw 1280-D row. |
| `radio_gs/configs/replica_explicit*` and Replica outputs | Replica explicit-feature experiments are not LERF same-evaluator baselines. |
| frame-wise RADIO reference row | This is a 2D frame-wise reference row, not a nearest-view 3D cache or explicit per-Gaussian 1280-D scene baseline. |

## Current Decision

Promote the nearest-view cache row only as a cache-only baseline, not as a 3D
scene memory. Promote the measured full per-Gaussian 1280-D explicit row as the
raw-feature 3D scene-memory control. The controlled evidence table should keep
the row below the frame-wise RADIO and nearest-view cache controls, and above
the compact GaussFM row, because it has 3D memory but is not compact and only
partially supports direct 3D query through registered primitive features.

## GPU State At Audit

Latest `nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader,nounits`
reported during the follow-up pass:

```text
0, 7494, 16588, 0
1, 812, 23270, 100
2, 470, 23612, 59
3, 762, 23320, 39
4, 7220, 16862, 0
5, 7418, 16664, 0
```

The final four-scene summary was generated from cached per-Gaussian features
after the parallel registration jobs completed.

## Resulting Table Status

The controlled evidence table now covers frame-wise RADIO, nearest-view cache, full
per-Gaussian 1280-D explicit scene memory, full compact GaussFM, core ablations,
direct-3D readouts, storage, and runtime. The remaining limitation is not a
missing row, but that the explicit row has low registered-Gaussian coverage
(0.2020 mean fraction), which should be reported as part of the raw-feature
baseline rather than hidden.
