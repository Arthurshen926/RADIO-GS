# Generic prompt/point-to-primitive discrimination — 2026-07-15

## Retained method changes

The canonical field, official DINO/SAM capability banks, support graph,
evidence weights, propagation solver, fixed threshold, and metric protocols are
unchanged.  Only typed query compilation is improved.

For image and registered-2D evidence, the prototype set is now anchored by the
responsibility-weighted spherical mean.  Remaining slots use deterministic
weighted farthest-point sampling.  The mean is a robust aggregate of the
complete prompt support; FPS preserves multi-modal detail.  No labels,
benchmark state, or learned parameters are involved.

For world-3D points, anisotropic Mahalanobis seed weights are evaluated only on
the 64 closest Euclidean primitives (four times the fixed k=16 support-graph
neighborhood).  This prevents a distant, very large Gaussian from dominating
a local world click.  Weighted FPS remains preferable for this sparse modality.

An attempted positive/negative overlap-removal rule changed all NVOS and SPIn
stage metrics by exactly zero because the upstream raster responsibility path
already makes positive and negative primitive seeds mutually exclusive.  It
was removed rather than retained as redundant complexity.

## Query-only ablation

All results use the corrected canonical-mpr-v1 fields and identical query,
seed, prototype count, solver, and fixed threshold.  There is no test-set
calibration.

| Task | Metric | canonical-mpr-v1 compiler | retained compiler | Delta |
|---|---|---:|---:|---:|
| ScanNet 3D point | unary mIoU | 0.183028 | 0.184140 | +0.001112 |
| ScanNet 3D point | propagated mIoU | 0.189502 | 0.190784 | +0.001282 |
| ScanNet 3D point | connected/core mIoU | 0.294615 | 0.299993 | +0.005378 |
| ScanNet 3D point | Acc@0.25 | 0.523810 | 0.523810 | 0.000000 |
| ScanNet 3D point | Acc@0.50 | 0.126984 | 0.126984 | 0.000000 |
| NVOS scribble | unary IoU | 0.720596 | 0.766626 | +0.046030 |
| NVOS scribble | propagated IoU | 0.720338 | 0.765273 | +0.044935 |
| NVOS scribble | connected IoU | 0.730200 | 0.744685 | +0.014485 |
| SPIn full mask | unary IoU | 0.503700 | 0.574983 | +0.071283 |
| SPIn full mask | propagated IoU | 0.940747 | 0.947549 | +0.006802 |
| SPIn full mask | connected IoU | 0.944605 | 0.948247 | +0.003642 |

The retained compiler improves every reported unary, propagated, and final IoU
without changing task protocols.  The larger unary gains on registered prompts
confirm that the main benefit is prompt-to-primitive discrimination rather
than graph propagation.

## Artifacts

- spherical-mean prompt ablations:
  `output/optimization_20260715/prompt_discrimination/spherical_mean_fps/`;
- local world-point ablation:
  `output/optimization_20260715/prompt_discrimination/weighted_fps_local64/`.

## Verification

The focused compiler/evaluator regression suite passes: 28 tests passed.  The
full repository suite reached 868 passed with three unrelated historical
failures (README baseline wording, a training-script line-count audit, and an
OpenCLIP frame-order expectation).  `git diff --check` passes.
