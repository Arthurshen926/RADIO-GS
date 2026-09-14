# Affinity domain ablation, 2026-09-14

Completed on physical GPUs 2 (full) and 3 (geometry only), with the September 4 sealed 12-train/4-validation ScanNet cohort. Both runs use 2,000 steps, identical seed, pair MLP, synthetic SAM-like perturbations and all-view pair supervision. The validation set also selects the F0.5 threshold; these are development results, not independent final test estimates.

The geometry-only model masks all four appearance-dependent channels inside the model. Its checkpoint records the feature mode, and the standard deployment loader restores the same mask. A checkpoint invariance test verifies that changing every appearance channel leaves predictions unchanged. Existing full checkpoints retain their behavior. Twelve targeted tests passed.

| Measurement | Full | Geometry only |
| --- | ---: | ---: |
| Pooled edge AP | 0.773469 | 0.732586 |
| Edge precision | 0.988722 | 0.978182 |
| Edge recall | 0.521825 | 0.533730 |
| Object best-hypothesis fragment recall | 0.621230 | 0.617758 |
| Mean object fragmentation | 2.385138 | 2.405972 |
| Fragment-weighted merge impurity | 0 | 0.001838 |
| PyTorch peak allocated bytes | 20767232 | 21061120 |
| PyTorch peak reserved bytes | 25165824 | 25165824 |

Allocator figures include training and evaluation, but exclude driver/context and non-PyTorch allocations. These fragment association measurements are not rendered LERF mIoU. Geometry-only performance largely retains the synthetic-cohort partition result while reducing edge ranking quality. This does not yet establish whether the deployment appearance mismatch causes the LERF degradation.

Outputs:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_affinity_domain_ablation_20260914/full_v2/`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_affinity_domain_ablation_20260914/geometry_only_v2/`

Each directory contains exact command arguments, log, successful exit receipt, checkpoint and full report. Initial directories without `_v2` retain failed startup logs: peak-memory reset ran before CUDA initialization. Initialization order was repaired before successful reruns.

Next: use the restored geometry-only checkpoint in the four-scene LERF object builder with the same association and rendering configuration as the full baseline, then compare extent purity and frozen text-query output. No new LERF result is claimed here.
