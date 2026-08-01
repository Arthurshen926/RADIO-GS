# OccamLGS LERF-2D strict reproduction (2026-08-01)

The released annotation/test frames were rendered directly from the three
iteration-30000 OccamLGS language checkpoints. The readout uses raw OpenCLIP
relevance, raw-peak feature-level selection, threshold 0.5, the released
30-by-30 OpenCV activation filter, 7-by-7 mask smoothing, and exact camera-name
matching. RGB geometry follows the released LERF intent and sees all registered
views; semantic feature lifting excludes the released `test.txt` frames.

All entries are `mIoU / localization accuracy` in percent.

| Scene | Paper | Local strict reproduction | Difference |
|---|---:|---:|---:|
| `figurines` | 58.6 / 80.4 | 61.12 / 78.57 | +2.52 / -1.83 |
| `ramen` | 51.0 / 74.7 | 59.60 / 73.24 | +8.60 / -1.46 |
| `teatime` | 70.2 / 93.2 | 72.92 / 93.22 | +2.72 / +0.02 |
| `waldo_kitchen` | 65.3 / 81.8 | 60.84 / 86.36 | -4.46 / +4.56 |
| **Scene-equal mean** | **61.3 / 82.5** | **63.62 / 82.85** | **+2.32 / +0.35** |

The query-weighted diagnostic (not the paper headline aggregation) is
63.92/81.73 over 208 annotated queries.

To avoid loading 100--300 unused high-resolution images, alpha masks, and ray
grids, the evaluator reads the full COLMAP metadata and released split first,
then materializes only the exact annotated cameras. Poses, intrinsics,
resolution, render path, checkpoints, and metric code are unchanged. This
reduced a failed 32 GB host-memory run to a stable direct evaluation.

Evidence:

- Per-scene JSON and telemetry:
  `/mnt/pool/sqy/results/RADIO-GS/output/protocol_audit_20260801/occamlgs_lerf2d_corrected_compat_v1`
  (JSON SHA-256: Figurines
  `eb56f50ac2fec82c84ba2380a5616cd0b9df3b55054bc2447a281e8a33b48e4b`,
  Ramen `19d7eefdff1c7e4e4abdb684a1bb9951e863c6bab18d7767511b8801b5b8f22e`,
  Teatime `4af8dfab2c6c9213a5cc3a4c82f021bb4c305b1078bab120f551a6828897567b`,
  Waldo Kitchen
  `e2f126f2225195a3a4388cb66cb0a5bf1663100944757a432c45214847a7cbdc`).
- Evaluator:
  `/root/RADIO-GS/radio_gs/scripts/eval_occamlgs_lerf_checkpoint.py`
- Peak GPU0 temperature over the four successful runs: 55 C.
- Peak GPU0 board power: 249.69 W under a 300 W cap.
- Peak GPU0 memory: 10,730 MiB.
- No thermal pauses or PCIe/driver failures occurred.

The aggregate reproduction reaches the paper result, so the OccamLGS LERF-2D
evaluation protocol is considered closed.
