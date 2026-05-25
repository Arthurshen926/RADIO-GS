# ScanNet OpenGaFF/VALA Published Context

Source: OpenGaFF arXiv v2, Table "Open-vocabulary semantic segmentation on
ScanNet-v2." External baseline rows are paper numbers and are not local reruns;
the OpenGaFF method row is intentionally omitted from the submission table.
CTF-GS rows are local VALA/OpenGaFF-8 evaluations on:
`scene0000_00`, `scene0062_00`, `scene0070_00`, `scene0097_00`,
`scene0140_00`, `scene0347_00`, `scene0400_00`, `scene0590_00`.

| Method | 19 mIoU | 19 mAcc | 15 mIoU | 15 mAcc | 10 mIoU | 10 mAcc |
|---|---:|---:|---:|---:|---:|---:|
| LangSplat | 2.45 | 8.59 | 3.45 | 13.21 | 6.48 | 21.89 |
| LangSplatV2 | 14.75 | 25.47 | 17.09 | 35.68 | 22.83 | 41.52 |
| OpenGaussian | 27.73 | 42.01 | 29.67 | 46.15 | 39.93 | 57.34 |
| Dr. Splat | 29.31 | 47.68 | 33.25 | 54.33 | 44.19 | 65.19 |
| OccamLGS | 31.93 | 48.93 | 34.25 | 53.71 | 45.16 | 64.39 |
| VALA | 32.11 | 50.05 | 35.10 | 54.77 | 46.21 | 65.61 |
| CTF-GS DINO-CV contextual kNN + spatial smoothing | 38.06 | 61.29 | 38.71 | 63.15 | 47.11 | 72.00 |

Paper-safe reading: CTF-GS is competitive with the published baseline rows
listed by OpenGaFF under the VALA/OpenGaFF split. This table does not compare
against OpenGaFF itself.
