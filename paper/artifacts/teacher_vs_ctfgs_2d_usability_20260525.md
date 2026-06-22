# Frame-wise RADIO vs GaussFM 2D Feature Usability

This report consolidates same-evaluator 2D evidence for selected downstream tasks. It supports a selected downstream tasks claim rather than universal feature superiority.

## LERF Rendered-View Text Grounding and Feature Memory

| Method | LocAcc | mIoU | Delta LocAcc vs frame-wise RADIO | Delta mIoU vs frame-wise RADIO |
|---|---:|---:|---:|---:|
| Frame-wise RADIO | 0.7985 | 0.4634 | +0.0000 | +0.0000 |
| Nearest-view RADIO cache | 0.2722 | 0.1545 | -0.5263 | -0.3089 |
| Per-Gaussian 1280-D RADIO memory | 0.5642 | 0.3182 | -0.2343 | -0.1452 |
| Full GaussFM | 0.8598 | 0.5707 | +0.0613 | +0.1073 |

## Frozen-Head Downstream Tasks

| Task | Primary | Frame-wise RADIO | GaussFM rendered | Delta | Secondary | Frame-wise RADIO | GaussFM rendered | Delta | N | Winner |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| SAM3 point prompt | mIoU | 0.3700 | 0.4173 | +0.0473 | LocAcc | 1.0000 | 1.0000 | +0.0000 | 208 | rendered |
| SAM3 box prompt | mIoU | 0.6560 | 0.6638 | +0.0079 | LocAcc | 0.8702 | 0.8221 | -0.0481 | 208 | rendered |
| SAM3 mask propagation | mIoU | 0.3583 | 0.3756 | +0.0173 | LocAcc | 0.7872 | 0.6596 | -0.1277 | 141 | rendered |
| DINOv3 dense matching | Mean score | 0.8547 | 0.9048 | +0.0501 | HitRate | 0.5723 | 0.5396 | -0.0327 | 3093 | rendered |
| DINOv3 mask propagation | mIoU | 0.4606 | 0.4677 | +0.0071 | LocAcc | 0.7660 | 0.7872 | +0.0213 | 141 | rendered |

## Claim-Safe Summary

- Primary rendered wins: 6 / 6.
- Universal superiority claim allowed: False.
- Recommended wording: GaussFM rendered features outperform the frame-wise RADIO reference on all selected primary downstream feature-usability metrics; secondary LocAcc/HitRate caveats are reported separately.

## Caveats

- SAM3 box prompt LocAcc remains frame-wise-RADIO-stronger (0.8221 vs 0.8702).
- SAM3 mask propagation LocAcc remains frame-wise-RADIO-stronger (0.6596 vs 0.7872).
- DINOv3 dense matching HitRate remains frame-wise-RADIO-stronger (0.5396 vs 0.5723).

## Sources

- controlled_evidence: `/root/RADIO-GS/paper/artifacts/controlled_evidence_table.json`
- sam_dino_formal: `/root/RADIO-GS/output/lerf_sam_dino_tasks/formal_v12c_dino_sam3_boundary_v9readout_gpu_20260528/lerf_sam_dino_task_aggregate.json`
- dino_homography_ransac: `/root/RADIO-GS/output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_all_20260514/lerf_sam_dino_task_aggregate.json`
- prototype_adaptor: `/root/RADIO-GS/output/lerf_adaptor_downstream/mainline/lerf_adaptor_downstream_aggregate.json`
