# Teacher vs CTF-GS 2D Feature Usability

This report consolidates same-evaluator 2D evidence for selected downstream tasks. It supports a selected downstream tasks claim rather than universal feature superiority.

## LERF Rendered-View Text Grounding and Feature Memory

| Method | LocAcc | mIoU | Delta LocAcc vs teacher | Delta mIoU vs teacher |
|---|---:|---:|---:|---:|
| Frame-wise RADIO teacher | 0.7985 | 0.4634 | +0.0000 | +0.0000 |
| Nearest-view RADIO cache | 0.2722 | 0.1545 | -0.5263 | -0.3089 |
| Per-Gaussian 1280-D RADIO memory | 0.5642 | 0.3182 | -0.2343 | -0.1452 |
| Full CTF-GS | 0.8712 | 0.5243 | +0.0727 | +0.0609 |

## Frozen-Head Downstream Tasks

| Task | Primary | Teacher | CTF-GS rendered | Delta | Secondary | Teacher | CTF-GS rendered | Delta | N | Winner |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| SAM3 point prompt | mIoU | 0.3700 | 0.4169 | +0.0470 | LocAcc | 1.0000 | 1.0000 | +0.0000 | 208 | rendered |
| SAM3 box prompt | mIoU | 0.6560 | 0.6638 | +0.0079 | LocAcc | 0.8702 | 0.8221 | -0.0481 | 208 | rendered |
| SAM3 mask propagation | mIoU | 0.3583 | 0.3756 | +0.0173 | LocAcc | 0.7872 | 0.6667 | -0.1206 | 141 | rendered |
| DINOv3 dense matching | Mean score | 0.8547 | 0.9048 | +0.0501 | HitRate | 0.5723 | 0.5393 | -0.0330 | 3093 | rendered |
| DINOv3 mask propagation | mIoU | 0.5119 | 0.4805 | -0.0314 | LocAcc | 0.7660 | 0.7943 | +0.0284 | 141 | teacher |

## Claim-Safe Summary

- Primary rendered wins: 5 / 6.
- Universal superiority claim allowed: False.
- Recommended wording: CTF-GS rendered features improve selected downstream feature-usability metrics over frame-wise RADIO teacher features, while DINOv3 caveats remain under the same frozen readout.

## Caveats

- SAM3 box prompt LocAcc remains teacher-stronger (0.8221 vs 0.8702).
- SAM3 mask propagation LocAcc remains teacher-stronger (0.6667 vs 0.7872).
- DINOv3 dense matching HitRate remains teacher-stronger (0.5393 vs 0.5723).
- DINOv3 mask propagation mIoU remains teacher-stronger (0.4805 vs 0.5119).

## Sources

- controlled_evidence: `/root/RADIO-GS/paper/artifacts/controlled_evidence_table.json`
- sam_dino_formal: `/root/RADIO-GS/output/lerf_sam_dino_tasks/formal_v9_dino_topk_area200_bg110_peak_20260514/lerf_sam_dino_task_aggregate.json`
- dino_homography_ransac: `/root/RADIO-GS/output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_all_20260514/lerf_sam_dino_task_aggregate.json`
- prototype_adaptor: `/root/RADIO-GS/output/lerf_adaptor_downstream/mainline/lerf_adaptor_downstream_aggregate.json`
