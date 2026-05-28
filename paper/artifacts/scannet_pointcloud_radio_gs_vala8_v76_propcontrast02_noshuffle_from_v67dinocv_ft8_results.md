# RADIO-GS VALA8 region-prototype contrast training + DINO-CV contextual kNN16/cand80 + spatial readout

Protocol: VALA/OpenGaFF ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3780 | 0.6104 |
| 15 classes | 0.3840 | 0.6289 |
| 10 classes | 0.4687 | 0.7181 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3154/0.5898 | 0.2974/0.5837 | 0.3559/0.6984 |
| scene0062_00 | 0.3967/0.7306 | 0.3967/0.7306 | 0.5451/0.8299 |
| scene0070_00 | 0.2218/0.3160 | 0.2405/0.3462 | 0.3807/0.5507 |
| scene0097_00 | 0.4617/0.7631 | 0.4348/0.7480 | 0.4864/0.7076 |
| scene0140_00 | 0.3543/0.5541 | 0.4068/0.6338 | 0.4361/0.7007 |
| scene0347_00 | 0.5932/0.7522 | 0.5590/0.7359 | 0.7084/0.7990 |
| scene0400_00 | 0.3926/0.7176 | 0.3926/0.7176 | 0.4266/0.7808 |
| scene0590_00 | 0.2882/0.4596 | 0.3442/0.5359 | 0.4102/0.6776 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1527 | picture 0.0246 | refrigerator 0.2969 |
| 15 classes | 0.1503 | table 0.1271 | door 0.2373 |
| 10 classes | 0.1509 | toilet 0.2145 | window 0.2423 |
