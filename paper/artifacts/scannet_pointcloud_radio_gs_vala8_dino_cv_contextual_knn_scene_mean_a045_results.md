# RADIO-GS VALA8 DINO-CV contextual kNN12 scene-mean alpha0.45

Protocol: VALA/OpenGaFF ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3715 | 0.6024 |
| 15 classes | 0.3784 | 0.6206 |
| 10 classes | 0.4585 | 0.7029 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3055/0.5830 | 0.2861/0.5730 | 0.3409/0.6781 |
| scene0062_00 | 0.4007/0.7335 | 0.4007/0.7335 | 0.5517/0.8312 |
| scene0070_00 | 0.2178/0.3122 | 0.2358/0.3417 | 0.3598/0.5224 |
| scene0097_00 | 0.4601/0.7601 | 0.4372/0.7472 | 0.4890/0.7101 |
| scene0140_00 | 0.3467/0.5356 | 0.3978/0.6122 | 0.4200/0.6691 |
| scene0347_00 | 0.5764/0.7337 | 0.5546/0.7235 | 0.6820/0.7789 |
| scene0400_00 | 0.3982/0.7186 | 0.3982/0.7186 | 0.4325/0.7814 |
| scene0590_00 | 0.2669/0.4422 | 0.3167/0.5150 | 0.3925/0.6518 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1502 | picture 0.0248 | refrigerator 0.2630 |
| 15 classes | 0.1497 | table 0.1229 | door 0.2396 |
| 10 classes | 0.1544 | toilet 0.2199 | window 0.2555 |
