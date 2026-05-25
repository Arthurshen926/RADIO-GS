# RADIO-GS VALA8 DINO-CV contextual kNN16/cand80 + spatial propagation + confidence proposal readout

Protocol: VALA/OpenGaFF ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3809 | 0.6134 |
| 15 classes | 0.3874 | 0.6321 |
| 10 classes | 0.4715 | 0.7207 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3165/0.5879 | 0.2965/0.5783 | 0.3590/0.6988 |
| scene0062_00 | 0.4088/0.7456 | 0.4088/0.7456 | 0.5512/0.8347 |
| scene0070_00 | 0.2190/0.3131 | 0.2400/0.3459 | 0.3762/0.5435 |
| scene0097_00 | 0.4657/0.7683 | 0.4421/0.7551 | 0.4934/0.7171 |
| scene0140_00 | 0.3553/0.5545 | 0.4080/0.6342 | 0.4372/0.7004 |
| scene0347_00 | 0.5950/0.7533 | 0.5613/0.7375 | 0.7110/0.8028 |
| scene0400_00 | 0.3986/0.7258 | 0.3986/0.7258 | 0.4328/0.7897 |
| scene0590_00 | 0.2882/0.4582 | 0.3442/0.5343 | 0.4117/0.6782 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1524 | picture 0.0172 | refrigerator 0.2978 |
| 15 classes | 0.1511 | table 0.1261 | door 0.2382 |
| 10 classes | 0.1531 | toilet 0.2158 | window 0.2445 |
