# RADIO-GS VALA8 DINO-CV contextual kNN16/cand80 scene-mean alpha0.45

Protocol: VALA/OpenGaFF ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3722 | 0.6025 |
| 15 classes | 0.3791 | 0.6207 |
| 10 classes | 0.4591 | 0.7025 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3066/0.5839 | 0.2873/0.5740 | 0.3414/0.6778 |
| scene0062_00 | 0.4009/0.7334 | 0.4009/0.7334 | 0.5521/0.8311 |
| scene0070_00 | 0.2188/0.3130 | 0.2369/0.3426 | 0.3611/0.5229 |
| scene0097_00 | 0.4600/0.7597 | 0.4370/0.7468 | 0.4887/0.7099 |
| scene0140_00 | 0.3467/0.5345 | 0.3978/0.6109 | 0.4203/0.6675 |
| scene0347_00 | 0.5757/0.7329 | 0.5539/0.7225 | 0.6817/0.7784 |
| scene0400_00 | 0.4026/0.7206 | 0.4026/0.7206 | 0.4359/0.7815 |
| scene0590_00 | 0.2665/0.4421 | 0.3161/0.5148 | 0.3912/0.6509 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1503 | picture 0.0247 | refrigerator 0.2628 |
| 15 classes | 0.1499 | table 0.1223 | door 0.2393 |
| 10 classes | 0.1546 | toilet 0.2203 | window 0.2558 |
