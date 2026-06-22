# RADIO-GS VALA8 contextual kNN16/cand80 + spatial smoothing + consensus-gated proposal readout

Protocol: VALA-aligned ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3808 | 0.6130 |
| 15 classes | 0.3874 | 0.6317 |
| 10 classes | 0.4713 | 0.7202 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3165/0.5880 | 0.2964/0.5784 | 0.3590/0.6989 |
| scene0062_00 | 0.4083/0.7449 | 0.4083/0.7449 | 0.5510/0.8344 |
| scene0070_00 | 0.2191/0.3133 | 0.2401/0.3460 | 0.3760/0.5431 |
| scene0097_00 | 0.4661/0.7683 | 0.4426/0.7550 | 0.4932/0.7169 |
| scene0140_00 | 0.3552/0.5542 | 0.4079/0.6338 | 0.4371/0.7000 |
| scene0347_00 | 0.5949/0.7528 | 0.5614/0.7371 | 0.7102/0.8021 |
| scene0400_00 | 0.3986/0.7246 | 0.3986/0.7246 | 0.4327/0.7882 |
| scene0590_00 | 0.2879/0.4582 | 0.3436/0.5342 | 0.4110/0.6778 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1524 | picture 0.0175 | refrigerator 0.2969 |
| 15 classes | 0.1512 | table 0.1262 | door 0.2382 |
| 10 classes | 0.1529 | toilet 0.2158 | window 0.2441 |
