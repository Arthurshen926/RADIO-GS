# RADIO-GS VALA8 DINO-CV contextual kNN scene-mean alpha0.75 diagnostic

Protocol: VALA/OpenGaFF ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3683 | 0.5957 |
| 15 classes | 0.3746 | 0.6136 |
| 10 classes | 0.4612 | 0.7036 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3037/0.5825 | 0.2853/0.5706 | 0.3411/0.6793 |
| scene0062_00 | 0.3910/0.7047 | 0.3910/0.7047 | 0.5602/0.8329 |
| scene0070_00 | 0.2100/0.3051 | 0.2368/0.3433 | 0.3653/0.5289 |
| scene0097_00 | 0.4625/0.7546 | 0.4312/0.7390 | 0.4937/0.7036 |
| scene0140_00 | 0.3496/0.5392 | 0.4005/0.6163 | 0.4242/0.6768 |
| scene0347_00 | 0.5783/0.7319 | 0.5569/0.7223 | 0.6837/0.7796 |
| scene0400_00 | 0.3792/0.6977 | 0.3792/0.6977 | 0.4257/0.7743 |
| scene0590_00 | 0.2722/0.4495 | 0.3163/0.5151 | 0.3951/0.6535 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1468 | picture 0.0000 | door 0.2379 |
| 15 classes | 0.1504 | table 0.1250 | door 0.2410 |
| 10 classes | 0.1545 | toilet 0.2284 | window 0.2554 |
