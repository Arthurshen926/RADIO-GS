# RADIO-GS VALA8 DINO-CV contextual kNN16/cand80 + spatial logit propagation

Protocol: VALA/OpenGaFF ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3806 | 0.6129 |
| 15 classes | 0.3871 | 0.6315 |
| 10 classes | 0.4711 | 0.7200 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3165/0.5881 | 0.2965/0.5785 | 0.3591/0.6990 |
| scene0062_00 | 0.4080/0.7448 | 0.4080/0.7448 | 0.5504/0.8339 |
| scene0070_00 | 0.2192/0.3134 | 0.2400/0.3460 | 0.3758/0.5428 |
| scene0097_00 | 0.4661/0.7680 | 0.4421/0.7544 | 0.4933/0.7167 |
| scene0140_00 | 0.3550/0.5537 | 0.4076/0.6333 | 0.4369/0.6996 |
| scene0347_00 | 0.5939/0.7522 | 0.5603/0.7363 | 0.7097/0.8015 |
| scene0400_00 | 0.3988/0.7253 | 0.3988/0.7253 | 0.4326/0.7887 |
| scene0590_00 | 0.2875/0.4577 | 0.3432/0.5336 | 0.4107/0.6774 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1523 | picture 0.0177 | refrigerator 0.2969 |
| 15 classes | 0.1509 | table 0.1262 | door 0.2378 |
| 10 classes | 0.1529 | toilet 0.2158 | window 0.2443 |
