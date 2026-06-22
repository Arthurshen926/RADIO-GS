# RADIO-GS VALA8 proposal-consistency + DINO-CV contextual kNN16/cand80 + spatial/proposal readout

Protocol: VALA-aligned ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3780 | 0.6104 |
| 15 classes | 0.3844 | 0.6291 |
| 10 classes | 0.4673 | 0.7173 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3144/0.5892 | 0.2956/0.5815 | 0.3549/0.6978 |
| scene0062_00 | 0.4027/0.7365 | 0.4027/0.7365 | 0.5415/0.8284 |
| scene0070_00 | 0.2183/0.3125 | 0.2400/0.3459 | 0.3784/0.5479 |
| scene0097_00 | 0.4566/0.7588 | 0.4305/0.7439 | 0.4778/0.7022 |
| scene0140_00 | 0.3525/0.5518 | 0.4047/0.6311 | 0.4329/0.6958 |
| scene0347_00 | 0.5960/0.7550 | 0.5622/0.7388 | 0.7095/0.8023 |
| scene0400_00 | 0.3949/0.7206 | 0.3949/0.7206 | 0.4319/0.7862 |
| scene0590_00 | 0.2888/0.4585 | 0.3449/0.5347 | 0.4116/0.6774 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1512 | picture 0.0153 | refrigerator 0.2982 |
| 15 classes | 0.1495 | table 0.1224 | door 0.2360 |
| 10 classes | 0.1512 | toilet 0.2141 | window 0.2420 |
