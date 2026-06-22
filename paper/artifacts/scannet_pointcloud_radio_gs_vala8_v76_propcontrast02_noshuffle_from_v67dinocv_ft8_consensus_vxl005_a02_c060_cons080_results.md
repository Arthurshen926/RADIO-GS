# RADIO-GS VALA8 region-prototype contrast + consensus-gated proposal readout

Protocol: VALA-aligned ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3782 | 0.6105 |
| 15 classes | 0.3843 | 0.6291 |
| 10 classes | 0.4689 | 0.7184 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3154/0.5898 | 0.2974/0.5837 | 0.3559/0.6984 |
| scene0062_00 | 0.3968/0.7296 | 0.3968/0.7296 | 0.5454/0.8300 |
| scene0070_00 | 0.2217/0.3159 | 0.2405/0.3463 | 0.3812/0.5514 |
| scene0097_00 | 0.4616/0.7633 | 0.4350/0.7486 | 0.4863/0.7077 |
| scene0140_00 | 0.3543/0.5544 | 0.4068/0.6340 | 0.4363/0.7012 |
| scene0347_00 | 0.5942/0.7526 | 0.5600/0.7364 | 0.7088/0.7994 |
| scene0400_00 | 0.3927/0.7177 | 0.3927/0.7177 | 0.4266/0.7811 |
| scene0590_00 | 0.2888/0.4604 | 0.3448/0.5367 | 0.4107/0.6780 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1528 | picture 0.0242 | refrigerator 0.2970 |
| 15 classes | 0.1505 | table 0.1273 | door 0.2373 |
| 10 classes | 0.1509 | toilet 0.2145 | window 0.2423 |
