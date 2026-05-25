# ScanNet VALA8 CTF-GS DINO-CV proposal-memory ablation

Protocol: VALA/OpenGaFF ScanNet-8 candidate split
Scenes: scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00, scene0347_00, scene0400_00, scene0590_00

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3931 | 0.6255 |
| 15 classes | 0.3837 | 0.6228 |
| 10 classes | 0.4612 | 0.7081 |

## Per-Scene

| Scene | split19 | split15 | split10 |
|---|---:|---:|---:|
| scene0000_00 | 0.3096/0.5795 | 0.2914/0.5753 | 0.3575/0.6993 |
| scene0062_00 | 0.4476/0.7659 | 0.4476/0.7659 | 0.5496/0.8338 |
| scene0070_00 | 0.2286/0.3272 | 0.2357/0.3404 | 0.3423/0.4870 |
| scene0097_00 | 0.4528/0.7601 | 0.4391/0.7484 | 0.4832/0.6992 |
| scene0140_00 | 0.4000/0.5849 | 0.3720/0.5524 | 0.4298/0.6970 |
| scene0347_00 | 0.5945/0.7554 | 0.5616/0.7378 | 0.7081/0.8009 |
| scene0400_00 | 0.3509/0.6801 | 0.3509/0.6801 | 0.4182/0.7793 |
| scene0590_00 | 0.3613/0.5507 | 0.3715/0.5818 | 0.4009/0.6685 |

## Category Stability

| Split | mean IoU std | worst class | most unstable class |
|---|---:|---|---|
| 19 classes | 0.1574 | table 0.0641 | refrigerator 0.2944 |
| 15 classes | 0.1427 | table 0.0635 | bed 0.2391 |
| 10 classes | 0.1591 | toilet 0.2138 | window 0.2435 |
