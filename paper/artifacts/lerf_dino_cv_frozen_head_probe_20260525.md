# DINO-CV Frozen-head Probe (2026-05-25)

Two-scene smoke test of DINO-CV trained checkpoints on frozen SAM3/DINO task probes

| Scene | Dense rendered Hit | Dense teacher Hit | Dense rendered Score | Dense teacher Score | DINO-prop rendered mIoU | DINO-prop teacher mIoU |
|---|---:|---:|---:|---:|---:|---:|
| figurines | 0.6500 | 0.7059 | 0.9089 | 0.8402 | 0.4458 | 0.5084 |
| waldo_kitchen | 0.6250 | 0.5882 | 0.9234 | 0.8467 | 0.3337 | 0.4113 |

Conclusion: DINO-CV checkpoints improve some dense-matching hit/score signals but still do not solve DINO mask propagation mIoU versus the frame-wise teacher; not promoted without a four-scene positive result.