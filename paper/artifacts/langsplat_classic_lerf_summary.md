# Classic LangSplat LERF Compatibility Summary

- root: `output/baselines/langsplat/lerf_compat_20260518`
- completed scenes: 4
- scene mean: LocAcc 0.7335, mIoU 0.4433
- object weighted: LocAcc 0.7356, mIoU 0.4613, queries 208
- caveat: local compatibility rerun after fp32 dim-3 feature conversion, chunked decoder eval, and split-aware train/test feature path fix. This is not a strict released-checkpoint macro.

| Scene | Queries | LocAcc | mIoU | Log |
|---|---:|---:|---:|---|
| figurines | 56 | 0.6607 | 0.3548 | `output/baselines/langsplat/lerf_compat_20260518/eval/figurines/20260519_215653.log` |
| ramen | 71 | 0.7324 | 0.5037 | `output/baselines/langsplat/lerf_compat_20260518/eval/ramen/20260519_214718.log` |
| teatime | 59 | 0.8136 | 0.5460 | `output/baselines/langsplat/lerf_compat_20260518/eval/teatime/20260519_220213.log` |
| waldo_kitchen | 22 | 0.7273 | 0.3687 | `output/baselines/langsplat/lerf_compat_20260518/eval/waldo_kitchen/20260519_220746.log` |
