# CAGS LERF Compatibility Summary

- root: `output/baselines/cags/lerf_compat_20260518`
- completed scenes: 4
- scene mean: mIoU 0.2627, Acc@0.25 0.3997, Acc@0.5 0.2552
- object weighted: mIoU 0.2394, Acc@0.25 0.3558, Acc@0.5 0.2260, objects 208, missing rendered masks 34
- caveat: local OpenGaussian-compatible CAGS LERF compatibility rerun. Missing rendered masks are counted in the source JSONs; this is a reproduced diagnostic row, not a SOTA or released-checkpoint claim.

| Scene | Iteration | Threshold | Objects | Missing | mIoU | Acc@0.25 | Acc@0.5 | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| figurines | 40000 | 10 | 56 | 2 | 0.4283 | 0.6071 | 0.4821 | `output/baselines/cags/lerf_compat_20260518/eval/cags_figurines_iter40000_lerf_iou.json` |
| ramen | 40001 | 10 | 71 | 24 | 0.0247 | 0.0000 | 0.0000 | `output/baselines/cags/lerf_compat_20260518/eval/cags_ramen_iter40001_lerf_iou.json` |
| teatime | 40000 | 10 | 59 | 2 | 0.2946 | 0.4915 | 0.2203 | `output/baselines/cags/lerf_compat_20260518/eval/cags_teatime_iter40000_lerf_iou.json` |
| waldo_kitchen | 40000 | 10 | 22 | 6 | 0.3031 | 0.5000 | 0.3182 | `output/baselines/cags/lerf_compat_20260518/eval/cags_waldo_kitchen_iter40000_lerf_iou.json` |
