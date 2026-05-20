# LEGaussians LERF Compatibility Summary

- root: `output/baselines/legaussians/lerf_compat_20260520`
- protocol: official render_mask.py outputs evaluated with local LERF JSON polygon GT
- scene mean: mIoU 0.2694, Acc@0.25 0.3974, Acc@0.5 0.2312
- object weighted: mIoU 0.2694, queries 208, missing 0

| Scene | mIoU | Acc@0.25 | Acc@0.5 | Objects | Missing |
|---|---:|---:|---:|---:|---:|
| figurines | 0.2562 | 0.3929 | 0.2321 | 56 | 0 |
| ramen | 0.1331 | 0.1268 | 0.0423 | 71 | 0 |
| teatime | 0.4623 | 0.6610 | 0.5593 | 59 | 0 |
| waldo_kitchen | 0.2259 | 0.4091 | 0.0909 | 22 | 0 |
