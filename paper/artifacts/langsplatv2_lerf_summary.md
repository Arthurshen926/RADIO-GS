# LangSplatV2 LERF Compatibility Summary

> **SUPERSEDED / DIAGNOSTIC ONLY.** This pre-correction summary contains the
> old camera-mapping result and is not a canonical LERF-2D baseline. Use
> `evaluation_protocol_freeze_20260801.yaml` and the OccamLGS row instead.

- root: `output/baselines/langsplatv2/lerf_compat_20260518`
- completed scene/index rows: 4
- scene mean: LocAcc 0.6176, mIoU 0.46010000000000006
- object weighted: LocAcc 0.6009562500000001, mIoU 0.4486783653846154, queries 208

| Scene | Index | Checkpoint | Mask Thresh | Queries | LocAcc | mIoU | Log |
|---|---:|---:|---:|---:|---:|---:|---|
| figurines | 0 | 10000 | 0.4000 | 56 | 0.8214 | 0.5965 | `output/baselines/langsplatv2/lerf_compat_20260518/eval/figurines_0/20260518_230019.log` |
| ramen | 0 | 10000 | 0.4000 | 71 | 0.7183 | 0.5913 | `output/baselines/langsplatv2/lerf_compat_20260518/eval/ramen_0/20260518_230211.log` |
| teatime | 0 | 10000 | 0.4000 | 59 | 0.2034 | 0.0968 | `output/baselines/langsplatv2/lerf_compat_20260518/eval/teatime_0/20260518_114759.log` |
| waldo_kitchen | 0 | 10000 | 0.4000 | 22 | 0.7273 | 0.5558 | `output/baselines/langsplatv2/lerf_compat_20260518/eval/waldo_kitchen_0/20260519_031249.log` |
