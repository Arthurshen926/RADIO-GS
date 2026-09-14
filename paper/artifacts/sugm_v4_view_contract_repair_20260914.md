# v4 视角一致性与接口修复

## 结论

本轮落实了可复现的实现错误修复，但没有恢复合理的完整精度，不能声称“所有 bug 已修复”。
同 native 源证据、冻结对象关联和范围、同文本缓存、同 0.2 阈值，修复原型保留及投票后四场景等权 mIoU **7.4154%**，对照 **7.4618%**。不是性能提升，也不是完整 LERF2D/3D 达标结果。

## 已落地

1. 新构建的对象原型优先保留不同源照片，写入逐原型 view ID；同一照片的嵌套掩码不再冒充多个视角投票。旧记忆没有 view receipt 时保留原 distinct-fragment 行为，以便回放。新记忆中的单视角对象仍可查询，没有设置必须多视角的硬门槛。
2. 冷加载检查有效原型的有限性、非零范数、fragment ID 上界、crop-kind 非负，以及同一 fragment 的视角一致性。过去部分无效原型可以通过记忆验证。
3. 显式 cannot-link 现在也约束非种子碎片；过去 restricted-seed 策略下只检查种子，非种子可加入已明确禁止的对象。当前 all-fragments 对照未改变此变量。
4. 新训练的 affinity checkpoint 内保存真实外观输入契约及未验证域信息，部署审计传递这些字段；旧 checkpoint 缺失时标为 unknown/未验证，不编造认证。这只修复信息丢失，不是训练/部署域差异已经解决。

## 四场景完整组合对照（development）

| 场景 | native 原组合 | 视角修复组合 |
|---|---:|---:|
| figurines | 5.4969% | 5.7924% |
| ramen | 4.3613% | 4.2363% |
| teatime | 6.2193% | 6.1829% |
| waldo_kitchen | 13.7698% | 13.4503% |
| 场景等权平均 | 7.4618% | 7.4154% |

22 帧、208 个 object-frame 观测。所有预测先生成三维 element field，再渲染 native masks；没有用 GT 选择原型、调整模型或选阈值。公开 mask 评分语义不等于完整上游协议认证。

运行器：`radio_gs/scripts/validate_lerf_view_consensus_repair.py`。
产物：`/mnt/pool/sqy/results/RADIO-GS/output/v4_view_consensus_repair_20260914/`，含 commands、四个原型修复记忆、日志、预测 field、掩码和 report。
使用两条并发任务队列，分配 GPU 2/3；主要渲染计算实际在 CPU。本轮没有启动代理输入上的重复训练。

## 尚未解决，不能冒充完成

- 真实 region-crop/SAM 分布上的关联训练及独立场景验证。
- 跨视角完整对象范围；已有稳定单候选诊断仍远低于历史精度。
- 查询前 top-2 压缩与重叠候选的语义冲突，以及独立 generic sigmoid 的误激活。
- 历史 dense identity 定位与 fallback 功能在新 carrier 上的有效独立实现。

上述是主线方法的实质缺口。本轮反证了“修复伪多视角投票就足以救回精度”，不把该修复发布为性能升级，不覆盖历史基线。

最终相关回归：294 passed、2 skipped；修改脚本编译及 git diff --check 通过。
