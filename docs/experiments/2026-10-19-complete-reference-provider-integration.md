# N14 完整提供器接入统一推理流程

日期：2026-10-19；活动分支 `codex/tmm-query-response`。
决策：[ADR0045](../adr/0045-integrate-complete-signed-reference-provider.md)。

## 实现进展

本轮把 N14 从带历史重放检查的实验脚本迁到显式推理入口。
旧研究脚本、既有结果及其主次方案身份均保留。新入口使用本次源视图推理生成的
`base_readout.pt`，不会读取历史 v10/N14 答案或评价标注来产生输出。

实际调用顺序：

1. 现有源视图路径生成语义响应和独立选集。
2. 原始正负涂画约束参考图 DINO 对应；SAM 产生语义与带约束两个完整参考候选。
3. 固定的 `quality` 或 `feasible` 策略按源端信息选择整张参考候选。
4. 相同两帧 SAM 实例传播读取声明的参考／目标 RGB，产生目标图像响应。
5. 按原分辨率重新构造实际高斯贡献，检查 alpha 和归一化算子与渲染一致。
6. 用既有 `lift_image_response`／公共响应系统拟合三维连续响应，再实际重渲染检查。

规范实现：`radio_gs/pipeline/reference_completion.py`、
`reference_completion_worker.py`、`reference_completion_contract.py`。
几何运算位于 `radio_gs/querying/image_response_geometry.py`，RGB 准备位于
`reference_transport.py`。公共核心不增加按 benchmark 分支的身份规则。

推理入口新增 `--reference-completion <profile.json>`；默认不启用。
它要求一个明确的 reference-instance 查询和 source-assisted 权限，不允许套到 UQIS map-only 输入。
原生网格 1008×756、每符号 16 点、120 步、0.5 阈值、gsplat CUDA footprint 等均保持 N14 配方。
本轮没有教师替换、训练、参数搜索或新的候选筛除。

## 输入与读出问题的修正

旧 NVOS 评分器曾把 `target_rgb_used=False` 写死。
这对原 v10 路径符合实际，但若直接用于 N14 会错误披露输入条件。
现在从 typed readout 的显式权限读取，核对目标帧／网格／阈值，记录负涂画与目标 RGB 使用。
这是一项为新集成路径修复的报告问题，不证明历史 N14 实验评分有误。

原独立选集保持源视图训练结果，明确标为
`inherited_source_only_selected_set_unchanged`。新连续响应与该选集不能互换解释。
直接图像响应只作为另一个输出进入相同评分器，主结果仍评实际三维响应的渲染。
两套 GT 边界口径分别保留；连续响应先双线性缩放再阈值化，AP 不使用量化后的图像。

## 全队列配置与追溯

- `configs/pipeline/reference_completion_20261019/quality.recipe.json`：保留的 N14 质量方案，历史次要对照。
- 同目录 `feasible.recipe.json`：N14 可行性方案，历史主策略。
- 两者各 31 个任务：8 个 NVOS 任务增加显式输入扩展，其他 23 个任务保持原配置。
- 八任务同策略；预检拒绝改变 initializer、改动其他 benchmark、遗漏任务或逐任务选策略。
- 权重、图像、涂画和几何拟合文件绑定 47 个去重资产哈希。
  配置与准备记录总计 **155,634 字节**，不复制权重或稠密结果。

两套有界执行配置位于 `configs/research/shared_evidence_n14_quality.json` 与
`shared_evidence_n14_feasible.json`。例如预检入口（不执行 GPU 推理）：

```bash
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/research/run_shared_evidence_migration.py \
  --config configs/research/shared_evidence_n14_quality.json \
  --run <新的预检名称>
```

只有显式 `--execute` 才会发起新推理，沿用已有临时预测回收和持久化预算。
当前未启动 GPU 工作。该完整队列入口不能代替先做固定代表样本的端到端验收。

## 已验证与尚未验证

[CPU 审计](2026-10-19-reference-completion-cpu-audit-v1.json)核对全部 16 份任务／策略配置：
原始涂画像素数、32 个选取提示点及两种源端选择均与历史 decoder 记录一致。
此处读取历史 decoder 元数据，不读取历史预测数组，不产生新的 benchmark 分数。
规范 worker 的主函数与原 N14 AST 完全一致；迁出的 RGB 类、相机参数与响应渲染算式
也通过结构等价检查。它们仍需真实 GPU 执行确认。

[输入准备检查](2026-10-19-reference-completion-input-preparation.json)实际执行了 16 次
`_prepare_inputs`，验证归一化后的场景／查询继续满足输入合同，临时配置已释放。
CPU 回归还覆盖：非法 RGB／GT 扩权、目标混入几何训练、错绑参考输入、
观察域归一化、公共后端数值一致、独立选集保留、评分口径与跨任务配置保持。

两种核心 profile 的全套测试均为 **253 passed、1 skipped**；跳过 CUDA/NVCC 渲染检查。
全 31 任务原有组件检查更新至 [v7](2026-10-19-shared-core-component-audit-v7.json)。
该组件检查覆盖原有内核，不等同于新 N14 提供器已经完成端到端验收。

[架构清单 v4](2026-10-19-n14-architecture-inventory-v4.json)记录新的 31 条计划，
其中仅 8 个 NVOS 任务声明目标 RGB 权限；共享建场仍标为未完全完成。

当前未获得新 GPU 推理结果，不能宣称迁移后的 N14 已复现旧精度或统一系统已全部验收。
后续优先固定 fern／horns_center 等代表任务的完整推理与数值差分，
再做同输入权限的 DINO/SAM 对照；最后冻结统一配方重评五项。
原有强配方已经有真实调用入口，不应再用较弱 v10 接口的成熟度替代它的迁移验收。
