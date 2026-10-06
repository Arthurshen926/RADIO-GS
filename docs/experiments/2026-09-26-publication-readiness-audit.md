# 投稿准备度、四项评估合约与方法优先级审计

2026-09-26。本审计不新增任务分数，不修改冻结评估器，也不将开发集反复迭代后的结果称为独立测试。结论是：**已有可复用的方法进展与完整开发评估，但尚未具备“统一四类查询的 SOTA 系统”证据。当前最有价值的工作仍是修复元素的语义写入与范围可靠性，而不是继续增加教师或维度。**

## 当前成绩能说明什么

| 评估 | 保留结果 | 必须保留的解释边界 |
|---|---:|---|
| LERF2D，4 场景 / 208 条观测 | 49.7408% IoU，68.4795% AP | 扩展源视角后的完整场，场景等权；开发结果 |
| LERF3D，4 场景 / 208 条观测 | 33.5676% IoU；加结构图控制 36.2258% | 选中元素的 2D 渲染掩码评价，不是物理三维体积/表面 GT |
| ScanNet OVS，paper8 | 54.5190 / 55.6667 / 69.8166% | 19 / 15 / 10 类，未标注网格输入、VALA Gaussian-domain 加权评估；3 场增加为 32 个训练视角 |
| NVOS，8 任务 / 7 张地图 | 81.1608% IoU | 正涂鸦、参考 SAM 范围与临时查询优化；禁止目标 RGB 和负涂鸦；不是纯余弦读出 |

来源：`paper/artifacts/full_field_rebuild_next19_complete.json`、`source_admission_adaptive_coverage_next29.json`、`nvos_reference_completion_next19_eight.json`。这些是不同已验证控制，**不能把它们拼成一个已经共同验证过的默认方法**。next30 残差只重测了原生读出/排序，不得用其 AP 代替这些最终 IoU。

四项 benchmark 也不等于四种 query：两项 LERF 和 ScanNet 主要验证文本；NVOS 验证参考 2D 正提示。LERF 的 GT 裁剪图像与注册点开发协议是类别并集，不能代替实例图像查询或完整 3D prompt benchmark。UQIS 按用户决定暂停；在恢复其数据与合约前，四 query 全能力声明保持未完成。

## 与历史和本地复现如何比较

1. **LUDVIG 的数字并非同输入上限。** 本地锁定代码的 NVOS 8 任务 × 3 种子结果为 91.2577%。官方路径在几何训练及逐查询多视角 SAM uplifting 中访问目标 RGB；当前 NVOS 路径不访问。约 10.10 点差距中，方法差与输入权限差尚未分离，不能全算给特征场。参考 `reproductions/ludvig/README.md` 与 `paper/artifacts/promptable_nvs_protocol_registry.yaml`。本审计未做新的联网排行榜核验，因此不主张这些方法是截至今天的最新 SOTA。
2. **LangSplatV2 / OccamLGS 的本地高分可作定位参照，不能直接作匹配排行榜。** 冻结 registry 中的 61.51% / 63.6200% 含几何、训练图权限、特征来源或实现身份差异；LangSplatV2 定位准确率还有 query-micro 与 scene-macro 的差异。沿用 `evaluation_protocol_registry_20260731.yaml` 的逐行 eligibility，不因数字高而删除限制。
3. **历史高分同样需要清洗。** 历史 retained LERF3D 四场等权为 42.93%，208 条加权为 43.74%；58.89% 是另一个带事后校准的记录，不能当有效对照。现有历史预测重评分只对齐栅格，没有对齐输入、几何、校准和渲染。ScanNet 的真值分配与权重随 Gaussian 几何变化，37.86 / 41.98 / 52.34% 历史数值不能作纯语义增益归因。

## 最值得迁移的方法思想

**第一优先是把融合后的实际读回纳入训练/准入。** 当前 native 前缀是源区域特征的加权平均，随后被保护；四教师结构训练不会自动纠正这条文本余弦路径。next30 的独立区域响应误差减少 21–39%，但融合读回文本误差在 7 / 12 图反升，说明区域层面保真不够。应在相同几何、相同源预算下，对可见性、区域身份和整体/部件相容性分组，使用未参与匹配/写入决策的视角检查写回后的文本与图像响应。只有区域 MSE 提升不能批准融合候选。

**保留 LUDVIG 的实际贡献加权和局部关系约束，限制其改动权。** 当前已有 renderer-adjoint、结构 32 图与 NVOS 正位置约束，均无需把目标 RGB 引入查询。应保持语义 unary，并让边界/结构只连接有可信对应的元素；未知区域不作负例、相近文本不等于同一对象。此前 native-only 图替代、单最高分种子传播、简单面积三档 max 均有负结果，不把它们重新包装为新方法。图只能传播现有证据，不能恢复已经平均掉的整体/部件区分。

**历史 hash branch 只能在算子一致后重用。** 旧实现“渲染 latent 后非线性解码”与“逐元素解码后渲染”不交换；本地合成审计最大差 0.02575，见 `historical_hybrid_operator_audit_20260917.json`。可以把显式/隐式残差作为单向量场的优化器，但必须用实际部署读出训练、导出后冷重放验证；不能直接拷贝旧 head 或用屏幕空间分数宣称元素集合恢复。

这些迁移方向与 ADR 0007 / 0008 / 0009 相容：几何先完成并冻结，语义原生空间关系明确，结构有单独容量，所有 query 仍读取同一个 Canonical Capability Feature。额外残差、全局范围头、图边缓存和临时优化都应计入总存储/查询时延。

## 本轮发现的真实发布问题与隔离修复

执行协议及源准入相关 34 项 CPU 测试：29 通过、5 失败。四个失败由七任务冻结器需要的 SPIn9 历史汇总缺失触发；一个失败是 LUDVIG 当前 launcher 的 hash 不再等于历史 registry。进一步逐项扫描发现另有 5 份旧 PFPR 外部 manifest 缺失。它们不属于当前四项任务，不能据此否定当前任务分数，但也不能宣称全局冻结检查全部通过。

LUDVIG 的精确历史 launcher 已从 git commit `4a668744707b93aa5394585810677e0a3f3e80ba` 恢复到隔离归档，SHA256 与历史 `1f1bc95d…` 完全一致。现行文件 `7c308c2f…` 保留原样；差异来自后续 GPU / materialize / full-carrier / transient-adapter 扩展。未更新任何历史 hash 来掩盖失败，也未把归档代码身份核验说成已恢复全部外部依赖或重新跑完 LUDVIG。原 README 仍描述旧两文件 patch，且其 `/root/baselines/LUDVIG` 路径在本机不存在，运行复现前须按新版本安装并核验。

完整检查五份保留结果/runtime seal 的 453 个直接绑定又发现一个实际断链：现行 LERF next19 评估入口为 `f453b560…`，旧封存为 `68dac7b7…`。依据两次原始修改记录恢复出精确旧代码，hash 与独立旧 seal 一致。修改只增加候选 field/output 参数、禁止候选复用基线缓存，以及结果 metadata；**评分、渲染、元素选择代码块逐字节相同**。因此这不是现有 IoU 被悄悄换算法的证据，但历史复现不能再引用未归档的可变入口。两份精确旧代码均新增归档到 `scripts/audits/frozen_sources_next31/`，工作文件和旧 seal 完全保留。

新增两项只读检查：

- `scripts/audits/audit_publication_readiness_next31.py`：扫描全部历史 authority 与 1773 个冻结源文件；保留缺失与 mismatch。
- `scripts/audits/verify_current_four_release_next31.py`：首轮完整核验，保留上述 LERF source mismatch 的失败报告。
- `scripts/audits/verify_current_four_release_next31_v2.py`：显式限定当前四项任务，仅允许固定映射的历史代码归档用**原 seal 的 hash**消解路径漂移；当前路径 mismatch 仍在报告。另核验原冻结 authority 中的三份协议与源 manifest，不能重新计算新 hash 来宣称旧 seal 通过。它不是替换七任务 freeze 的绿色开关，不认证统一方法或 SOTA。

报告：`output/publication_readiness_next31/audit.json`、失败的 `current_four/report.json`、修复后的 `current_four_v2/report.json`。**v2 已实际执行完成，退出码 0：453 个直接绑定全部通过（其中一个旧 LERF source 通过精确归档解析）、8 份四任务外部 authority 结果匹配、3 份冻结协议匹配、1773 个冻结源文件零变化，LUDVIG 历史代码身份通过精确归档核验。** 总计读取约 30.4 GiB 绑定资产，没有以文件存在代替 hash 校验。新增归档身份、错误归档拒绝、未登记替代拒绝及 LERF metric block 一致性测试，与当前 registry / source-admission 测试合计 **23 项通过**。这是当前四任务证据身份的通过，不是七任务全局 freeze、统一方法或 SOTA 的通过。

## 可执行验收清单

- [ ] **融合方法先过源侧验证。** 预先固定候选、源视角/概念划分和准入规则；记录可见、支持、冲突、未知质量。检查实际融合读回的文本/图像排序，概率图与实际元素选择同时验收。禁止根据已见 benchmark 分数逐场择优。
- [ ] **同一方法闭环。** 通过源侧检查后，重建所有依赖校准/范围头，封存同一方法的模态开关与参数来源。依次完整跑 LERF4/208、ScanNet8/3 splits、NVOS8；源拒绝和失败任务不得离开分母。
- [ ] **正确地报告选择与可见性。** LERF 概率渲染和 selected-only 渲染各自保留冻结阈值。遮挡后显露的合法物体表面不能直接被当成“物理3D误检”；UQIS 缺席时不宣称完整3D身份被验证。
- [ ] **建立可复现的方法包。** 固定代码、依赖、teacher 身份、scene field、query encoder、geometry、所有可选全局状态及运行命令；冷进程无 teacher 缓存，A/B/A 查询状态隔离。将现有大量 audit 分支收敛成一个明确配置，禁止临时挑旧模块组合后直接报分。
- [ ] **保留开发/测试边界。** 已反复查看的四项结果一律开发评估；新方法冻结后，才在此前未用于方法选择的场景/查询上一次性验收。源侧留视角并不自动等于独立最终测试，范围头已经见过的源图也不能称全模型留出。
- [ ] **公平比较与成本。** 同任务权限、同几何轨道、同 scene/query 分母与聚合方式；若预算增加，报告等预算消融和总计算/显存/资产字节/查询延迟。随机部分重复种子并逐场报告差异，避免将微小均值波动写成稳定增益。
- [ ] **清理历史 authority 断链。** 找回 SPIn/PFPR 原文件及完整外部源码依赖后再运行全局 validator；在此之前它们独立隔离，当前四项 inventory 可单独审计，但不能据此把历史全局状态改成通过。

实际运行命令（新输出目录，不覆盖前次报告）：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/audits/audit_publication_readiness_next31.py \
  --output output/publication_readiness_next31_repeat/audit.json
OMP_NUM_THREADS=2 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/audits/verify_current_four_release_next31_v2.py \
  --output-root output/publication_readiness_next31_repeat/current_four
OMP_NUM_THREADS=2 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_current_four_release_next31.py tests/test_evaluation_protocol_registry.py \
  tests/test_source_accepted_category.py
OMP_NUM_THREADS=2 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python -m pytest -q \
  tests/test_evaluation_protocol_freeze.py tests/test_evaluation_protocol_registry.py \
  tests/test_ludvig_promptable_protocol_registry.py tests/test_source_accepted_category.py
```

最后一项当前预期仍有上述 5 个失败，不能把失败移除来获得“全绿”。达到投稿状态的关键产物应是**可解释且稳定的同合约方法增益、完整四任务复现包和诚实的能力边界**，而不是保证一个未测得的 SOTA 数字。
