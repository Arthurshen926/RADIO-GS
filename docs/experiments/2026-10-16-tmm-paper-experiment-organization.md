# TMM 五任务对比、机制消融与论文组织

日期：2026-10-16。工作树 `/root/RADIO-GS-query-field`，分支 `codex/tmm-query-response`。
本轮遵循用户要求：直接复用可信论文比较值与已经完成的本地协议复现，不为了补表重复训练外部基线。
研究对象仍是完整 query 响应的保留与替换，教师表征、特征场维度和存储不成为本篇系统消融变量。

## 论文入口与组织

- [主稿](../../paper/tmm_response/main.tex)、[PDF](../../paper/tmm_response/build/main.pdf)。
- 主线：问题与动机 → 相关工作 → 完整响应替换方法 → 数据/输入/训练设置 → 五项主对比 → 核心机制消融 → 候选能力与决策误差 → 局限。
- [实验源登记](../../configs/research/tmm_paper_experiments.json) 为论文比较值的来源；生成 [JSON](../../paper/tmm_response/paper_experiments.json) 和 [CSV](../../paper/tmm_response/artifacts/paper_comparisons.csv)。
- 五张主对比表、机制控制表均由脚本生成。旧按实验轮次组织的正文快照保存在 `paper/tmm_response/archive/main_before_paper_organization_20261016.tex`，历史结果与诊断没有删除。
- P = 论文报告，R = 本地复现，B = 既有完整提供器，M = 新机制已实际评估。待完成不写成 0，也不以 B 冒充 M。

## 复用的外部结果及核验

| 任务 | 本地已完成复现 | 核验后的结果（%） | 使用边界 |
|---|---|---|---|
| LERF-2D | OccamLGS 四场景 | mIoU 63.6200，定位 82.8487 | 发布语义路径，兼容 RGB 几何 |
| LERF-3D selection | VALA 四场景 | mIoU 54.1249，Acc@0.25 79.3526 | VALA 语义与评估路径，Occam RGB 几何；评分是所选 Gaussian 的 2D 投影 |
| ScanNet OVS | VALA paper8 | mIoU19/15/10 34.5269 / 37.9606 / 47.3642 | Gaussian 伪 GT 与 opacity-volume 权重；不能直接比较十场景 mesh 指标 |
| NVOS | LUDVIG-SAM 八任务×三 seed | IoU 91.2577 | 发布的正笔画、all-view RGB、在线 target RGB 输入 |
| UQIS | 无对应的外部论文行 | 使用既有区域读出、v10 与 L11 内部对照 | 自定义 UQIS9 v0.3.1，不能移用 NVOS/SPIn-NeRF 数值 |

八个权威复现文件的 SHA256 均与 2026-08-01 冻结记录吻合。
局部聚合重新从逐场景/逐任务结果计算；ScanNet 冻结摘要有极小小数差异，当前表采用原始逐场景值，四舍五入后的论文展示不变。
论文上下文数值来自 [VALA 表 1/2](https://arxiv.org/html/2509.05515v1) 和
[LUDVIG 表 1/D](https://arxiv.org/html/2410.14462v5)。其中被 VALA/LUDVIG 引用的他法数字标为“该论文报告”，不误写为我们全部重新复现。

### 必须保留的历史引用纠错

旧协议摘要把 **Dr. Splat 的 43.29 / 64.30** 误标为 VALA 的 LERF-3D 论文结果。
原论文 Table 1 的 **VALA 为 58.02 / 82.85**。
因此本地 VALA 相对论文是 **−3.8951 / −3.4974 pp**，不是旧报告所写的 +10.83 / +15.05 pp。
本地运行结果 54.1249 / 79.3526 没有变化。纠错登记只覆盖当前稿件的引用，不覆盖/伪造历史封存文件。

## A14：补齐 ScanNet 八场景评分

新增 `scripts/research/score_tmm_scannet_paper8.py`，对已冻结 v10 类别预测评分。
完整八场景和三个子菜单的预测先哈希封存，再读取 GT；没有再训练、阈值搜索、菜单切换或按结果挑场景。
使用原已有 `assign_vala_pseudo_labels` 与 `volume_weighted_split_metrics`：五倍最大尺度半径、top-1000、类别均衡 Mahalanobis 投票、opacity × sx × sy × sz 权重。

| 评分轨道 | mIoU19 | mIoU15 | mIoU10 |
|---|---:|---:|---:|
| 原始十场景 mesh | 61.0115 | 62.5276 | 72.4861 |
| 八场景 Gaussian 指标补评 | 66.9480 | 67.8641 | 78.0653 |

八场景 mAcc19/15/10 为 77.9368 / 79.6133 / 87.8914%。
本项目仍使用给定无标签网格构建的固定 Gaussian，VALA 使用 RGB 优化几何。
**这是评分对齐，不是输入等价或新机制提升；两个分数的差也不是算法收益。**
论文 ScanNet 表按两个轨道分组，来源和输入差异在表注与正文中明确列出。

运行目录 `output/tmm_response_20261016/study/scannet_paper8_metric_bridge_v1/`。
完整执行约 261 秒，落盘 **96,507 bytes**，无持久稠密预测/伪标签；保留逐场景、逐类别结果、输入哈希、预测哈希与完整汇总。
原输出、教师缓存和 v10 默认均未修改。

## 核心消融如何对应方法

| 机制问题 | 已完成控制 | 当前判断 |
|---|---|---|
| 候选是否提供可用能力 | 原答案、有限完整候选 label-oracle、区域读出与完整提供器 | oracle 仅作上界归因；提供器差值不属于选择器贡献 |
| 查询是否改善完整答案选择 | full、query-blind、同视图联合置乱 | query-blind 64 树，full 128 树，包含容量差异；置乱保持完整模型更适合检验查询对应性 |
| null 查询正则是否有效 | full 与同容量/同权重 no-null | 144/144 决策相同，当前不能写成有效贡献 |
| 保留门限是否避免破坏 | 同一 full 模型，正门限 vs 零门限 | 零门限恢复文本/图像收益但损害 ScanNet；门限有取舍，不能简单降低 |
| 缺口在候选、排序还是门限 | 候选增益 = 排序遗憾 + 有符号门限代价 + 实际增益 | 图像主要损失在排序；类别完整候选空间较小 |

上述最新机制控制联合覆盖全部十个 ScanNet 房间与 268 个 UQIS 查询，使用同一模型/共同门限。
最新 full 图像 IoU/AP 为 40.9859/60.5771%，相对 v10 +1.0880/+1.2683 pp；1 个查询改善、0 个查询受损。
其余当前已评分输出持平。query-control 的九场景描述性区间仍包含零。
原全队列 E01–E04 是部分证据/融合/求解器的历史归因，保留在补充材料，不能代替新机制跨五任务的消融。

## 还需完成的最小实验集

1. LERF-2D：在原源观察契约下形成完整响应候选与可用证据，统一运行 full / blind / permuted / zero-margin。一次 query 决策跨视图保持一致。
2. LERF-3D：必须提供真正不同的独立 selected-Gaussian 候选并渲染评分；不能用语义分支消融代替。
3. NVOS：当前每视图仅一个目标 region，需建立能适用的参考/轨迹证据，再运行同组控制；不把原 86.51 分复制为新方法结果。
4. ScanNet/UQIS：已有同候选控制；补固定提供器下的跨教师配置迁移、未打开房间的验证及查询耗时。
5. 所有任务使用统一决策机制；任务可具有合法的不同观察适配，但不能按任务选择结果最好的策略。

每次方法更新必须共同反馈五任务；尚未实现的输出单列 pending。当前工作把比较、消融、归因和缺口具体化，不宣称新机制已经五项闭环。

## 复现入口

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  /root/miniconda3/envs/cybersim_agent/bin/python -m radio_gs.response_study refresh
/root/miniconda3/envs/cybersim_agent/bin/python -m radio_gs.response_study verify
/root/miniconda3/envs/cybersim_agent/bin/python -m radio_gs.response_study test
/root/miniconda3/envs/cybersim_agent/bin/python -m radio_gs.response_study paper
```

单独重建对比表：`scripts/research/build_tmm_paper_experiments.py`，只读取已有结果。
单独补评使用 `score_tmm_scannet_paper8.py --run <新名称>`；既有封存运行不会覆盖。
九个新增测试覆盖场景宏平均、八场景集合错用、AP/定位误替代、pending 冒填、VALA/Dr. Splat 引用错位和缺失引用来源。

## 本轮验证完成

120 项相关测试通过（新增 9 项）；对比表 42 行、100 个来源/生成文件经绑定核验。
`verify` 与三份 PDF 编译通过，主稿 8 页，无未解析引用或 LaTeX 引用警告。
八个外部复现权威文件的原冻结哈希全部匹配。历史 E01 仍有 8 个源版本缺少对应文本，
这一既有精确重放限制继续披露，不影响本轮外部比较文件的完整性核验。
实验目录当前 2,135,104,038 bytes，距 2 GiB 上限余 12,379,610 bytes；本轮仅增加 A14 的 96,507 bytes。
