# TMM：完整响应的保留与替换

更新：2026-10-16。分支 `codex/tmm-query-response`。
**问题、完整响应决策架构与实验框架已凝练；跨教师、跨 query 的有效性仍未全部闭环。**
本目录是当前课题的唯一论文入口。特征场重建、维度和压缩不作为本篇研究变量。

**当前推进重点已调整：** 根据“不要小修小补”的要求，停止锚点／手工排名规则的局部试探，
开始训练一个完整响应决策机制：原答案和两个冻结后端生成的完整候选 → 查询证据与答案一致性 →
学习实际替换收益 → 超过独立校准门槛才整份替换。见
[ADR0023](../../docs/adr/0023-learned-complete-response-replacement.md)、
[方法定义](replacement.tex) 和 [方法 PDF](build/replacement.pdf)。
下面的部分证据融合及历史正向控制是背景与提供器依据，不能当作新学习器的贡献结果。
L01 独立训练／开发和 L02 文字／图像迁移单独登记，不混入 E01–E04 的完整五任务实验。

**评估审计 A15/A16（10 月 16–17 日）：** LERF 208 张选集投影、ScanNet 十场景网格／八场景高斯评分
独立复算一致；LERF 语义结果只重聚合，未重新栅格化。NVOS 发现灰度 GT 二值化口径差异（约 0.17 pp），
主要差距仍集中在 flower/trex/orchids。原始源 RGB 已使用，目标 RGB 权限与 LUDVIG 不同，贡献尚未因果分离。
详见 [正确性与差距审计](../../docs/experiments/2026-10-16-tmm-protocol-correctness-audit.md)。

**联合监督阶段 D01/L03/L04** 已实现真实图像数据构造、双教师局部匹配、文本／图像联合收益学习和
六个房间－模态分组共同约束的单一校准门槛。配置为
[`response_replacement_joint.json`](../../configs/research/response_replacement_joint.json)。
模型不输入模态标识，两个完整解码器不重训。进度以运行目录 `summary.json` 为准；
完成后 [`joint_replacement_feedback.json`](joint_replacement_feedback.json) 和
[`联合实验报告`](../../docs/experiments/2026-10-16-tmm-joint-response-replacement.md) 由原始结果自动生成。
该阶段仍不能替代 LERF/NVOS 源辅助适配与 ScanNet 完整类别菜单迁移。
L05/L06 进一步在相同监督和完整候选上测试集合共同效用估计，保证候选顺序等变及比较一致性；
这些性质不自动意味着精度提升。成对模型的校准反馈参与了这一架构设计，不能把重复使用的
校准集称为未打开的模型选择检验。旧部分证据正文已存入 `archive/`；当前正文和方法定义
共用 `sections/replacement_method.tex`。

**跨任务审计 A08/A09 已完成：** 22 个源任务按完整答案单位重算证据并封存选择，
归因区分候选能力与决策能力。原始源拟合分数未通过对应关系控制；旧候选池的多指标
余量也很小。29 个独立房间的官方完整菜单／并集目标已准备好。
D02 已补齐原 Mask-Adapter：2,643 视角、97,532 区域；D03 已完成原类别 SAM3
和前景／背景证据，共 29 房间、116 个完整菜单样本。校准集仍缺 6 类。
见 [跨任务报告](../../docs/experiments/2026-10-16-tmm-cross-task-capacity.md) 与
[生成反馈](cross_task_feedback.json)。没有新五任务成功主张。

L07 已完成：文本、图像、完整类别菜单共用一个效用模型；同一个门限保护
3 房间 × 5 输出共 15 个校准单元。每个菜单的 19/15/10 类指标始终使用同一候选。
拟合使用全量 D03；进度与结果由
[类别监督反馈](category_supervision_feedback.json) 自动生成。
运行 `python scripts/research/run_tmm_multitask_program.py --sam-gpu 2` 可顺序完成
D03、L07、刷新与核验；GPU 通过 `CUDA_VISIBLE_DEVICES` 指定，失败时停止。
已完成的运行不能覆盖，续接前应检查程序进度与数据目录。

L07 开发集图像 IoU 增加 8.7794 pp，但 query-blind 增加 9.7302 pp，故查询贡献仍不成立。
四个开发类别池没有更优的完整菜单；这部分零增益不能全部归因于学习失败。
L09 的相对差异对照在 L07/L08 结果前封存，保持其余因素不变；校准结果为全部保留，
不采纳为主线。A11 完成固定效用误差与无标签输入域分析，见
[联合归因报告](../../docs/experiments/2026-10-16-tmm-joint-utility-diagnosis.md)。
111 项相关测试通过。正文现以共同效用为唯一主线，成对模型和相对差异模型归入对照。

D04 已在全部 10 个 ScanNet 原 Gaussian 场景上恢复三套完整菜单，覆盖 1,158
视角、73,905 区域。原生／融合回放最大概率误差 7.16e-7，所有 19/15/10 子菜单
均零标签变化。L08 已完成，以同一模型／门限联合评估 ScanNet
和 UQIS；A10 已区分候选上界与决策损失。运行程序为
`scripts/research/run_tmm_multitask_transfer_program.py`，当前目录
`multitask_transfer_program_v2`；本轮程序已完成，不能覆盖已封存运行。
见 [联合迁移报告](../../docs/experiments/2026-10-16-tmm-shared-category-instance-transfer.md)
与 [五输出反馈](multitask_transfer_feedback.json)。ScanNet 三项持平，UQIS 图像 IoU/AP
为 40.8668/60.4594%，文本与点查询不变；查询对照的差异区间仍包含零。
LERF/NVOS 仍待迁移，区域覆盖与几何可见域的差异也明确保留。

**L10/L11 查询证据残差对照已完成：** 五折按房间交叉拟合结构先验，再学习查询条件下的效用修正；
置乱查询约束零修正，并配套同容量、同权重的无置乱约束对照。五组策略联合评分全部
10 个 ScanNet 场景和 268 个 UQIS 查询。正式图像 IoU/AP 为 **40.9859/60.5771%**，
与此前 L06 最好水平相同；1 条查询改善、0 条受损，其他输出保留。
但 full 与无置乱约束对照的 **144/144** 个决策相同，开发门槛也未通过，故约束贡献未成立。
零门限下文本／图像 IoU 提升到 22.1666/41.1659%，却降低 ScanNet19 至 60.5368%；不采用该跨任务退化。
A12/A13 进一步把候选空间分解成排序损失、有符号门限代价和实际收益；图像主要损失在排序，
不能用降低门限解决。见 [完整机制对照与归因](../../docs/experiments/2026-10-16-tmm-residual-utility.md)
和 [生成反馈](residual_utility_feedback.json)。本轮没有增加持久稠密预测，临时数组已全部释放。

## 五项对比与机制消融的论文组织

主稿已按「引言 → 相关工作 → 单一机制 → 实验设置 → 五项主对比 → 机制消融 → 归因与局限」重排。
[当前 PDF](build/main.pdf) / [主稿](main.tex) / [实验章节](sections/experiments.tex)。
原按轮次组织的正文留在 `archive/main_before_paper_organization_20261016.tex`；历史诊断和结果仍保留。

- [对比登记](../../configs/research/tmm_paper_experiments.json)：引用 VALA、LUDVIG 原论文表格，复用已有 Occam/VALA/LUDVIG 本地复现。
- [生成表格与来源](paper_experiments.json) / [长表 CSV](artifacts/paper_comparisons.csv)：P 论文、R 复现、B 原提供器、M 新机制严格区分；不手改数值。
- 已纠正 VALA LERF3D 的历史引用错位：**58.02/82.85** 才是 VALA；43.29/64.30 是 Dr. Splat。旧封存记录不覆盖。
- A14 完成八场景 ScanNet Gaussian 指标补评，66.9480/67.8641/78.0653%。它只对齐评分，不消除给定网格与 RGB Gaussian 的输入差异。
- UQIS 加入既有区域读出基线；仅 v10 与同输入 L11 的差值归于新选择机制。没有捏造外部 UQIS 论文结果。
- 核心消融固定候选和教师，检验查询证据、置乱、null 约束与保留门限。查询盲分支为 64 树，full 为 128 树，容量差异明确披露。
- LERF 两项和 NVOS 的新机制迁移仍待完成；保留原分数不能冒充完成消融。下一步按五项共同反馈推进，不新增表征层面的系统消融。

详见 [实验组织与补全清单](../../docs/experiments/2026-10-16-tmm-paper-experiment-organization.md)。

## 方法及其边界

我们研究：**已有完整答案是否值得替换，以及替换是否修复错误而没有丢失原有能力？**
固定教师、场表示与完整解码器，按以下步骤完成一个决策：

1. 保留原答案，并用有效解码器生成有上限的完整候选，概率和实际选集一起保留。
2. 提取查询、观察与候选的一致性，区分支持、排除、未知；源拟合分数不能直接冒充效用。
3. 在独立房间学习真实答案效用的相对差异，当前控制版本使用候选集合共同效用。
4. 经共同校准后保留原答案或返回一个完整候选；类别菜单整体选择，跨视图共享查询选择。

历史部分证据融合、范围一致性及逆求解器是背景／控制，不堆叠为最终方法模块。
当前数学定义以 [replacement_method.tex](sections/replacement_method.tex) 为准；
[method.tex](sections/method.tex) 仅保留旧部分证据背景。
[方法清单](method_manifest.json) 将每项主张绑定到代码符号和对应证据。

**本次纠正的归因：** 九个 v10 UQIS 地图实际使用 `DirectAnchorMembership`，
图像通过 `infer_region` 选估计锚点，再调用与实际点相同的 `infer`。
这仍会把估计锚点强制置为正例。`QueryConditionedMembership` 是可选候选，
没有部署在 v10 这九个地图中。不能声称已解决估计身份与实际点击的权限区分。
后续 A06 已量化这处缺口：仅取消强制正值，实例集合完全不变，图像 AP 仅变化
+0.000144 pp。因此，完整响应的身份条件比单点赋值更值得研究；默认推理仍未改变。

## 实现职责

| 代码 | 作用与状态 |
|---|---|
| `radio_gs/querying/source_identity.py` | 已有文字候选身份支持；局部正向控制 |
| `radio_gs/querying/typed_evidence.py` | 正／负／可见观察契约 |
| `radio_gs/querying/local_element_fusion.py` | 保留类别路径的双原生先验及来源权重 |
| `radio_gs/querying/direct_anchor_membership.py` | v10 实际完整成员提供器；包含估计锚点局限 |
| `radio_gs/querying/response_refinement.py` | 兼容融合及首轮控制；具体符号状态见方法清单 |
| `radio_gs/querying/response_reconstruction.py` | 原求解器兼容实现及同证据求解器控制 |
| `response_permissions.py` / `visibility_refinement.py` / `support_agreement.py` | 位于上述 querying 目录；诊断候选，未晋升默认 |
| `response_study_adapters.py` / `response_stage_adapters.py` | 位于上述 querying 目录；后置／原响应阶段的实验适配 |
| `radio_gs/response_study/` | 登记、核验、导出和复现入口；不新增推理算法 |
| `scripts/research/build_tmm_*.py` / `summarize_tmm_response_study.py` | 从保留证据构建论文材料 |

数值代码保留在原位置，避免移动文件破坏历史哈希和运行依赖。新增入口负责组织，
不是替换所有任务的另一套后端。

## 实验登记与证据层次

唯一手工登记表：[configs/research/tmm_response.json](../../configs/research/tmm_response.json)。
完整原始目录：`output/tmm_response_20261016/study/`。

| 编号 | 原运行目录 | 科学问题 | 结论 |
|---|---|---|---|
| E01 | post_response_v1 | 部分支持与补集负例权限 | 仅限制负权限不足；未晋升 |
| E02 | post_response_v2 | 可用观察域归一化 | 恢复 LERF IoU，但仍有其他任务退化 |
| E03 | post_response_v3 | 范围一致性是否胜过等量更新 | UQIS 不及匹配预算控制；未晋升 |
| E04 | stage_reconstruction_v4 | 同准入证据下重建更准是否有效 | 源拟合更好但任务收益混合；UQIS 仅兼容 |

每轮均包含 31 个任务、UQIS 268 查询，先完成全队列预测再评分。
E04 前三次中断编号 F01–F03，不能作为额外实验；A01–A06 是兼容和归因审计。
所有独立选集冻结。LERF 的另一输出是 **3D 选集的 2D 投影指标**，不是有真实 3D GT 的 IoU。
UQIS 真实实例选集 IoU 与连续分数 AP 分开报告。

生成的 [指标 CSV](artifacts/metrics.csv) 包含 128 条候选指标（8 个候选 × 16 项），
每条都有固定参考值、差值、描述性区间、输入权限和是否实际干预的标记。
[证据目录](artifacts/study_catalog.json) 记录来源哈希、代码版本可用性、失败尝试和提供器审计。
不计算跨 benchmark 总均分，不把每项最佳值拼成一个“新方法”。

| 核心结果（%） | v10 | E04 对角 | E04 平方重建 | E04 Huber |
|---|---:|---:|---:|---:|
| LERF2D IoU | 70.40 | 70.40 | 70.80 | 70.82 |
| LERF2D AP | 87.62 | 87.62 | 87.00 | 86.58 |
| NVOS IoU | 86.51 | 86.57 | 86.52 | 86.51 |
| ScanNet 19 类 mIoU | 61.01 | 61.01 | 60.28 | 60.73 |
| UQIS 图像 AP | 59.31 | 59.31（兼容） | 59.31（兼容） | 59.31（兼容） |

完整 16 项及配对效应见 [E04 反馈](stage_reconstruction_v4_feedback.json)；
源目标下降与类别分解见 [E04 归因](stage_reconstruction_v4_analysis.json)。
22/22 源辅助任务的平方目标降低，并未带来全面精度提高，故不默认推广逆求解器。

正向历史结果另列于 [evidence.json](evidence.json)：文字身份准入带来选集投影
+2.541 pp；获准观察进入响应带来语义 IoU +19.789 pp。二者来自不同控制，不可相加，
也不是本轮新增精度。局部图像对应与完整成员收益归入保留提供器。

### A06：估计锚点的权限与身份条件

67 个图像干预值先封存，再对全部 268 个 UQIS 查询评分；其余任务列为保留兼容参考，
没有冒充第五轮共享机制实验。取消单点强制正值后，模型自身分数仍为 0.914–0.939，
全部高于原 0.35 阈值；图像选集 IoU 仍为 39.897960%。图像 AP 的变化区间跨零。

事后诊断：44 个估计锚点正确、23 个错误；错误者中 16 个来自包含目标的混合区域，
7 个所选区域完全没有目标。模型自成员分数辨别身份正确性的 AUROC 仅 0.5316，
不能直接把它当作查询身份置信度。条件分组不是因果效果，也不用于运行时挑选答案。
详见 [研究报告](../../docs/experiments/2026-10-16-tmm-estimated-anchor-authority.md)、
[受控结果](estimated_anchor_authority_feedback.json) 和 [置信度诊断](anchor_self_score_diagnosis.json)。

## 统一操作与可复现范围

新增定向机制实验 **Q01**：保留原双教师身份区域与完整成员分类器，通过其他视角的
查询证据定位锚点，并与同视角内证据权重打乱的控制比较。图像实际 IoU 为
**39.8980 → 38.7674%**，控制为 **38.7483%**；查询证据相对控制仅 +0.0191 pp，
区间跨零。候选不升级。全部 67 条图像干预先封存，再评分 268 条查询；其他 benchmark
仍是历史参考。Q01 不计入 E01–E04 的完整共同实验数。
见 [完整报告](../../docs/experiments/2026-10-16-tmm-query-witness-anchor.md)、
[定向反馈](query_witness_anchor_feedback.json) 与 [独立指标表](artifacts/targeted_metrics.csv)。
本次持久新增约 2.12 MB，无新权重或持久大预测。

**Q02** 进一步保留 v10 完整答案，并增加现有两个教师各自身份赢家对应的完整答案，
比较同一组视角上的查询证据后只选择一个结果。图像 AP **59.3088→59.5690%**，
实际 IoU 却 **39.8980→38.2163%**，不升级默认。三个候选中，27 条查询有更好的答案；
标注选优为 **45.0975%**，对应 5.1995 pp 候选空间和 6.8811 pp 选择损失。
45.0975% 是有限候选诊断，绝不是自动成绩。单教师全局选择也均低于联合基线。
本次新增持久文件约 2.18 MB；全部 67 条图像预测先封存，再评分 268 条查询。
详见 [Q02 报告](../../docs/experiments/2026-10-16-tmm-complete-response-verification.md)
和 [机制反馈](complete_response_verification_feedback.json)。定向指标 CSV 现在另存 32 条记录，
不与 E01–E04 的 128 条完整共同实验指标混合。

在仓库根目录、现有 `cybersim_agent` 环境运行：

```bash
python -m radio_gs.response_study status     # 当前登记状态，同时核验结果
python -m radio_gs.response_study verify     # 报告封存、完整队列、聚合指标、方法来源
python -m radio_gs.response_study results    # 导出 JSON / CSV
python -m radio_gs.response_study refresh    # 重建证据、表格、图和方法清单
python -m radio_gs.response_study test       # 相关数值、协议、存储及登记测试
python -m radio_gs.response_study paper      # 构建 main.pdf / diagnostics.pdf
python -m radio_gs.response_study reproduce E04 --run e04_replay_new
python -m radio_gs.response_study reproduce Q01 --run q01_replay_new
python -m radio_gs.response_study reproduce Q02 --run q02_replay_new
```

`reproduce` 命令仅打印重跑计划；追加 `--execute` 后执行各自登记队列。新输出不能覆盖已有目录，
源码哈希必须符合所重跑的实验。现有运行器仍负责预算和预测后评分的约束。
`verify` 检查保留报告及聚合关系，不会重新加载大预测与 GT；不能称为独立复验。
`refresh` 后再运行 `paper`，生成 PDF 的哈希记录于 [method_validation.json](method_validation.json)。

| 实验 | 源码版本 | 预测保留与重跑条件 |
|---|---|---|
| E01 | 8 条版本记录缺少对应源码文本，含阶段间重复版本 | 保留磁盘预测／结果；精确历史推理尚不能认证 |
| E02 | 原始源码快照可用 | 保留磁盘预测；旧大文件写入器需受限迁移后才能重新运行 |
| E03 | 登记的预测源码与当前一致 | RAM 预测已释放；重新评分须重算，进程退出无法续用 RAM |
| E04 | 登记的预测源码与当前一致 | 私有 tmpfs 预测已释放；新运行期间可恢复，完成后清除 |
| Q01 | 登记的预测源码与当前一致 | 9 景定向推理、268 查询评分；RAM 预测已释放，需重算 |
| Q02 | 登记的预测源码与当前一致 | 保留原完整响应的候选选择；9 景、268 查询；RAM 已释放 |

源码可用不等于已完成端到端复现：仍依赖原数据、教师／地图资产和运行环境。
E01/E02 自动启动关闭；不会拿当前改过的代码冒充旧实验。
持久预算仍为研究目录 2 GiB、单运行 256 MiB、单文件 16 MiB、空闲保留 1 GiB。
已有目录接近上限；后续保留小型清单、日志和指标，引用公共资产，不重复保存教师、
大权重或全量预测。大预测仅在 RAM／私有 tmpfs 临时生成，成功后清除。

## 文稿及后续收敛

正文 [main.tex](main.tex) 组织问题、核心机制、正向对应证据与必要控制；
[diagnostics.tex](diagnostics.tex) 保存四轮细节及负结果；
[related_work.tex](sections/related_work.tex) 保存已核对的相关工作。
`tables/`、`figures/`、`*_feedback.json`、`method_manifest.json`、`artifacts/` 是生成产物；
不要手动修改数字。旧 8 页稿仍在 `build/stage_reconstruction_v4_manuscript.zip`。

下一阶段补齐源任务有功能差异的完整候选、独立类别菜单监督与类别覆盖，检验效用证据
是否能辨别实际修复和损伤。UQIS 的候选空间较大，主要瓶颈仍是决策；源任务旧数值
变体的候选空间较小，不能一概归因于同一个阈值。需共同检验 AP、实际集合及菜单子集
指标，不能逐任务拼接最佳变体。当前未完成统一迁移或独立测试闭环，论文仍是工作稿。

完整旧说明与上下文见 [历史快照清单](../../docs/history/2026-10-16-normalization-snapshots.json)。
