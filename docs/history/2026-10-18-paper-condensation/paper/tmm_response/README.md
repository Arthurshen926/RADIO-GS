# TMM：查询证据融合与 3D 响应形成

更新：2026-10-17；分支 `codex/tmm-query-response`。
**v10 是我们的方法主体。** 其高分、身份准入、响应融合、局部对应与完整成员能力都纳入论文。
此前全部围绕替换器的定位已由
[ADR0024](../../docs/adr/0024-v10-method-ownership-and-nvos-evidence.md) 修正。
修正不改变数值，也不将旧收益算作新实验收益。

## 方法、结果与证据

- [正文](main.tex) / [PDF](build/main.pdf)：问题、方法、五项比较、核心对照与局限。
- [方法定义](sections/method.tex)：查询适用性 → 支持／排除／未知 → 几何响应及完整输出。
- [实现与主张绑定](method_manifest.json)：代码符号、来源哈希及贡献边界。
- [v10 指标与配对控制](evidence.json)：31 个任务，禁止逐任务挑最好结果。
- [对比实验](paper_experiments.json)：P=论文，R=本地复现，O=我们方法，M=可选扩展。
- [诊断附录](diagnostics.tex)：E01–E04 全部正负反馈。
- [可选完整响应替换](replacement.tex)：L01–L11 单独分析，不能代替 v10 贡献。
- [评分与输入审计](../../docs/experiments/2026-10-16-tmm-protocol-correctness-audit.md)。

v10 有不同类型的有效实现：文字候选准入、参考跟踪证据、类别双原生先验和条件完整成员。
共同原则不等于已经验证同一个求解器或网络适合所有任务。
高分必须同时交代教师、几何、源访问、监督和指标定义。

已有局部受控证据：LERF 身份准入 +2.541 pp、同证据响应使用 +19.789 pp；
UQIS 图像局部对应 +4.137 pp、完整成员复用 +5.022 pp；ScanNet 三房间双原生 +2.348 pp。
不同干预和队列不能相加，也不是 L11 的收益。
L11 图像 IoU +1.088 pp 是独立扩展；其余已测输出保留，LERF/NVOS 扩展未闭环。

## 本轮 NVOS

N01 固定八任务，交叉比较源分辨率与边界证据。教师、正提示、几何、源访问、
120 步 Huber 求解和 0.5 阈值不变。全部预测封存后评分，分别报告两种 GT 边界口径。
其他 benchmark 是哈希核验的未改动参考，不能称为全队列新推理。
GPU v1 的显存上限失败记录保留；CPU CSR v2 先对照 v10 数值一致性。
结果：旧低网格 86.5134、覆盖低网格 86.7069、旧高网格 85.6138、覆盖高网格
87.7776 IoU。组合干预提升 7/8 项，AP 98.0222 → 98.1527；orchids AP 下降
2.0161 点，不能宣称所有指标共同提升。候选尚未切换默认。

A17/A19 核验 v10 原连续响应、独立成员场及两种读出：原独立选集投影 85.2245，
没有被遗漏的更高 NVOS 总分；交叉读出揭示成员场与语义场的目标／读出不能互换。
A18 在 N01 的同一批 32 张封存响应上补齐 AP/AUROC。
见[本轮完整分析](../../docs/experiments/2026-10-17-nvos-evidence-and-v10-method.md)
及[逐任务反馈](nvos_coverage_feedback.json)。本轮六个运行／准备目录合计约
0.61 MiB，私有临时预测已全部释放。

N02 已完成：同一响应场下，固定／拟合内／留出源估计分别为 87.7776／87.5151／
87.5836 IoU。源端目标都提高，但 trex 增加误报导致整体未超过 N01；不晋升默认。
A20 完整分数诊断发现，orchids 的 AP 损失主要对应背景分数变化（−2.2373 pp），
前景变化贡献为正（+0.2212 pp）。这项使用 GT 的拆分只用于解释，不参与推理。
见[ADR0025](../../docs/adr/0025-source-only-response-readout-calibration.md)、
[本轮完整分析](../../docs/experiments/2026-10-17-nvos-source-readout.md)及
[八任务误差图](../../output/tmm_response_20261016/study/nvos_score_changes_v1/errors.png)。
本轮新增约 0.93 MiB 记录，临时预测已释放。

## 实现入口

| 实现 | 职责 |
|---|---|
| `radio_gs/querying/source_identity.py` | 文字候选身份准入 |
| `typed_evidence.py` / `local_element_fusion.py` | 观察权限、类别先验及源权重 |
| `reference_response.py` / `response_reconstruction.py` | NVOS 响应及受控求解 |
| `direct_anchor_membership.py` | v10 实际完整成员能力 |
| `coverage_evidence.py` | NVOS 覆盖证据候选，未晋升默认 |
| `response_calibration.py` | 根据源证据估计响应决策阈值，目标 GT 不参与 |
| `radio_gs/response_study/` | 登记、核验、生成和受限复现 |

短文件名均位于 `radio_gs/querying/`。九个 v10 UQIS 地图实际用
`DirectAnchorMembership`，不是可选 `QueryConditionedMembership`。估计图像锚点
强制正值的单点消融已完成，实际集合不变；不能声称它解决了身份不确定性。

## 统一操作与存储

使用现有 `cybersim_agent` 环境，在工作树根目录运行：

```bash
python -m radio_gs.response_study status
python -m radio_gs.response_study verify
python -m radio_gs.response_study results
python -m radio_gs.response_study refresh
python -m radio_gs.response_study test
python -m radio_gs.response_study paper
```

登记表：[tmm_response.json](../../configs/research/tmm_response.json)。
`verify` 核验报告与聚合，不是重新推理；`refresh` 生成表格，不手填数字。
`paper` 构建正文、诊断、可选替换三个 PDF。复用外部复现，不为排版重训。

持久预算：研究目录 2 GiB、运行 256 MiB、单文件 16 MiB；保存小型配置、源码快照、
封存哈希和结果。大预测在有上限的私有 tmpfs 中，完成后释放，不重复保存教师／地图。
E01 的 8 条历史源码版本缺少文本，精确旧推理不可认证，不能用现代码冒充。
历史细节在 docs/experiments；此次调整前的完整说明见
[说明快照](../../docs/history/2026-10-17-v10-attribution/manifest.json)。

## 局部响应修正 N03

已完成全八任务、六组规则及 A21 空间诊断。留出局部 IoU/AP 为
87.5538/98.2704；相对 v10 八项 IoU 都提高，比匹配改动量对照多 0.3209 pp，
但比 N01 少 0.2238 pp，未升级默认。N01 仍为不访问目标 RGB 条件下的最佳 IoU 候选。
A22 源参考掩码与跟踪重合度为 96.49%–99.82%，不代表 GT 精度或完整范围。
[完整结果与收敛决策](../../docs/experiments/2026-10-17-nvos-local-credit.md)。
正文已补齐真实成员解码公式和 22/3 房间的监督／校准条件，没有新增训练。


## 图像条件响应 N04–N06

用户允许合理调整输入和模块。N04 用固定旧响应分别引导重建 RGB 与真实
目标 RGB，在完整八任务上隔离输入效应。真实 RGB 图像响应为 89.0636，
实际三维回投后为 88.3201 IoU，较 v10 提高 1.8067 pp；AP 则下降。
这是视角条件更新，还不是统一全视角响应场，也尚未达到 SOTA。
[结果、协议与复跑审计](../../docs/experiments/2026-10-17-nvos-image-response.md)。

N05 的正向密集提示候选为 85.3829 IoU，低于完整正负提示 88.0159；
删除负提示对八项均有损失。[完整正负反馈](../../docs/experiments/2026-10-17-nvos-partial-response.md)。
N06 已完成固定身份点的去框对照，三维 IoU 为 84.1524；有密集先验但去框为
87.1612，均未超越 N04。[完整归因与后续方向](../../docs/experiments/2026-10-17-nvos-unboxed-response.md)。
所有规则在完整八任务统一使用；不按任务挑选结果。N05/N06 不测 AP。

新模块为 `image_conditioned_response.py`、`partial_response_prompt.py` 和
`unboxed_response_prompt.py`。正式 N04 为 v3，v2 源码快照例外保留；
两次完整推理在最终评分尺寸的二值掩码完全相同。报告保存参数、源码、
输入哈希、逐任务混淆计数和八任务图；不新增权重或持久密集预测。

## 参考记忆 N07

原参考查询经冻结教师记忆模块直接传至目标视角。静态、两帧、序列证据的
三维 IoU 分别为 88.3201、87.9449、87.8852；增加记忆未超过静态方案。
[完整结果](../../docs/experiments/2026-10-17-nvos-memory-response.md)同时记录
源码审计、复跑浮点差异和 VCAR 的七任务／提示／alpha 读出差异。
此结果保留为完整负对照，不将各任务的赢家拼接成新指标。

## 原分辨率写回 N08

固定同一批图像证据，静态分支从粗网格的 88.3201 提高到 **88.8424 IoU**，
八任务全部改善；相对 v10 提高 2.3290 pp。原预设主要候选（序列记忆）
为 88.1124，仍未超过静态分支。最佳完整对照与主要候选分开报告。
图像到三维的差距从 0.7435 缩小到 0.2212 pp。N09 完整复跑补测 AP 为
96.5817，低于 v10 的 98.0222；两种 GT 下逐任务混淆计数与 N08 相同，
三维连续响应并非逐位相同。尚未达到 SOTA。
[实现修复、完整结果和归因](../../docs/experiments/2026-10-17-nvos-native-writeback.md)。
164 项登记测试通过；原高斯足迹的数值问题有独立渲染器回归测试和完整
同网格对照，修复本身的平均 IoU 变化小于 0.000006 pp。

## 原始正负证据迁移 N09–N10

N09 增加冻结 DINOv2 和官方参考负涂画，比较仅涂画授权与参考 SAM 掩码／
补集授权，统一经过 SAM 及原分辨率三维写回；完整八任务 IoU 为
81.5803／86.0812。前者补回 flower、orchids 的范围，却混入其他 horns、
leaves 实例。参考端留一准确率接近 100% 不能证明目标端身份正确。
N10 对完全相同的亲和概率直接写回，IoU 为 77.7307／84.2098；跳过 SAM
改善 AP 但没有超过静态方案。新增两项均不晋升默认。

[完整机制分析及下一步设计边界](../../docs/experiments/2026-10-17-nvos-signed-affinity.md)；
[N09 逐任务反馈](nvos_signed_affinity_feedback.json)、
[N10 逐任务反馈](nvos_direct_affinity_feedback.json)。全部预测先封存，再评分；
同时报告两种 GT、IoU/AP、独立混淆计数及源端变化。其他四个 benchmark
仅核验既有结果，不称为新增全套推理。代码、来源及失败尝试均保留；无新增权重
或持久密集预测。N09/N10 阶段的最佳完整三维对照为 N08 静态分支 88.8424 IoU。

## 实例约束下的范围补全 N11

完整八任务实际三维 IoU 提高到 **89.9734**，比静态方案多 1.1310 点，
六任务改善、两任务下降；AP 为 96.4276，略低于静态方案的 96.5817。
无约束补全／v10 响应锚点／原始涂画几何锚点三维 IoU 分别为
83.5805／89.9673／89.9734。实例约束保留 flower 的范围收益并抑制其他
horns、leaves 实例；两种锚点接近，未证明几何版本有独立优势。
orchids 下方范围和 fern／trex 内部误报仍是缺口，尚未达到 SOTA。

[完整结果、空间诊断与方法决策](../../docs/experiments/2026-10-17-nvos-instance-extent.md)；
[逐任务反馈](nvos_instance_extent_feedback.json)。72 张预测先封存再评分，
其他四项 benchmark 仍为未改动参考；170 项登记测试通过。新增记录共
287,337 字节，所有私有临时数组已释放；原研究额度仅余 7,696 字节。
默认五 benchmark 配方未更换，也不按任务拼接最佳分支。

## 参考响应补全与采纳 N12–N14（2026-10-18）

N14 阶段最强固定 NVOS 方案为源端质量采纳：实际三维 IoU **93.1730**、AP **98.9451**。
预先指定的可行性主候选为 92.9424 / 98.8599。N12 完整参考替换为 91.4774；
N13 概率残差 90.7469 且相对完整替换八任务全降，故不保留为必要模块。
三轮均完整八任务、两种 GT，全部输出先封存再评分。N14 在 N12 分析后设计，
为开发结果；源图质量规则不读取场景名或目标分数，重新调用教师并实际三维写回。

方法正文凝练为参考范围补全、参考端采纳、实例传播、原生三维求解；旧 N11 公式
移到诊断附录。所有新分支仍增加原源教师残差，查询全局一致性未闭环。
其他四 benchmark 为未改动参考，无默认切换或 SOTA 认证。180 项测试通过。
[完整研究记录](../../docs/experiments/2026-10-18-nvos-reference-completion.md)。
新输出受独立 8 MiB 预算管理，旧 2 GiB 研究目录保持不变，密集预测已释放。


## 多视角能力迁移与停止决定 N15（2026-10-18）

完整八任务和两种 GT 已完成。补全参考的仅源融合为 **92.7461 IoU / 98.8149 AP**，
相对同流程原始参考的 88.0061 / 98.2586，验证了不使用目标图像证据的能力迁移。
源＋目标等权融合为 **93.0302 / 98.9797**，相对 N14 的 IoU 下降、AP 略升；
未通过预定改善条件，因此停止该联合融合路线。截至 N15，最高 IoU 候选为 N14 93.1730。
这不是全局性能上限证明，也未取得同协议 SOTA 认证；近期论文报告还包括
LEGO 94.2、WildSeg3D 94.1，不能只把超过 VCAR 93.5 当作目标完成。
184 项登记测试通过，N15 持久实验记录增加 0.5583 MiB，临时密集数组已释放。
[结果、归因与协议边界](../../docs/experiments/2026-10-18-nvos-multiview-completion.md)。


## 目标重解码与当前取舍 N16（2026-10-18）

局部重解码主候选把实际三维 IoU 提高到 **93.2805**，AP 降到 **98.4663**；
全图重解码为 93.0998 / 98.2626。前者相对 N14 的 IoU 增加 0.1075 pp，
AP 损失 0.4787 pp，4 项改善、4 项下降；去掉整个 horns 场景后平均 IoU 增益消失。
因此停止重解码分支，保留 N14 作为双指标更均衡的方案。N16 是当前最高 IoU
的实测对照，不能继续把 N14 写成最高 IoU；也不能将 N16 称为双指标改进或 SOTA。
188 项测试通过；新实验记录 0.2662 MiB，所有临时密集数组已释放。
[完整结果与停止依据](../../docs/experiments/2026-10-18-nvos-target-redecode.md)。

## 数值归因与优化收束 A24（2026-10-18）

八任务、24 张图像概率响应与 logits 已精确重放，两种 GT 的原 AP 和混淆计数
均复现。双精度及稳定概率混合排序对平均 AP 的影响小于 **0.002 pp**，全部
硬掩码不变。局部重解码相对 N14 的图像 AP 差仍为 **−2.2124 pp**，不能归因
于怀疑的 float32 sigmoid/插值因素。本轮没有新三维求解或性能提升。
维持 N14 均衡方案、N16 最高 IoU 对照，以及 N15 仅源能力结果；停止当前融合
与重解码细化搜索。该经验平台期不等于全局上限，SOTA 目标仍未完成。
192 项登记测试通过；含失败记录的新实验输出 0.1724 MiB，临时数组全部释放。
[完整审计与停止判断](../../docs/experiments/2026-10-18-nvos-score-precision.md)。
