# LERF / ScanNet OVS 正确性与 NVOS 差距审计

日期：2026-10-16 至 17。当前工作树 `/root/RADIO-GS-query-field`，分支 `codex/tmm-query-response`。
审计对象是当前论文使用的**保留 v10 提供器结果**；不把这些结果当作 L11 新学习机制的收益。

## 结论与适用范围

本轮没有发现导致 LERF 投影或 ScanNet 高分的算分错误、漏掉失败样本、类别错位或均值汇总错误。
LERF 208 张选集投影图全部独立复算一致；ScanNet 十场景网格和八场景高斯指标独立复算一致。
但“数字算对了”不等于“相对 VALA 的全部差额来自本文新机制”：教师、源视图查询访问、几何载体不同，
当前高分主要属于 v10 提供器。L11 在 ScanNet 全部保留原类别菜单，在 LERF/NVOS 尚未完成迁移。

NVOS 不是“完全没用原始 RGB”。本方法使用原始源 RGB 构建几何、参考 SAM 掩码和源跟踪，
排除目标帧；本地三种子 LUDVIG-SAM 复现使用 all-view RGB 几何，并在含目标帧的视图上运行 SAM。
双方都是正向 scribble；缺少负向 scribble 不能解释与这条 LUDVIG-SAM 对照的差距。
此外发现 GT 灰度边界二值化的真实差异，但只解释约 0.17 pp，不能解释约 4.74 pp 的总体差距。

这不是从训练原图到最终预测的全链路重新训练/推理，也不是对全部历史开发过程无泄漏的证明。
历史 benchmark 已多次用于研发反馈，不能称为未触碰测试集。共享 GPU 持续有其他任务；本轮数值复算用 CPU，
未重新栅格化 LERF 语义响应。其 70.4007% 是逐对象记录重聚合及实现审查通过，证据强度低于投影图的逐像素复算。

## 审计结果

| 对象 | 本轮检查 | 数值结果 |
|---|---|---|
| LERF-2D | 22 帧、208 对象，与原始 JSON 查询集合逐项对齐；无重复、无漏项；IoU/localization 重聚合 | 70.4007 / 92.4204% |
| LERF selected-set projection | 208 张已有 PNG 与独立 GT JPEG，`>10` 阈值逐像素重算；对象 IoU 最大差 0 | mIoU 66.9633%，Acc@.25 90.0178% |
| ScanNet mesh10 | 原始 NYU40 类别、完整菜单 argmax、不可观测 void、GT 范围外忽略、GT 出现类与场景等权 | 19/15/10 类 61.0115 / 62.5276 / 72.4861% |
| ScanNet mesh8 | 上述相同预测，只取论文八场景 | 61.0567 / 62.2573 / 72.8852% |
| ScanNet Gaussian8 | 独立 Mahalanobis 类均衡投票、opacity-volume 权重、混淆矩阵 | 66.9480 / 67.8641 / 78.0653% |
| NVOS | 八张连续响应恢复到原始 GT 分辨率，独立 TP/FP/FN；与历史 float32 IoU 差小于 1e-7 | 86.5134% |

ScanNet 独立实现使用 SciPy 四元数旋转、半径邻域和 float64 类均衡密度投票，没有复用原评估 helper。
八场景共 **1,134,207 个高斯**的伪标签 SHA256 全部一致；每场景固定采样 32 个高斯，调用 VALA 发布的
原函数（仅设备设为 CPU），共 256 个标签全部一致。八场景的最大邻域小于 1000、空邻域为 0，
因此本地 helper 的空邻域 fallback 与数值稳定化差异没有影响本次结果。

全部十场景的高斯中心与提供网格对应顶点坐标误差为 **0**。这来自明确提供的未标注 `points3d.ply`，
不是把语义标签当作预测；检查了 PLY 顶点属性和几何 state，均未含语义标签。
但这种载体与 VALA 的优化高斯不同，高斯指标的伪 GT 和体积权重也依赖载体。
八场景 19 类从 mesh 的 61.0567% 到 Gaussian 的 66.9480%，**+5.8913 pp 是评估域变化，不是算法提升**。
同理 15/10 类为 +5.6067/+5.1801 pp。不能把十场景 mesh、八场景 Gaussian 或不同载体的结果直接当作同条件消融。

## 高分来自哪里，哪些贡献已有证据

1. 当前 LERF 使用 SAM3 文本条件源分割、SigLIP2 giant 原生匹配、身份筛选、完整响应融合及独立选集。
   它与以 SAM/CLIP 特征提升为主的 VALA 不共享提供器或全部输入条件。每场景两标量校准使用源 query detections，
   是源查询监督，不能描述成 query-independent training；也不是目标帧 GT 监督。
2. 当前 ScanNet 使用 SigLIP2 + Mask-Adapter 双原生语义、SAM3 类别源证据、可见性投票及提供网格。
  本轮核查代码路径把预测与 GT 评分分开，未发现语义标签进入这些预测计算。未对全部外部教师预训练数据做溯源认证。
  虽然 `local_element_fusion.py` 也实现了监督残差头，当前配置走 `average_prior`，不能仅凭模块存在就说该头正在使用测试标签。
3. 历史匹配控制已显示：LERF 身份准入对投影 IoU +2.5411 pp；相同证据的输出融合对语义 IoU +19.7890 pp；
   ScanNet 三房间双原生对照 +2.3476 pp。这些分别有具体干预依据，**不是可相加的全系统收益分解**。
4. VALA 的 LERF-3D 论文值是 **58.02%**，不是此前曾误标成 VALA 的 Dr. Splat 43.29%。
   当前 66.9633% 比 VALA 论文高 8.9433 pp；LERF-2D 的 70.4007% 比 VALA 61.7% 高 8.7007 pp。
   ScanNet 19 类八场景高斯 66.9480% 与 VALA 论文 32.11% / 本地复现 34.5269% 差距很大，
   评分复算通过仍不能消除载体与教师差异，尚无匹配实验将该差距全部归因于响应机制。

源访问检查覆盖 LERF 四场景与 NVOS 八任务：语义源列表、几何 RGB 拟合列表、MoGe manifest 与 source authority
均不包含对应测试/目标帧。LERF 实际 identity cache 文件集合与声明源帧完全一致（122/293/169/180 帧）；
NVOS 源 RGB 字典与 tracking 列表也排除目标帧。这里核查的是源读取及深度/拟合记录；注册相机/SfM 可使用的
姿态元数据和全部历史程序读文件行为，并未被本轮重新追踪。

## NVOS：先分开评分口径、输入权限与响应缺陷

本地 LUDVIG-SAM 取三种子固定阈值结果，均值 **91.2577%**，不是其 `oracle_target_iou`。
我们的原始记录为 **86.5134%**，相差 **4.7443 pp**。SAGA 论文 92.6% 是另一方法的外部参考，差 6.0866 pp，
不能仅凭其论文数值判断某个实现环节的因果贡献。

| 任务 | 当前 IoU | 本地 LUDVIG-SAM | 差距 pp | 当前 Precision / Recall |
|---|---:|---:|---:|---:|
| fern | 82.19 | 84.51 | 2.32 | 89.45 / 91.00 |
| flower | 86.53 | 97.57 | 11.05 | 98.81 / 87.44 |
| fortress | 97.23 | 98.49 | 1.26 | 99.92 / 97.30 |
| horns_center | 95.73 | 97.97 | 2.24 | 98.52 / 97.13 |
| horns_left | 92.70 | 92.65 | -0.05 | 95.87 / 96.56 |
| leaves | 94.08 | 96.65 | 2.56 | 99.58 / 94.46 |
| orchids | 68.14 | 74.56 | 6.42 | 99.91 / 68.18 |
| trex | 75.51 | 87.66 | 12.15 | 83.53 / 88.72 |

flower、trex、orchids 占净均值差距的 **78.04%**；这是错误集中度，不是原因占比。

### 评分口径的实际影响

所有以下控制使用**同一批已保留响应**，没有重跑教师、改变几何或访问目标 RGB。

| 仅改变的因素 | 八任务平均 IoU | 相对当前 pp |
|---|---:|---:|
| 原协议：GT `>127`，OpenCV 双线性，响应 `>=0.5` | 86.5134 | 0 |
| 仅使用 PyTorch 双线性（与 LUDVIG 相同） | 86.5134 | +0.000019 |
| 仅把 GT 改为 LUDVIG 的 `gt // gt.max()` | 86.6835 | +0.1701 |
| 仅借用 LUDVIG 响应 min-max、PyTorch 双线性、固定 `>0.25` | 86.6085 | +0.0951 |
| 同时借用上述 GT 与读出规则 | 86.6131 | +0.0997 |
| 每任务查看 GT 后，从 0.01–0.99 选最好阈值，诊断用 | 90.0921 | +3.5787 |

灰度 GT 差异涉及 flower 6075、horns_center 19778、horns_left 10819、orchids 23615 个像素；
其他四任务的前景相同。这是真实协议不一致，需要在论文中明确，**不能悄悄改写历史 86.5134%**。
最后一行使用测试 GT，是 99 阈值网格内的 oracle 诊断，不是可部署性能，也不是所有可能方法的上限。
各项增益不独立，不可相加。统一套用 LUDVIG 的读出会使 trex 从 75.51% 降至 65.78%。

### 方法环节的诊断

- **flower / orchids：完整目标响应不足。** 高 precision、低 recall，误差图显示花簇的整块漏检，不只是轮廓几像素。
  orchids 的弱响应仍落在一部分遗漏花朵上；GT 事后选低阈值可使其达到网格内 85.70%，证明当前响应有可利用信号。
  但如何仅凭合法源证据恢复这些区域而不伤害 trex，仍是方法问题；不能逐场景手调阈值。
- **trex：细结构与背景分离不足。** 当前同时有肋骨等细结构漏检、背景响应；网格 oracle 仅 76.56%，
  远低于本地 LUDVIG-SAM 87.66%。因此仅改标量阈值不足。响应图中的空间错位/外溢与几何、遮挡、源掩码或融合均可能有关，
  本轮不能在这些原因间作最终因果判定。
- **输入权限确实不同。** 我们源辅助链路排除目标帧；LUDVIG-SAM 的 all-view 几何和 SAM 特征包含目标 RGB。
  当前几何源拟合为 3000 步且使用低分辨率源 raster；本地 LUDVIG 是 30k all-view 几何和另一套处理尺度。
  当前目标响应栅格为 1008×756，源响应拟合为 256×192，几何 RGB 拟合宽 320；LUDVIG 配置尺寸为 1600×1199。
  “OpenCV/PyTorch 双线性几乎等价”只检验同一响应的插值，**没有证明这些生成分辨率差异无关**。
  这些不是简单更换插值算子可以模拟的差异。目标 RGB 或更好几何究竟贡献多少 pp，尚未测量。
- **未发现观测域归一化污染。** 八任务不可观测元素的原始 scores 全为 0，当前渲染分子未显式再乘 observed
  不会在这些已测结果中加入额外响应。这不保证未来任意新响应都满足该前置条件。

误差图：灰色 TP，橙色 FP，青色 FN；只对显示下采样，全部数值按原始 GT 分辨率计算。

![NVOS 响应与错误](../../output/tmm_response_20261016/study/protocol_input_lineage_audit_v2/nvos_error_diagnosis.png)

## 接下来的可识别实验

先冻结一个公开可比的评分口径，并保留当前历史轨道；不能混合 GT 二值化和阈值后择优。
利用已有 LUDVIG 复现资产，在相同参考正向 scribble、相同 SAM/响应解码器、固定评分下进行：

1. 固定 all-view 几何，仅从 SAM/源语义构造中去掉目标帧：识别目标 RGB 语义证据的影响。
2. 固定同一组合法源证据与读出，对比两套冻结几何：识别几何/投影影响，并明确 all-view 几何本身的目标 RGB 权限。
3. 在上述对齐后，用统一源证据规则修复“部分观察误当完整排除”和细结构空间分离；仍一起反馈全部 benchmark。

本轮未运行这些新推理实验，没有推定它们会自动抹平 4.74 pp，也没有推广新默认方法。

## 可复查产物

- A15：`output/tmm_response_20261016/study/protocol_correctness_audit_v2/{summary,lerf,nvos,scannet,prediction_seal}.json`。
- A16：`output/tmm_response_20261016/study/protocol_input_lineage_audit_v2/`，源权限、GT/输入哈希、固定 upstream 源快照及误差图。
- CPU 主审计约 44 秒；新增输出仅紧凑 JSON、源码快照与一张图，无模型/特征/密集预测副本。
- 首次审计在“所有 GT 已二值化”的检查处终止，正是上述灰度边界差异；保留 failure receipt，v2 显式测量其影响。
- `scripts/research/audit_tmm_protocol_correctness.py --run <新名称>` 与
  `scripts/research/audit_tmm_protocol_lineage.py --run <新名称>` 可复跑；解释器和线程配置见根 CONTEXT。

原始依据：[VALA 论文与附录](https://arxiv.org/html/2509.05515v2)、
[VALA 评分实现](https://github.com/changandao/VALA/blob/48902a541333d65aeb0aebf64ad664777a27c3fc/eval/evaluator3D.py)、
[LUDVIG NVOS 实现](https://github.com/naver/ludvig/blob/4461fc515439bb498a75d71738a1e73cf7a452ed/evaluation/spin_nvos/base.py)、
[LUDVIG SAM 源视图处理](https://github.com/naver/ludvig/blob/4461fc515439bb498a75d71738a1e73cf7a452ed/predictors/sam.py)。
