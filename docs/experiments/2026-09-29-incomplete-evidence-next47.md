# next47：同证据分责与不完整观测下的排除机制

2026-09-29。完成 4 个 LERF 场景 × 9 个对照；全部使用同一 next44 检测缓存、源图集合、冻结几何、原生 query、相同 60 步求解和原 evaluator。没有新增 teacher 调用或人工标签来产生这些预测。研究目标收敛到“如何区分可排除区域与漏检造成的未知区域”。本轮证明了既有机制的同证据贡献，但新的简单 censor 没有提升，未替换主线。

## 主结果

| arm | 2D IoU % | AP % | selected 投影 IoU % | localization % |
|---|---:|---:|---:|---:|
| plain | 51.340 | 68.426 | 35.259 | 79.073 |
| consistency_only | 52.379 | 69.986 | 36.881 | 79.204 |
| detector_only | 51.911 | 69.050 | 35.393 | 79.073 |
| identity_only | 53.598 | 70.536 | 36.143 | 81.305 |
| detector_consistency | 52.667 | 70.162 | 37.102 | 79.205 |
| both | 53.914 | 71.211 | 37.710 | 81.234 |
| no_prior | 42.396 | 52.043 | 34.679 | 84.197 |
| censored | 53.902 | 70.274 | 37.172 | 80.413 |
| matched_confidence | 53.759 | 71.063 | 37.816 | 81.234 |

主 2×2 因子表是 detector_only / identity_only / detector_consistency / both：所有项都保留 detector 最大置信度；identity 项额外乘 native foreground/context 相对置信度；consistency 项使用留一视角连续一致性。plain 是单位置信度参考，不应把 plain→identity 的变化全部归于 native identity。

同检测结果下，detector_only→both 提升 IoU **2.002** 点、selected **2.318** 点。因此收益并非完全来自多调用 SAM3。不过，二者不是超加性的协同效应，也没有证明同预算超过公开 SOTA。完整逐场景见 `output/incomplete_evidence_next47/summary.json`；不能把四场平均当独立重复实验的显著性结论。

no_prior 同时移除原场初始化与先验正则：观测域初始化 0.5、非观测域 0，prior_strength=0，仍为相同 60 步。IoU 42.397、AP52.043，比 both 降 11.517/19.169 点；selected 降3.031点，但 localization反而更高。说明原场先验对当前求解确有作用，不能据此证明原场内每个字典/结构/压缩模块都必要。该结果只涉及当前优化预算，不能声称无先验的最优解必然更差。

both 重现 next44 的宏观结果。元素概率最大差 2.81e-6（GPU sparse accumulation）；无测试集驱动重校准。

## 最关键的补充分责：身份选择还是更新强度？

`matched_confidence` 去掉逐视角 foreground/context 差异，使用每个 query 的固定系数乘 detector confidence；固定系数使进入一致性之前的已知像素加权证据总量与 both 相同。保留 source-only 的 query 级 native 质量信息，不使用 GT。

结果为 IoU **53.759**、AP **71.063**、selected **37.816**。原始逐视角身份权重比它仅高约 **0.155** 点 IoU，selected 反而低 **0.105** 点。未进行显著性论证，不能据此证明 native identity 完全无用；但它明显削弱了“主要收益来自逐视角挑对身份”的解释。当前更多收益可由 query 级更新强度和原场先验解释，不能把 detector_only→both 的 2 点增益全部包装成身份推理创新。

这个控制仍需 native 分数估计 query 级总量，不等价于完全移除 native 编码成本，也不证明一个统一常数对所有 query 同样有效。

## 新机制实际改变了什么

在 `typed_evidence.py` 中增加可选 `censor_positive`：

1. 每个源视角将腐蚀后的正范围反投影为元素支持；
2. 对当前视角，取其他视角中的最大正支持并重新投影；
3. 仅对当前负区乘 `1-other_positive_support`；正证据和完整前向 renderer 不变；
4. 原先一致性与身份权重保持不变，默认参数保持旧行为。

这改变的是排除强度，并没有把所有负例关掉。也没有把重复的负标签当成完整性的证据。它仍是最小的正证据保护控制，**不是完整性变量推断，也不是已经建立的新颖方法贡献**。

结果：IoU 53.914→53.902，AP71.211→70.274，selected37.710→37.172，未通过。其他视角的正响应也可能是错误身份、错误范围或几何泄漏；直接保护它会保留假阳性。这个结果反驳的是该简单策略的充分性，不是否定不完整观测问题，更不证明完整范围已经无误。

## 科研判断与下一项必要证据

- next44 已有一定同证据算法价值，但增益仍不足以支持“领先强基线”的主张。
- 下一机制不能只是再换一个负例权重；需要独立区分身份可靠、范围完整、几何对应可靠三件事。当前 censor 只利用其他视角正响应，尚未获得额外独立的完整性判断。
- 小规模可信范围干预仍未完成。60–100 个人工确认 query-view 样本不能用自动伪标签或 assistant 自称人工审阅代替。已有助手审阅材料可作为辅助诊断，需明确其来源。
- official baseline、几何/语义交换表、独立泛化、完整成本表尚未完成。当前不能宣称 CVPR-ready。保留离线场与在线证据增强两条部署口径。

## 官方基线推进

已下载 [VALA 官方实现](https://github.com/changandao/VALA)，固定版本 `48902a541333d65aeb0aebf64ad664777a27c3fc`，及其三个子模块。官方 extractor 使用真实官方 fork renderer；它的实现/默认阈值与 next46 的适配版不同，不能将 next46 失败称为 VALA 失败。

本地已有 LERF RGB Gaussian PLY、COLMAP 相机和语言缓存。figurines pilot 已完成官方三层语义提取、渲染和评分：295 个源视角、4 个渲染视角、56 个标注 query-view，mIoU **57.53%**、Acc@0.25 **89.29%**、Acc@0.5 **64.29%**，没有缺失预测。结果为 `output/vala_official_next47/all_metrics_30000_0.6.json`。

这使用历史 RGB Gaussians 接官方语义提取，不是重新训练完成的作者协议复现。历史 geometry 的训练输入与 teacher 缓存 provenance 未重新确立，不能当同预算正式比较。官方评估对保存的 alpha PNG 与 JPEG GT 做 byte>10 二值化，不能把其 IoU、Acc@0.5 与当前 probability-map IoU、localization 混为一谈。几何、源视角预算及读出均不同；该单场景结果提供可运行的强机制锚点，还不能用于公平 SOTA 排名。完整四场景与固定几何交换仍未完成。

旧 `paper/radio_gs_tpami.tex` 已加历史版本说明，不将在线结果填入旧离线部署主张。当前候选研究问题与证据要求写入 `paper/CVPR_RESEARCH_PLAN.md`。

## 复现入口

`PYTHONPATH=. .../cybersim_agent/bin/python scripts/research/fuse_incomplete_evidence_next47.py --key lerf_figurines --modes detector_only,identity_only,detector_consistency,both,no_prior,censored`

每个输出 arm 使用原 `evaluate_scope_dictionary.py --pipeline --calibrated --readout anchored --element-results <arm>/<key>/elements.pt`。36 个场景评估完整落盘。selected 指标是先选元素再二维投影评分，不能写成物理三维 GT IoU。图像、NVOS、ScanNet 本轮未重跑或声明增益。没有新增防御性测试或审计框架。
