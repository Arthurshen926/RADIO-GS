# next46：干净语义写入、patch 语言头与多原型的实际对照

2026-09-28。附件建议的第一阶段完成四场景对照；第二阶段完成 figurines/ramen 原始 patch 与滑窗试点；另完成双原型和固定背景对照读出。共 38 项场景评估。没有候选达到现有保留方法，更未达到用户要求的主流水平。因此不提升默认主线、不按场景拼接赢家。基础地图、几何、原生 query encoder 与 evaluator 不变。

## 已实现的方法改动

- 新增 `radio_gs/querying/visibility_semantic_writing.py`：真实缓存 αT 写入；每条射线按贡献排序，保留覆盖 0.7 总质量的最短前缀，再与全视图 top-half contribution quantile 和 1e-4 floor 取交。只过滤语义写入，保持完整前向渲染。固定参数未用 benchmark query/GT 搜索。
- 新增原生高维 semantic sidecar 构建器。三个固定写入对照使用同一教师：全部 αT、mass/quantile gate、gate 加 5 步 normalized Weiszfeld。逐视角先形成原生 1536D 观测，聚合不经过 64D。区域教师对照保留相同 mask overlap 平均，因此它测试 gate，不声称消除了二维池化污染。
- 原始 dense 对照直接将 C-RADIOv4 的 raw patch token 送入 `SigLIP2SummaryHead`，与 next43 的 spatial adaptor + local native pooling head **不同**。这是 LESV 启发的实验性 patch→summary-head 用法；不是官方 spatial adaptor，不声称与 LESV checkpoint/文本空间实现完全相同。
- 滑窗对照重新访问合法源 RGB：工作图宽 1024、保持比例，512 窗口、256 步长，在每个窗口提取真实 RADIO token 后应用语言 MLP；Gaussian window × 与全图语言特征的非负 cosine 融合。不是仅对旧 token 做插值。随后仍用同一 96×128 写入算子。
- 双原型对照：在 gate 后逐高斯保存两个观测方向，按原生 cosine 进行固定 3 步球面聚类，第二簇至少有两个视角才保留；查询取两个原型的最大 cosine。这只是视角多模态，不证明识别出了物理实例或整体/部件。
- 固定背景词读出：原生文本编码 object/things/stuff/texture，以 query cosine 减最大背景 cosine，固定 scale=20；不做目标阈值搜索、不改变 evaluator。源监督校准读出作为对照。无新观测元素保留原场。

每个 source-calibrated arm 均在原 source training cache 的非 held 视角拟合两项标量：scale 和 offset，100 步。特征构建本身不读取 benchmark query，完整 native query 编码只在评估时使用。构建不训练新 MLP。Image/prompt 分支不接入本实验 sidecar；本轮没有重跑它们，不能声称全 query 指标已验证。

## 全部结果

所有值为百分数；4 场景为完整 LERF4，2 场景固定为 figurines/ramen。不可把两种 cohort 均值横向比较。

| 实验/arm | 场景数 | 2D IoU | AP | selected 投影 IoU |
|---|---:|---:|---:|---:|
| clean_writing_next46/alpha | 4 | 33.812 | 66.251 | 19.991 |
| clean_writing_next46/contrast_alpha | 2 | 23.404 | 53.405 | 27.465 |
| clean_writing_next46/gate | 4 | 34.614 | 67.317 | 19.477 |
| clean_writing_next46/robust | 4 | 32.782 | 66.223 | 18.319 |
| clean_writing_next46/contrast_gate | 2 | 26.855 | 55.062 | 25.652 |
| clean_writing_next46/contrast_robust | 2 | 29.913 | 54.777 | 25.499 |
| dense_language_writing_next46/alpha | 2 | 15.308 | 60.192 | 6.507 |
| dense_language_writing_next46/contrast_gate | 2 | 29.808 | 57.626 | 20.338 |
| dense_language_writing_next46/contrast_robust | 2 | 31.889 | 57.463 | 19.603 |
| dense_language_writing_next46/robust | 2 | 14.800 | 60.284 | 6.131 |
| dense_language_writing_next46/contrast_alpha | 2 | 27.334 | 56.442 | 20.366 |
| dense_language_writing_next46/gate | 2 | 15.824 | 60.838 | 6.793 |
| sliding_language_writing_next46/gate | 2 | 15.862 | 61.767 | 6.865 |
| sliding_language_writing_next46/robust | 2 | 15.040 | 61.118 | 6.188 |
| sliding_language_writing_next46/alpha | 2 | 15.362 | 60.985 | 6.596 |
| multimodal_writing_next46/multimodal | 2 | 21.968 | 57.432 | 7.483 |

现有 next44 online-source 候选 LERF4 为 **53.914 / 71.211 / 37.710**（IoU/AP/selected）；其 figurines/ramen 两场景对应 **48.209 / 64.895 / 28.876**。它使用 query-time source detector，计算预算高于本轮 query-independent sidecar，作为保留结果列出，不是等预算教师替换基线。

区域教师四场景的纯 gate 对全 αT：IoU 33.812→34.614，AP 66.251→67.317，selected 19.991→19.477。过滤发生了，但没有得到期望的 selected 提升，不能宣称证明 semantic bleeding 已解决。单中心 robust 更差。

直接写入 sidecar 同时绕过了现有 scope/dictionary 的覆盖区域读出，因此 next38/44 与它的差距不能全归因于某一新模块。严格机制归因应来自本轮相同 teacher/readout 的 alpha/gate/robust 对照。

固定背景词改善若干 IoU/selected，但同时损失 AP，仍低于保留方法。不同读出对同一原生特征产生明显不同的范围，表明部署概率判定仍是实际问题；这不是已有排序/身份问题被解决的证据。两个读出的 unsupported fallback 与冻结 next38 仅有浮点误差，检查的最大差为 2.4e-7。

## 源侧分责证据

Dense MLP 的源 held-supervision-view direct AP：figurines 70.933、ramen 68.567；滑窗后 71.118、69.192。相应 gate 融合 readback 为 71.241/65.842，滑窗为 70.652/66.458。这里使用同一组已知源伪正负区域与各 arm 共同支持像素，**不是完整对象 GT，也不是独立最终测试**；这些源图参与地图构建，仅监督校准留出。

源侧直接特征与融合结果并未出现普遍、灾难性的排序损失。因此本次控制不支持把所有低分都归因于几何写入。新 MLP 在这项源诊断中高于之前 local-pooling-head 报告，但跨轮共同支持掩码不同，不应把差值当作严格匹配的教师增益。

## 对附件建议的修正与落实边界

[VALA 原文](https://arxiv.org/html/2509.05515v2) 支持 visibility gating 与跨视角鲁棒汇聚。本实现明确使用 per-ray mass + per-view quantile，是方法适配而非逐行复现；量化截断以后也不保证仍保留 70% 质量。均值记录应按实际 retained_mass 解读。

附件给出的 `argmin sum w*d_cos` 若 d_cos=1-cos，单位球上解就是归一化加权平均，不能直接称为鲁棒中位数。这里使用 sqrt(2-2cos) 的 chord-distance Weiszfeld，仍不声称等价于 VALA 的 streaming optimizer。

[LESV 原文](https://arxiv.org/html/2604.01388v1) 的语言 MLP 与滑窗启发已实际执行；但原论文还使用不同语义几何载体。保留原 Gaussian、原 encoder 的移植不能直接承诺复现公开分数。当前新 dense 的终端定位失败，也不支持在其上立即加字典就会恢复语义。

双原型只保留两个视角簇，没有完成独立实例 ID、整体/部件包含或可验证的 component extent。直接对相关的多个 proposal 用 noisy-OR 会重复计算同一证据，后续 identity/extent factorization 不能照公式直接堆上去。

## 下一阶段集中项（未完成）

1. **可部署的 dense 身份与范围判定**：当前源有限概念校准不能可靠泛化到细粒度 query；固定背景对照也未解决。先用源侧未参与对应决策的观测，检验 local dense、原生 crop 与 query 的匹配和完整范围，分开报告 AP、正负概率及实际元素集合。不是继续扫 benchmark 阈值。
2. **有独立身份证据的 component，而非更多任意 prototype**：先建立可验证的跨视角同一实例关系，整体/部件包含不强制并簇；未知范围不作负例。prototype 的支持数只证明重复观测，不能替代身份验证。
3. **语义 super-surface 与 sparse dictionary**：本轮尚未完成。待前两项得到可靠证据再实现。当前不把几何分桶或本轮双簇称为 super-surface；字典用于压缩保留能力，不用于恢复已经混合的信息。
4. **VASD 式 query-time identity evidence**：next44 已有源检测和 foreground/context confidence；独立 component 上的可见视角投票、identity/extent 严格分解仍未实现。与 query-independent field 分开报预算和成绩。

本轮明确拒绝“gate、dense、median、两个 prototype 已接入，所以整套建议完成”的结论。没有显著性能提升；既有高分候选保持不变，实验性能力和失败证据均保存。

## 评估与复现

沿用 `evaluate_scope_dictionary.py`，读取 hierarchical_pipeline。selected/LERF3D 仍是选中 Gaussian 的 2D 重投影评分，并非物理三维 GT。不能把公开论文数值不经协议对齐就当作已达成目标。

- `scripts/research/run_clean_writing_next46.py`：区域教师 LERF4 三 arm。
- `scripts/research/build_clean_writing_next46.py --key lerf_figurines --teacher radio_language`：raw patch 语言 MLP。
- `scripts/research/extract_sliding_language_next46.py --key lerf_figurines`：真实滑窗；再以 `--dense-cache output/sliding_language_next46 --output output/sliding_language_writing_next46` 写入。
- `--two-prototypes --output output/multimodal_writing_next46`：双原型构建，评估 multimodal arm。
- `scripts/research/measure_dense_readback_next46.py`：源已知区域分责读出。
- `scripts/research/contrastive_clean_writing_next46.py`：固定背景词对照；使用保存 elements.pt 进入原 evaluator。
- `output/clean_writing_next46/summary.json`：38 项结果及宏平均；其余各 output 根下保留 execution、source_readback、extraction 与完整评估文件。

大场景首次构建遇到显存不足，已改为 CPU 累积与 GPU 分块求解后完成；没有用稀疏抽样冒充全图。没有新增防御性测试或大型审计设施。所有结果是开发证据，没有提交/部署主线替换。
