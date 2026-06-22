# GaussFM 投稿项目阶段汇报材料

日期：2026-06-09

用途：导师组汇报 / 投稿前中期检查 / 基金申请书式项目梳理

## 一、项目名称与总体定位

**项目名称：** GaussFM：面向开放词汇三维场景理解的紧凑基础特征高斯场

**一句话概括：** 本项目希望把二维基础视觉模型中的高维语义特征，编码为三维高斯场上的紧凑隐变量，并在查询时按需重建出可用于开放词汇理解的 foundation feature，使同一个紧凑三维特征记忆同时具备二维新视角定位、三维 primitive 级选择和跨数据集点级查询能力。

**投稿定位：** 目标按 TPAMI 级别期刊论文准备。本文不是单一 benchmark 的增量方法，而是围绕“紧凑基础特征三维场景记忆”构建了一套方法、协议、消融、定性、效率和风险分析完整闭环。

## 二、立项背景：为什么这个问题重要

近年来，RADIO、DINO、SigLIP、SAM 等视觉基础模型在二维图像上表现出很强的语义、定位和匹配能力。但是这些能力主要存在于单帧图像或二维特征图中。当任务从二维图像扩展到三维场景时，现有范式面临三个核心矛盾。

第一，**二维基础特征强，但视角局部且存储昂贵。** 逐帧提取或缓存高维特征虽然可以保留强语义信息，但无法自然形成一个可部署、可复用的三维场景表示。若把 1280 维甚至更高维的 dense feature 直接存到每个 Gaussian 或每个点上，存储和训练成本很高，也不一定带来稳定的三维查询能力。

第二，**开放词汇三维理解需要同时服务二维和三维接口。** LERF/LangSplat 类任务重视 rendered-view heatmap；OpenGaussian、Dr. Splat、VALA-aligned 等方向逐渐强调直接在 Gaussian、primitive 或 point level 查询目标对象。只解决 rendered 2D heatmap 并不足以支撑新一代 open-vocabulary 3D scene understanding。

第三，**三维表示不能退化成特定任务的分类器。** 如果每个下游任务都单独训练一个 head 或缓存一个特征库，方法就难以成为通用场景记忆。理想形式应是：一个紧凑三维特征场，在统一的查询执行层中按需完成特征重建、文本匹配、几何投影和对象支持形成，而不是为 2D/3D benchmark 拼接互不相关的分支。

因此，本项目的核心问题是：

> 如何把冻结二维基础模型的高维特征能力，压缩到一个紧凑、可渲染、可直接查询、可跨任务复用的三维高斯场景记忆中？

## 三、国内外研究现状与不足

### 3.1 二维开放词汇特征场

LERF、LangSplat 等工作证明，利用 CLIP 或其他视觉语言特征可以在三维场景中实现 rendered-view open-vocabulary localization。其优势是视觉效果直观、评价协议成熟；不足是查询通常发生在渲染后的二维 feature map 上，对三维 primitive 级对象理解支持不足。

### 3.2 三维 Gaussian 开放词汇理解

OpenGaussian、Dr. Splat、VALA-aligned 等工作进一步强调 Gaussian-level、point-level 或 instance-level querying。特别是 the unpublished protocol source 已经明确把 LERF-OVS 的 2D 和 3D open-vocabulary query 作为共同实验口径。因此，本文不能把“同时展示 2D/3D 查询”作为独占新意，而应把它视为领域正在形成的标准评价维度。本文的差异应落在：如何用紧凑隐变量表示高维 foundation features，如何在查询时按需重建可用特征，如何同时获得存储压缩和下游质量提升，以及如何把多视角注册、边界支持和多头一致性内化为一个统一三维特征记忆。

### 3.3 基础模型蒸馏与多头下游可用性

RADIO、DINO、SAM、SigLIP 等基础模型提供了强大的通用特征空间。现有方法常将其作为监督来源或评价 head 使用，但三维场景表示如何在强压缩后仍保持这些特征对下游任务的可用性，是一个尚未充分解决的问题。

### 3.4 本项目切入点

本项目不把三维 Gaussian 仅作为渲染载体，也不把二维 feature cache 当作最终答案，而是提出：

> 学习一个 compact foundation-feature Gaussian field，使其在存储上紧凑，在语义上兼容冻结基础模型，在接口上同时支持 rendered-view、primitive-level 和 point-level 查询。

## 四、拟解决的关键科学问题

### 问题一：高维基础特征如何压缩成紧凑隐变量，同时保留下游可用性？

二维 foundation feature 往往维度高、噪声来自单帧、视角间不一致。本文要解决的不是把高维特征离线完整存进三维场景，而是学习一个 compact latent memory：显式存储的是低维隐变量和全局解码器，查询发生时再按需重建 RADIO-compatible features。这样既避免每个 Gaussian 存储完整 1280D 特征，又保留文本匹配、DINO/SAM adaptor 等下游任务需要的 feature geometry。

### 问题二：同一个紧凑三维特征记忆如何统一支撑二维、三维和点级查询？

Rendered-view grounding 查询的是像素级特征图，direct 3D object selection 查询的是 Gaussian primitive，ScanNet point-query 查询的是离散点云位置。三者输出形态不同，但底层都可以表述为：从 compact Gaussian memory 中按需重建 foundation features，然后进行文本相似度计算、空间聚合和对象支持形成。本文要证明的不是“两个接口”的机械并列，而是一个统一查询执行层在不同采样域上的自然实例化。

### 问题三：direct 3D 查询如何从 primitive similarity 走向 object support？

仅用文本和单个 Gaussian feature 的相似度，容易出现小物体漏检、碎片化、低可见区域不稳定等问题。本文把 prompt ensemble、score component guard 和 label-free color-edge support calibration 作为统一查询执行层中的对象支持形成机制，而不是只服务某一个表格的后处理。它的作用是把局部相似度场转化为稳定 object support，从而改善 2D/3D 查询中的边界、完整性和小物体召回。

### 问题四：如何证明 compact field 不是单纯压缩，而能优于原始逐帧 RADIO 特征？

如果 compact field 只是低维近似原始 RADIO feature，其贡献有限。本文通过 frame-wise RADIO vs. reconstructed scene field 的评估证明：多视角重建后的三维 feature field 在若干冻结下游 head 上可以强于原始逐帧 RADIO 特征，说明三维场景级聚合具有去噪和增强作用。这里不建议在汇报中使用“学生模型全面超过教师模型”的口径，而应说“原始逐帧 foundation features 与重建场景特征的下游可用性对比”。

## 五、研究目标

本项目的总体目标是形成一篇可投稿顶刊的完整论文，围绕 GaussFM 建立一个逻辑闭环：

1. 提出 compact foundation-feature Gaussian field 的统一表示；
2. 支持 LERF rendered-view 2D open-vocabulary grounding；
3. 支持 LERF direct 3D primitive-level object selection；
4. 支持 ScanNet VALA-aligned ScanNet-8 direct point-query；
5. 证明相对原始逐帧 RADIO 特征和显式高维 memory 的下游可用性与存储优势；
6. 给出系统消融、存储效率、失败案例、定性结果和协议边界；
7. 将论文整理为 TPAMI 期刊投稿格式。

## 六、总体技术路线

GaussFM 的核心思想可以分成四层。

### 6.1 离线基础特征监督

对训练视角 RGB 图像提取冻结 RADIO-compatible foundation feature，并引入 SigLIP2、DINO、SAM3 adaptor 等冻结或轻量下游 head 作为多头一致性监督。这里的基础模型只提供监督和评价，不在推理时变成额外 per-scene cache。

### 6.2 紧凑高斯特征场

在每个 Gaussian 上学习 compact latent code，同时引入空间上下文分支、visibility / quality 估计和几何一致性约束。相比直接存储 1280D RADIO feature，compact code 更节省存储，也能通过多视角训练获得更稳定的场景级特征。这个“又小又好”的证据应作为主线之一重点强调：本文不是为了压缩而牺牲质量，而是在存储占用显著降低的同时提升 selected downstream usability。

### 6.3 Foundation-space reconstruction

通过 compact-to-feature decoder 将紧凑 code 解码回 RADIO-compatible feature space。这样文本相似度、DINO matching、SAM adaptor 等下游 head 可以继续使用冻结基础模型接口，而不需要重新定义任务特定类别空间。更准确地说，GaussFM 是一个**按需重建的特征记忆**：离线存储的是 compact latent memory，查询时才根据视角、点位置或 primitive 位置重建所需特征。

### 6.4 统一查询执行层

同一个 compact field 通过统一查询执行层服务三类采样域：

- 图像域：渲染并重建 dense feature map，用于 LERF 2D open-vocabulary grounding；
- primitive 域：在 Gaussian center 重建特征并形成对象支持，用于 LERF direct 3D object selection；
- 点云域：对 ScanNet points / vertices 查询紧凑场特征，用于跨数据集开放词汇点级理解。

训练阶段，Multiview Primitive Registration 是方法内部的多视角监督模块，用于把跨视角一致的对象证据压回 compact field；推理阶段则按查询域选择相应的采样和投影方式。汇报中只需说明注册证据已经被吸收到 compact field 中。

## 七、论文贡献组织

建议论文主文贡献控制在 **3 个**，最多不超过 4 个。当前工作内部模块较多，如果贡献写 5 个以上，审稿人容易觉得方法是模块堆叠；如果压缩成 3 个，主线会更像一个完整思想。建议采用下面三条贡献：

### 贡献一：紧凑、按需重建的 foundation-feature Gaussian memory

区别于逐帧 feature cache 或每 Gaussian 高维显式存储，本文学习一个 compact foundation-feature Gaussian map。它既是三维场景表示，也是基础特征空间的**隐式可重建记忆**：存储紧凑隐变量，查询时通过解码和渲染 / 采样算子重建任务所需特征。这一贡献同时覆盖“高维特征压缩”和“按需特征重建”两个核心点。

### 贡献二：面向开放词汇三维理解的统一查询执行层

本文不是只做 rendered 2D heatmap，也不是为每个 benchmark 单独设计出口，而是将同一 compact memory 接入图像域、primitive 域和点云域三种查询场景。Multiview Primitive Registration、prompt ensemble、score component guard、label-free color-edge support calibration 等模块，都应内化为统一查询执行层：它们共同完成按需特征重建、文本匹配、空间聚合和对象支持形成。

### 贡献三：完整的多协议实验闭环，证明“又小又好”

本文在 LERF rendered-view OVS、LERF direct 3D object selection、ScanNet VALA-aligned ScanNet-8 point-query 三类协议上与公开 SOTA 结果对比，同时提供原始逐帧 RADIO 特征 vs. 重建场景特征、多任务 frozen-head usability、核心模块消融、storage / efficiency 和定性分析。实验目标不是只证明某个指标高，而是证明 compact memory 在存储占用和下游质量上同时成立。

如果导师希望贡献更细，也可以拆成 4 条：把“重建场景特征强于原始 RADIO + storage/efficiency”单独作为第四条实验贡献。但正式论文主文建议优先采用 3 条，叙事更凝练。

## 八、全文实验设计与结果

### 8.1 LERF rendered-view open-vocabulary grounding

任务：在 LERF-OVS 场景中渲染新视角 dense feature map，通过文本查询得到 heatmap 和 mask。该实验用于证明按需重建的图像域 foundation feature 可以支持二维开放词汇定位。

主定量对比：

| Method | Figurines | Ramen | Teatime | Waldo | Macro LocAcc |
| --- | ---: | ---: | ---: | ---: | ---: |
| LERF | 0.795 | 0.625 | **0.938** | 0.815 | 0.793 |
| LangSplat | 0.804 | 0.732 | 0.881 | **0.955** | 0.843 |
| LEGaussians | 0.767 | 0.737 | 0.683 | 0.523 | 0.678 |
| GaussFM | **0.821** | **0.901** | 0.898 | 0.818 | **0.860** |

同时，GaussFM 的主图像域 mask 结果达到 0.8598 LocAcc / 0.5889 mIoU。汇报时要把这张表作为主结果，而不是把原始 RADIO 对比表当成主结果；原始 RADIO 对比更适合放在“重建特征质量分析”一节。

### 8.2 LERF direct 3D object selection

任务：直接在 Gaussian primitive 上根据文本查询选择目标，再把选中的 primitives 渲染到标注视角计算 mIoU 和 Acc@0.25。

主定量对比：

| Method | mIoU | Acc@0.25 |
| --- | ---: | ---: |
| OpenGaussian | 38.36 | 51.43 |
| Dr. Splat | 43.29 | 64.30 |
| InstanceGaussian | 45.30 | 58.44 |
| GaussFM + MPR diagnostic | 48.01 | 67.60 |
| GaussFM compact support-calibrated query | **50.14** | **70.44** |
| GaussFM + official SAM3 box control | 57.05 | 68.35 |

结论：GaussFM compact support-calibrated query 在不使用 official SAM3 decoder 作为主结果的情况下，达到 50.14 mIoU / 70.44 Acc@0.25。官方 SAM3 box control 可以作为边界辅助上限分析，不应作为核心方法主结果。

### 8.3 ScanNet VALA-aligned ScanNet-8 direct point-query

任务：在 ScanNet VALA-aligned ScanNet-8 场景上做开放词汇点级查询，验证 compact field 的跨数据集 3D usability。

主定量对比：

| Method | 19 mIoU | 19 mAcc | 15 mIoU | 15 mAcc | 10 mIoU | 10 mAcc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LangSplat | 2.45 | 8.59 | 3.45 | 13.21 | 6.48 | 21.89 |
| LangSplatV2 | 14.75 | 25.47 | 17.09 | 35.68 | 22.83 | 41.52 |
| OpenGaussian | 27.73 | 42.01 | 29.67 | 46.15 | 39.93 | 57.34 |
| Dr. Splat | 29.31 | 47.68 | 33.25 | 54.33 | 44.19 | 65.19 |
| OccamLGS | 31.93 | 48.93 | 34.25 | 53.71 | 45.16 | 64.39 |
| VALA | 32.11 | 50.05 | 35.10 | 54.77 | 46.21 | 65.61 |
| GaussFM | **38.06** | **61.29** | **38.71** | **63.15** | **47.11** | **72.00** |

结论：在 VALA-aligned 8-scene point-query protocol 下，GaussFM 在 19/15/10 class split 上均取得最优结果。汇报和论文中可以写“we compare with reported results under the same VALA-aligned protocol”，无需把这张表讲成不可靠的外部协议；但数据来源仍应在 caption 或脚注中标明为 prior-paper reported numbers。

### 8.4 重建场景特征 vs. 原始逐帧 RADIO 特征

这一节建议单独作为全文实验的一节，**不要混进消融实验**。原因是它回答的是“本文学到的 compact reconstructed feature 是否真的比原始基础特征更有用”，属于核心 claim 验证，而不是某个模块是否有效。

| Task | Metric | 原始逐帧 RADIO | GaussFM | Δ |
| --- | --- | ---: | ---: | ---: |
| LERF text grounding | mIoU | 0.4634 | 0.5707 | +0.1073 |
| SAM3 point prompt | mIoU | 0.3700 | 0.4173 | +0.0473 |
| SAM3 box prompt | mIoU | 0.6560 | 0.6638 | +0.0079 |
| SAM3 mask propagation | mIoU | 0.3583 | 0.3756 | +0.0173 |
| DINOv3 dense matching | Mean score | 0.8547 | 0.9048 | +0.0501 |
| DINOv3 mask propagation | mIoU | 0.4606 | 0.4677 | +0.0071 |

结论：这一节支撑“重建场景特征不是低维退化版 RADIO，而是在多视角三维聚合后对 selected downstream tasks 更有用”。它和 ablation 的区别是：这里比较的是**原始基础特征 vs. 本文重建特征**；ablation 比较的是**本文内部模块开关**。

### 8.5 Storage / efficiency 定量分析

这一节建议放在主文实验部分，不要只放补充材料。它直接支撑“compact”这个关键词。

当前 storage 表采用的是非常保守的 **full deployable checkpoint** 口径：不仅统计每个 Gaussian 的 compact latent，也把 checkpoint 中携带的 3DGS geometry/RGB tensors、hash field、CTR/HCD codec、VFA/refiner 等都计入 compact footprint。因此 saving 只有 1.74x--4.04x，看起来不如“低维 latent”直观。

正式论文建议把 storage 拆成三种口径：

1. **Semantic latent payload**：只统计每 Gaussian 的 compact latent。当前 latent 大约是 64D fp16，因此相对 1280D fp16 是约 20x 压缩。
2. **Feature-memory package**：latent + global context/hash + decoder/head。这个口径更接近方法新增的 feature memory，占用比 latent-only 大，但仍能说明按需重建的部署成本。
3. **Full checkpoint footprint**：当前表格口径，包含 carried geometry/RGB、codec、refiner，是最保守但横向不一定最有竞争力的口径。

推荐主文表格不要只放 full checkpoint，否则会削弱 compact claim；最好放 `Latent payload`, `Feature-memory package`, `Full checkpoint`, `Direct 1280D feature` 四列。这样读者能看到：compact latent 本身确实很小，而 full checkpoint saving 不夸张是因为我们采用了保守部署统计。

| Scene | #G | Direct 1280D feature | Compact ckpt | Saving |
| --- | ---: | ---: | ---: | ---: |
| Figurines | 168,791 | 412.1 MiB | 237.0 MiB | 1.74x |
| Ramen | 382,687 | 934.3 MiB | 311.2 MiB | 3.00x |
| Teatime | 460,157 | 1123.4 MiB | 338.1 MiB | 3.32x |
| Waldo Kitchen | 691,728 | 1688.8 MiB | 418.5 MiB | 4.04x |

效率证据可以更简洁地放一张小表：

| Evidence | Scope | Cost | VRAM |
| --- | --- | ---: | ---: |
| LERF eval | 4 scenes | 124.77 s / 31.19 s-scene | 2076 MiB |
| ScanNet eval | 10-scene superset | 150.90 s / 15.09 s-scene | 1666 MiB |
| Storage | 4 scenes | 1.74x--4.04x saving | -- |

结论：本文不是简单把特征存在 3DGS 上，而是用 compact latent memory 实现“更小 + 更好”。

此外，当前 efficiency 表是**整场景评估吞吐**，不是交互式单次 query latency。LERF profile 包含 teacher/rendered 双路评价、多个 annotated views、所有 text queries 和保存可视化，因此不能直接代表 single-query realtime。投稿前更建议补一个小的 latency 表：

| Query type | Reported unit | 建议统计方式 |
| --- | --- | --- |
| LERF 2D query | ms / view-query | warm GPU，固定一帧一条文本，排除 disk IO 和可视化保存 |
| LERF direct 3D query | ms / text query | primitive scoring + support calibration + selected-primitives render |
| ScanNet point query | ms / class query 或 points/s | 固定点数，报告 kNN query + class scoring |
| Batched queries | queries/s | 同一 scene 多文本批量评分，展示实际评估吞吐 |

这张 latency 表会比“整场景评估总耗时”更有说服力。

## 九、消融实验与机制解释

消融实验建议独立成一节，专门回答“贡献来自哪里”。不要把 SOTA 对比、原始 RADIO 对比、storage/efficiency 混进消融，否则主线会显得散。

### 9.1 核心架构消融

消融显示，仅用简单线性投影或去掉 foundation-space reconstruction 会明显损害 LocAcc 和 mIoU。说明本文不是把 RADIO feature 简单降维，而是通过上下文高斯场和非线性解码器学习可按需重建的基础特征。

建议主文只放一张精简表：

| Variant | Macro LocAcc | Macro mIoU | 说明 |
| --- | ---: | ---: | --- |
| Full GaussFM | 0.858 | 0.485 | 完整架构 |
| w/o FGC warm-start | 0.802 | 0.424 | 几何/基础特征 warm-start 重要 |
| w/o VFA | 0.840 | 0.480 | 视角校准影响定位 |
| w/o HGCF | 0.839 | 0.507 | region overlap 有时上升，但 peak stability 下降 |
| w/o CTR | 0.531 | 0.260 | foundation-space reconstruction codec 最关键 |

### 9.2 查询执行层 / support calibration 消融

Direct primitive score 的高低并不等价于最终 object mask 好坏。小物体、细长物体、多实例和低可见目标容易出现 zero prediction 或 fragmented support。support-calibrated query module 把 feature similarity 转化成稳定 object support，是 Acc@0.25 和 Waldo small-object recovery 的关键。

| Variant | mIoU | Acc@0.25 | 说明 |
| --- | ---: | ---: | --- |
| Compact single prompt, pure one-map | 0.449 | 0.672 | 最严格 compact-only |
| Compact prompt ensemble, pure one-map | 0.457 | 0.685 | 文本查询更稳 |
| Compact prompt ensemble + RGB component guard | 0.500 | 0.705 | 小物体 / 碎片支持改善 |
| Compact prompt ensemble + RGB/score-component guard | 0.501 | 0.704 | 主结果 |

### 9.3 Multiview Primitive Registration 在方法中起什么作用？

Multiview Primitive Registration 提供多视角注册证据，是方法内部训练模块，有助于 compact field 学到稳定 primitive support。汇报时不需要围绕 cache 展开，只要说明：注册证据在训练阶段被压回 compact latent memory，最终查询仍从 compact field 出发。

这一部分可以在消融总结中讲，不必作为主文大表单独展开。MPR 的作用是提供跨视角一致对象证据，并通过训练内化到 compact latent memory 中。汇报时避免把它讲成“推理时另一个特征库”，否则会削弱 compact memory 的主线。

## 十、定性结果安排

定性图的原则是：每张图都必须服务一条主贡献，不能只是“看起来效果还可以”。建议主文保留 3 张主图 + 1 张消融图，更多样例放 appendix。

### 主文图一：整体框架图

使用 `paper/figures/radio_gs_framework.pdf`。图中按四列组织：

1. Offline supervision；
2. Stored compact map；
3. Unified query execution layer；
4. Protocol evidence。

汇报时重点说明：本文不是为三个任务拼三个模型，而是一个 compact foundation-feature Gaussian map 加统一查询执行层；2D、3D、point-query 只是同一特征记忆在不同采样域上的使用方式。

### 是否需要额外模块图？

需要，但不建议画太多。除了整体框架图，最值得补一张 **“按需特征重建与查询执行”机制图**，可以作为 Figure 2 或 supplementary Figure：

```text
compact latent Gaussian memory
        |
        | sampling / rasterization / point query
        v
RADIO-compatible feature reconstruction
        |
text / SAM / DINO compatible scoring
        |
geometry-aware aggregation + support calibration
        |
2D mask / 3D selected primitives / point labels
```

这张图能解决两个潜在误解：

1. 我们不是离线完整存储高维特征，而是查询时按需重建；
2. 2D、3D、point-query 不是三个临时出口，而是同一查询执行层在不同采样域上的实例。

如果版面紧张，这张模块图可以合并到框架图中，用一个放大的 inset 展示。不要再为每个小模块单独画图，否则会强化“模块堆叠”的印象。

### 主文图二：LERF 2D and 3D Open-Vocabulary Query

使用 `paper/figures/lerf_2d3d_ovs_qualitative.png`。每个 scene 展示 2D rendered query 和 3D direct query 的区别。

讲法：

- 2D OVS：在图像域按需重建 feature map，输出 heatmap / mask；
- 3D OVS：在 primitive 域按需重建 feature 并形成对象支持，渲染 selected primitives 评价 mask；
- 两者评价流程不同，但都由同一个 compact latent memory 支持。

现有图需要重做，而不是简单微调。主要问题：

1. 缺少每个 query 的明确 GT mask/outline，读者很难判断 2D/3D mask 是否正确；
2. 2D prior 与 ours 的可视化风格不统一，一个偏 mask，一个偏 heatmap；
3. 3D OVS 的 zoom box 线条和裁剪区域不统一，对比方法一侧尤其不清楚；
4. 白底 3D cut-out 适合展示 selected primitives，但需要和 GT crop 对齐。

建议新版排版：

```text
RGB+query | GT | Prior 2D mask | Ours 2D mask | Prior 3D selection | Ours 3D selection
```

或者每个 query 两行：

```text
2D: RGB+query | GT | prior heatmap/mask | ours heatmap/mask
3D: RGB+query | GT | prior selected primitives | ours selected primitives
```

颜色规范必须统一：GT 用绿色轮廓，prior 用蓝色，ours 用橙色；如果使用 heatmap，prior/ours 都用同一 colormap 和同一 alpha。

### 主文图三：ScanNet Open-Vocabulary 3D Query

使用 `paper/figures/scannet_openvocab_3d_query_qualitative.png`。主文采用 binary query point cloud，更容易展示某一开放词汇类别是否被找出。全类别彩色点云更适合放 appendix。

对比方法最好用“可获得输出中最强的新方法”，而不是默认 OpenGaussian。当前 OpenGaussian 可视化虽然清楚，但方法偏早且部分样例差距过大，容易被认为 cherry-pick。优先级建议：

1. 如果能拿到 VALA / OccamLGS / Dr. Splat 的同协议点云预测，主文用其中一个作为 prior；
2. 如果拿不到，则保留本地 OpenGaussian reproduction，但 caption 写清楚是 reproduced baseline，并在 appendix 增加更多非极端样例；
3. 主文样例可以选择我们表现好、baseline 有可解释失败的 case，但不要只挑 baseline 完全崩的图。最好的 qualitative 是“baseline 找到一部分，但 ours 更完整/更干净”。

### 主文图四或 appendix 首图：Direct3D support calibration ablation

使用 `paper/figures/lerf_direct3d_support_policy_ablation_qualitative.png`。展示 base compact score 与 final support-calibrated query 在 small / fragmented object 上的差别，用来解释 direct 3D 指标提升来自哪里。

这张消融定性最有必要，因为它直接对应贡献二中的“对象支持形成”。建议选 Waldo / small object / fragmented support 示例，列为：

```text
RGB + query | GT | compact score only | + prompt ensemble | + support calibration | final overlay
```

重点展示：base compact field 已能找到语义区域，但 support calibration 让 mask 从碎片化/漏检变成更完整对象。

现有图的问题是背景太亮、mask 不突出、标签框遮挡较多、右侧 3D support zoom 比例不统一。重做时建议：

- RGB 背景降亮度到 35%--45%；
- GT/prior/ours mask 用高饱和实色加 2--3 px 粗轮廓；
- 每行固定同一个 GT-centered crop，避免不同列 zoom 到不同位置；
- 右侧 3D support 用同一 crop 尺寸，去掉复杂斜线连接，只保留 inset 边框；
- 文字只保留 query 和 IoU，减少画面遮挡。

### 其他消融定性是否需要？

建议最多再放一张到 appendix，不进主文：

**Rendered feature boundary calibration qualitative**  
内容：threshold heatmap -> peak component -> feature-only SAM-adaptor boundary -> GT。  
作用：解释 mIoU 为什么从 heatmap threshold 到 peak component、再到 feature-only boundary 有提升。  
是否进主文：建议可以放主文半页或 appendix 首图，因为它对应图像域对象边界质量，能支撑“统一查询执行层中的对象支持形成”。

**Reconstructed feature vs. frame-wise RADIO qualitative**  
这一张也建议加入，至少放 appendix，如果版面允许可以做成主文小图。它对应贡献三中的“重建特征不是低维退化，而是 selected downstream tasks 更好”。建议做成 2--3 行小图：

```text
Task | RGB/GT | frame-wise RADIO | GaussFM reconstructed feature
LERF text grounding | mask/heatmap | teacher heatmap/mask | ours heatmap/mask
SAM3 adaptor prompt | GT | teacher mask | ours mask
DINO dense matching | source-target pairs | teacher matches | ours matches
```

DINO 可以保留少量错误匹配，但要选择 ours 正确匹配明显更多、响应更集中的样例。不要只放热力图，最好用 correspondence lines 或 matched points，让“正确匹配更多”直观可见。

不建议主文放：

- DINO matching 可视化：除非 topology adaptor 指标非常强，否则容易暴露匹配错误，适合 appendix diagnostic。
- official SAM3 box 可视化：容易让审稿人误解主方法依赖官方 RGB SAM decoder。
- VPR / registration 过程大图：会把注意力从 compact latent memory 拉回 cache / registration。
- 全类别 ScanNet 彩色点云：视觉复杂，主文不如 binary query 清楚。

### Rendered boundary calibration 和 Direct3D support calibration 是否互相提升？

概念上，它们都属于“对象支持形成”：前者在图像域把 heatmap/coarse mask 变成更好的 2D boundary，后者在 primitive 域把 similarity score 变成更完整的 3D object support。因此可以在方法章节里统一叙述。

但从实验证据上，不能默认说二者已经互相提升。需要区分两种情况：

1. **如果只是 evaluation-time calibration / boundary head**：它主要改善对应任务的输出，不一定改变 compact latent field，因此不能声称会间接提升另一个任务。
2. **如果作为训练 loss 反向优化 shared compact field**：理论上可以让 2D boundary supervision 改善 direct 3D feature support，也可以让 direct 3D support distillation 改善 rendered heatmap stability。但这需要 cross-task ablation 证明。

建议补一个小型交叉消融，如果时间允许：

| Training / calibration | LERF 2D mIoU | Direct3D mIoU | 结论 |
| --- | ---: | ---: | --- |
| base compact field | -- | -- | baseline |
| + rendered boundary supervision only | -- | -- | 看是否提升 Direct3D |
| + Direct3D support supervision only | -- | -- | 看是否提升 rendered 2D |
| + both | -- | -- | 验证是否互补 |

如果没有这张表，论文里就写“both are unified as object-support formation modules, evaluated in their corresponding protocols”，不要写“互相提升”。

## 十一、风险边界与应对策略

### 风险一：公开对比数字的来源表述

应对：如果对比数字来自 VALA-aligned 或相关论文中同一 benchmark protocol 的表格，主文可以写“reported under the same benchmark protocol”，不要在汇报里主动弱化成“协议不完全同源”。但论文 caption 仍应标明数据来源，例如“prior reported results under the VALA-aligned protocol; unpublished protocol-source row omitted”。这比“external baseline protocol 不可靠”更适合投稿叙事，也不会误导为我们本地复现了所有方法。

### 风险二：RGB/GrabCut support calibration 可能被质疑为后处理

应对：明确它是 label-free classical color-edge support calibration，不调用 learned RGB segmentation network，也不调用 official RGB SAM decoder。主表同时保留 strict pure one-map ablation，说明 compact field 本身已具备 direct 3D ability；support calibration 用于提升 object support 和边界稳定性。

### 风险三：official SAM3 box row 可能混淆主 claim

应对：把 official SAM3 box row 定位为 boundary-assisted control / upper-bound analysis，不作为主 compact direct-field claim。

### 风险四：Waldo small objects 仍是 failure mode

应对：主文 failure analysis 主动说明 small / fragmented / low-visibility objects 是最困难情况，并用 support-calibrated query module 的定性和定量改善展示本文已针对该问题做出改进。

### 风险五：DINO mask propagation 不宜过度宣称

应对：避免写 universal superiority over frame-wise RADIO。表述为 selected downstream tasks improve，DINO topology-sensitive propagation 作为分析项或 caveat。

## 十二、当前投稿完成度

从技术内容角度，项目已经形成完整投稿包：

- TPAMI-style main paper 已编译；
- supplementary material 已编译；
- 框架图、主定性图、ScanNet 定性图、消融定性图已整理；
- 三条主 benchmark 结果已进入 `final_rows.yaml`；
- storage、efficiency、ablation、failure analysis 已有主文或补充材料表格；
- claim validator、final rows registry validator、checksum 和 artifact manifest 已通过；
- cover letter draft、submission checklist、submission mode guide 已准备。

尚需导师和作者团队确认的内容：

1. 是否正式以 TPAMI 为目标期刊；
2. 投稿是否匿名；
3. 作者顺序、单位、通讯作者、ORCID；
4. funding、COI、acknowledgement、AI/tool disclosure；
5. 是否接受 external published-context comparison 的当前口径；
6. 是否需要在投稿前再补一个严格 same-evaluator baseline 作为保险。

## 十三、下一阶段计划

### 近期计划：投稿前 polish

1. 导师确认核心 claim 和结果口径；
2. 根据导师意见压缩或调整主文表格；
3. 对 abstract、introduction、discussion、limitations 做最终人工精修；
4. 填写作者和投稿系统所需信息；
5. 最终编译匿名或非匿名版本。

### 中期计划：进一步增强审稿防御

1. 补更多 same-evaluator baseline 或复现实验，如果导师认为必要；
2. 扩展 appendix failure analysis；
3. 完善 reproducibility package 和 large asset release；
4. 对 DINO topology-sensitive task 做进一步方法探索，但不阻塞当前投稿。

### 长期计划：扩展为更完整的三维基础模型场景记忆方向

1. 从 3DGS 扩展到 mesh / point cloud / neural field；
2. 引入更多 foundation heads 和跨任务统一查询执行层；
3. 支持动态场景和机器人交互式查询；
4. 形成开放词汇三维场景记忆 benchmark 和工具链。

## 十四、建议汇报 PPT 结构

建议控制在 15 页左右。

1. 标题页：GaussFM 与一句话 claim；
2. 背景：二维 foundation features 很强，但三维部署缺紧凑 scene memory；
3. 痛点：feature cache 昂贵、rendered-only 不够、direct 3D support 不稳；
4. 核心科学问题：如何把 foundation feature 压缩成可查询三维场；
5. 方法总览图：one compact latent memory + unified query execution；
6. 模块一：compact Gaussian feature field；
7. 模块二：foundation-space reconstruction 与 multi-head consistency；
8. 模块三：support-calibrated query module；
9. 实验组织总表：SOTA 对比、重建特征质量、消融、存储效率；
10. 主结果一：LERF rendered-view；
11. 主结果二：LERF direct 3D；
12. 主结果三：ScanNet VALA8；
13. 重建特征质量 + storage/efficiency：证明“又小又好”；
14. 定性结果与消融可视化：2D/3D OVS、ScanNet、support calibration；
15. 投稿状态、风险边界、请导师决策的问题。

## 十五、建议汇报开场稿

这次汇报我希望按投稿项目而不是普通实验进展来讲。本文的核心目标是构建一个紧凑的 foundation-feature Gaussian latent memory：它不显式存储每个 Gaussian 的完整高维 RADIO 特征，而是在查询时按需重建 RADIO-compatible features，并通过统一查询执行层服务 LERF rendered-view 2D open-vocabulary query、LERF direct 3D primitive query 和 ScanNet point-level query。

目前我们已经把论文整理成 TPAMI 方向的完整投稿包，主文、补充材料、图表、三条 benchmark、消融、存储效率、failure analysis 和 artifact provenance 都已经闭环。今天主要想请老师判断三件事：第一，本文的主线和 claim 是否足够清楚；第二，当前实验和风险边界是否支撑顶刊投稿；第三，是否可以进入正式投稿流程，以及是否还需要补充某些 baseline 或协议验证。

## 十六、导师可能追问与建议回答

### 问：本文真正的新意是什么？

答：新意不是简单把 RADIO 特征蒸馏进 3DGS，也不是单纯声称支持 2D/3D 查询。更准确地说，本文提出 compact foundation-feature Gaussian latent memory：显式存储低维隐变量，查询时按需重建 RADIO-compatible features，并通过统一查询执行层完成文本匹配、空间聚合和对象支持形成。因此它同时解决存储压缩、foundation-space reconstruction、多协议评价和小物体/碎片化 support calibration。

### 问：为什么不是只投会议？

答：目前工作量已经超过单一 benchmark 方法。论文包含完整方法体系、三类协议、原始逐帧 RADIO 与重建场景特征对比、storage/efficiency、failure analysis、qualitative、supplement 和 reproducibility package，更适合按 TPAMI 期刊论文组织。

### 问：direct 3D 主结果是否依赖额外模型？

答：主 compact support-calibrated query 使用 compact primitive scores、prompt ensemble 和 label-free color-edge support calibration，不调用 official RGB SAM decoder。Multiview Primitive Registration 是训练时吸收多视角证据的内部模块；official SAM3 box 是 control row，不作为主 claim。

### 问：重建场景特征是否全面强于原始 RADIO？

答：不能无边界宣称全面强于原始 RADIO。准确表述是：在 selected frozen-head downstream tasks 上，reconstructed scene field 超过原始逐帧 RADIO features；DINO topology-sensitive mask propagation 等任务仍保留为 caveat。

### 问：对比方法数字如何组织，是否能直接作为 SOTA 对比？

答：主文应该按 benchmark protocol 组织 SOTA 对比表。对于 VALA-aligned 或相关论文中已经按同一 protocol 报告的结果，可以作为 prior reported results under the same benchmark protocol 放入主表；caption 标明来源即可，不需要在汇报中主动弱化为“协议不一致”。但也不要写成我们本地完整复现了所有方法，除非确实本地复现。

### 问：最大风险是什么？

答：最大风险有三个：公开对比数字来源表述、RGB edge calibration 的措辞边界、small/fragmented object failure。论文已通过 protocol provenance、strict ablation、failure analysis 和 support-calibrated query module 对这些风险做了防御。

## 十七、汇报时的核心把控

汇报时不要把重心放在“我试了很多模块”，而要始终围绕一条主线：

> 二维基础模型特征很强，但不能直接成为可部署三维场景记忆；GaussFM 把高维 foundation features 编码成 compact Gaussian latent memory，并在查询时按需重建特征，使同一个三维 map 支持二维、三维和点级开放词汇查询。

所有实验都服务于这条逻辑：

- LERF 2D 证明图像域按需特征重建有效；
- LERF direct 3D 证明 primitive 域按需特征重建和对象支持形成有效；
- ScanNet 证明跨数据集 point-query usability；
- 原始逐帧 RADIO vs. 重建场景特征证明 compact field 不只是压缩，还能提升 selected downstream usability；
- Storage/efficiency 证明不是靠显式高维 memory；
- Failure/provenance 证明投稿口径可信。
