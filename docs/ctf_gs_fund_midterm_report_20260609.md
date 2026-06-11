# CTF-GS 项目基金申请书式中期汇报材料

日期：2026-06-09  
项目名称建议：**面向开放词汇三维场景理解的紧凑基础特征高斯场研究**  
论文名称建议：**CTF-GS: Compact Foundation-Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding**

## 一、汇报定位

本材料面向导师汇报、课题中期检查或基金本子雏形。写作目标不是简单罗列实验，而是把项目组织成一条完整的科研逻辑链：

> **二维视觉基础模型已经具备强大的语义、匹配和分割能力，但这些能力主要存在于离散图像帧中。真实三维场景需要的是一个可部署、可查询、可压缩的三维基础特征记忆。本文研究如何将帧级基础特征重建到 3D Gaussian 场景中，使同一个紧凑场同时支持二维新视角查询、三维 primitive 查询和跨数据集点级查询。**

汇报时应避免把重点放在“我们做了很多模块”。更清楚的讲法是：

1. 先讲**为什么需要三维基础特征记忆**；
2. 再讲**现有开放词汇 3DGS 方法为什么还不够**；
3. 然后讲**本文提出一个紧凑、可查询、多读出的 foundation-feature Gaussian field**；
4. 最后用**三类协议和消融实验**证明主张成立。

## 二、项目摘要

近年来，以 RADIO、DINO、SAM、SigLIP 等为代表的视觉基础模型在二维图像理解中表现突出，能够提供密集语义、区域边界、跨视角匹配和开放词汇对齐能力。然而，现有三维场景表示大多仍围绕 RGB 重建、几何可视化或单一语义标签优化展开，难以直接承载高维基础模型特征。若直接在每个 3D Gaussian 上存储原始高维特征，会带来显著存储开销；若只训练特定任务头，又会损失基础模型的泛化能力和多任务可用性。

本项目提出 **CTF-GS：紧凑基础特征高斯场**。其核心思想是：不直接保存每帧或每个 Gaussian 的完整基础特征，而是在 3D Gaussian 场景中学习一个紧凑的 foundation-feature field，通过全局解码与校准模块重建 RADIO-compatible 特征，并提供多个下游读出接口。该表示同时支持：

- **二维新视角开放词汇定位**：渲染 dense feature map，在像素层面进行文本查询；
- **三维直接对象选择**：在 Gaussian primitive 层面计算文本相似度，选择三维对象支持区域；
- **跨数据集点级语义查询**：在 ScanNet 点云上进行 direct point-query open-vocabulary evaluation；
- **冻结头下游能力验证**：在 SigLIP2、SAM3 adaptor、DINOv3 等 frozen-head tasks 中比较重建场与帧级 teacher 特征的可用性。

当前项目已形成 TPAMI 方向投稿包。主文、补充材料、框架图、定量表格、定性图、provenance、checksum 和 submission checklist 均已整理。实验结果显示，CTF-GS 在 LERF-OVS rendered-view grounding、LERF direct 3D object selection 和 ScanNet VALA8 direct point-query 三个协议上形成了完整证据链。

## 三、立项依据与研究背景

### 3.1 现实需求：三维场景不能只依赖二维帧级特征

二维视觉基础模型的能力已经非常强，但这些能力天然是 **view-local** 的。一个室内场景如果只保留训练图像或最近邻图像特征，面对新视角、遮挡、视角变化和三维对象选择时会出现三个问题：

1. **查询不连续**：每个视角特征独立，不能保证新视角语义响应稳定；
2. **存储不经济**：直接缓存每帧高维特征或每个 Gaussian 的 1280-D feature 会使场景记忆膨胀；
3. **三维不可查询**：二维 heatmap 可以定位图像区域，但无法自然支持 primitive-level object selection 或 point-level semantic query。

因此，项目的核心需求不是再训练一个二维分割器，而是构建一个可部署的三维场景记忆：

> **将二维基础模型的高维知识压缩、重建并绑定到三维高斯场景中，使三维场景本身具备开放词汇、多任务、可查询的基础特征表达能力。**

### 3.2 学术背景：开放词汇 3DGS 正从二维热力图走向三维 primitive 查询

已有 LERF、LangSplat、LEGaussians 等工作验证了在三维场景中嵌入语言特征并进行 rendered-view open-vocabulary querying 的可行性。这类方法的典型输出是二维渲染视角上的 heatmap 或 mask。

近两年 OpenGaussian、Dr. Splat、InstanceGaussian、OpenGaFF 等工作进一步强调：开放词汇三维理解不应只停留在二维 rendered pixel-level parsing，还应支持 **3D Gaussian / point / instance-level querying**。也就是说，查询对象应从“渲染图像上的像素”进一步移动到“三维场景中的 primitive 或 object support”。

本项目正处于这条研究主线中，但关注点有所不同：

- 不是仅在 Gaussian 上绑定一个低维 language code；
- 不是只做 rendered feature heatmap；
- 不是依赖显式 feature cache 作为推理主接口；
- 而是学习一个紧凑的 **foundation-feature scene memory**，用同一个三维表示支持多类读出。

## 四、拟解决的关键科学问题

项目围绕三个相互关联的问题展开。

### 问题一：如何把帧级基础模型特征转化为紧凑三维场景记忆？

二维 foundation features 维度高、空间密集、视角相关。如果直接存储，会使 3DGS 表示失去紧凑性；如果过度压缩，又会损失开放词汇和下游 frozen-head 可用性。

需要解决的问题是：

> 如何设计一种紧凑的 Gaussian feature field，使其既能保存多视角基础特征中的稳定语义，又能在渲染和直接查询时恢复 teacher-compatible feature？

### 问题二：同一个三维特征场能否同时支持二维和三维查询？

开放词汇 3DGS 的评估协议至少包含两类不同查询：

- rendered-view protocol：先渲染 feature map，再在图像平面上做文本相似度；
- direct-3D protocol：先在三维 Gaussian primitive 上做文本查询，再渲染 selected primitives 用于评估。

二者流程不同，证明的能力也不同。项目需要回答：

> 一个 compact foundation-feature field 能否同时支持 rendered-view readout 和 direct primitive readout，而不是为每个任务单独训练一个模块？

### 问题三：重建后的三维特征是否只是 teacher 的近似复制，还是能提升下游可用性？

如果 CTF-GS 只是低维压缩 RADIO 特征，那么贡献会偏工程压缩。更强的主张是：多视角重建、可见性约束和支持区域校准可以削弱单帧噪声，使 reconstructed scene field 在某些 frozen-head downstream tasks 中优于 frame-wise foundation features。

需要验证：

> 在相同 frozen evaluator 下，重建的三维场景特征是否能在开放词汇定位、区域分割、DINO matching/propagation 等任务中表现出更好的下游可用性？

## 五、总体目标与研究内容

### 5.1 总体目标

构建一个面向开放词汇三维场景理解的紧凑基础特征高斯场，使三维 Gaussian 场景从 RGB/geometry representation 升级为可查询的 foundation-feature scene memory，并形成面向顶刊投稿的完整方法、实验和理论叙事。

总体目标可拆为四点：

1. **紧凑表示**：避免直接存储每个 Gaussian 的原始高维 foundation feature；
2. **特征重建**：从 compact codes 重建 RADIO-compatible foundation-space features；
3. **多协议读出**：支持 rendered-view、direct primitive、direct point-query 三种查询接口；
4. **证据闭环**：通过 LERF、ScanNet、teacher-vs-field、storage/efficiency、ablation、qualitative 和 failure analysis 形成投稿级证据链。

### 5.2 研究内容一：Contextual Gaussian Feature Field

该部分解决“基础特征如何存”的问题。

项目在已有 3D Gaussian geometry 上附加紧凑 latent code，并引入空间上下文分支，将 per-Gaussian compact code 与 voxel/spatial context 融合，得到 **Contextual Gaussian Feature Field**。该 field 不只输出 feature code，还预测 feature quality 和 visibility，使模型能够识别哪些区域的重建特征更可靠。

关键设计：

- per-Gaussian compact code：负责 primitive-level 局部特征承载；
- spatial context branch：补充场景上下文，缓解单 primitive 特征碎片化；
- reliability/visibility heads：提供特征可信度和可见性估计；
- geometry/semantic decoupled fusion：避免几何可见性和语义特征相互干扰。

### 5.3 研究内容二：Foundation-Space Reconstruction 与 View-Conditioned Feature Calibration

该部分解决“紧凑特征如何还原为可用基础特征”的问题。

模型通过 **Foundation-Space Reconstruction** 将 compact feature map 解码回 RADIO-compatible feature space，再通过 **View-Conditioned Feature Calibration** 利用 alpha、depth、visibility 等渲染信号进行视角条件校准。

这一设计的意义是：

- 保留 RADIO/SigLIP/DINO/SAM 等 frozen heads 的下游接口；
- 避免为每个任务训练 task-specific classifier；
- 让新视角渲染出的 feature map 能直接用于文本查询、mask readout 或 dense matching。

### 5.4 研究内容三：Multi-Head Foundation Consistency

该部分解决“基础特征不仅要像 teacher，还要保留下游结构”的问题。

仅用 feature cosine reconstruction 可能不足以保留 DINO 的 matching topology、SAM 的 region sensitivity 和 SigLIP 的 text alignment。因此项目引入 **Multi-Head Foundation Consistency**，利用冻结的 SigLIP2、DINOv3、SAM3 adaptor heads 对重建特征施加辅助约束。

其定位不是调用 official SAM3 RGB decoder 作为外部后处理，而是在训练或 task probe 中验证：

- 重建 field 是否保持 text-aligned querying；
- 是否保留 DINO-style affinity/correspondence；
- 是否能驱动 SAM-compatible adaptor-space boundary readout；
- 是否在 selected frozen-head tasks 中优于 frame-wise foundation features。

### 5.5 研究内容四：Multiview Primitive Registration 与 Support-Calibrated Primitive Readout

该部分解决“direct 3D object selection 如何稳定”的问题。

仅靠 Gaussian-center feature 与 text embedding 的相似度，容易出现小物体漏选、support 碎片化和边界不稳定。项目因此引入两个层面的设计：

1. **Multiview Primitive Registration**：将多视角 rendered foundation evidence 注册回可见 Gaussian primitive，用作训练桥梁和分析读出；
2. **Support-Calibrated Primitive Readout**：在 compact primitive scores 基础上，通过 prompt ensemble、score-component guard 和 label-free color-edge support calibration，将 primitive score 转化为稳定 object support。

这里要强调边界：

- MPR 是 training/analysis bridge，不是 deployed compact direct readout 的必需 cache；
- label-free color-edge support calibration 是经典支持区域校准，不是额外 learned RGB segmentation model；
- official SAM3 box readout 作为 boundary diagnostic，不承担本文核心 direct-field claim。

## 六、总体技术路线

整个项目可以用一条“从二维基础特征到三维可查询场景记忆”的流程讲清楚。

### 6.1 输入与监督

输入：

- 已训练的 3D Gaussian scene geometry；
- posed RGB training views；
- frozen RADIO / SigLIP2 / DINOv3 / SAM3 adaptor heads；
- LERF、ScanNet 等评估协议中的 query 和 annotation。

监督来源：

- RADIO dense feature reconstruction；
- frozen-head geometry consistency；
- multi-head adaptor consistency；
- multiview registration evidence；
- label-free reliability/visibility targets。

### 6.2 存储层

每个 scene 存储一个 compact foundation-feature Gaussian map，包括：

- compact per-Gaussian codes；
- spatial context branch；
- quality/visibility heads；
- global decoder/calibration heads。

不直接持久化每个 Gaussian 的完整 1280-D teacher feature，也不把 MPR feature cache 计入 compact scene memory。

### 6.3 读出层

同一个 compact field 支持三种读出：

1. **Rendered-view readout**：渲染 dense feature map，用于 LERF 2D open-vocabulary grounding；
2. **Direct primitive readout**：在 Gaussian primitive 上直接计算 text score，并通过 support calibration 得到 object support；
3. **Direct point-query readout**：在 ScanNet points/vertices 上查询 compact field feature，用于 open-vocabulary point-level evaluation。

### 6.4 评估层

实验分成四组：

- LERF-OVS rendered-view grounding；
- LERF direct 3D object selection；
- ScanNet VALA8 direct point-query transfer；
- frame-wise foundation features vs reconstructed scene field 的 frozen-head downstream comparison。

这种组织方式保证读者不会把二维 heatmap protocol、三维 primitive protocol 和点云 semantic query protocol 混在一起。

## 七、主要创新点

### 创新点一：从 teacher-feature compression 升级为 compact foundation-feature scene memory

本文不是简单压缩 teacher 特征，而是把二维基础模型能力转化为三维场景中的紧凑、可查询、可部署记忆。该记忆既能在新视角渲染出 dense foundation-space features，也能在三维 primitive/point 层面直接查询。

### 创新点二：Contextual Gaussian Feature Field 同时兼顾 per-primitive 细节和 scene-level context

与单纯 per-Gaussian feature embedding 不同，本文使用 compact code 与 spatial context 融合，缓解 primitive-level feature fragmentation，为 direct 3D selection 和 ScanNet point-query 提供更稳定的三维语义支持。

### 创新点三：多读出统一在同一个 compact field 上，而不是多个任务分支拼接

本文强调 **one compact foundation-feature Gaussian map supports multiple protocols**。Rendered 2D grounding、direct 3D primitive selection 和 ScanNet point-query 只是不同评估协议下的 readout，而不是三套互不相关的方法。

### 创新点四：MPR 作为训练桥梁，压缩多视角注册证据到 deployed compact field

Multiview Primitive Registration 借鉴 direct Gaussian registration 类工作的思想，将多视角 foundation evidence 注册到 Gaussian primitive 上，但本文进一步把该证据压缩回 compact direct field，避免把 feature cache 作为主推理接口。

### 创新点五：Support-Calibrated Primitive Readout 解决 direct 3D 中 score-to-object-support 的瓶颈

实验表明 direct 3D 的瓶颈不只是 feature cosine，而是从 primitive score 到 object support 的转换。本文通过 prompt ensemble、component guard 和 label-free color-edge support calibration 显著改善 small/fragmented object 的 Acc@0.25 和 mask support。

## 八、阶段性进展与实验结果

### 8.1 论文与投稿包状态

当前项目已整理成 TPAMI 方向投稿包：

- 主文：`paper/radio_gs_tpami.tex`，已编译为 11 页 PDF；
- 补充材料：`paper/radio_gs_tpami_supplement.tex`，已编译为 9 页 PDF；
- 框架图：`paper/figures/radio_gs_framework.pdf`；
- 主定性图：LERF 2D/3D OVS、ScanNet binary query、direct3D support calibration ablation；
- 结果表：LERF 2D、LERF direct 3D、ScanNet VALA8、teacher-vs-field、ablation、storage、efficiency；
- provenance：`paper/artifacts/final_rows.yaml`、`checksums.txt`、`paper_assets_manifest.json`、claim validator。

### 8.2 LERF-OVS rendered-view grounding

任务目标：验证 compact field 是否能在 novel view 渲染出可用于开放词汇定位的 dense feature map。

主结果：

| 指标 | CTF-GS |
| --- | ---: |
| Macro LocAcc | 0.8598 |
| Macro mIoU | 0.5889 |

该结果使用 frozen SigLIP2 text head、label-free peak-relative threshold、peak-connected-component coarse mask 和 feature-only prompt-conditioned SAM3 boundary readout。重要的是，该 readout 不调用 official RGB SAM3 decoder。

解释：

- LocAcc 说明文本查询峰值位置稳定；
- mIoU 说明 rendered feature response 能形成可用 mask support；
- 相比 frame-wise RADIO RGB feature，CTF-GS rendered field 在同一 evaluator 下提升了 both LocAcc and mIoU。

### 8.3 LERF direct 3D object selection

任务目标：验证查询是否能发生在三维 Gaussian primitive 层面，而不是只在 rendered 2D heatmap 上。

主结果：

| Readout | mIoU | Acc@0.25 |
| --- | ---: | ---: |
| CTF-GS MPR analysis | 0.480 | 0.676 |
| CTF-GS compact direct field | 0.501 | 0.704 |
| Official SAM3 box diagnostic | 0.570 | 0.684 |

关键结论：

- compact direct field 已经不依赖 MPR cache 或 official RGB SAM decoder；
- support-calibrated primitive readout 使 compact direct field 的 Acc@0.25 超过注册 multiview analysis readout；
- Waldo Kitchen 的 small/fragmented object support 从原先最危险的 failure mode 转为可解释、可缓解的剩余风险。

需要向导师讲清楚：

> Direct 3D 不是把 2D heatmap 换个背景图展示。它是在 3D primitives 上先做文本查询，再把 selected primitives 渲染出来与 mask 对齐评估。因此它证明的是 compact field 的 primitive-level queryability。

### 8.4 ScanNet VALA8 direct point-query transfer

任务目标：验证该 field 是否能离开 LERF object-query 场景，在 ScanNet 点云 direct point-query 协议中仍然可用。

主结果：

| Split | mIoU | mAcc |
| --- | ---: | ---: |
| 19-class | 0.3806 | 0.6129 |
| 15-class | 0.3871 | 0.6315 |
| 10-class | 0.4711 | 0.7200 |

实验口径：

- follow VALA/OpenGaFF ScanNet-8 scene subset；
- every-20-frame ScanNet training split；
- direct point-query feature probe；
- 不是 full ScanNet semantic segmentation leaderboard。

解释：

该实验证明 compact field 不只是 LERF 场景上的查询技巧，而是具备跨数据集三维 point-level feature usability。

### 8.5 Frame-wise foundation features vs reconstructed scene field

任务目标：证明 CTF-GS 不是 teacher 的低质量近似复制，而是通过多视角重建形成更稳定的场景级特征。

主结果：

| Feature Source | Macro LocAcc | Macro mIoU |
| --- | ---: | ---: |
| RADIO RGB frame-wise features | 0.8065 | 0.4597 |
| CTF-GS rendered field | 0.8598 | 0.5707 |

解释：

- CTF-GS 由 RADIO supervision 训练，但 inference 时利用了三维场景中的多视角聚合；
- 同一 frozen evaluator 下，reconstructed scene field 能削弱单帧噪声；
- 因此可以谨慎表述为：CTF-GS improves selected downstream usability over frame-wise foundation features。

### 8.6 Frozen-head downstream probes

在 SAM3 adaptor 和 DINOv3 task probes 上，项目验证了 rendered field 的 selected downstream usability：

- SAM3 point prompt mIoU：0.4173 vs teacher 0.3700；
- SAM3 box prompt mIoU：0.6638 vs teacher 0.6560；
- SAM3 mask propagation mIoU：0.3756 vs teacher 0.3583；
- DINOv3 mask propagation + SAM-boundary readout：0.4677 vs teacher 0.4606；
- DINO dense matching score：0.9048 vs teacher 0.8547。

这里建议汇报时保持边界：

> 这些结果支持 selected frozen-head downstream tasks 上的下游可用性提升，但不应写成所有 DINO/SAM 指标都全面超越 teacher。

### 8.7 存储和效率

项目显式比较 compact checkpoint 与直接 per-Gaussian 1280-D fp16 feature storage：

- compact representation 在四个 LERF 场景中均节省存储；
- 场景 Gaussian 数越多，节省越明显；
- optional MPR cache 单独统计，不计入 compact scene memory。

该结果用于支撑“compact foundation-feature scene memory”的可部署性，而不是单纯追求 mIoU。

## 九、消融结论

当前完整消融显示，各模块贡献可以按重要性理解：

1. **Multiview-registration-to-field primitive feature transfer**：direct 3D 中最大幅度提升，说明 primitive feature 需要多视角 registered evidence；
2. **Foundation-Space Reconstruction codec**：rendered architecture 中最核心模块，去掉后 LocAcc 和 mIoU 大幅下降；
3. **Frame-wise RADIO vs CTF-GS rendered field**：证明重建场不是低质量复制，而是有下游任务收益；
4. **Frozen-Head Geometry Consistency warm-start**：提升 feature geometry compatibility；
5. **Support-Calibrated Primitive Readout**：显著改善 direct 3D small/fragmented object support；
6. **Feature-only SAM3 boundary readout / peak-component readout**：主要改善 mask boundary 和 region support。

汇报时可用一句话总结：

> 核心提升来自“把 foundation features 重建进三维场景”和“把多视角 primitive evidence 压缩回 compact field”；边界和 support calibration 是把 feature score 转化成可评估 object mask 的必要 readout。

## 十、目前逻辑链是否闭环

当前项目已经形成较完整闭环：

| 环节 | 是否闭环 | 证据 |
| --- | --- | --- |
| 研究动机 | 已闭环 | 2D foundation features 强但 view-local，3D 场景需要 compact queryable memory |
| 方法主线 | 已闭环 | Contextual Gaussian Feature Field + Foundation-Space Reconstruction + Multi-Protocol Readout |
| 2D 协议 | 已闭环 | LERF rendered-view LocAcc/mIoU |
| 3D primitive 协议 | 已闭环 | LERF direct 3D mIoU / Acc@0.25 |
| 点云 transfer 协议 | 已闭环 | ScanNet VALA8 direct point-query |
| teacher-vs-field | 基本闭环 | selected frozen-head tasks 显示 reconstructed field 优于 frame-wise features |
| 存储效率 | 已闭环 | direct 1280-D storage vs compact checkpoint |
| 定性展示 | 已闭环 | LERF 2D/3D OVS、ScanNet binary query、direct3D ablation |
| 风险边界 | 已明确 | external baselines、small objects、ScanNet scope、SAM3 diagnostic 分开表述 |

因此，当前材料可以支撑一篇完整顶刊稿件的主叙事。剩余主要是投稿前 human metadata、最后文字润色和可选外部 baseline 严格复现。

## 十一、主要风险与应对

### 风险一：外部 baseline 不是全部 same-evaluator rerun

风险：

LERF、OpenGaussian、Dr. Splat、OpenGaFF context 等外部数字来自 published context 或 official-source rows，不一定与本项目本地 evaluator 完全一致。

应对：

- 主文明确写 external results are source-anchored context；
- 不做无边界 universal SOTA claim；
- 本方法核心 claim 绑定本地固定 protocol、final_rows 和 same-protocol ablations。

### 风险二：direct 3D 中 small/fragmented object 仍是困难点

风险：

Waldo Kitchen、细长物体、小物体、多实例对象仍容易出现 support 不完整或 boundary 不精细。

应对：

- 用 Support-Calibrated Primitive Readout 缓解；
- 主文保留 failure analysis；
- 后续可继续研究 proposal-aware / object-level grouping。

### 风险三：SAM3 / DINO claim 需要谨慎

风险：

如果写成“官方 SAM3 decoder 直接由 CTF features 驱动”或“所有 DINO 指标全面超过 teacher”，容易被审稿人质疑。

应对：

- 主文使用 “SAM3 adaptor-space probe / feature-only boundary readout”；
- official SAM3 box row 作为 diagnostic；
- DINO 写 selected frozen-head downstream usability，不写 universal superiority。

### 风险四：ScanNet 不能表述为 full segmentation leaderboard

风险：

ScanNet 当前是 VALA/OpenGaFF-8 direct point-query feature probe，不是完整 ScanNet semantic segmentation benchmark。

应对：

- 表格和正文明确 direct point-query transfer；
- 对外部 numbers 做 context comparison；
- 不写 full ScanNet SOTA。

## 十二、下一步工作计划

### 短期：投稿前完善

1. 完成导师审稿意见吸收；
2. 检查主文所有 claim 是否和 final_rows 对齐；
3. 补充作者信息、funding、COI、AI/tool disclosure；
4. 最后一轮 PDF proofread；
5. 若导师要求，补充一页 strict external baseline protocol discussion。

### 中期：进一步增强顶刊说服力

1. 加强 small-object / fragmented-object direct 3D support；
2. 探索更正式的 object/proposal-aware primitive grouping；
3. 对 DINO topology adaptor 做更系统验证；
4. 如果有资源，选择 1-2 个外部方法做严格 same-evaluator rerun；
5. 扩展到更多室内/真实场景，验证 compact foundation-feature field 的泛化能力。

### 长期：可扩展研究方向

1. 从 static scene memory 扩展到 dynamic scene foundation-feature memory；
2. 从 text query 扩展到 instruction-level 3D scene reasoning；
3. 将 compact Gaussian feature field 与机器人导航、交互式编辑、具身问答结合；
4. 研究 foundation-feature field 的理论压缩边界和多任务信息保持机制。

## 十三、建议汇报 PPT 结构

建议 15 页左右。

1. **题目页**：面向开放词汇三维场景理解的紧凑基础特征高斯场研究；
2. **一句话问题**：二维基础模型很强，但三维场景缺少可部署的 foundation-feature memory；
3. **研究背景**：LERF/LangSplat/OpenGaussian/Dr. Splat/OpenGaFF 的发展趋势；
4. **科学问题**：如何紧凑存储、如何多协议读出、如何超过 frame-wise teacher；
5. **总体思路**：one compact foundation-feature Gaussian map；
6. **框架图**：使用 `paper/figures/radio_gs_framework.pdf`；
7. **方法一**：Contextual Gaussian Feature Field；
8. **方法二**：Foundation-Space Reconstruction + View-Conditioned Calibration；
9. **方法三**：MPR + Support-Calibrated Primitive Readout；
10. **实验协议**：LERF 2D、LERF 3D、ScanNet point-query、teacher-vs-field；
11. **主结果**：三张核心定量表合并讲；
12. **定性结果**：LERF 2D/3D OVS + ScanNet query；
13. **消融与机制分析**：哪些模块贡献最大；
14. **风险与边界**：external baseline、small object、SAM/DINO、ScanNet scope；
15. **下一步与投稿计划**：TPAMI 投稿包状态和导师需要决策事项。

## 十四、汇报开场建议

可以这样开场：

> 本项目关注的问题是：二维视觉基础模型已经很强，但它们的特征主要存在于离散图像帧中，难以直接作为三维场景的可部署记忆。我们希望把这些基础特征压缩并重建到 3D Gaussian 场景中，使同一个 compact field 同时支持新视角二维开放词汇定位、三维 primitive-level object selection 和 ScanNet 点级语义查询。当前我们已经完成 TPAMI 方向投稿包，并通过三类 benchmark、teacher-vs-field 对比、存储效率和消融实验形成了较完整的证据链。

## 十五、导师可能会问的问题与回答口径

### 问题一：这是不是只是把 RADIO 特征压缩一下？

回答：

不是。压缩只是必要条件，核心是把 view-local foundation features 转换成 scene-level queryable memory。实验上，CTF-GS rendered field 在同一 frozen evaluator 下超过 frame-wise RADIO RGB features，说明多视角重建和支持校准带来了下游可用性提升。

### 问题二：为什么要做 direct 3D object selection？

回答：

因为 2024-2025 年开放词汇 3DGS 主线已经从 rendered 2D heatmap 转向 primitive/point/instance-level query。没有 direct 3D selection，文章只能说明 feature rendering 可用，不能说明三维场景本身可查询。

### 问题三：MPR 会不会削弱 compact-only claim？

回答：

不会。MPR 在本文中是 training bridge 和 analysis control。主 compact direct readout 不读取 MPR feature cache。MPR 的作用是把多视角注册证据压缩回 compact field，而不是作为最终推理主存储。

### 问题四：RGB color-edge calibration 算不算额外模块？

回答：

它是 label-free classical support calibration，不是 learned segmentation network，也不是 official RGB SAM decoder。它用于把 primitive score 转化为稳定 mask support，类似固定后处理/读出校准。主文同时报告 strict no-RGB ablation，保证 claim 边界清楚。

### 问题五：现在能否说 SOTA？

回答：

建议不做无边界 universal SOTA claim。可以说：在本地固定协议和 source-anchored context 下，CTF-GS compact direct readout 具有竞争力，并在三类任务上形成了完整证据链。若要做严格 leaderboard claim，需要对外部方法进行 same-evaluator rerun。

## 十六、最终一句话总结

> 本项目的核心贡献是把二维基础模型的帧级特征转化为紧凑、可查询、多协议读出的三维 Gaussian scene memory；它不是单一 open-vocabulary heatmap 方法，而是面向开放词汇三维场景理解的 foundation-feature field 表示学习框架。
