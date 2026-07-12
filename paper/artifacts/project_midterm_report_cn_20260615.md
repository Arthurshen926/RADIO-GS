# 面向开放词汇三维场景理解的紧凑基础特征高斯记忆

项目汇报材料  
日期：2026-06-15  
项目代码名：RADIO-GS / GaussFM
建议投稿目标：IEEE TPAMI 或同级别计算机视觉顶刊

---

## 一、执行摘要

本项目面向一个正在快速发展的方向：如何把二维基础模型的强表征能力可靠、紧凑、可查询地迁移到三维场景表示中，使三维场景不仅能渲染 RGB 图像，还能支持开放词汇查询、三维目标选择、点云语义查询以及多类基础特征下游任务。

现有方法大多沿两条路线推进：一类将 CLIP/SigLIP 等语义特征蒸馏到 3DGS，用于二维渲染视图的开放词汇定位；另一类把文本对齐特征注册到三维 Gaussian、point 或 instance 上，用于直接三维查询。这两条路线分别解决了部分问题，但仍存在三个核心不足：高维基础特征存储代价大，多视角特征不一致会导致三维基元语义碎片化，且二维渲染查询和三维直接查询往往被设计成不同任务接口，缺少一个统一、紧凑、可复用的三维基础特征记忆。

本项目提出 GaussFM：将三维 Gaussian 场景扩展为一个紧凑基础特征三维记忆。方法并不显式存储完整 1280 维 RADIO 特征，而是在每个 Gaussian 上存储低维 compact latent，并结合空间上下文、可见性、可靠性与全局解码器，在查询或渲染时按需重建 RADIO-compatible foundation feature。该记忆通过两类监督共同形成：一是稠密渲染视图上的 RADIO 特征重建和结构一致性约束；二是多视角基元注册提供的稀疏语义锚定，使三维 Gaussian primitive 具有稳定的直接查询能力。

当前结果显示，该方法在三个主要任务上形成闭环证据：

- LERF-OVS rendered-view open-vocabulary query：GaussFM 本地可追溯结果为 58.89 mIoU / 85.98 Acc；外部方法行目前仅作为论文报告值背景。
- LERF-OVS direct 3D open-vocabulary query：GaussFM 本地可追溯结果为 50.14 mIoU / 70.44 Acc@0.25；统一 readout 后再进行严格排名。
- ScanNet-v2 八场景本地点查询：GaussFM 达到 19-class 38.06 mIoU / 61.29 mAcc，15-class 38.71 / 63.15，10-class 47.11 / 72.00；该 GT-point 查询域尚未与论文中 Gaussian-domain 外部结果统一。

此外，在 feature usability 层面，重建出的三维场景特征在所选 frozen-head 2D 下游任务主指标上均超过 frame-wise RADIO，包括 SAM3 prompt/mask 相关任务、DINOv3 dense matching 和 DINOv3 mask propagation。存储方面，compact latent payload 相比直接存储 1280-D fp16 特征约 20 倍压缩；即使计入固定 decoder/refiner，feature-memory package 随场景规模增长时仍具有明显优势。

一句话概括：本项目不是简单把二维特征贴到三维高斯上，而是学习一个 compact RADIO feature field：用低维 latent、空间上下文、可靠性建模和解码器构成可压缩、可重建、可多协议查询的三维基础特征记忆，使同一三维场景表示在二维、三维和点云开放词汇理解任务中均具备下游可用性。

---

## 二、研究背景与意义

### 2.1 二维基础模型很强，但天然是视角局部的

近年来，RADIO、DINO、SAM、SigLIP/CLIP 等二维基础模型已经具备很强的图像级、patch 级和 dense feature 表达能力。这些特征能支撑开放词汇检索、语义定位、mask 生成、跨视图匹配等下游任务。

然而，二维基础特征本质上依赖单帧图像输入。对于一个真实三维场景，如果每次查询都重新选择视角、提取图像特征、再做多视角融合，会带来三个问题：

1. **存储与计算重复**：每个视角都保留高维特征，存储和查询成本随视角数增长。
2. **跨视角不一致**：同一物体在不同视角下的 foundation feature 可能受遮挡、光照、尺度、背景影响而不稳定。
3. **三维查询困难**：单帧特征可以做二维 heatmap，但不能天然回答“哪些三维 Gaussian/point 属于这个开放词汇目标”。

因此，一个自然但尚未充分解决的问题是：能否把二维基础特征转化为三维场景自身的可查询记忆？

### 2.2 3DGS 提供了高效场景表达，但语义理解仍未充分解决

3D Gaussian Splatting 已成为高效神经渲染和三维场景表达的重要技术路线。它通过大量 Gaussian primitive 表示场景几何、颜色、透明度和视角相关外观，具有渲染速度快、可显式访问、易于编辑和查询等优势。

但原始 3DGS 主要表达 RGB 外观，而不是 foundation feature。若直接为每个 Gaussian 存储高维语义特征，会迅速带来存储问题。例如对 1280-D RADIO fp16 特征，Waldo Kitchen 场景仅特征存储就需要约 1.69 GiB。更重要的是，直接把多视角二维特征投到 Gaussian 上并不能自动解决可见性、遮挡、特征冲突和小物体碎片化问题。

### 2.3 开放词汇三维理解正在从二维渲染查询走向三维基元查询

早期 LERF/LangSplat 类工作主要关注 rendered-view open-vocabulary localization，即先渲染二维特征图，再和文本做相似度得到 heatmap。近期 OpenGaussian、Dr. Splat、VALA 等公开工作进一步强调 direct 3D object selection、point-level query 和 instance/primitive-level understanding。

这意味着论文不能只证明“渲染出的 feature map 能做二维定位”，还需要证明三维场景表示本身具备直接查询能力。也就是说，理想的三维 foundation-feature 表示应同时满足：

- 能渲染成 dense feature map，服务 novel-view 2D open-vocabulary grounding；
- 能在 Gaussian primitive / point level 被直接查询，服务 3D object selection 和 point-level understanding；
- 存储紧凑，查询高效，且不依赖部署时的高维多视角 cache。

这正是本项目的研究定位。

---

## 三、拟解决的关键科学问题

本项目围绕一个核心问题展开：

> 如何将二维基础模型的高维、多视角、不稳定特征，压缩为一个紧凑、可重建、可直接查询的三维 Gaussian 场景记忆？

围绕该核心问题，可以拆解为四个关键科学与技术问题。

### 问题一：高维基础特征如何压缩进三维场景而不丢失下游可用性？

直接存储 1280-D RADIO 特征代价高，且不一定比重建特征更好。项目需要回答：低维 compact latent 是否能通过解码重建出 RADIO-compatible feature，并在开放词汇定位、SAM/DINO 下游任务中保持甚至超过原始 frame-wise RADIO 的可用性。

### 问题二：多视角二维特征如何形成稳定的三维基元语义？

同一 Gaussian 在不同视角中的语义证据可能不一致，且小物体、细长物体和低可见区域容易出现漏检和碎片化。项目需要将多视角注册证据压回 compact field，使 Gaussian primitive 不是孤立地学习单点特征，而是学习跨视角、跨空间上下文一致的语义支持。

### 问题三：二维 rendered query 与三维 direct query 如何统一到同一个场景记忆？

如果每个任务都设计一套专用模块，方法会显得像任务特调。项目需要把不同查询任务解释为同一基础特征记忆的不同使用方式：二维任务通过渲染恢复 dense feature map，三维任务通过 Gaussian/point 位置直接恢复或聚合 compact feature，并使用统一的语义锚定和支持校准机制。

### 问题四：如何让结果具有投稿级实验闭环？

顶刊论文不仅要求方法有效，还要求协议清晰、对比充分、消融完整、定性可信、存储与效率可解释。项目需要在 LERF rendered-view、LERF direct 3D、ScanNet point-query、frame-wise feature usability、storage/latency、qualitative analysis 等多个层面形成闭环证据。

---

## 四、总体研究思路

项目的总体思路可以概括为：

> 先用标准 3DGS 获得几何和 RGB 外观基础，再在该三维结构上学习一个紧凑 foundation-feature memory；该 memory 不完整存储高维特征，而是在需要时重建 RADIO-compatible feature，并通过多视角注册和结构一致性监督增强其二维与三维查询能力。

该思路有三个关键转变。

### 4.1 从“存储高维特征”转为“存储可重建特征记忆”

传统方式可以理解为：把每个 Gaussian 直接绑定一个高维语义向量。但这既费存储，也容易把多视角噪声固化到三维中。

本项目改为存储 compact latent。查询时通过上下文场和解码器重建所需的 RADIO-compatible feature。这样做的意义是：

- 存储上，per-Gaussian 低维 latent 约 20 倍压缩；
- 表达上，解码器和空间上下文可学习多视角去噪和结构补全；
- 使用上，同一 memory 可用于 dense rendered feature 和 sparse primitive feature。

这里“记忆”不是离线保存完整 feature bank，而是一个可重建的场景特征记忆：显式存储的是 compact code，完整 feature 在查询和渲染时按需恢复。

### 4.2 从“单帧监督”转为“稠密重建 + 稀疏语义锚定”

项目将监督分为两类。

第一类是稠密 rendered supervision：在训练视角渲染 compact feature map，解码到 RADIO feature space，与 frozen RADIO 特征进行 dense reconstruction，并结合 mask boundary、local topology、cross-view structure 等 dense structural signals。

第二类是稀疏 primitive regularization：通过 Multiview Primitive Registration 将多视角证据注册到 Gaussian primitive 上，在稀疏 summary space 中约束 primitive 的语义可查询性。最新 2x2 消融显示，Sparse SigLIP/MPR 对 Direct3D 有明显增益，而 2D dense SigLIP/text loss 不是核心贡献。这支持一个更清晰的叙事：SAM/DINO 类结构信息适合稠密 rendered 监督，SigLIP/MPR 更适合作为稀疏 primitive semantic anchor。

### 4.3 从“像素热力图”转为“对象支持校准”

开放词汇三维查询的瓶颈不只是 feature 相似度，还包括 score-to-mask selection、组件支持完整性和边界稳定性。尤其在 Waldo Kitchen 等小物体、低可见场景中，直接阈值容易出现 zero prediction 或碎片化。

因此项目引入 support-calibrated primitive selection：在 compact primitive score 的基础上，利用 GT-free score component、可见性、面积、边界和 RGB edge 信息进行支持区域校准。该模块不调用 official RGB SAM decoder，也不需要 VPR inference cache；它服务于把 primitive score 转成稳定 object support。

---

## 五、方法体系

### 5.1 前置几何重建

项目首先使用标准 3DGS 对每个场景进行 RGB/几何重建，获得 Gaussian centers、scales、opacity、color 等基础场景结构。这一步是三维表示的几何基础，不是本文的主要创新。

后续 feature memory learning 在该 Gaussian 几何结构上进行。换言之，本文不是重新提出 RGB reconstruction 方法，而是在已有 3DGS 场景上学习紧凑 foundation-feature memory。

### 5.2 紧凑上下文高斯特征场

每个 Gaussian 存储一个低维 compact latent，而不是完整 1280-D RADIO feature。模型还包含：

- per-Gaussian fine latent code；
- 空间上下文 / hash-like branch，用于补充局部几何和场景上下文；
- opacity、scale、view count、visibility、confidence 等可靠性信息；
- compact-to-RADIO 的 HCD/CTR decoder；
- rendered-view 的 screen-space calibration/refiner，用于提升渲染特征图质量；
- point summary adapter，用于 primitive-level summary-space regularization。

因此，“Compact Feature Memory Learning”并不只是一个 decoder，而是一个由 compact latent、空间上下文、可靠性建模、特征解码和视图校准共同构成的上下文高斯特征场。

### 5.3 稠密 rendered RADIO reconstruction

训练时，对每个训练视角渲染 compact feature map，再通过 decoder 重建 RADIO-compatible dense feature。该重建目标包括：

- 与 frozen RADIO dense feature 对齐；
- 通过 HCD/CTR 保持高维 feature 的可重建性；
- 利用 dense structural signals 保持边界、局部拓扑和区域结构；
- 在 rendered-view 上形成可用于 open-vocabulary grounding 的特征图。

这部分是项目的基础能力：没有稳定的 RADIO-compatible dense reconstruction，后续二维和多下游 frozen-head usability 都无法成立。

### 5.4 Multiview Primitive Registration 与稀疏语义锚定

Multiview Primitive Registration 的作用不是在推理时维护一个 VPR cache，而是在训练中把多视角注册证据压回 compact field。它将可见视角中的 feature evidence 聚合到 Gaussian primitive 上，为 direct primitive query 提供稀疏语义监督。

最新消融说明，3D sparse supervision 中选择 SigLIP summary space 作为语义锚更合理：它提供的是 primitive-level semantic usability regularization，而不是把整个 compact field 特化成 SigLIP feature field。稠密 feature 重建仍然以 RADIO-compatible foundation space 为主。

这个设计可解释为：

- 稠密 rendered reconstruction 学习“场景中每个视角下的基础特征结构”；
- 稀疏 primitive regularization 学习“哪些三维基元具有稳定语义支持”；
- 二者共同优化同一个 compact feature memory。

### 5.5 Support calibration 与边界稳定

Direct 3D object selection 中，primitive score 本身并不等于高质量 mask。项目将 selection 过程扩展为 score support calibration：

- prompt ensemble 提升文本查询稳定性；
- opacity/view confidence 控制低可靠 primitive；
- score component guard 修复碎片化与小物体漏检；
- RGB/GrabCut boundary snapping 作为轻量 GT-free 边界校准，不引入额外 foundation model 或 official RGB SAM decoder。

该模块的目标不是用额外网络替代 feature field，而是把 compact field 的 primitive score 转化为更稳定的 object support，尤其缓解 Waldo small-object failure。

### 5.6 Frozen-head 下游任务验证

项目还使用 SAM3 adaptor、DINOv3 adaptor、SigLIP2 summary head 等 frozen-head 或 adaptor-space probes 来验证重建特征的下游可用性。这部分不应表述为“本文直接学习了 DINO/SAM/SigLIP 原生特征”，更准确的说法是：

> 本文学习 RADIO-compatible compact scene features，并通过冻结或轻量 adaptor 的下游任务检验这些重建特征是否保留了可用于 segmentation、matching、text grounding 的基础模型能力。

---

## 六、主要创新点

建议对外论文贡献压缩为三个主贡献，汇报时可展开为四点。

### 创新一：紧凑可重建的 compact RADIO Gaussian feature field

项目提出 compact foundation-feature Gaussian memory，其核心实现就是 compact RADIO feature field。它不显式存储完整高维 RADIO feature，而是在每个 Gaussian 上学习低维 compact latent；同时引入空间上下文分支、visibility/confidence 可靠性建模、HCD/CTR feature decoder 和 rendered-view calibration，使三维 Gaussian 场景能够按需重建 RADIO-compatible dense feature 或 primitive feature。

因此，第一个贡献不只是“把特征压缩存起来”，而是提出一个紧凑、上下文化、可重建的 RADIO feature field。它使三维场景从“RGB/几何表示”扩展为“可查询基础特征记忆”，兼具压缩性、跨视角去噪能力和下游可用性。

### 创新二：稠密 rendered 重建与稀疏 primitive 语义锚定的联合训练框架

项目将 dense rendered RADIO reconstruction 与 sparse MPR/SigLIP primitive regularization 分开建模，又共同作用于同一个 compact field。该设计避免把方法解释成某个下游任务的特化，同时解释了为什么 dense SAM/DINO-style structural signals 和 sparse SigLIP semantic anchors 各有位置。

### 创新三：面向三维开放词汇查询的支持校准机制

项目发现 direct 3D object selection 的关键瓶颈不只是 feature cosine，而是 primitive score 到 object support 的转换。通过 support-calibrated primitive selection，方法显著提升 Acc@0.25 和边界指标，尤其修复小物体、低可见、碎片化查询。

### 创新四：跨协议完整实验闭环

项目同时验证：

- LERF rendered-view 2D open-vocabulary grounding；
- LERF direct 3D object selection；
- VALA-aligned ScanNet-8 point-level query；
- frame-wise RADIO vs reconstructed scene features；
- storage / efficiency / qualitative / ablation / failure analysis。

这使论文不是单点指标提升，而是围绕“紧凑三维基础特征记忆”形成多维证据链。

---

## 七、实验体系与当前结果

### 7.1 LERF 2D rendered-view open-vocabulary query

任务：在 novel rendered view 上，给定开放词汇文本 query，输出二维 heatmap/mask，与 LERF-OVS 标注比较。

| Method | Mean mIoU | Mean Acc | Figurines mIoU / Acc | Teatime mIoU / Acc | Ramen mIoU / Acc | Waldo mIoU / Acc |
|---|---:|---:|---:|---:|---:|---:|
| LangSplat | 51.40 | 84.30 | 44.70 / 80.40 | 65.10 / 88.10 | 51.20 / 73.20 | 44.50 / 95.50 |
| GAGS | 54.12 | 81.66 | 53.59 / 78.57 | 60.29 / 88.14 | 46.81 / 69.01 | 55.80 / 90.91 |
| OccamLGS | 61.30 | 82.50 | 58.60 / 80.40 | 70.20 / 93.20 | 51.00 / 74.70 | 65.30 / 81.80 |
| GOI | 42.00 | 59.20 | 23.90 / 44.60 | 55.80 / 67.80 | 33.70 / 56.30 | 54.50 / 68.20 |
| GALA | 55.49 | 73.43 | 59.35 / 82.14 | 76.73 / 88.14 | 35.13 / 50.70 | 50.75 / 72.73 |
| LangSplatV2 | 59.90 | 84.10 | 56.40 / 82.10 | 72.20 / 93.20 | 51.80 / 74.70 | 59.10 / 95.50 |
| GaussFM | 58.89 | 85.98 | 52.43 / 82.14 | 65.15 / 89.83 | 63.25 / 90.14 | 54.75 / 81.82 |

说明：GaussFM 行来自本地原始输出；外部行是 source-paper reported context。当前表支持 feature usability 讨论，但在 readout 完全统一前不作严格排名。

### 7.2 LERF 3D direct open-vocabulary query

任务：给定文本 query，直接在 Gaussian primitive 上计算相关性并选择三维基元，再将选中基元渲染到标注视角计算 mask IoU 和 Acc@0.25。

| Method | Mean mIoU | Mean Acc | Figurines mIoU / Acc | Teatime mIoU / Acc | Ramen mIoU / Acc | Waldo mIoU / Acc |
|---|---:|---:|---:|---:|---:|---:|
| OpenGaussian | 38.36 | 51.43 | 39.29 / 55.36 | 60.44 / 76.27 | 31.01 / 42.25 | 22.70 / 31.82 |
| SuperGSeg | 35.94 | 52.02 | 43.68 / 60.71 | 55.31 / 77.97 | 18.07 / 23.94 | 26.71 / 45.45 |
| OccamLGS | 47.22 | 74.84 | 52.90 / 78.57 | 61.02 / 93.22 | 32.01 / 54.92 | 42.95 / 72.72 |
| Dr. Splat | 43.29 | 64.30 | 54.42 / 80.36 | 57.35 / 77.97 | 24.33 / 35.21 | 37.05 / 63.64 |
| GALA | 36.71 | 59.71 | 45.25 / 69.64 | 53.27 / 84.75 | 17.08 / 25.35 | 31.22 / 59.09 |
| LangSplatV2 | 35.87 | 55.80 | 45.15 / 67.86 | 49.30 / 79.66 | 19.01 / 21.13 | 30.00 / 54.55 |
| GaussFM | 50.14 | 70.44 | 51.04 / 67.86 | 56.40 / 76.27 | 59.99 / 83.10 | 33.12 / 54.55 |

说明：该本地结果支撑 compact RADIO Gaussian feature field 可进行 primitive-level 查询；外部行尚未在完全相同的 selector/readout 下重跑，因此不作严格排名。

### 7.3 VALA-aligned ScanNet-8 direct point query

任务：在 ScanNet 8 个官方/对齐场景上做开放词汇 point-level semantic query，比较 split19/split15/split10。

| Method | 19 classes mIoU / mAcc | 15 classes mIoU / mAcc | 10 classes mIoU / mAcc |
|---|---:|---:|---:|
| LangSplat | 2.45 / 8.59 | 3.45 / 13.21 | 6.48 / 21.89 |
| LangSplatV2 | 14.75 / 25.47 | 17.09 / 35.68 | 22.83 / 41.52 |
| OpenGaussian | 27.73 / 42.01 | 29.67 / 46.15 | 39.93 / 57.34 |
| Dr. Splat | 29.31 / 47.68 | 33.25 / 54.33 | 44.19 / 65.19 |
| OccamLGS | 31.93 / 48.93 | 34.25 / 53.71 | 45.16 / 64.39 |
| VALA | 32.11 / 50.05 | 35.10 / 54.77 | 46.21 / 65.61 |
| GaussFM | 38.06 / 61.29 | 38.71 / 63.15 | 47.11 / 72.00 |

说明：结果表明 compact RADIO feature field 具备跨数据集的点级查询能力；由于本地 GT-point 域和外部 Gaussian-domain 协议不同，当前不据此宣称全面最优。

### 7.4 Reconstructed scene features vs frame-wise RADIO

项目专门比较了重建三维场景特征与原始 frame-wise RADIO 在多个 frozen-head 下游任务中的表现。

| Task | Primary metric | Frame-wise RADIO | GaussFM | Delta |
|---|---:|---:|---:|---:|
| LERF text grounding | mIoU | 0.4673 | 0.5889 | +0.1217 |
| SAM3 point prompt | mIoU | 0.3700 | 0.4173 | +0.0473 |
| SAM3 box prompt | mIoU | 0.6560 | 0.6638 | +0.0079 |
| SAM3 mask propagation | mIoU | 0.3583 | 0.3756 | +0.0173 |
| DINOv3 dense matching | Mean score | 0.8547 | 0.9048 | +0.0501 |
| DINOv3 mask propagation | mIoU | 0.4606 | 0.4677 | +0.0071 |

结论：在所选 primary metrics 上，重建三维场景特征均超过 frame-wise RADIO。这是支撑“重建场景特征不仅更紧凑，而且更有下游可用性”的关键证据。

### 7.5 Storage footprint

| Scene | #Gaussians | Direct 1280-D fp16 | Latent payload | Latent saving | Feature-memory package | Package saving |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 168,791 | 412.1 MiB | 20.6 MiB | 20.00x | 199.0 MiB | 2.07x |
| Ramen | 382,687 | 934.3 MiB | 46.7 MiB | 20.00x | 225.1 MiB | 4.15x |
| Teatime | 460,157 | 1123.4 MiB | 56.2 MiB | 20.00x | 234.5 MiB | 4.79x |
| Waldo Kitchen | 691,728 | 1688.8 MiB | 84.4 MiB | 20.00x | 262.8 MiB | 6.43x |

解释：latent payload 是纯 per-Gaussian compact feature storage；feature-memory package 还包含 decoder/refiner 等固定模块。固定模块不随 Gaussian 数线性增长，因此场景越大，compact memory 的优势越明显。论文中应主动区分 latent payload、feature-memory package 和 full checkpoint，避免 full checkpoint 中普通 3DGS 几何/RGB 张量掩盖 feature compression 的贡献。

### 7.6 最新 Dense/Sparse SigLIP 消融

为回应方法叙事中的关键问题，项目做了 Ramen 8-epoch 2x2 消融：

| Variant | Dense SigLIP/text | Sparse SigLIP/MPR | Direct3D mIoU | Direct3D Acc@0.25 | 2D LocAcc | 2D mIoU |
|---|---:|---:|---:|---:|---:|---:|
| dense0_sparse0 | no | no | 0.2389 | 0.3099 | 0.9014 | 0.6389 |
| dense1_sparse0 | yes | no | 0.2239 | 0.3239 | 0.8732 | 0.6416 |
| dense0_sparse1 | no | yes | 0.3129 | 0.4225 | 0.8451 | 0.5954 |
| dense1_sparse1 | yes | yes | 0.2913 | 0.4085 | 0.8451 | 0.6142 |

该结果支持如下判断：

- Sparse SigLIP/MPR 对 Direct3D 有明显正贡献；
- Dense rendered SigLIP/text loss 不是 rendered-view feature reconstruction 的核心；
- 论文主叙事应避免说 compact latent 在三维监督里直接特化到 SigLIP 任务空间；
- 更合理的写法是：dense rendered RADIO reconstruction + dense structural regularization，辅以 sparse primitive semantic regularization。

---

## 八、定性结果与图表安排

### 8.1 主图安排

建议顶刊稿件采用如下图表结构。

**Figure 1: Overall Framework**  
强调从 posed RGB + frozen RADIO evidence，到 compact foundation-feature Gaussian memory，再到二维/三维/点云查询任务。图中应明确区分：

- RGB/geometry reconstruction 是前置 3DGS；
- feature training 学习 compact latent + contextual field；
- RADIO 是主 foundation feature；
- SAM/DINO/SigLIP 是 adaptor/frozen-head consistency 或 evaluation space，不应画成并列 foundation cues。

**Figure 2: Method Details**  
建议分三块：

1. Compact contextual Gaussian feature field：latent、spatial context、visibility/confidence、decoder。
2. Dense rendered RADIO reconstruction：render compact map，decode，dense feature loss，boundary/topology regularization。
3. Sparse MPR semantic anchoring：multiview registration，primitive summary target，support-calibrated selection。

**Figure 3: LERF 2D and 3D Open-Vocabulary Query**  
每个 scene 展示 2D OVS 和 3D OVS 的区别。2D OVS 展示 heatmap/mask over RGB；3D OVS 应展示选中 Gaussian primitive 渲染到空白背景的 mask/support，而不是和 2D heatmap 表现完全相同。2D baseline 可用 LangSplatV2，3D baseline 可用 Dr. Splat。

**Figure 4: ScanNet Open-Vocabulary 3D Query**  
主文建议做 binary query point cloud：RGB/overview、GT class mask、baseline、GaussFM。避免全类别彩色点云在主文造成信息过载。

**Figure 5: Reconstructed Scene Features vs Frame-wise RADIO**  
展示 SAM prompt、DINO matching、mask propagation 等代表性下游任务。重点不是展示所有错误，而是让读者直观看到 GaussFM 重建特征在边界、匹配稳定性或区域传播上优于 frame-wise RADIO。

### 8.2 消融定性

最值得保留的消融定性是两类。

第一类：Direct3D support calibration。展示 Base compact、+prompt ensemble、+support calibration、GT。应挑选 Waldo/小物体/低可见样本，突出 support calibration 如何减少漏检和碎片化。

第二类：compact memory / feature reconstruction ablation。展示没有 spatial context、没有 MPR、没有 support calibration 时的差异。相比 rendered boundary calibration，这类图更能和核心贡献挂钩。

Rendered boundary calibration 可以放 appendix 或作为次要消融；如果可视化差异不明显，不建议占主文篇幅。

---

## 九、当前完成度评估

从完整顶刊投稿角度看，项目已经具备较完整闭环。

### 已基本完成

- 方法主线：compact foundation-feature Gaussian memory；
- 三个核心 benchmark：LERF rendered-view、LERF direct 3D、VALA-aligned ScanNet-8；
- frame-wise RADIO vs reconstructed scene feature 对比；
- storage footprint；
- 多轮 direct 3D support calibration 和 Waldo failure 修复；
- feature-only SAM/SAM3 adaptor 相关实验；
- Dense/Sparse SigLIP 监督叙事消融；
- qualitative 主图和若干消融图初稿；
- TPAMI 方向论文结构和素材基础。

### 当前已完成的关键收尾

1. **图像质量**：Figure 1/2 已按最终叙事重画，DINO/SAM/SigLIP 不再被画成与 RADIO 并列的原始 foundation cues，而是作为 adaptor/probe spaces。
2. **术语统一**：主文已统一为“compact foundation-feature scene memory / contextual Gaussian feature field / support calibration”等表达，`readout` 不再作为顶层方法叙事。
3. **Direct3D 文字边界**：主结果明确为 no multiview-registration feature cache、no official RGB SAM decoder；RGB/GrabCut boundary snapping 被表述为 GT-free lightweight support calibration。
4. **DINO/SAM/SigLIP 叙事**：主文已避免声称学习官方 DINO/SAM/SigLIP 原生特征，统一写作 RADIO-compatible scene feature 经过 frozen/adaptor heads 检验下游可用性。
5. **论文压缩**：主文保留本地结果与 published-context 表、机制消融、feature usability、storage/efficiency 和核心定性图；详细审计放入补充材料或 artifacts。

总体判断：

- 作为 CVPR/ICCV/ECCV 级会议投稿：完成度约 90%；
- 作为 TPAMI 级顶刊投稿：方法和实验已具备完整基础，完成度约 88-90%。剩余主要是人工层面的最终作者信息、致谢/基金/合规声明，以及提交系统要求的格式确认。

---

## 十、建议论文主线

论文标题可考虑：

> Compact Foundation-Feature Gaussian Memory for Open-Vocabulary 3D Scene Understanding

或更强调可重建记忆：

> Reconstructive Foundation-Feature Gaussian Memory for Open-Vocabulary 3D Scene Understanding

核心故事线建议如下。

### 第一段：动机

二维基础模型有强表征能力，但在三维场景中仍是视角局部、存储昂贵、跨视角不稳定的。开放词汇三维理解需要一个能被场景自身携带的基础特征记忆。

### 第二段：问题

直接为 3DGS 存储高维 feature 不经济，也不能解决多视角不一致和 primitive-level 查询碎片化。二维 rendered grounding 与 direct 3D selection 之间存在协议和表示断裂。

### 第三段：方法

GaussFM 学习 compact foundation-feature Gaussian memory：在 Gaussian 上存低维 latent，通过上下文场和 decoder 重建 RADIO-compatible feature；通过 dense rendered RADIO reconstruction 学习基础特征结构，通过 MPR sparse semantic anchoring 增强 primitive-level 查询，通过 support calibration 稳定 object support。

### 第四段：贡献

贡献建议写三点：

1. 提出紧凑可重建的 compact RADIO Gaussian feature field，以低维 latent、空间上下文、可靠性建模和特征解码替代显式高维 feature 存储。
2. 提出稠密 rendered RADIO reconstruction 与稀疏 MPR semantic anchoring 结合的训练框架，使同一 compact memory 同时服务二维渲染查询和三维基元/点查询。
3. 在 LERF 2D、LERF direct 3D、ScanNet point query、frame-wise feature usability、storage/efficiency 上建立系统实验闭环，证明方法又小又好。

---

## 十一、导师汇报建议流程

建议汇报按“基金申请书 + 中期进展”的方式讲，而不是按代码模块堆叠。

### 11.1 15 页汇报版本

1. 标题页：项目名称、目标顶刊、当前完成度。
2. 背景：二维 foundation features 强，但三维部署困难。
3. 痛点：高维存储、多视角不一致、二维/三维查询断裂。
4. 核心科学问题：如何学习紧凑、可重建、可查询的三维基础特征记忆。
5. 总体方案图：GaussFM overall framework。
6. 方法一：compact contextual Gaussian feature field。
7. 方法二：dense rendered RADIO reconstruction。
8. 方法三：MPR sparse semantic anchoring。
9. 方法四：support-calibrated primitive selection。
10. 实验一：LERF rendered-view OVS。
11. 实验二：LERF direct 3D object selection。
12. 实验三：ScanNet point-level query。
13. 实验四：reconstructed scene features vs frame-wise RADIO。
14. Storage/efficiency + qualitative。
15. 当前风险、收尾计划、投稿路线。

### 11.2 汇报时要主动强调的点

- 本文不是又做一个 LangSplat/LERF-style rendered heatmap，而是把基础特征变成三维场景记忆。
- compact latent 不是简单压缩，而是结合空间上下文和多视角监督的可重建 memory。
- VPR/MPR 不是推理 cache，而是训练时把多视角证据压回 compact field 的桥梁。
- SAM/DINO/SigLIP 不是并列原始输入特征；主学习对象是 RADIO-compatible foundation feature。
- Direct3D 最终结果说明 compact field 已能直接做 primitive query，不需要 VPR cache 作为主接口。

### 11.3 导师可能会问的问题与回答

**问：已有方法是否也能同时做 2D 和 3D query？**  
答：部分方法已经覆盖 2D rendered grounding 或 direct 3D selection，但本文强调的是紧凑可重建 foundation-feature memory：不显式存储高维 per-Gaussian feature，不依赖推理时多视角 cache，并系统证明同一 compact memory 在 2D、3D、ScanNet point-query 和 frozen-head usability 上均有效。

**问：为什么不用“readout”作为主叙事？**  
答：readout 容易让人感觉是为不同任务外挂接口。本文建议把它内化为“同一 feature memory 的不同查询方式”：二维任务通过渲染恢复 dense feature，三维任务通过 primitive/point 位置恢复或聚合 feature，再经过统一的 semantic/support calibration。

**问：为什么 3D sparse supervision 用 SigLIP，而不是 SAM/DINO？**  
答：当前叙事不应说“因为 direct 3D OVS 是文本任务”。更合理的解释是：SAM/DINO 表达边界、mask、局部拓扑和 correspondence，更适合 dense rendered structural regularization；SigLIP summary 提供更稀疏、更语义化的 primitive-level anchor，适合作为 MPR 的 sparse semantic regularization。最新 2x2 消融支持这一点。

**问：RGB/GrabCut boundary snapping 是否削弱 pure-map claim？**  
答：它不是额外 foundation model，也不是 official SAM RGB decoder，而是 GT-free lightweight support calibration。主文可以同时报告 strict pure one-map ablation 和 best support-calibrated result，术语上不要说成完全 no-RGB。

**问：storage 为什么看起来没有 20 倍？**  
答：需要区分 latent payload、feature-memory package 和 full checkpoint。纯 per-Gaussian latent 相比 1280-D fp16 feature 是 20 倍压缩；package 中 decoder/refiner 是固定开销，场景越大越被摊薄；full checkpoint 还包含 3DGS 几何/RGB，不应单独用于衡量 feature compression。

---

## 十二、投稿前最终工作计划

### 12.1 论文包最终审阅

1. 固定当前 TPAMI 主线配置，避免继续大范围试模块。
2. 最终通读主文 PDF，确认 14 页主文中的图表顺序、跨页浮动和 caption 表达。
3. 核对所有大图的显示质量，尤其是 LERF 2D+3D OVS、ScanNet binary query、Direct3D support calibration 三张定性图。
4. 准备作者、基金、致谢、数据/代码/模型许可证、AI/tool disclosure 等投稿系统必填信息。
5. 根据最终投稿路径确认是否需要匿名版或非匿名版 PDF。

### 12.2 可复现与提交材料

1. 保持 `final_rows.yaml`、主文表格和 reproducibility package 三者一致。
2. 上传或归档 large-asset manifest 中列出的 checkpoint、mask/logit、qualitative source assets。
3. 投稿前运行 LaTeX 编译、claim validator、final-row registry validator、checksum 和 `git diff --check`。
4. 准备 cover letter，突出 compact reconstructive RADIO Gaussian feature memory、可追溯的三任务本地结果、storage/efficiency 和 feature usability 证据链；统一协议完成前不写三任务最优。

### 12.3 写作计划

建议按 TPAMI 标准推进：

- 第 1 周：完成 Introduction、Related Work、Method 初稿；
- 第 2 周：完成 Experiments、Ablation、Discussion 初稿；
- 第 3 周：统一图表、术语、引用和 appendix；
- 第 4 周：内部 review，压缩篇幅，准备投稿版本。

---

## 十三、总结

本项目目前已经从一个“将 RADIO 特征蒸馏到 3DGS”的工程方向，发展为一个更完整的顶刊级研究故事：

> 学习一个紧凑、可重建、可多协议查询的三维基础特征 Gaussian memory，使三维场景表示具备二维 rendered grounding、三维 primitive selection、点云开放词汇查询和多类 frozen-head 下游任务能力，同时显著降低高维 feature storage。

当前最重要的不是继续无限扩展模块，而是把主线叙事、图表、术语和实验组织收束到这一核心贡献上。只要最终稿避免过度声称、明确协议边界，并用现有三类 benchmark 与 feature usability 结果支撑“又小又好”的 compact scene memory claim，本文已经具备冲击顶级视觉期刊的基础。
