# RADIO-GS 当前算法框架与实现细节总览

> 更新时间：2026-04-14  
> 目标：把当前会话中已经跑通并达到阶段性目标的 RADIO-GS 方法体系，整理成一份可直接阅读、可继续维护的技术文档。

## 1. 文档定位

这份文档总结的是 **RADIO 特征 + 3D Gaussian Splatting** 的当前主线框架，重点不是历史版本回顾，而是：

1. 当前到底在解决什么问题。
2. 训练和推理链路分别如何工作。
3. 每个关键模块在代码里由什么实现。
4. 目前已经验证有效的配置和工程经验是什么。

它可以视为一份“当前体系结构说明 + 方法流程说明 + 代码导航”。

---

## 2. 总体目标与任务定义

### 2.1 总体目标

RADIO-GS 的目标是：  
**把冻结的 RADIO C-RADIOv4-H 的高维空间特征（1280d）蒸馏进一个 3DGS 场景表示中，使得任意新视角都能渲染出可被下游 foundation heads/adaptors 直接消费的特征图。**

这意味着模型不是只生成 RGB，也不是只重建深度，而是要重建一种“**可迁移的 foundation feature field**”。

### 2.2 要支持的下游任务

当前主线覆盖三类任务：

1. **深度估计**（Replica `room_0`）
2. **语义分割**（Replica / LERF-OVS）
3. **Text grounding**（主要在 `lerf_ovs`）

设计原则是：  
**尽量不重新训练下游基础 head，而是让新视角渲染出的特征直接兼容 RADIO 体系已有能力。**

### 2.3 当前最重要的研究问题

当前这条线最核心的问题不是“能不能渲染出特征”，而是：

1. 渲染出来的新视角特征是否还保留 RADIO 的语义结构？
2. 这些特征做深度/分割/grounding 时是否还能达到强性能？
3. 几何深度明显优于特征深度时，能否用几何信号去反向提升特征质量？
4. 当前瓶颈究竟是几何、分辨率、特征蒸馏不足，还是 **feature/head domain gap**？

---

## 3. 当前算法框架：一句话概括

**几何由 Gaussian Splatting 提供，语义由 RADIO 教师特征蒸馏提供，最终通过 HCD codec + hybrid feature field + screen-space refiner 重建新视角 1280d foundation features，再接冻结的下游 heads/adaptors 完成任务。**

可以用下面这张抽象流程图概括：

```text
训练阶段
RGB / pose / depth / semantics
    -> 冻结 RADIO 编码器
    -> 1280d 教师特征
    -> 高斯场特征渲染（explicit 或 hybrid）
    -> compact latent / hybrid fused features
    -> HCD decoder 恢复到 1280d
    -> screen-space refiner 做像素级修正
    -> 多种监督联合优化
         - feature L2 / cosine
         - depth / geom-depth
         - segmentation
         - grounding / SigLIP alignment
         - frozen depth head supervision (FDH)

推理阶段
新视角 pose
    -> 高斯场渲染 compact feature
    -> hybrid / codec / refiner
    -> 新视角 1280d feature map
    -> 下游 task heads
         - depth
         - segmentation
         - text grounding
         - direct depth head / DM head
```

---

## 4. 当前方法的核心组成模块

## 4.1 几何骨架：3DGS

场景几何不是从零学习的，而是依赖一个已经收敛的 Gaussian scene backbone。

当前 `room_0` 主线配置中使用的是：

- `use_2dgs: false`
- `ply_path: /root/RADIO-GS/output/3dgs_models/room_0/v8_fixed_poses_3dgs/point_cloud/iteration_30000/point_cloud.ply`

它提供三类关键信息：

1. 新视角投影关系
2. 几何深度图
3. alpha / visibility / boundary cues

这部分决定了：  
**RADIO-GS 不是纯隐式神经场，而是“几何显式、特征可学习”的 3DGS 特征渲染框架。**

## 4.2 教师特征：RADIO 1280d

教师特征来自冻结的 `RADIO C-RADIOv4-H`，维度为 `1280`。

数据预处理链路是：

1. 用 `radio_gs/scripts/extract_radio_features.py` 抽取每帧 RADIO 特征。
2. 按 frame id 顺序落盘为 `.pt` 文件。
3. 训练时从 `feature_dir / val_feature_dir` 读取这些教师特征作为监督目标。

在当前 `room_0` 最优配置中：

- train split: `Sequence_1`
- val split: `Sequence_2`
- mixed split: `true`
- feature resolution: `60x80`

## 4.3 特征场主干：explicit 与 hybrid

当前代码支持两类特征场：

1. **explicit**
   - 每个 Gaussian 直接携带可学习特征
   - 渲染后直接得到 latent feature map
   - 简单稳定，但上限有限

2. **hybrid**
   - 每个 Gaussian 学习一个 per-Gaussian latent
   - 同时使用 3D spatial hash grid 查询 coarse feature
   - 最后融合 fine/coarse 两路特征

当前主线基本使用 **hybrid**。

---

## 5. Hybrid 架构的工作方式

`radio_gs/models/hybrid_gaussian.py` 是当前主线最关键的结构实现。

### 5.1 Hybrid 的三条内部路径

Hybrid 不是单一分支，而是三段式结构：

1. **Fine path**
   - 由 per-Gaussian latent 渲染到屏幕空间
   - 再由 `FineDecoder` 解码成 fine feature

2. **Coarse path**
   - 通过新视角深度/位置图把像素反投影到 3D
   - 使用 `SpatialHashField` 查询多分辨率 hash grid
   - 再由 `CoarseDecoder` 解码成 coarse feature

3. **Fusion path**
   - 用 `FusionHead` 或 `DecoupledFusionHead`
   - 自适应融合 fine / coarse
   - 输出最终的 hybrid feature map

### 5.2 当前 room_0 最优配置的结构超参

来自 `radio_gs/configs/replica_hybrid_v14_room_0_fdh_ws240_w005.yaml`：

| 项 | 当前值 |
|----|--------|
| `architecture` | `hybrid` |
| `hybrid_latent_dim` | `32` |
| `hash_levels` | `16` |
| `hash_features_per_level` | `4` |
| `hash_output_dim` | `96` |
| `fine_dim` | `96` |
| `coarse_dim` | `96` |
| `hybrid_output_dim` | `192` |
| `hybrid_decoupled_heads` | `true` |
| `hybrid_semantic_adaptor` | `true` |

### 5.3 为什么 hybrid 有优势

它把“局部纹理细节”和“场景级语义上下文”拆开处理：

1. fine path 更偏局部、视角相关、边界敏感
2. coarse path 更偏空间先验和结构一致性
3. decoupled heads 让 geometry-heavy 与 semantic-heavy 信号可以分别建模

这也是为什么 hybrid 能成为当前主线，而不是只做 explicit feature splatting。

---

## 6. HCD codec：从紧凑空间回到 1280d foundation feature

`radio_gs/models/hcd_codec.py` 负责把可学习特征场和 RADIO 教师空间连接起来。

### 6.1 它解决的问题

如果直接让高斯场学习 1280d 特征：

1. 参数量太大
2. 渲染与优化都更难
3. 容易在监督不足时学到噪声通道

所以当前做法是：

1. 场景表示只负责更紧凑的 latent / bottleneck 空间
2. 用 HCD decoder 把它恢复到 RADIO 1280d
3. 在解码后的高维空间上做监督与任务评估

### 6.2 当前设置

在当前 `room_0` best config 中：

| 项 | 当前值 |
|----|--------|
| `bottleneck_dim` | `192` |
| `dual_stream` | `true` |
| `symmetric_decoder` | `true` |

这意味着当前主线不是极限压缩，而是偏向 **保真优先** 的中等压缩率配置。

---

## 7. Screen-space refiner：像素级修正模块

`radio_gs/models/screen_refiner.py` 是当前系统里非常关键但容易被忽略的一环。

它的作用是：  
**在渲染后的屏幕空间 feature map 上，再做一次带几何/边界引导的局部修正。**

### 7.1 当前 refiner 使用的引导

在当前 best config 中，refiner 是强开启状态：

| 项 | 当前值 |
|----|--------|
| `use_refiner` | `true` |
| `refiner_hidden_dim` | `256` |
| `refiner_num_blocks` | `8` |
| `refiner_rgb_guide` | `true` |
| `refiner_depth_guide` | `true` |
| `refiner_depth_grad` | `true` |
| `refiner_alpha_guide` | `true` |
| `refiner_boundary_guide` | `true` |
| `self_guided` | `true` |

### 7.2 refiner 的意义

它在工程上主要缓解三类问题：

1. 渲染后特征的局部模糊
2. 几何边界附近的特征泄漏
3. 由 Gaussian alpha blending 带来的边缘混叠

当前经验表明：  
**refiner 对深度、语义、grounding 的定性质量都非常重要。**

---

## 8. 训练监督是如何组合的

当前训练不是单一 reconstruction loss，而是一个多目标联合优化框架。

实现主入口：`radio_gs/scripts/train_feature_field.py`

## 8.1 特征重建主损失

最基础的 feature distillation 目标包括：

1. `l2_weight`
2. `cosine_weight`
3. `adaptor_weight`

当前 best config：

| 项 | 值 |
|----|----|
| `l2_weight` | `1.0` |
| `cosine_weight` | `2.0` |
| `adaptor_weight` | `0.5` |

其中 cosine loss 在当前系统里尤其重要，因为最终目标不是逐通道数值完全一致，而是 **在 foundation feature 空间中保持方向和语义结构**。

## 8.2 结构正则项

当前主线会叠加以下结构正则：

1. `tv_weight`
2. `gradient_loss_weight`
3. `depth_guided_feature_weight`
4. `geometric_edge_loss_weight`

当前 best config：

| 项 | 值 |
|----|----|
| `tv_weight` | `0.03` |
| `gradient_loss_weight` | `0.3` |
| `depth_guided_feature_weight` | `0.2` |
| `geometric_edge_loss_weight` | `0.15` |

这些项的共同目标是：  
**让特征不仅“像教师”，还要在几何边界、平滑面和过渡区域具备更合理的空间结构。**

## 8.3 深度与语义辅助监督

当前主线在训练时直接挂了深度和语义头：

| 项 | 值 |
|----|----|
| `depth_loss_weight` | `0.2` |
| `geom_depth_loss_weight` | `0.1` |
| `seg_loss_weight` | `0.3` |
| `hybrid_semantic_aux_weight` | `0.2` |

这些监督不是为了最终“依赖这些头做结果”，而是为了在训练期把任务可用性压回到特征里。

## 8.4 Text grounding / SigLIP 对齐

当前框架里 grounding 不是一个独立支线，而是被并入整体蒸馏框架：

1. `grounding_query_loss_weight`
2. SigLIP2 feature projection
3. SigLIP2 summary-head 对齐（可选）

在当前 best room_0 config 中：

- `grounding_query_loss_weight: 0.10`
- `grounding_query_loss_downsample: 4`

在代码中对应：

- `QueryGroundingAuxLoss`
- `SigLIP2Adaptor`
- `SigLIP2SummaryHead`

它们共同起到的作用是：  
**让特征不仅保留 closed-set semantics，还保留 text-aligned open-vocabulary capability。**

---

## 9. Frozen Depth Head Supervision (FDH)：当前核心创新线

这是整个项目里最有研究含义的一条线。

### 9.1 直觉

用户提出的核心想法是：

> 3DGS 重建出来的几何深度质量，明显好于直接由新视角特征预测出来的深度；  
> 那么就可以把一个冻结的深度 head 作为“几何到特征”的蒸馏桥梁，让几何监督反向提升特征质量。

### 9.2 在代码中的实现

`train_feature_field.py` 中有一条显式的 frozen-depth-head 分支：

1. 加载一个预训练 depth head checkpoint
2. 将其参数全部冻结
3. 用 decoded feature 输入这个 frozen head
4. 用 GT depth 计算 frozen-depth loss
5. 把该损失加回主训练目标

当前 best room_0 config：

| 项 | 值 |
|----|----|
| `frozen_depth_head_weight` | `0.05` |
| `frozen_depth_head_path` | `room_0_seq1_depth_head.pth` |
| `frozen_depth_loss_type` | `scale_invariant` |
| `frozen_depth_gradient_weight` | `0.1` |

### 9.3 当前结论

FDH 确实有效，但它的主要贡献更像是：

1. 提升 rendered feature 的结构性
2. 改善 geometry-aware feature learning
3. 为 direct depth 进一步优化提供更好的底座

而不是单靠 frozen head 就直接把 direct depth 拉到最优。

---

## 10. 当前 direct depth 的关键结论：瓶颈是 domain gap

当前最关键的诊断结论已经非常明确：

1. **不是简单分辨率问题**
   - `120x160` 在 direct head 上明显更差
2. **不是 multitask frozen-head proxy 能自动解决的问题**
   - 某些 proxy 变好，但真正 direct oracle 指标反而变差
3. **主瓶颈是 feature/head domain gap**
   - 把旧 head 直接迁移到当前 feature domain，会有失配

因此，当前 direct depth 的最强路线变成：

1. 先用 FDH/feature-field 把 current-best checkpoint 做好
2. 再导出该 checkpoint 自己渲染出的 train/val features
3. 用这些 exact-domain rendered features 训练一个 **current-best exact-domain DM head**
4. 再做 direct-only 评估

这就是当前达到 `0.0361` 的那条路线。

---

## 11. Direct DM Head：为什么它现在重要

### 11.1 作用

它不属于 feature field 主干，而是一个后续诊断 / 优化模块：

1. 用 `render_codec_features.py` 导出当前 checkpoint 的 rendered features
2. 用 `pretrain_oracle_head.py` 在这些 rendered features 上训练 depth head
3. 在 `eval_rendered.py --direct_depth_only --depth_head_checkpoint ...` 上评估

### 11.2 当前状态

当前 best direct head 是：

- `output/radio_gs/oracle_heads/room_0_fdh_ws240_w005_best_dm_depth_head.pth`
- exact-domain rendered-val best epoch: `E10`
- 50-frame direct-only: **AbsRel = 0.0361**

这个结果的意义是：

1. 已优于旧 transferred DM head (`0.0406`)
2. 已优于 matched-resolution geometry (`0.0548`)
3. 当前最现实地证明了“rendered foundation features 可以支撑 geometry-level direct depth”

### 11.3 它与原始 frozen-head 理念的关系

从研究理想上说，项目更希望直接保持“预训练 frozen head 的泛化性”；  
但从当前实证结果看：

- frozen oracle head 仍然偏弱
- direct exact-domain DM head 是把 direct depth 拉到目标区间的最有效路径

所以现阶段要把它视为：

**一条重要的诊断与上界逼近路线，而不是替代整个 foundation-feature 主方法。**

---

## 12. 一次训练迭代的真实数据流

当前训练循环可以概括成：

1. 从数据集读入：
   - pose
   - RADIO reference feature
   - depth
   - semantics
   - 可选 RGB
2. 用 Gaussian feature renderer 按当前 pose 渲染 latent feature map
3. 同时得到 geometry depth / alpha / boundary guides
4. Hybrid 分支把 latent + hash-grid coarse feature 融合
5. FeatSharp / refiner 做屏幕空间修正
6. HCD decoder 把特征恢复到 1280d RADIO space
7. 计算：
   - feature reconstruction losses
   - depth losses
   - segmentation losses
   - grounding / SigLIP losses
   - frozen depth head losses
8. 反向传播，只更新允许训练的模块

### 12.1 当前真正参与更新的模块

当前主线中通常会更新：

1. Gaussian feature params / latent params
2. hybrid hash field
3. fine decoder / coarse decoder / fusion head
4. HCD codec
5. screen-space refiner
6. 训练期辅助 heads（如果配置开启）

而通常保持冻结的包括：

1. RADIO encoder
2. geometry backbone（PLY / 高斯结构）
3. frozen depth head
4. SigLIP projection / summary heads

---

## 13. 推理与评估链路

## 13.1 `eval_rendered.py`

这是最核心的综合评估入口。

它可以做：

1. rendered feature 的多任务评估
2. geometry-depth / fused-depth 评估
3. cross-domain 诊断
4. direct depth head 评估

目前项目里最重要的几种 depth mode 包括：

1. **Rendered**
2. **Fused**
3. **Geom same-res / full-res**
4. **Head @ GT feat**
5. **Head @ rendered**

## 13.2 `eval_grounding.py` / `eval_lerf_grounding.py`

主要负责 grounding 和零样本文本分割评估，尤其用于：

- `lerf_ovs` 四个场景
- 每 scene 最优温度/scale 配置探索
- 论文中的 grounding qualitative / quantitative

## 13.3 `generate_visualizations_v2.py`

负责最终 qualitative figure 生成。

当前它已经支持：

1. `feature_pca/`
2. `depth/`
3. `segmentation/`
4. `grounding/`
5. `grounding_seg/`
6. `composite/`

并且在本次会话中新增了：

- `--depth_head_checkpoint`

这样 depth 面板可以直接可视化 **latest exact-domain Direct DM Head**，不再只依赖 probe 代理。

---

## 14. 当前最优配置与结果（面向理解，而非完整实验表）

### 14.1 Room_0 depth 主线

当前 room_0 的主线配置是：

- config: `radio_gs/configs/replica_hybrid_v14_room_0_fdh_ws240_w005.yaml`
- checkpoint: `output/radio_gs/room0_hybrid_v14_fdh_ws240_w005/checkpoints/best.pth`

对应代表性结果：

| 指标 | 当前值 |
|------|--------|
| Rendered depth | `0.0562` |
| Fused depth | `0.0335` |
| Geom full-res | `0.0223` |
| Direct exact-domain DM head | `0.0361` |

### 14.2 LERF-OVS 主线

当前最强结论包括：

| 任务 | 当前结果 |
|------|----------|
| Text grounding macro LocAcc | `0.867` |
| Formal segmentation macro mIoU | `0.386` |

这说明 RADIO-GS 当前不只是一个 depth feature renderer，而是已经具备：

1. 新视角语义保真
2. 文本对齐能力迁移
3. 跨任务 foundation feature rendering 能力

---

## 15. 当前代码实现的核心文件映射

如果需要顺着代码读，建议按下表：

| 文件 | 作用 |
|------|------|
| `radio_gs/config.py` | 全局配置 schema |
| `radio_gs/scripts/train_feature_field.py` | 主训练入口 |
| `radio_gs/models/hybrid_gaussian.py` | hybrid 特征场主干 |
| `radio_gs/models/hcd_codec.py` | HCD 编解码器 |
| `radio_gs/models/screen_refiner.py` | screen-space refiner 与 guides |
| `radio_gs/rendering/feature_renderer.py` | 高斯特征渲染 |
| `radio_gs/heads/depth_head.py` | depth head / frozen depth head / DM head 结构 |
| `radio_gs/heads/segmentation_head.py` | segmentation head |
| `radio_gs/heads/grounding_head.py` | grounding 辅助头与 loss |
| `radio_gs/scripts/eval_rendered.py` | 多模式定量评估 + direct-head 诊断 |
| `radio_gs/scripts/render_codec_features.py` | 导出 rendered features 训练 exact-domain head |
| `radio_gs/scripts/pretrain_oracle_head.py` | 训练 oracle / DM depth head |
| `radio_gs/scripts/generate_visualizations_v2.py` | 论文图与定性图生成 |

---

## 16. 当前工程经验与关键实践

### 16.1 已验证有效的做法

1. `ws240` warmstart pipeline 是当前基础底座
2. `FDH w=0.05` 在 room_0 depth 上是当前最好折中
3. `60x80` 是当前更稳定的 feature-resolution
4. 强 refiner + 多 guide 信号非常关键
5. direct depth 若想接近 geometry，必须认真处理 head-domain gap

### 16.2 已验证无效或不值得继续的方向

1. 单纯把 feature resolution 提到 `120x160`
2. 只看 multitask proxy 而不看真正 direct oracle 指标
3. 假设 frozen head 一定会自然适配当前 feature domain

### 16.3 当前最稳妥的理解

当前 RADIO-GS 已经形成一个“两层结构”：

1. **主方法层**
   - Gaussian feature rendering + hybrid + HCD + refiner + FDH
2. **诊断/增强层**
   - exact-domain DM head
   - fused depth
   - direct-head qualitative / quantitative analysis

二者共同定义了当前系统的真实能力边界。

---

## 17. 当前仍然存在的限制

1. frozen pretrained head 的“零适配泛化”仍未完全实现
2. direct exact-domain DM head 仍高于 full-res geometry `0.0223`
3. LERF grounding 很强，但 segmentation 尤其在 `waldo` 上仍受 mask extent 问题限制
4. feature-domain quality 与 downstream robustness 之间仍存在 depth/semantics trade-off

---

## 18. 一句话总结

当前的 RADIO-GS 已经不是“把 RADIO 特征塞进 3DGS”这么简单，而是一个完整的 **foundation feature rendering system**：

- 用 Gaussian geometry 提供可微三维渲染骨架  
- 用 hybrid latent + hash field 提供高表达特征场  
- 用 HCD codec + refiner 恢复高保真 1280d RADIO 特征  
- 用 depth / segmentation / grounding / FDH 等联合监督维持任务可用性  
- 再用 exact-domain DM head 诊断并逼近 geometry-level direct depth

如果只看当前最核心的工程和算法事实，那么可以把它概括成：

> **一个面向新视角 foundation feature rendering 的 3DGS 蒸馏框架，并已在深度、分割、grounding 三类任务上形成了完整训练、评估和可视化闭环。**
