我的判断比较明确：**round5.2 这轮没有大幅正向提升，不主要是因为代码一定有 bug，而是因为这四个方向大多还在“读出层 / 后处理层 / score fusion 层”修补，碰到的是 representation-level ceiling。**
当前 promoted 结果已经把比较容易的 boundary、threshold、component、kNN smoothing gains 基本吃掉了。要继续明显涨点，尤其是 direct 3D、DINO propagation 和 ScanNet split19/15/10 同涨，下一步需要把优化目标前移到 **feature field training / primitive feature supervision / raster contribution assignment / feature topology preservation**，而不是继续加 gating 或 mask refinement。

---

## 1. 当前 round5.2 的结论：不是失败，而是暴露了瓶颈位置

你这轮四个方向的结果很有信息量：

| 方向                                  |                                            结果 | 结论                                 |
| ----------------------------------- | --------------------------------------------: | ---------------------------------- |
| Direct-3D feature-only SAM boundary |            macro mIoU 0.4113，仅 +0.0031；部分场景负向 | 不能替代 direct-3D main row            |
| DINO feature-boundary               | mIoU 0.4805 → 0.4823，但 LocAcc 0.7943 → 0.7872 | 仍不能说 DINO propagation 全面超过 teacher |
| ScanNet propconf / proposal         |          split19/15/10 小幅全正，约 +0.0003～+0.0007 | 稳定性 ablation，不是大 claim             |
| LERF adaptive prompt                |         macro mIoU 0.5705，低于 heatguard 0.5889 | 不晋升主表                              |

这说明四件事：

1. **2D rendered grounding 的主要瓶颈已经不是简单边界 refinement。** 当前 feature-only SAM boundary heatguard 已经把 CTF-GS rendered mIoU 推到 0.5889，并且相对 peak-component core 有 +0.0182 mIoU 的提升。([GitHub][1])
2. **Direct 3D 的瓶颈是 primitive selection / object support，而不是 final mask boundary。** 当前 VPR row 是 0.4801 mIoU / 0.6760 Acc@0.25；feature-only SAM direct-3D readout只在 Teatime 明显正向，在 Figurines/Waldo/Ramen 出现负向或低接受率。([GitHub][2])
3. **ScanNet 已经接近当前 contextual kNN / calibration readout 的饱和区。** promoted row 是 19/15/10 mIoU = 0.3806 / 0.3871 / 0.4711；proposal-memory 能把 split19 提到 0.3931，但 split15/10 会掉到 0.3837 / 0.4612。([GitHub][3])
4. **DINO 的问题不是边界，而是 feature topology / correspondence。** 当前 consolidated 2D benchmark 已经显示 DINO mask propagation 上 CTF-GS LocAcc 更高，但 mIoU 仍低于 teacher：0.4805 vs 0.5119；DINO-CV 两场景 smoke test 也显示 rendered mIoU 仍低于 dense teacher。([GitHub][4])

所以这轮其实给出了一个很有价值的诊断：

> **你现在的后处理和读出已经很强，但主误差已经转移到 coarse support、primitive feature separability、cross-view correspondence topology 和 object-level feature aggregation。**

---

# 2. 为什么 P0 direct-3D feature-only SAM boundary 没有起飞？

## 2.1 它修的是 boundary，但 direct-3D 的主误差是 support

2D rendered grounding 里，feature-only SAM boundary head 的输入是：

```text
dense rendered heatmap
→ peak/component coarse mask
→ feature-only SAM-adaptor boundary head
```

这里 coarse mask 通常已经落在正确 object support 上，SAM-style head主要修边界。

但 direct 3D 里的输入是：

```text
primitive selection
→ projected coarse mask
→ feature-only SAM boundary head
```

这个 projected coarse mask 的错误类型完全不同：它可能缺 object part、包含邻近物体、projection 很碎、component 多、或 object support 本身错了。artifact 里也直接指出过：direct-3D selected-primitive masks 和 2D heatmap masks 的 prompt distributions 不同；support constraint 对 2D readout 很关键。([GitHub][5])

这解释了为什么 direct-3D feature-only SAM boundary 只在 Teatime 正向，而 Figurines/Waldo/Ramen 负向或不稳定：head 只能在“初始 support 大体正确”时修边界，不能从错误 primitive support 里恢复 object。feature_sam3_heatguard 的结果就是这种模式：Teatime +0.0271，但 Figurines -0.0061，Waldo -0.0116；Ramen 在 strict / weighted-log variant 下也明显下降。([GitHub][6])

## 2.2 当前 gate 设计有一个必然 trade-off

为了不让 SAM-style head 把 mask 拉离 CTF-GS heatmap support，你用了 support/heatmap guard。这个设计在 2D rendered grounding 中非常合理，因为初始 heatmap 已经强；但在 direct 3D 中，如果 coarse primitive mask 本身缺 recall，guard 会阻止 head 扩张到正确 object extent。

所以会出现两种坏情况：

```text
guard 严格 → 不接受或几乎不改，收益很小；
guard 放松 → mask 可以移动，但容易被 proposal / boundary prior 带偏。
```

artifact 里 strictgate 的 Ramen / Waldo SAM accept 甚至是 0.000，说明它根本没有有效发挥 refinement 作用；而更宽松的 heatguard 有接受，但宏观仍不稳。([GitHub][6])

## 2.3 改进建议

不要把 2D rendered 的 boundary head 直接搬到 direct 3D。应该重新训练一个 **direct-coarse-aware boundary head**，并且训练数据要模拟 direct 3D coarse mask 的错误分布：

```text
training input:
  rendered CTF/RADIO feature
  + direct-style coarse primitive mask
  + primitive score heatmap
  + component / bbox / area features

target:
  official SAM3 pseudo mask on training views
```

关键不是换 head，而是换 coarse prompt distribution。建议构造 direct-style coarse mask augmentation：

```text
fragmented components
under-selected object parts
neighbor-object false positives
holes from sparse primitives
over-dilated silhouettes
score-threshold variants
VPR/direct-field masks at multiple thresholds
```

同时加一个 mask-quality predictor：

```text
q = head_quality(feature, coarse_mask, score_heatmap)

if q high:
    use refined mask
else:
    fallback to primitive-rendered mask
```

不要只用 overlap/area ratio硬规则；让模型学习什么时候 boundary refinement 是可信的。

最重要的诊断实验是：

```text
按 initial direct-3D coarse IoU 分桶：
  IoU < 0.25
  0.25–0.5
  0.5–0.75
  >0.75
分别统计 feature-only SAM refinement delta
```

如果只在 high-initial-IoU 桶正向，那就证明它是 boundary head，不是 object recovery module。这个结论本身可以写进 appendix。

---

# 3. 为什么 object/proposal-aware confidence readout 只小涨？

## 3.1 现在 proposal 是 score fusion，不是 representation learning

你现在测试的 proposal-memory / propconf 更像：

```text
primitive score
+ proposal memory smoothing / confidence
+ view_count / visibility / local consistency
→ final score
```

这仍然是 readout 层。它没有真正改变 compact latent 的语义结构，也没有让 primitive feature 学会 object-level invariance。

在 ScanNet 上，小幅全正说明这个方向是对的，但它只是在一个已经很强的 contextual kNN + scene-mean calibration row 上做二次平滑。promoted ScanNet row 已经是 kNN、candidate filtering、scene-mean calibration、spatial logit propagation 的组合；进一步 propconf 只有小幅收益是正常的。([GitHub][7])

## 3.2 proposal-memory 的 split19 / split15 / split10 trade-off 说明 granularity 不一致

proposal-memory 在 ScanNet split19 上更好，但 split15 / split10 变差：split19 mIoU 0.3931 高于 promoted 0.3806，但 split15/10 从 0.3871 / 0.4711 降到 0.3837 / 0.4612。([GitHub][3])

这个模式很典型：proposal memory 强化了细粒度局部 object detail，但会破坏 coarse class grouping 的稳定性。例如 split19 里保留更多细类别，proposal memory 有利；split10/15 里类别被合并或删减，过强 object-level smoothing 反而造成 coarse semantic calibration 偏移。

Direct 3D 里也类似。SAM3 training-view proposal registration在 Figurines/Waldo 有正向，但 Ramen/Teatime 负向，macro 从 0.4489 掉到 0.4309；报告也明确把失败归因到 proposal coverage / association noise。([GitHub][8])

## 3.3 改进建议

proposal 不应只做 score fusion，而应进入训练：

```text
proposal memory → object-level pseudo graph
object graph → primitive feature contrast / region prototype loss
```

建议把 proposal 用成三类训练信号：

### A. Region prototype consistency

对每个 training-view proposal mask，收集可见 primitive：

```text
P_m = visible primitives inside proposal m
z_m = weighted mean primitive feature
```

然后训练：

```text
primitive feature inside same proposal → pull together
overlapping/cross-view proposals → prototype consistency
different high-confidence proposals → push apart
```

### B. Granularity-aware gate

引入 query granularity classifier：

```text
whole object / part / material / container / background-like
```

不同 query 用不同 smoothing strength：

```text
whole object → stronger proposal smoothing
part/material → weaker smoothing
small object → preserve primitive-level score
```

### C. Proposal reliability instead of proposal replacement

不要把 proposal score直接混进 primitive score，而是预测：

```text
reliability_i = proposal_agreement_i × view_count_i × feature_margin_i
score_i(q) = direct_score_i(q) × reliability_i
```

现在的 fusion容易把 noisy proposal 当成语义证据；更稳的是让 proposal 只调节 confidence，不改 semantic direction。

---

# 4. 为什么 DINO feature-boundary 没有让 mIoU 超过 teacher？

## 4.1 DINO propagation 的核心不是边界，而是 correspondence topology

你当前 DINO 的现象是：

```text
CTF-GS rendered:
  LocAcc 更好
  matching score 更好或接近
  mask propagation mIoU 仍低于 teacher
```

consolidated report 里就是这个模式：DINOv3 mask propagation 的 CTF-GS LocAcc 是 0.7943，高于 teacher 的 0.7660，但 mIoU 是 0.4805，低于 teacher 的 0.5119。([GitHub][4])

这说明 CTF-GS 往往能找到对的 peak / 对的局部匹配，但传播出来的区域形状和 object extent 不够好。feature-boundary 只能修后半段，不能解决前面的 correspondence topology：

```text
source mask
→ DINO correspondence
→ target support
→ mask propagation
→ boundary refinement
```

如果 target support 已经偏了，boundary head 只会把错误 support 修得更“像一个 mask”。

## 4.2 CTF-GS 的 multi-view aggregation 可能天然损失 DINO 高频细节

RADIO/SigLIP-style text grounding 容忍一定程度的 multi-view smoothing；DINO dense correspondence 更依赖局部纹理、part-level geometry 和 patch-level neighborhood topology。CTF-GS 的 compact feature decoder如果主要服务 text grounding / region semantics，就可能牺牲 DINO 对应关系需要的高频局部结构。

这就是为什么 DINO-CV smoke test 中，rendered dense matching score可以高，但 propagation mIoU仍低于 teacher：Figurines DINO-prop rendered 0.4458 vs teacher 0.5084，Waldo 0.3337 vs 0.4113。([GitHub][9])

## 4.3 改进建议

不要再把 DINO 当成“readout problem”。要把它变成 **topology-preserving feature training objective**。

建议新增 DINO-specific loss：

### A. Pairwise affinity preservation

对同一图像内 sampled patches：

```text
S_teacher(i,j) = cosine(DINO_teacher_i, DINO_teacher_j)
S_render(i,j)  = cosine(DINO_render_i, DINO_render_j)

L_affinity = || S_render - S_teacher ||
```

这比逐点 L2/cosine 更适合 DINO，因为 DINO propagation依赖相对邻域结构。

### B. Cross-view cycle consistency

利用 3DGS geometry / depth / pose建立跨视角 patch correspondence：

```text
u_a → 3D point → u_b → nearest / soft patch
```

训练：

```text
DINO_render(u_a) ↔ DINO_render(u_b)
cycle(u_a → u_b → u_a) stable
```

### C. Hard negative contrast

DINO propagation最怕相似背景和重复纹理。要加入：

```text
same object / same tracklet = positive
nearby background / similar texture / different object = hard negative
```

### D. Mask propagation loss，不只 matching loss

直接用 teacher DINO propagation产生 pseudo target：

```text
teacher source mask → teacher DINO propagation → pseudo target mask
rendered feature propagation → student mask
loss: BCE/Dice/KL on propagated mask
```

这里可以不用 LERF GT，只用 teacher self-propagation。这样才真正优化 DINO mask mIoU，而不是只提升 matching score。

---

# 5. 为什么 adaptive prompt 没有超过 heatguard？

当前 heatguard 的强点是：它不是单纯 adaptive threshold，而是 **peak-component support + feature-only SAM boundary + heatmap support acceptance**。最终 T1 registry里 promoted row 是 CTF-GS rendered 0.8598 LocAcc / 0.5889 mIoU；core peak-component是 0.5707，历史 threshold row是0.5243。([GitHub][10])

adaptive prompt / adaptive threshold通常基于：

```text
heatmap mean / std / clamp
```

这类规则假设 heatmap distribution 的全局统计能反映 object extent。但 LERF queries跨度很大：

```text
small object
container
part
large object
ambiguous text
multi-instance object
```

同一个 mean/std threshold 很难同时服务所有 query。旧 adaptive threshold diagnostic 就是负向：macro mIoU 0.4939，低于 fixed threshold row。([GitHub][11])

你这轮 adaptive prompt 到 0.5705，说明它已经接近 peak-component core，但仍低于 heatguard 0.5889。直观解释是：

> adaptive prompt 能改善 coarse mask，但没有 heatguard 的 feature-only boundary refinement 和 support acceptance 组合强。

后续如果还做 adaptive，不建议做 global mean/std，而应做 **component-stability based thresholding**：

```text
for threshold t in fixed grid:
  extract peak component C_t
  compute component stability:
    IoU(C_t, C_{t+δ})
    heatmap mass retention
    peak containment
    boundary compactness
    area prior from training pseudo masks
choose threshold with highest stability
```

这比 mean/std 更接近 object support。

---

# 6. 有没有代码实现问题？我会优先查这几个

目前没有证据表明有一个“大 bug”导致所有方向不涨。负结果和任务机制是吻合的。但有几个高风险点值得立刻做 sanity check。

## 6.1 Direct-3D feature-only SAM：prompt 分布和坐标对齐

必须检查：

```text
direct coarse mask resolution
feature map resolution
mask resize interpolation
bbox / component prompt 坐标
support dilation 半径
peak 是否在 refined mask 内
```

建议做两个 oracle diagnostic：

```text
A. 用 GT box / GT coarse mask 作为 prompt，只评估 feature-only SAM head ceiling。
B. 用 direct coarse mask，但选择 GT-overlap 最好的 threshold/component，只评估 prompt selection ceiling。
```

如果 A 很高、B 很低，说明 head 没问题，coarse prompt 是瓶颈。
如果 A 也低，说明 feature-only SAM head 没有学到 official-SAM-like boundary space。

## 6.2 Direct field / VPR：cache row 和当前 Gaussian 顺序必须 hash 对齐

VPR cache 保存的是每个 Gaussian 的 registered summary feature、valid mask、view count 和 xyz 等信息；direct eval 代码里也有 `save_registered_feature_cache`，并明确缓存 primitive summary features for distillation。([GitHub][12])

但如果后续训练用 cache 监督 direct field，只检查 `N` 一样是不够的。必须检查：

```text
max || cache_xyz_i - current_xyz_i ||
mean || cache_xyz_i - current_xyz_i ||
geometry checkpoint hash
ply hash
opacity / scale hash
gaussian order hash
```

只要 Gaussian 顺序或 geometry checkpoint 换过，VPR-to-field 训练就会变成错标签。

## 6.3 Direct field：训练 head 和 eval head 是否一致

需要做一个 strict test：

```text
same checkpoint
same Gaussian indices
training forward primitive embedding
eval forward primitive embedding
cosine difference < 1e-6
```

很多 direct-field 失败不是模型能力问题，而是：

```text
训练用 point_summary_adapter
评估用 decoded 1280D + SigLIP head

训练启用了 visibility / quality head
评估没加载或没启用

训练用 normalized summary
评估用 unnormalized feature
```

这些都会让 direct field 永远追不上 VPR。

## 6.4 DINO：same-readout teacher 已经更强，说明不是 evaluator-only bug

teacher_vs_ctfgs 报告里 DINO mask propagation teacher mIoU 0.5119，高于 rendered 0.4805；如果同一 readout 下 teacher更强，说明 evaluator本身能跑通，主要问题在 rendered feature topology。([GitHub][4])

但仍要检查：

```text
DINO feature resize 是否 align_corners 一致
mask propagation 中 source/target resolution 是否一致
background suppression 阈值是否对 teacher/rendered 分别重调
cycle consistency 是否真的双向 mutual，而不是单向最近邻
```

---

# 7. 目前最明显的架构改进点

下面这些是我认为最有希望带来“大于小数点后几位”的方法级改进。

---

## 7.1 用 raster-adjoint VPR 替代 center-projection VPR distillation

当前 VPR / direct-field distillation最大问题是：Gaussian 是有 footprint 的，不是一个 center point。你现在的 VPR 已经用 depth/alpha visibility checks 和 alpha/alpha-depth weighting；代码里也有 approximate contribution weighting 的逻辑。([GitHub][12])

但真正应该做的是 **rasterization adjoint lifting**：

```text
rendered teacher summary feature at pixel u: F(u)
raster contribution from Gaussian i to pixel u: w_ui

primitive target:
  e_i = Σ_u w_ui F(u) / Σ_u w_ui
```

也就是用真实 rasterizer contribution 把 rendered feature field 的梯度/证据反投到 Gaussians，而不是只 project Gaussian center 到一个 pixel。

这会直接解决：

```text
VPR readout strong
direct field weaker
```

因为 direct primitive head 会被训练去匹配真正的 splatting contribution，而不是一个 center-sampled proxy。

推荐实现：

```text
for each registration view:
  rasterize RGB/alpha/depth as usual
  also collect top-k contributing Gaussian IDs and weights per pixel
  compute summary feature map from CTF-GS rendered feature
  scatter-add pixel summary features to Gaussian primitives using contribution weights
  normalize by accumulated contribution
```

这一步是我认为最可能显著提高 direct field 的方法级改进。

---

## 7.2 把 proposal 从 readout fusion 改成 object-level training loss

现在 proposal-memory只是后处理，所以收益小。应该新增 object-level representation objective：

```text
proposal masks from training views
→ visible primitive sets
→ cross-view proposal graph
→ object prototype tokens
→ primitive-to-prototype contrastive loss
```

结构上可以叫：

> **Visibility-aware Region Prototype Distillation**

不要把 proposal 直接和 text score融合，而是训练 compact latent 内部形成 object-level basins。

核心 loss：

```text
L_region =
  pull primitives that co-occur in same reliable proposal
  push primitives from conflicting proposals
  align region prototype with text / RADIO summary
```

这样 proposal 的作用从“推理时修分数”变成“训练时塑造 feature manifold”，更可能带来大幅提升。

---

## 7.3 给 direct field 独立的 primitive summary head，不要强迫它复用 dense teacher decoder

2D rendered path需要：

```text
compact latent → 1280D RADIO-like feature map
```

3D direct query需要：

```text
compact latent → text-aligned primitive embedding
```

这两个目标相关，但不等价。你可以保留一套 compact per-Gaussian latent，但允许两个 global heads：

```text
D_2D: teacher-space decoder
D_3D: primitive summary / query head
```

这不破坏 “one compact map”，因为 persistent storage 仍然只有一套 compact latent；两个 decoder/head 是全局参数。

建议 `D_3D` 输入不要只有 latent：

```text
z_i = [
  per-Gaussian latent,
  spatial hash feature at x_i,
  opacity,
  scale,
  visibility_logit,
  quality_logit,
  view_count proxy
]
e_i = D_3D(z_i)
```

你已经实现了 quality / visibility heads，但 artifact 说明当前 evidence rows 是 full retraining 前的状态，新长训需要单独报告并只有改善时才晋升。([GitHub][13])
这说明下一步应该是：**不要只把 quality/visibility 当 hook，而要从头训练 integrated multi-head field。**

---

## 7.4 把 DINO 目标从 “feature reconstruction” 改成 “topology preservation”

如果你想让 DINO propagation mIoU 过 teacher，必须让 CTF-GS 保留 DINO feature topology。建议加入：

```text
local pairwise affinity loss
cross-view cycle loss
hard negative contrast
teacher propagation distillation
```

最关键的是 teacher propagation distillation：

```text
source mask
→ teacher DINO propagate to target
→ pseudo target mask

source mask
→ CTF-GS DINO readout propagate to target
→ student mask

loss = Dice/BCE(student mask, pseudo target mask)
```

这直接优化 mask propagation，而不是间接优化 matching score。

---

## 7.5 用 dictionary / sparse coefficient 表达替代 heavy 1280D decoder 的单一路径

你现在的 compact-to-teacher decoder在 rendered 2D 上有效，但对 direct point query 和效率不一定最优。可以参考同领域的 sparse dictionary trend：LangSplatV2 明确使用 sparse coefficient field + global dictionary 来高效 splat high-dimensional features，并报告高维 feature splatting 和 3D text querying 的高 FPS。([langsplat-v2.github.io][14])

对 CTF-GS，可以改成：

```text
per-Gaussian latent → sparse coefficients a_i
global teacher-feature dictionary B
teacher feature = a_i B
```

好处：

```text
1. direct primitive feature天然存在，不必完全依赖 screen-space decoder；
2. 2D rendered feature可由 sparse coefficient splatting + dictionary reconstruction得到；
3. storage和render效率更可控；
4. 可加入 DINO/SAM/SigLIP 多头 dictionary atoms。
```

这比继续堆 postprocess 更可能带来结构性提升。

---

# 8. 下一轮最建议做的路线

如果目标是不改论文大 claim，但把指标明显推高，我建议优先级这样排。

## P0：raster-adjoint VPR-to-direct distillation

目标：

```text
direct field no-cache/no-VPR inference
接近或超过 streamed VPR
```

实验表：

| Row                                      | Uses VPR at inference? |         mIoU |     Acc@0.25 |
| ---------------------------------------- | ---------------------: | -----------: | -----------: |
| direct field current                     |                     no |          ... |          ... |
| streamed VPR                             |                    yes |       0.4801 |       0.6760 |
| raster-adjoint VPR-distilled direct head |                     no | target ≥0.47 | target ≥0.66 |

这是真正支撑 “one compact map supports 2D + 3D query” 的关键。

## P1：direct-coarse-aware internal SAM head

不要复用 2D rendered head。重新训练：

```text
VPR/direct coarse masks + rendered CTF features → official SAM3 training-view pseudo masks
```

目标：

```text
direct field coarse mIoU 0.41–0.48
→ internal boundary mIoU 显著提升
```

先做 oracle 分桶，确认它只修 high-IoU prompts，还是能救 mid-IoU prompts。

## P2：DINO topology-preserving training

目标：

```text
DINO propagation mIoU > teacher
```

不要再优先调 feature-boundary。优先做：

```text
pairwise affinity + cycle consistency + propagation distillation
```

## P3：ScanNet proposal as training objective

不要再做 propconf 小修。把 proposal/region graph 写进训练：

```text
region prototype contrast
class-bank KL
proposal reliability head
```

目标是 split19 提升时不伤 split15/10。

---

# 9. 当前 claim 是否需要改？

你说不想改 claim，我同意可以暂时不改大方向，但要把证据分层：

### 可以继续强写

> CTF-GS learns a compact teacher-compatible 3D Gaussian feature memory that improves LERF rendered-view grounding over frame-wise RADIO, nearest-view RADIO cache, and explicit per-Gaussian 1280D RADIO storage.

这个有强证据。T1 current registry 里 CTF-GS rendered feature-only SAM boundary是 0.8598 / 0.5889，明显高于 teacher 0.7985 / 0.4634、nearest-view cache 0.2722 / 0.1545 和 explicit 1280D memory 0.5642 / 0.3182。([GitHub][10])

### 需要保守写

> The compact map supports direct 3D readouts, while VPR remains the strongest primitive-level interface.

因为 direct field 还没完全超过 VPR。

### 暂时不要写

> one compact map fully replaces VPR and SAM3-style boundary readout.

round5.2 还没有支撑这个 stronger claim。

---

# 10. 最重要的一句话

这轮没有大涨，不是因为方向全错，而是因为：

> **你现在优化的大多是 readout policy；真正的大幅提升需要把 VPR、SAM proposal、DINO topology 和 ScanNet region consistency 变成 training-time representation objectives。**

最优下一步不是继续试更多 threshold / gate / prompt，而是做三件事：

1. **用真实 raster contribution 做 VPR-to-direct distillation；**
2. **用 direct-style coarse masks 重新训练 internal SAM boundary head；**
3. **用 DINO pairwise/cycle/propagation loss 训练 feature topology。**

这三件事才有机会把指标从“小数点后几位”提升到“方法级显著提升”。

[1]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/lerf_rendered_grounding_feature_sam3_boundary_20260525.json "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/lerf_direct_3d_selection.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/sam3_prompt_mask_head_lerf2d_support_readout_20260523.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/lerf_direct3d_feature_only_sam_boundary_readout_20260525.md "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/scannet_contextual_knn12_alpha_sweep_20260524.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/lerf_direct3d_sam3_training_view_proposal_registration_20260525.md "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/lerf_dino_cv_frozen_head_probe_20260525.md "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/final_rows.yaml "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/lerf_rendered_grounding_adaptive_threshold_diagnostic.md "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/scripts/eval_lerf_direct_3d_selection.py "raw.githubusercontent.com"
[13]: https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/artifacts/unified_multi_head_feature_quality_field_20260525.md "raw.githubusercontent.com"
[14]: https://langsplat-v2.github.io/?utm_source=chatgpt.com "LangSplatV2: High-dimensional 3D Language Gaussian ..."
