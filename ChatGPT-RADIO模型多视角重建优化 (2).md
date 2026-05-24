# RADIO模型多视角重建优化

**User:** Anonymous  
**Created:** 5/16/2026 19:05:44  
**Updated:** 5/22/2026 16:37:03  
**Exported:** 5/22/2026 16:37:49  
**Link:** [https://chatgpt.com/c/6a084f88-5b88-83ea-941d-2478e7835cb5](https://chatgpt.com/c/6a084f88-5b88-83ea-941d-2478e7835cb5)  

## Prompt:
2026/5/22 16:02:56

git仓库已更新，我有几个改进方向，但是结果都一般，是不是代码实现有问题，按理说不应该。帮我分析一下问题所在，并且指出如何改进下面几个部分：
1）我想把当前使用sam精修mask的方法从在 query / evaluation RGB image 上调用 frozen official SAM3 改成“重建 RADIO 特征 → 转成 SAM3 特征 → 官方 SAM3 decoder 直接出 mask。这样sam mask精修的能力就属于方法本身，但实验结果比较差
2）Direct field 的3d query 能力仍然没有超过VPR/registered readout ，无法实现”one compact foundation-feature Gaussian map supports both 2D rendered query and 3D primitive query“

## Response:
2026/5/22 16:02:58

我会先核对最新代码中 SAM3 feature-to-mask、VPR-to-field/direct field 的训练与评估链路，再区分实现 bug、目标错配和设计上限。重点会放在可操作修复方案上。

## Response:
2026/5/22 16:03:02

我已经看到两个更具体的断点：官方 SAM3 强结果来自 RGB image-state 路径，而当前 RADIO→SAM3 更像 adaptor/shape bridge，不等价于 decoder 所需表征。Direct field 也已有提升但仍弱于 streamed VPR，说明瓶颈更可能在接口约束与监督目标。

## Response:
2026/5/22 16:06:11

> **思考SAM与RADIO适配器特征映射**
>
> 需要提到SAM是否需要提示。根据搜索结果，某些资料表明，图像嵌入、提示编码器和掩膜解码器约在50ms内通过提示预测。我不需要过度依赖现有的库来进行网络搜索，更多的是思考可能的代码问题。关于SAM3特征到解码器，RADIO适配器和官方SAM3解码器可能有所不同。
>
> **检查适配器与解码器的兼容性问题**
>
> 当前的适配器损失使用了规范化的投影特征和扁平化的锚点，但它与解码器不兼容，因此表现较差。一个可能的代码问题是，我们可能提供了NCHW格式的输出特征图，而官方解码器期望的是NHWC格式。在训练过程中，检查状态注入是否正确至关重要。同时，要验证解码路径中的“boxes_xyxy”和“normalized center-size”的一致性。
>
> **考虑渲染与一致性约束的改进方案**
>
> 目前，不仅仅是一个目标，它在查询时间内使用了完整的渲染和细化。为了优于VPR，直接场景学生必须将基础特征和所有可见性上下文压缩成单一的向量。由于部分信息依赖视角，无法用静态基础向量完全匹配。为了解决视透明度和标签模糊问题，可以采用“渲染与基础特征双重一致性”的方法来改进。
>
> **考虑特征压缩和POOLING方法的改进方案**
>
> 如果想要一个统一特征，可以考虑使用相同的紧凑代码并通过两个头部来处理。如果严格要求“单一表示”，则可以考虑采用多槽原始特征，而不是每个高斯分布一个向量。这样高斯分布可以覆盖多个对象语义，但会增加存储需求。
>
> **考虑硬负样本挖掘和调试问题**
>
> 直接损失从有效样本均匀采样，需要使用“硬负样本挖掘”。对于SAM特征解码器，如果结果比SAM3适配器任务差很多，可能是由于特征到解码器桥接没有训练，不是代码问题，而是管道问题。对于直接字段，若结果接近0.436，不是代码错误，反映已知差距。
>
> **检查温度/规范化及缓存问题**
>
> VPR目标是规范化总结，直接头输出可能也需规范化，否则余弦相似度会异常。有可能存在缓存问题，尤其是`direct_point_teacher_features`的点数检查，而不是坐标对齐。如果Gaussion顺序或检查点几何形状与缓存不同，可能导致不正确的直流场蒸馏结果。
>
> **检查几何更新和缓存问题**
>
> 几何如果被冻结的话，特征微调不会影响几何结构。但如果与新几何进行对比，则可能有差异。需要验证是否VPR到场训练更新了几何，还是仅更新了特征字段。同时，要确保训练和评估使用相同的特征键，以保证正确的解码路径。
>
> **解决训练检查和缓存问题**
>
> 训练过程中只做了形状检查，并且没有提及XYZ对齐问题。可以引用feature_supervision部分，在第2855-2865行进行点数检查。如果目标是保持最终地图一致，VPR缓存应用于训练，并通过归一化oracle测试来检验直接字段查询是否与VPR匹配。
>
> 已思考 10m 3s

## Response:
2026/5/22 16:13:02

我的判断：**这两个方向结果一般，不一定是代码写错；更可能是当前目标比现有训练信号强很多。** 但我确实看到几个高风险实现点，尤其是 **SAM3 decoder 输入接口不匹配** 和 **VPR cache 与 Gaussian 几何/顺序对齐不足**，这两个需要优先排查。

下面分两部分说。

---

# 1. “重建 RADIO 特征 → SAM3 特征 → 官方 SAM3 decoder 出 mask”为什么差？

## 1.1 当前 official SAM3 readout 本质上仍是 RGB-image path

你现在仓库里的 official SAM3 cache/readout 路径明确是：

```text
RGB image_path
→ Image.open(image_path).convert("RGB")
→ processor.set_image(image)
→ processor.set_text_prompt(query, state)
→ masks_logits / scores / boxes
```

代码里 `run_sam3_on_image` 就是这样做的：它读取 RGB 图像，调用 `processor.set_image(image)`，再对每个 query 调 `processor.set_text_prompt`，最后返回 `masks_logits`、`scores` 和 `boxes`。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/radio_gs/scripts/build_sam3_foundation_cache.py)) 这说明之前效果好的 SAM3-box row，本质是 **CTF-GS/VPR 给出 3D coarse evidence，official SAM3 在 evaluation RGB image 上做边界精修**。

你现在想改成：

```text
CTF-GS rendered RADIO feature
→ SAM3-adaptor feature
→ official SAM3 decoder
→ mask
```

这个想法方向是对的；如果成功，它确实属于方法本身。但问题是：**RADIO 的 SAM3 adaptor feature 并不自动等价于 official SAM3 image encoder state**。

当前项目里的 SAM3-adaptor loss 主要是 adaptor-space 约束。论文草稿明确说，SAM3 adaptor 的 region / mask-logit loss 是 teacher adaptor tokens 定义 deterministic anchors，然后匹配 soft region prototypes 或 token assignment distribution；它还明确说这不是 official frozen SAM3 mask-decoder loss，official SAM3 weights 只在 separate promptable readout/cache path 使用。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

代码里的 `radio_adaptor_loss.py` 也能看到同样逻辑：region loss 是“without an external SAM mask cache”时，用 teacher SAM3 adaptor tokens 通过 deterministic anchor tokens 定义 soft regions；mask-logit fallback 也是 teacher adaptor tokens 定义 deterministic mask anchors，然后匹配 decoded features 的 soft assignment logits。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/radio_gs/losses/radio_adaptor_loss.py))

所以目前的 SAM3-adaptor 表征更像：

> **“SAM3 teacher-space semantic / region-compatible feature”**

而不是：

> **“official SAM3 image encoder 输出的完整 decoder state”**

这两者差别很大。

---

## 1.2 这条路径差，最可能不是“精修能力不行”，而是 decoder state 不对

官方 SAM/SAM2/SAM3 类模型的 mask decoder 通常依赖：

```text
image encoder state
+ positional encoding / RoPE
+ prompt encoder tokens
+ high-resolution / multi-scale features
+ original image size / resize metadata
+ correct dtype / normalization
```

你的 official SAM3 cache 代码里甚至明确检查 resolution：当前 official SAM3 image model 预期 1008 resolution with RoPE setup，非 1008 需要显式 `--allow_unsafe_resolution`。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/radio_gs/scripts/build_sam3_foundation_cache.py)) 这意味着它对输入 state 的几何尺度、位置编码和内部布局是敏感的。

所以如果你只是做：

```text
RADIO 1280-D feature
→ linear / adaptor
→ 伪装成 SAM3 feature
→ decoder
```

通常会失败。原因不是 SAM3 decoder 不强，而是 decoder 接收到的是 **out-of-distribution image embedding**。

这类 failure 的典型表现是：

- GT box prompt 下 mask 仍然差；
- mask 边界非常碎或塌缩；
- score 高但 mask 空洞 / 偏移；
- 在 teacher RADIO feature 上也不行；
- threshold / postprocess 怎么调都救不回来。

如果出现这些现象，问题基本不在 CTF-GS，而在 **RADIO→SAM3 decoder-state bridge 没有学到 official image encoder manifold**。

---

## 1.3 需要先做四个 sanity check

这四个实验能快速判断是代码 bug 还是方法假设不成立。

### Check A：official state identity injection

先不要用 RADIO。做：

```text
RGB image
→ official SAM3 processor.set_image(image)
→ 捕获 state / image embeddings
→ 手动重新喂给 official decoder
→ mask
```

这个结果必须和原始 `processor.set_text_prompt` 路径几乎一致。如果不一致，说明你手动注入 state 的代码有 bug，例如 state 字段缺失、shape 错、dtype 错、resolution metadata 错、postprocess 错。

### Check B：teacher RADIO upper bound

做：

```text
RGB image
→ frozen RADIO teacher feature
→ RADIO-to-SAM3 bridge
→ official SAM3 decoder
→ mask
```

这里还没有 CTF-GS。如果这一步都很差，说明问题不是 3D reconstruction，而是 **RADIO/SAM3 bridge 不足以驱动 official decoder**。

### Check C：GT box prompt diagnostic

用 GT box 只做诊断：

```text
teacher RADIO / rendered RADIO → SAM3 bridge → official decoder + GT box
```

如果 GT box 下仍然差，说明 image embedding/state 有问题；如果 GT box 好、VPR box 差，说明是 coarse prompt / 3D selection 问题。

### Check D：state statistics audit

对 official SAM3 image encoder state 和 RADIO-to-SAM3 bridge state 做：

```text
per-layer mean / std / norm
per-token cosine
PCA distribution
feature resolution
channel layout
positional encoding alignment
mask-logit distribution
```

如果 mean/std/norm 差很多，official decoder 基本不可能稳定。

---

## 1.4 目前最推荐的修法：不要直接喂 official decoder，先做 internal SAM-adaptor mask head

如果你想让“mask 精修能力属于方法本身”，最稳的路线不是直接调用 official SAM3 decoder，而是新增一个 **CTF-GS internal SAM-adaptor mask head**：

```text
CTF-GS rendered RADIO feature
→ frozen / learned SAM3 adaptor projection
→ lightweight promptable mask head
→ refined mask
```

训练信号可以来自：

```text
official SAM3 RGB masks on training views
VPR / CTF-GS coarse masks as prompts
LERF labels只用于最终评估，不用于训练
```

损失建议：

```text
L_mask = BCE / Dice / focal
L_boundary = boundary / level-set / trimap loss
L_logit = official SAM3 mask-logit distillation
L_prompt = box / point prompt consistency
L_region = current SAM3-adaptor region consistency
```

这个 head 属于方法本身，因为 inference 时只用：

```text
compact map → rendered RADIO feature → SAM-adaptor mask head
```

不用再看 evaluation RGB image。

论文里可以把它叫：

> **CTF-GS-SAM adaptor readout**

不要叫 official SAM3 decoder readout。

---

## 1.5 如果坚持 official decoder，需要学的是 “SAM3 image-encoder state emulator”

如果你一定要用 official SAM3 decoder，那就不能只训练 adaptor feature，要训练完整的 state bridge：

```text
M_sam : RADIO feature map → official SAM3 image encoder state
```

训练目标不是仅仅 feature cosine，而是：

```text
L_state = 多层 image embedding / high-res feature regression
L_norm = feature norm / variance matching
L_mask = frozen SAM3 decoder output mask distillation
L_prompt = box / point / text prompt consistency
L_boundary = boundary loss
```

流程应该是：

```text
Stage 0: official RGB SAM3 upper bound
Stage 1: teacher RADIO → M_sam → frozen decoder
Stage 2: rendered CTF-GS RADIO → M_sam → frozen decoder
Stage 3: VPR coarse prompt + M_sam decoder mask
```

只有当 Stage 1 接近 official RGB SAM3，Stage 2 才有意义。否则你是在让 CTF-GS 修一个本身就没学好的 cross-model interface。

---

## 1.6 一个很具体的实现坑：box / prompt 坐标约定

你现在 direct-3D 代码里有 `mask_to_sam3_box_prompt`，注释写的是把 binary mask 转成 SAM3 normalized `[cx, cy, w, h]` box prompt；SAM3 candidate selection 明确只用 rendered prediction overlap，不用 GT。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/scripts/eval_lerf_direct_3d_selection.py))

但是 official cache 里保存的是 `boxes_xyxy`。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/radio_gs/scripts/build_sam3_foundation_cache.py))

如果你新实现里把 prompt 注入 official decoder，一定要确认 decoder 当前接口到底要：

```text
normalized cxcywh
absolute xyxy
normalized xyxy
processor-internal resized coordinates
original-image coordinates
```

这个错误非常常见，而且会造成“方法看起来完全不工作”。

---

# 2. Direct field 为什么还是没超过 VPR / registered readout？

## 2.1 这不是反常，当前结构下很正常

VPR 的强度来自它用了完整的 rendered-view measurement operator：

```text
posed views
→ render teacher-space feature map
→ VFA / screen-space refiner
→ frozen SigLIP2 head
→ depth / alpha visibility check
→ register text-aligned feature back to visible Gaussian centers
```

论文草稿现在也是这么定义 VPR 的：它从 posed views render teacher-space features，应用 VFA 和 frozen SigLIP2 head，再用 depth/alpha checks 把 text-aligned features 注册回 visible Gaussian centers。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

而 direct field 是：

```text
Gaussian center compact code
→ decode / direct adaptor
→ text query
```

它没有天然经过：

- screen-space refiner；
- multi-view aggregation；
- pixel footprint；
- alpha blending；
- depth ordering；
- visibility confidence；
- query-view boundary correction。

所以它弱于 VPR 是合理的。

你当前结果也印证了这一点：raw direct Gaussian-center scoring 几乎不可用，只有 0.0458 mIoU / 0.0707 Acc@0.25；VPR-to-field consistency 把它提升到 0.4119 / 0.5876，confidence weighting 进一步到 0.4363 / 0.6191，但 streamed registered VPR 仍是 0.4799 / 0.6760。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

这说明 direct field 不是“还是很差”，而是：

> **已经从不可用变成可用，但还没有完全吸收 VPR 的多视角、screen-space、visibility-aware 信息。**

---

## 2.2 但这里有一个很重要的代码风险：VPR cache 与当前 Gaussian 是否严格对齐？

你当前 VPR cache 会存：

```text
xyz
summary_features
valid
view_counts
metadata
```

仓库里 `save_registered_feature_cache` 明确保存了这些内容，并说明 cache 存的是 Gaussian granularity 的 SigLIP2 summary-space features，作为 direct 3D readout 的训练 target。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/scripts/eval_lerf_direct_3d_selection.py))

但是在训练 direct field 时，代码片段里对 teacher cache 的检查主要是：

```text
teacher_features.shape[0] == num_source_points
```

也就是只检查点数是否一致；如果不一致就报错。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/radio_gs/training/feature_supervision_mixin.py))

这不够。因为下面这些情况都会让 VPR target 失效：

- geometry checkpoint 换了；
- Gaussian 顺序变了；
- densification / pruning 后索引变了；
- cache 来自另一个 run；
- 同一个 scene 但 PLY / checkpoint 不一致；
- `query_gaussian_points` 使用的 `valid_indices` 和 cache row 顺序不一致。

**即使 N 一样，row alignment 也可能错。**

这个问题非常可能造成 direct field 学不到，甚至越训越差。

建议立刻加：

```python
cache_xyz = payload["xyz"].to(device)
model_xyz = model.get_xyz().detach()

max_l2 = (cache_xyz - model_xyz).norm(dim=-1).max()
mean_l2 = (cache_xyz - model_xyz).norm(dim=-1).mean()

assert max_l2 < 1e-5 or max_l2 / scene_scale < 1e-6
```

同时 cache metadata 里加入：

```text
geometry_checkpoint_path
geometry_checkpoint_sha256
ply_path
ply_sha256
num_gaussians
xyz_hash
scale_hash
opacity_hash
feature_checkpoint_hash
registration_views
registration_assignment_mode
```

如果这个 audit 失败，当前 VPR-to-field 结果不可靠。

---

## 2.3 第二个风险：训练路径和推理路径可能不是同一个 feature head

你当前 direct supervision 支持多种路径：

```text
query_compact_points
query_gaussian_points
direct_point_feature_key
point_summary_adapter
siglip_summary_head
direct_point_summary_alignment
direct_point_summary_adapter_weight
direct_point_text_loss
```

代码里 direct point loss 会根据 `direct_point_feature_key` 从 `query_compact_points` 或 `query_gaussian_points(return_aux=True)` 取 compact / aux feature，然后 decode，再可选地通过 SigLIP2 summary head 或 point summary adapter 做 summary alignment / text loss。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/radio_gs/training/feature_supervision_mixin.py))

这里最容易出现的问题是：

```text
训练时监督的是 A_siglip(D(c_i))
推理时用的是 B(c_i)

或者：

训练时用 point_summary_adapter
推理时仍然用 decoded 1280-D + SigLIP2 head

或者：

训练 checkpoint 里保存了 adapter，但 eval 脚本没有加载 / 没有启用
```

你要做一个强制一致性测试：

```text
同一个 checkpoint
同一批 Gaussian indices
训练 loop 里 forward 出来的 primitive embedding
eval loop 里 forward 出来的 primitive embedding
cosine difference < 1e-6
```

如果不一致，direct-field 结果没有意义。

---

## 2.4 第三个风险：你在用 feature cosine 学 VPR，但 direct 3D query 需要 ranking

当前 VPR-to-field loss 主要是：

```text
1 - cos(A_SigLIP2(D(c_i)), stopgrad(e_i^vpr))
+ λ L_text
```

论文草稿也是这样写的：用 cached normalized VPR summary feature 和 view-count confidence weight 训练 compact field 和 lightweight summary adaptor。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

这个 loss 能让 embedding 接近，但 direct selection 最终看的是：

```text
score_i(q) = cos(e_i, text_q)
threshold / rank / top-k
```

也就是说 direct 3D 的关键不是 average cosine，而是 **query ranking / margin / calibration**。

如果 `e_i` 和 `e_i^vpr` 的平均 cosine 已经不错，但 mIoU 仍差，说明问题不是 feature reconstruction，而是：

- 正负 query margin 不够；
- background primitives score 偏高；
- text score temperature 不同；
- threshold 不能直接沿用 VPR；
- ambiguous primitives 没有 objectness / confidence gating；
- Waldo 这类 clutter scene 的 false positive 太多。

因此 direct field 需要加 query-distribution / ranking loss，而不是只加 feature cosine。

---

# 3. 我建议如何把 direct field 改到真正支持 “one compact map, 2D+3D query”

你的最终目标应该是：

```text
persistent map:
  compact per-Gaussian latent c_i
  global decoders / heads

2D query:
  render c_i → teacher feature map → text / adaptor query

3D query:
  c_i → primitive summary head → text query
```

这仍然是 **one compact map**。两个 readout head 不算两种 stored feature representation，因为它们是全局网络参数，不是 per-scene cache。

---

## 3.1 不要强迫同一个 1280-D decoded feature 同时承担 2D pixel feature 和 3D primitive feature

我建议加一个独立但轻量的 primitive head：

```text
e_i = B(
  fine_code_i,
  coarse_code(x_i),
  opacity_i,
  log_scale_i,
  rotation_i,
  optional learned confidence
)
```

输出：

```text
e_i ∈ SigLIP2 summary space
normalized
```

这个 head 只负责 direct 3D query。2D rendered-view 仍然走：

```text
render compact latent → teacher-space decoder → RADIO/SigLIP2 feature map
```

这样不会互相伤害。

---

## 3.2 加 render-level summary consistency，这是当前最缺的约束

现在 VPR-to-field 主要是 point-level：

```text
B(c_i) ≈ e_i^vpr
```

但 direct field 最终要和 2D rendered feature统一，应该加：

```text
render(B(c_i)) ≈ A_SigLIP2(D(render(c_i)))
```

也就是：

```text
per-Gaussian primitive summary head
→ 用同一 3DGS rasterizer render 成 summary map
→ 匹配 rendered teacher-feature map 经过 SigLIP2 head 的 summary map
```

这个 loss 很重要，因为它让 primitive summary feature 在 splatting composition 下也成立，而不是只在孤立 Gaussian center 上成立。

建议 loss：

```text
L_render_summary =
  1 - cos(
    render_summary(B(c_i)),
    stopgrad(A_siglip(D(render(c_i))))
  )
```

再加 VPR cache：

```text
L_vpr_point =
  1 - cos(B(c_i), stopgrad(e_i^vpr))
```

这两个合起来才能把 VPR 的多视角 evidence 压回 compact field。

---

## 3.3 加 query-distribution distillation，而不是只对齐 feature

对一个 text bank：

```text
T = {LERF queries, ScanNet classes, background/negative nouns, object parts}
```

让 direct field 复制 VPR 的 query distribution：

```text
p_i^vpr = softmax(e_i^vpr T^T / τ)
p_i     = softmax(B(c_i) T^T / τ)

L_query = KL(p_i^vpr || p_i)
```

这比单纯 feature cosine更直接优化 direct 3D selection。

同时加 margin：

```text
L_margin = max(0, m - s_pos + s_hard_neg)
```

hard negatives 可以来自：

- 同 scene top competing text；
- nearby primitives with different VPR top label；
- high-score background primitives；
- Waldo false-positive clusters。

---

## 3.4 学一个 confidence / objectness scalar，但不要存大 cache

VPR 的 `view_counts` 和 score margin 很有用。你现在已经用 `log1p(view_counts)` 做 training weight，并把 direct field 提到 0.4363 / 0.6191。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/PROJECT_MAINLINE.md))

下一步可以让模型预测：

```text
rho_i = confidence / objectness / queryability
```

inference 时：

```text
score_i(q) = rho_i * cos(B(c_i), text_q)
```

这个 `rho_i` 可以作为：

- head 输出；
- 或者每个 Gaussian 一个 fp16 scalar。

如果严格坚持 one representation，不想存 scalar，也可以让 `rho_i = sigmoid(R(c_i))`。这样不算第二套 feature。

---

## 3.5 训练时做 “one-scene overfit” 诊断

为了判断是不是代码问题，做一个极小过拟合实验：

```text
Scene: ramen 或 figurines
Target: VPR cache
Data: all valid Gaussians
Loss: only L_vpr_point + L_query
No rendered feature loss
No SAM3
No ScanNet
Train primitive head B only
```

目标：

```text
direct field score should reproduce cached VPR score almost exactly
```

如果这都不能接近 VPR mIoU，说明：

- feature path / eval path 不一致；
- cache 和 geometry row alignment 错；
- head 容量太小；
- target normalization 或 text embedding normalization 错；
- checkpoint 没正确加载 primitive head。

如果能过拟合但 full training 不行，说明是 loss conflict、sampling、confidence weighting、negative mining 或 generalization 问题。

---

# 4. 对两个方向的具体修改建议

## 4.1 SAM3 feature-to-mask readout：建议分成三条结果线

### Line A：official SAM3 RGB readout，保留为 assisted upper bound

```text
CTF-GS/VPR coarse 3D selection
→ box / mask prompt
→ official SAM3 on RGB image
```

这个继续作为：

> **image-assisted boundary readout**

不要把它写成 pure 3D 或 pure map output。

---

### Line B：CTF-GS-SAM adaptor mask head，作为方法主线

```text
compact map
→ rendered RADIO feature
→ SAM3 adaptor feature
→ lightweight promptable mask head
```

这个才是你想要的：

> **method-internal SAM-style boundary readout**

它可以进主表，但名字要准确，不能说 official SAM3 decoder。

---

### Line C：official SAM3 decoder state emulator，作为高风险强版本

```text
compact map
→ rendered RADIO feature
→ learned SAM3 image-state emulator
→ frozen official SAM3 decoder
```

这个如果成功最漂亮，但必须通过 identity injection 和 teacher RADIO upper-bound 诊断。

---

## 4.2 Direct field：建议从 “VPR cache readout” 改成 “VPR teacher distillation”

最终论文主线最好改成：

```text
VPR is a training teacher, not a deployed representation.
```

最终 deployed map：

```text
stored:
  compact latent c_i

global:
  teacher decoder D
  primitive summary head B
  optional SAM-adaptor mask head H

not stored:
  VPR cache
  1280-D per-Gaussian teacher features
  SAM3 RGB masks
```

现在 storage table 也支持这个边界：compact checkpoint 相比 direct 1280-D fp16 feature storage 有 1.74×-4.04× saving；但如果把 optional VPR cache 持久化，saving 会变成 0.56×-0.68×，所以 VPR cache 必须报告为非 persistent artifact。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/PROJECT_MAINLINE.md))

---

# 5. 我认为当前最可能的问题排序

## SAM3 方向

| 优先级 | 问题 | 判断 |
|---:|---|---|
| P0 | 把 SAM3-adaptor token 当成 official SAM3 decoder state | **最可能** |
| P0 | 没有做 official state identity injection | **必须补** |
| P0 | resolution / RoPE / resize metadata 不匹配 | **高风险** |
| P0 | prompt box 坐标格式错 | **高风险** |
| P1 | feature norm / dtype / channel layout 错 | 高风险 |
| P1 | mask logits threshold / postprocess 错 | 中高 |
| P2 | CTF-GS rendered feature本身不够好 | 不是首要，因为 teacher-RADIO bridge 先要过关 |

## Direct field 方向

| 优先级 | 问题 | 判断 |
|---:|---|---|
| P0 | VPR cache row 和当前 Gaussian row 没有 xyz/hash 对齐检查 | **必须排查** |
| P0 | 训练用的 primitive feature head 和 eval 用的 head 不一致 | **必须排查** |
| P0 | direct loss 只优化 feature cosine，不优化 query ranking | **方法瓶颈** |
| P1 | 没有 render-level primitive-summary consistency | **方法瓶颈** |
| P1 | threshold 直接沿用 VPR，direct score calibration 不同 | 高风险 |
| P1 | invalid / low-view-count Gaussians 推理时没 gated | 高风险 |
| P2 | head 容量不足 / geometry context 没输入 | 中高 |
| P2 | Waldo clutter / multi-instance ambiguity | 真实难点，不是单纯代码问题 |

---

# 6. 当前论文 claim 应该如何调整

在这两个方向还没有完全成功之前，主 claim 不要写成：

> one compact map already fully replaces VPR and official SAM3 RGB readout.

更稳的版本是：

> CTF-GS stores a compact teacher-compatible feature map. VPR provides a strong label-free teacher for primitive-level readout. VPR-to-field distillation already closes much of the direct-3D gap, and the next step is to make the distilled primitive head the deployed 3D readout. SAM3-adaptor probes show feature compatibility, while official SAM3 RGB readout remains an assisted boundary upper bound.

如果你按上面的改法把 direct field 提到接近或超过 VPR，并把 SAM3 readout 改成 internal SAM-adaptor mask head，那么 claim 可以升级为：

> **one compact foundation-feature Gaussian map supports both rendered-view 2D query and direct 3D primitive query, with an internal SAM-style boundary readout.**

目前还差两个关键证据：

1. `CTF-GS direct primitive head` 不用 VPR inference cache，接近或超过 streamed VPR；
2. `CTF-GS SAM-adaptor mask head` 不用 evaluation RGB official SAM3 encoder，达到接近 official SAM3-box readout 的边界质量。

---

# 7. 最建议你下一轮优先做的 6 个实验

1. **SAM3 identity injection test**  
   official RGB state 手动注入 decoder，结果必须等于原始 `processor.set_image + set_text_prompt`。

2. **Teacher RADIO→SAM3 decoder upper bound**  
   不经过 CTF-GS，先看 RADIO feature 是否能驱动 official decoder。

3. **GT box prompt diagnostic**  
   如果 GT box 仍差，是 image embedding/state 问题；如果 GT box 好，是 VPR prompt 问题。

4. **VPR cache xyz/hash alignment audit**  
   cache `xyz` 和当前 `model.get_xyz()` 做 max/mean L2 检查，并加入 geometry hash。

5. **Direct head one-scene overfit**  
   只训练 primitive head 拟合 VPR cache，不加其他 loss。如果不能过拟合，优先查代码路径。

6. **Add render-level summary consistency**  
   让 primitive head 的 rendered summary map 匹配 rendered RADIO→SigLIP2 summary map，这是实现 one compact map 的关键约束。

一句话总结：

> **SAM3 失败的主要原因很可能是“adaptor feature ≠ official decoder state”；direct field 不如 VPR 的主要原因是“center feature 没有被 screen-space multi-view measurement 充分约束”。先用 identity / teacher upper-bound / geometry-alignment / one-scene-overfit 四个诊断把代码风险排掉，然后把 SAM3 做成 internal mask head，把 VPR 做成 training teacher，论文逻辑就会清楚很多。**

---

**Sources:**

- [RADIO-GS/radio_gs/scripts/build_sam3_foundation_cache.py at main · Arthurshen926/RADIO-GS · GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/radio_gs/scripts/build_sam3_foundation_cache.py)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/scripts/eval_lerf_direct_3d_selection.py)

## Prompt:
2026/5/22 16:25:03

把 compact per-Gaussian latent转换成teacher feature map的能力只能在2d 空间 渲染稠密特征图去解码吗？能否直接支持稀疏特征解码出稀疏teacher feature？

## Response:
2026/5/22 16:25:06

我会先区分“稀疏采样 teacher feature”和“直接生成 primitive-level 可查询特征”：理论上可以 sparse decode，但关键在监督定义、坐标对齐以及是否与现有 2D 渲染训练目标一致。

## Response:
2026/5/22 16:26:21

另外需要把“稀疏像素/射线特征”和“稀疏 3D primitive 特征”分开，因为它们的 target、训练约束和能支撑的 claim 不同。

## Response:
2026/5/22 16:30:01

不是只能在 2D 空间渲染稠密 feature map 后再解码。**从表示能力上讲，compact per-Gaussian latent 完全可以直接支持稀疏 feature 解码**，包括：

```text
1. 稀疏 3D Gaussian center feature
2. 稀疏 3D point / point-cloud feature
3. 稀疏 image-plane pixel / ray feature
4. 稀疏 object proposal / mask 内 feature
```

但有一个核心区别：

> **RADIO teacher feature 原生定义在 2D image plane 上，不原生定义在 3D Gaussian primitive 上。**  
> 所以 2D dense teacher feature 是直接监督；3D sparse teacher feature 是通过 visibility / projection / VPR / point aggregation 诱导出来的监督。

这也是为什么当前 dense rendered feature 表现强，而 direct Gaussian-center readout 比 VPR 弱：**dense path 的监督是直接的，sparse 3D path 的目标本身需要额外定义。**

---

## 1. 当前代码其实已经有“稀疏 feature 解码”的雏形

你当前 `HybridFeatureGaussian` 的主架构是：

```text
per-Gaussian compact latent
+ spatial hash field
+ screen-space decoder
→ high-dimensional RADIO feature
```

文件开头也明确写了当前模型把 per-Gaussian compact latent 和 multi-resolution spatial hash grid 结合，通过 screen-space decoding 重建高维 RADIO features。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/models/hybrid_gaussian.py))

但关键是：代码里已经不仅支持 dense map，也支持 point query。`SpatialHashField.forward()` 支持输入 `[N,3]` 的 3D positions，`forward_screen_space()` 则是把 dense position map 展平后查询 hash field。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/models/hybrid_gaussian.py)) 另外，`_decode_point_features()` 会把 `latent_points [N, latent_dim]` 和 `normalized_points [N,3]` reshape 成 `[N,C,1,1]`，然后调用 `decode_screen_space`，最后再取回 `[N, output_dim]`。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/models/hybrid_gaussian.py))

也就是说，从代码能力上：

```text
dense decode:
  latent_map [B, C, H, W]
  position_map [B, 3, H, W]
  → teacher feature map [B, D, H, W]

sparse decode:
  latent_points [N, C]
  point_xyz [N, 3]
  → teacher feature points [N, D]
```

是可以共用同一套 decoder 的。

---

## 2. 为什么当前 sparse direct decode 仍然弱？

原因不是“不能稀疏解码”，而是 **dense rendered target 和 sparse primitive target 不是同一个统计对象**。

### 2.1 Dense teacher feature 是 image-plane target

训练主目标通常是：

```text
render compact latent map
→ decode teacher feature map
→ match RADIO teacher feature map
```

这里的 teacher feature 是每个 pixel 上的 RADIO feature。这个监督非常明确。

### 2.2 Gaussian center feature 没有天然 teacher label

如果你直接问：

```text
Gaussian i 的 RADIO teacher feature 是什么？
```

这个问题本身没有唯一答案。一个 Gaussian 可能：

- 在多个 view 中可见；
- 在不同 view 中覆盖不同 image pixels；
- 对一个 pixel 只是 alpha-blending 中的一小部分；
- 在边界处同时参与 object 和 background；
- 与多个 Gaussians 共同合成同一个 pixel feature。

所以 dense rendered loss 只约束：

```text
sum_i alpha_i(u) * feature_i  ≈ teacher_feature(u)
```

它不唯一约束每个 `feature_i`。这就是 direct center readout 弱的根本原因。

你的论文草稿也已经反映了这个现象：raw Gaussian-center diagnostic 很弱，而 VPR-to-field consistency 可以显著提升 direct Gaussian-center readout，但仍没有完全追上 streamed VPR。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

---

## 3. 稀疏解码应该分成三种，不要混在一起

### A. 稀疏 image-plane decode

这是最接近 dense feature map 的版本。

```text
给定 camera 和一批 sampled pixels / rays
→ 只对这些 pixels 计算可见 Gaussian contributions
→ 得到 sparse latent at pixels
→ decode sparse teacher features
```

它等价于“不渲染整张 feature map，只渲染采样 pixels”。这个 target 仍然是 2D RADIO teacher feature，所以监督最干净。

适合用途：

```text
降低训练显存
加速 feature distillation
只监督 object masks / hard pixels / boundary pixels
```

这条线最安全，因为它没有改变 teacher feature 的定义。

---

### B. 稀疏 3D point decode

这是对 arbitrary 3D point / ScanNet point cloud 做 feature query。

```text
给定 point x
→ 找附近 K 个 Gaussians
→ 按 Gaussian density / opacity / Mahalanobis distance 加权 latent
→ decode sparse teacher feature
```

你当前 `query_compact_points()` 就接近这个思路：它对 query points 做 KNN，结合 Gaussian density 和 opacity 得到权重，再聚合 latent 后解码。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/models/hybrid_gaussian.py))

适合用途：

```text
ScanNet point-cloud semantic segmentation
3D point query
mesh / point cloud feature transfer
```

但注意：这个 feature 的 teacher target 不是原生 RADIO feature，而是需要由多视角 teacher observation 注册到 3D point 上。

---

### C. 稀疏 Gaussian primitive decode

这是你想要的 direct 3D query：

```text
Gaussian i
→ compact latent c_i
→ direct primitive head
→ feature_i / text-aligned embedding_i
→ text query
```

这是最理想的 “one compact map supports 2D and 3D query” 路径。

但它最难，因为 primitive-level feature target 最不直接。VPR 本质上就是在构造这个 target：

```text
rendered teacher-compatible feature
→ visibility-aware registration
→ primitive pseudo-feature
```

论文草稿中 VPR 的定义就是：从 posed views 渲染 teacher-space features，经过 VFA 和 frozen SigLIP2 head，再用 depth/alpha checks 注册回 visible Gaussian centers。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

所以更准确的说法是：

> **VPR 不是最终想存的第二套特征；VPR 应该是给 sparse primitive decoder 提供 teacher target 的方式。**

---

## 4. 最推荐的最终架构：single storage, dual sparse/dense readout

你最终想要的目标可以这样定义：

```text
Persistent map:
  per-Gaussian compact latent c_i
  spatial hash field H(x)
  global decoders / heads

Not stored:
  VPR cache
  raw 1280-D per-Gaussian RADIO feature
  SAM3 RGB mask cache
```

然后有两个 readout：

### 2D dense / sparse pixel readout

```text
camera + pixels
→ splat compact latents to pixels
→ decode teacher feature
→ text query / SAM-adaptor / downstream head
```

这个可以是 dense：

```text
all pixels H×W
```

也可以是 sparse：

```text
sampled pixels / mask pixels / boundary pixels
```

### 3D sparse primitive / point readout

```text
Gaussian center or 3D point
→ compact latent + hash context + geometry context
→ primitive / point feature
→ text query
```

这条路径不需要渲染 dense feature map，但需要独立的 primitive-level supervision。

关键是：**只存一套 compact latent，不代表只能有一个 decoder head。**

你可以有：

```text
D_2D: pixel teacher-feature decoder
D_3D: primitive summary / teacher-feature decoder
```

这两个是全局网络参数，不是 per-scene feature cache。它们不会破坏 “one compact map” 的 claim。

---

## 5. 是否应该直接 sparse decode 成 1280-D teacher feature？

可以，但我建议分两层输出：

### 第一层：sparse RADIO teacher-space feature

```text
z_i = D_teacher(c_i, H(x_i), geometry_i)
z_i ∈ R^1280
```

用途：

```text
保持和 dense teacher map 同空间
支持 RADIO/SigLIP2/SAM3/DINO adaptor
做 feature reconstruction / compatibility analysis
```

### 第二层：sparse text-aligned summary feature

```text
e_i = A_summary(z_i) 或 B_direct(c_i, H(x_i), geometry_i)
e_i ∈ R^d
```

用途：

```text
直接 3D query
ScanNet class matching
LERF text-to-Gaussian selection
```

对于 3D query，**直接输出 summary/text-aligned feature 往往比强迫输出完整 1280-D teacher feature更有效**。因为 3D object selection 看的是 text ranking / margin，不是每个 teacher channel 的 L2 reconstruction。

所以推荐：

```text
2D rendered path:
  decode 1280-D RADIO feature map

3D primitive path:
  decode compact summary embedding
  optionally regularize it through 1280-D teacher-space consistency
```

---

## 6. 具体应该怎么改

### 6.1 增加明确的 sparse decoder API

现在 `_decode_point_features()` 已经有雏形，但建议把它变成 paper-facing / evaluator-facing 的清晰接口：

```python
def decode_sparse_teacher_features(
    self,
    gaussian_indices: torch.Tensor | None = None,
    points_xyz: torch.Tensor | None = None,
    mode: str = "gaussian",  # "gaussian", "point_knn", "pixel_sparse"
    k: int = 8,
    return_summary: bool = False,
    return_aux: bool = False,
) -> dict:
    """
    Returns:
      teacher_features: [N, 1280]
      summary_features: [N, D] optional
      confidence: [N] optional
      aux: geometry / weights / view_count
    """
```

并区分三个入口：

```text
decode_gaussian_features(indices)
decode_point_features(points_xyz)
decode_sparse_pixel_features(camera, pixels)
```

---

### 6.2 新增 direct primitive head，不要只复用 dense decoder

建议：

```text
input_i =
  [
    latent_i,
    hash_field(x_i),
    opacity_i,
    log_scale_i,
    view_count_i or learned confidence,
    optional geometry embedding
  ]

primitive_summary_i = B_3D(input_i)
primitive_teacher_i = D_3D(input_i)    # optional 1280-D
```

这里 `B_3D` 是 direct 3D query 的主 head。

dense rendered path 仍然使用：

```text
render latent map
+ render / unproject position map
→ D_2D
```

这样可以避免一个 decoder 同时服务两个分布不同的对象。

---

### 6.3 加 sparse-to-dense consistency，这是最关键的 loss

仅仅让 sparse primitive feature 拟合 VPR cache 还不够。应该加一个 render consistency：

```text
primitive_summary_i = B_3D(c_i)

render primitive_summary_i with same Gaussian rasterizer
→ summary_map_from_primitives

dense rendered path:
  render compact latent
  → decode RADIO feature map
  → SigLIP2 / summary head
  → summary_map_from_dense

Loss:
  summary_map_from_primitives ≈ stopgrad(summary_map_from_dense)
```

也就是：

$$
\mathcal{L}_{s2d}
=
1 - \cos
\left(
R(B_{3D}(c_i)),
\operatorname{sg}[A(D_{2D}(R(c_i)))]
\right)
$$

这能把 3D sparse feature 和 2D rendered feature 绑定起来。

这一步是实现：

> **one compact map, two readouts**

的关键。

---

### 6.4 加 VPR-to-sparse distillation，但把 VPR 只作为 teacher

当前论文已经有 VPR-to-field consistency：

```text
cached VPR primitive feature
→ supervise compact direct field
```

草稿里写的是用 cached normalized VPR summary feature 和 view-count confidence weight 监督 direct compact field，并匹配 scene-query distribution。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

建议进一步明确成：

```text
VPR cache is training-only
not deployed
not counted as storage
```

loss：

```text
L_vpr_sparse =
  w_i * (1 - cos(B_3D(c_i), stopgrad(e_i^VPR)))

L_query =
  KL(
    softmax(e_i^VPR T^T / tau),
    softmax(B_3D(c_i) T^T / tau)
  )
```

其中 `T` 是 text bank：

```text
LERF queries
ScanNet classes
background nouns
hard negative object names
```

这样直接优化 3D query ranking，而不仅是 feature cosine。

---

### 6.5 对 sparse 3D teacher feature 做 confidence modeling

不是所有 Gaussian 都应该被强监督。低 view count、边界、透明/反射、遮挡区域都很不稳定。

建议输出：

```text
rho_i = sigmoid(C(c_i, H(x_i), geometry_i))
```

query score：

```text
score_i(q) = rho_i * cosine(e_i, text_q)
```

训练：

```text
rho_i ≈ normalized VPR view_count / VPR agreement / feature margin
```

这不算第二套 feature，只是一个 scalar confidence head。

---

## 7. 你应该做的 sanity tests

### Test 1：dense decoder 和 sparse pixel decoder 一致性

从 dense rendered latent map 里随机采 pixels：

```text
dense:
  D_2D(latent_map, position_map)[:, :, y, x]

sparse:
  D_2D(latent_map[:, :, y, x].reshape(N,C,1,1),
       position_map[:, :, y, x].reshape(N,3,1,1))
```

两者应该几乎完全一致。如果不一致，说明 decoder / normalization / dtype / reshape 有 bug。

---

### Test 2：Gaussian sparse decoder overfit VPR cache

只选一个 scene，比如 `ramen`：

```text
freeze all except B_3D
target = VPR cached primitive embedding
train B_3D until train cosine nearly saturated
```

如果单 scene 都无法 overfit，说明：

```text
cache alignment
feature head
normalization
checkpoint loading
loss implementation
```

有问题。

---

### Test 3：sparse 3D feature render-back check

把 sparse primitive feature 再 render 回 image：

```text
B_3D(c_i)
→ rasterize to summary map
→ compare with VPR / dense summary map
```

如果 primitive cosine 很高但 render-back map 很差，说明单点特征没有考虑 footprint / opacity / visibility composition。

---

### Test 4：teacher feature target ambiguity audit

对每个 Gaussian：

```text
收集所有 VPR views 中它覆盖 / 投影到的 teacher features
计算方差
```

如果一个 Gaussian 的 multi-view teacher feature variance 很大，不应该强行用一个 deterministic feature 去拟合。应该降权或输出 uncertainty。

---

## 8. 最终回答

**可以直接支持稀疏 feature 解码。**

但要把问题说清楚：

```text
稠密 2D teacher feature:
  原生定义明确，直接监督，性能容易好。

稀疏 3D teacher feature:
  不是 RADIO 原生输出，需要通过 visibility / VPR / multi-view aggregation 定义。
```

你当前代码已经有 sparse point decode 的雏形；`_decode_point_features()` 会把 sparse latent 和 normalized point reshape 成 1×1 map 后调用 screen-space decoder，`query_compact_points()` 也会用 KNN / density / opacity 聚合附近 Gaussians 再解码。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/models/hybrid_gaussian.py))

但为了实现你想要的：

> **one compact foundation-feature Gaussian map supports both 2D rendered query and 3D primitive query**

我建议不要只复用 dense decoder 做 Gaussian-center decode，而是改成：

```text
single stored compact latent
+ dense 2D teacher-feature decoder
+ sparse 3D primitive summary decoder
+ sparse-to-dense consistency
+ VPR training-only distillation
```

这样最终地图仍然只存一套 compact per-Gaussian latent；VPR cache 只作为训练 teacher，不作为部署地图；2D query 走 rendered teacher feature，3D query 走 sparse primitive head。这个逻辑比“必须先渲染整张 dense map 再做 3D query”更干净，也更符合你要投稿的核心 claim。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/models/hybrid_gaussian.py)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)