# 项目总结与优化

**User:** Anonymous  
**Created:** 5/8/2026 13:39:01  
**Updated:** 5/12/2026 3:07:38  
**Exported:** 5/12/2026 4:29:56  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fd76f5-1e00-83ea-956c-f27ef66858cf](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fd76f5-1e00-83ea-956c-f27ef66858cf)  

## Prompt:
2026/5/12 02:59:49

按照你的建议进行了一波推进，包括方法改进和实验验证，git仓库已更新，总结如下：
已按“Dr. Splat-style registration”方向做完一轮方法级升级和验证，核心结果已经写回主线文档和 LaTeX 草稿。

主要改动：
- 在 [eval_lerf_direct_3d_selection.py](/root/RADIO-GS/radio_gs/scripts/eval_lerf_direct_3d_selection.py) 增加 `--score_source registered_view`：
  渲染 VFA-refined RADIO feature → SigLIP2 projection → 用 depth/alpha 可见性把特征注册回 Gaussian primitives → 在 3D primitive 上做文本查询。
- 新增测试 [test_lerf_direct_3d_selection.py](/root/RADIO-GS/tests/test_lerf_direct_3d_selection.py)，覆盖 registration frame 选择、text-space scoring、unregistered Gaussian fallback。
- 更新主报告、表格和论文草稿：
  - [lerf_direct_3d_selection.md](/root/RADIO-GS/output/radio_gs/reports/lerf_direct_3d_selection.md)
  - [lerf_direct_3d_selection_table.tex](/root/RADIO-GS/paper/lerf_direct_3d_selection_table.tex)
  - [radio_gs_draft.tex](/root/RADIO-GS/paper/radio_gs_draft.tex)
  - [PROJECT_MAINLINE.md](/root/RADIO-GS/docs/PROJECT_MAINLINE.md)
  - [submission_status.md](/root/RADIO-GS/docs/submission_status.md)

关键指标提升：
- 原始 Gaussian-center direct readout：macro mIoU `0.0804`，Acc@0.25 `0.0932`
- 新 registered-view + softmax_scene + 24 views 固定协议：macro mIoU `0.3421`，Acc@0.25 `0.5547`
- best-by-scene diagnostic：macro mIoU `0.3619`，Acc@0.25 `0.6100`
- OpenGaussian 官方参考：macro mIoU `0.3836`，Acc@0.25 `0.5143`

结论：direct 3D object selection 已经从“明显短板/limitation”提升到“可作为 dual-readout 证据”的水平，Acc@0.25 已超过 OpenGaussian 官方宏平均，mIoU 接近但仍受 Waldo Kitchen 拖累。Waldo 48-view softmax probe 可到 `0.1756`，仍低于 OpenGaussian `0.2270`，后续主要瓶颈是该场景的 object fragmentation / registration coverage。

验证已跑：
- `pytest tests/test_lerf_direct_3d_selection.py tests/test_direct_point_supervision.py -q`：`28 passed`
- `py_compile` 覆盖新增/修改的 evaluator 和 report builder：通过
- GPU4/5 当前已空闲。
从论文投稿角度是否完整？是否还能继续完善改进？不管是从claim论证还是方法抑或是精度指标

## Response:
2026/5/12 02:59:51

我会先把这次 registered-view direct 3D 提升重新定性：它已足够支撑 dual-readout 证据，但仍不能把 Acc@0.25 优势写成全面 SOTA。

## Response:
2026/5/12 03:00:07

接下来重点不是证明“能进主实验”，而是收束命名、公平协议和消融闭环；同时清理 draft 中仍把 direct 3D 写成负结果的旧表述。

## Response:
2026/5/12 03:07:38

结论：**现在已经可以按“dual-readout CTF-GS”论文推进，但还不能按“primitive-level SOTA”论文推进。** 这轮 registered-view readout 把项目从“direct 3D 是 limitation”推进到了“direct 3D 是有效证据”，这是质变；但 mIoU 仍低于 OpenGaussian official context，且若纳入 2025 InstanceGaussian 这类工作，SOTA 说法仍不成立。更稳的投稿定位是：

> **CTF-GS learns a compact teacher-compatible feature field that supports two complementary readouts: dense rendered-view feature maps for 2D grounding, and registered primitive-level features for direct 3D object selection.**

中文：

> **CTF-GS 学习一个 compact teacher-compatible feature field，既能渲染 dense feature map 做 2D grounding，也能通过 view-to-primitive registration 得到 primitive-level feature 做 direct 3D object selection。**

这已经比原先 PPT 里“reusable foundation-feature memory”的说法更扎实、更能对接 2025 年工作。旧 PPT 的主线还停留在 LERF rendered grounding、ScanNet point query 和 HCD/FDH/refiner 消融闭环；现在 direct 3D selection 已经补进主线，论文结构应该随之升级。`radio_gs_submission_presentation.pptx`

---

## 1. 投稿完整性判断

我会这样判断当前状态：

| 路线 | 当前完整度 | 是否建议推进 |
|---|---:|---|
| **保守 CTF-GS：compact teacher feature field + rendered LERF + ScanNet point query** | 很完整，约 92-95% | 可以推进 |
| **dual-readout CTF-GS：再加入 registered direct 3D object selection** | 基本完整，约 88-92% | 建议推进，但要补协议和消融 |
| **primitive-level SOTA paper** | 还不够 | 除非继续提升 Waldo/mIoU 并补同协议 baseline |

你仓库的 submission status 也把当前项目评为 conservative route 约 92%，strict dual-readout / primitive-level CTF-GS 约 87%，并明确说 registered direct-selection upgrade 后 2025-style primitive-level paper 已经 credible，但还需要围绕 Waldo Kitchen 和 protocol provenance 谨慎 framing。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md))

现在最大的好消息是：direct 3D object selection 已经从 Gaussian-center fixed result 的 **0.0804 mIoU / 0.0932 Acc@0.25**，提升到 registered softmax24 fixed protocol 的 **0.3421 mIoU / 0.5547 Acc@0.25**。这说明之前的问题不是 teacher feature 本身完全不可用，而是 Gaussian-center readout 太弱；通过 VFA-refined rendered feature 注册回 primitives，primitive-level readout 变得可用。仓库 mainline 也把这个实验定义为 OpenGaussian-style query-select-render protocol，并记录 registered softmax24 的完整四场景结果。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

---

## 2. 现在能 claim 什么，不能 claim 什么

### 可以强 claim

**第一，CTF-GS 支持 dual-readout。**

这轮结果足以支撑：

> **The same compact teacher feature field supports rendered-view dense feature readout and registered primitive-level direct 3D readout.**

注意这里是 **registered primitive-level readout**，不是 raw Gaussian-center readout。方法描述必须明确：direct 3D selection 的强结果来自 `registered_view`，即先渲染 VFA-refined RADIO features，再投影到 SigLIP2 space，并用 depth/alpha 可见性注册回 Gaussian primitives。仓库文档也明确把这个过程写成 rendered SigLIP2-aligned features registered back to visible primitives with depth/alpha checks。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

**第二，registered readout 让方法进入 OpenGaussian-style 3D object selection 对话。**

OpenGaussian 官方 README 对 LERF object selection 的流程是先渲染 text-selected 3D Gaussians 到多视角图像，再计算评估指标；它也提醒 LERF 样本有限，指标可能不稳定。([GitHub](https://github.com/yanmin-wu/OpenGaussian)) 你们现在的 evaluator 正好对齐这个 query-select-render-evaluate 接口。

**第三，Acc@0.25 可以说超过 OpenGaussian official macro reference。**

按你们当前表，registered softmax24 fixed protocol 是 **0.5547 Acc@0.25**，OpenGaussian official context 是 **0.5143**。这个可以写，但必须加限定：

> compared with OpenGaussian official-source reference under an OpenGaussian-style protocol, not a locally rerun same-evaluator reproduction.

### 不能强 claim

**第一，不能 claim direct 3D object selection SOTA。**

因为 mIoU 仍是 **0.3421**，低于 OpenGaussian official **0.3836**。如果纳入 InstanceGaussian 2025，它在 LERF 3D object selection and rendering 表里报告 **45.30 mIoU / 58.44 mAcc**，也高于你们当前的 34.21 / 55.47。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_InstanceGaussian_Appearance-Semantic_Joint_Gaussian_Representation_for_3D_Instance-Level_Perception_CVPR_2025_paper.pdf)) 所以现在更准确的表述是：

> **competitive direct 3D object selection with higher Acc@0.25 than OpenGaussian official reference, but lower mIoU and still behind recent instance-aware methods.**

**第二，不能说 raw hybrid feature field 本身天然适合 3D primitive selection。**

现在结果证明的是：

> **HGCF + CTR + VFA 的 rendered features 可以通过 view-to-primitive registration 转化为有效 primitive features。**

不是：

> Gaussian-center compact feature without registration already solves 3D selection.

这点非常重要，因为原始 center readout 只有 0.0804 mIoU。

**第三，不能说 student feature 全面强于 original RADIO。**

仍然只能说：在同一 frozen SigLIP2 evaluator 下，rendered features 改善 LERF localization stability。仓库 status 记录的 teacher-vs-rendered 是 RADIO RGB macro LocAcc 0.7985 vs rendered 0.8712，mIoU 0.4922 vs 0.4941。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md)) 这不是“全面强于 teacher”，而是“多视角 3D 蒸馏提升 task-relevant usability”。

---

## 3. 方法主线应该怎么升级

我建议把新增模块正式命名为：

> **VPR: View-to-Primitive Registration**

不要叫 Dr. Splat-style registration。论文里可以说 inspired by direct registration methods，但模块名要体现你自己的方法：**从 rendered teacher-compatible feature 到 primitive-level feature 的注册桥梁**。

最终方法模块建议如下：

| 代码/旧名 | 投稿名 | 作用 |
|---|---|---|
| Hybrid Gaussian feature field | **HGCF: Hybrid Gaussian Code Field** | compact code storage |
| HCD codec | **CTR: Compact-to-Teacher Reconstruction** | compact code → RADIO teacher space |
| Screen-space refiner | **VFA: View-Space Feature Aligner** | 修正 rendered feature 的 peak/boundary |
| FDH warm-start | **FGC: Frozen Geometry-Head Consistency** | 几何感知 frozen-head regularization |
| registered_view direct readout | **VPR: View-to-Primitive Registration** | rendered feature → primitive feature → direct 3D selection |

这样主线非常清楚：

1. **HGCF** 解决高维 teacher feature 的 compact storage。
2. **CTR** 解决 compact code 到 teacher-compatible feature space 的恢复。
3. **VFA / FGC** 提升 rendered-view feature quality。
4. **VPR** 把 view-space feature quality 转化为 primitive-level direct query ability。

Dr. Splat 直接把 CLIP embeddings 分配到 pixel-ray 上的 dominant Gaussians，以绕过 rendering stage；它的核心是 direct language embedding registration。([arXiv](https://arxiv.org/html/2502.16652v1)) 你们的方法不完全一样：你们不是绕过 rendered feature，而是利用已经优化好的 rendered teacher-compatible feature，再通过 depth/alpha visibility 注册回 primitives。这个差异要主动写清，否则审稿人会说这是 Dr. Splat 的弱改版。

推荐写法：

> **Unlike pure direct-registration methods that attach language embeddings to Gaussians, VPR uses the reconstructed teacher-compatible rendered feature as an intermediate consistency signal and registers it back to visible primitives. This bridges the rendered-view strength of CTF-GS and the primitive-level query protocol.**

---

## 4. 论文题目是否要改？

现在可以考虑把 “dual-readout” 写进标题，但我仍建议标题不要太激进。

最稳标题：

> **CTF-GS: Compact Teacher Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding**

如果你想突出这轮新增贡献，可以改成：

> **CTF-GS: Compact Teacher Feature Fields with View-to-Primitive Registration for Open-Vocabulary 3D Gaussian Scene Understanding**

这个标题更长，但更准确：它把 direct 3D selection 的提升归因到 **View-to-Primitive Registration**，而不是暗示原始 compact field 自然强。

我不建议标题写：

> Dual-Readout Teacher Feature Fields ...

除非正文里有足够空间解释 dual-readout 的 registered nature。否则读者可能默认 direct readout 是 raw Gaussian feature，而这和实验事实不一致。

---

## 5. 实验结果怎么放最合理

建议主实验章节改成下面顺序。

### 5.1 LERF rendered-view open-vocabulary grounding

这是主结果，保留：

| Metric | Macro |
|---|---:|
| LocAcc | 0.8712 |
| mIoU | 0.4941 |

它证明 CTF-GS 的核心 teacher-feature reconstruction 有效。

### 5.2 Teacher vs rendered feature usability

保留这个表，因为它是你们和普通 language-GS 最大的差异：

| Source | Macro LocAcc | Macro mIoU |
|---|---:|---:|
| Original RADIO RGB | 0.7985 | 0.4922 |
| CTF-GS rendered | 0.8712 | 0.4941 |

结论写成：

> 3D distillation sharpens task-relevant localization responses under the same frozen evaluator.

### 5.3 LERF direct 3D object selection

新增主表，但表标题要准确：

> **OpenGaussian-style Direct 3D Object Selection**

表里必须分清：

| Method | Text head | Protocol | mIoU | Acc@0.25 |
|---|---|---|---:|---:|
| OpenGaussian | CLIP | official reference | 0.3836 | 0.5143 |
| CTF-GS center readout | SigLIP2 | fixed local | 0.0804 | 0.0932 |
| CTF-GS + VPR | SigLIP2 | fixed softmax24 | 0.3421 | 0.5547 |
| CTF-GS + VPR | SigLIP2 | best-by-scene diagnostic | 0.3619 | 0.6100 |

主文只放 fixed softmax24，best-by-scene 放 appendix。仓库 mainline 也是这个边界：fixed protocol 是当前最强主结果，best-by-scene 是 diagnostic。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

### 5.4 ScanNet direct point query

保留作为另一个 3D readout 证据。这里证明 feature field 不只是 LERF object selection，也能做 point-level class query。

### 5.5 Ablation

现在必须新增 VPR 相关消融，否则 direct 3D selection 会显得像工程 hack。

建议至少放这几个：

| Variant | 目的 |
|---|---|
| Gaussian-center readout | 证明 raw primitive readout 不够 |
| registered cosine24 | 证明 registration 本身有效 |
| registered softmax24 | 证明 scene-softmax calibration 有效 |
| registered softmax24 w/o VFA | 证明 VFA-refined rendered feature 对 registration 有贡献 |
| registered softmax24 w/o depth/alpha visibility check | 证明 geometry-aware registration 不是普通投影 |
| registered 8/24/48 views | 证明 view count 和 coverage tradeoff |
| train-pose-only vs all-pose | 排除 test-view leakage 质疑 |

最后两项尤其关键。

---

## 6. 当前最需要补的不是新模块，而是 protocol hygiene

从投稿角度，最容易被审稿人抓的不是 0.3421 是否够高，而是 **VPR 是否公平、是否泄漏、是否把 evaluation views 用进了 registration**。

你需要在论文里明确回答：

1. **registration views 来自哪里？**  
   是 train views、all dataset poses，还是 evaluation annotated views？

2. **VFA 输入是否使用 GT RGB？**  
   如果 VFA-refined feature 依赖 observed RGB，而 registration 用了 evaluation frame RGB，那会有 test-image leakage 风险。更安全的是：registration 只使用 rendered RGB/depth/alpha 或训练视角。

3. **是否使用 LERF GT masks 或 query labels？**  
   必须明确没有。LERF object masks 只能在最后 metric stage 用。

4. **threshold / top-p 是否固定？**  
   fixed top0p02 可以做主结果；best-by-scene 只能 diagnostic。

5. **外部 baseline 是否本地复现？**  
   如果 OpenGaussian 仍是 official reference，就不能说 strict same-evaluator SOTA。

建议写一个小表：

| Protocol item | CTF-GS + VPR setting |
|---|---|
| Feature source | rendered CTF-GS feature, no teacher rerun |
| Registration views | fixed 24 views, specify train/all |
| Registration supervision | depth/alpha visibility only |
| Text head | frozen SigLIP2 |
| GT masks | metric only |
| Selection | fixed softmax_scene + top0p02 |
| External baseline | official-source reference unless locally rerun |

这个表非常值。它能让 reviewer 放心。

---

## 7. 是否还能继续提高精度？

可以，但下一步要针对 **Waldo Kitchen fragmentation / coverage**，不是泛泛调阈值。

你当前 Waldo fixed mIoU 0.1413，OpenGaussian 是 0.2270；即使 48-view softmax probe 到 0.1756，仍低于 OpenGaussian。这说明瓶颈不是简单 view count，而是 object fragmentation / registration coverage。

优先级如下。

### P0：做 coverage audit

先不要盲目训练。给每个 query 统计：

- selected Gaussians 的 registration coverage；
- GT mask 内可见 pixels 中有多少对应已注册 primitives；
- 每个 Gaussian 的 registered view count；
- selected primitives 是否集中在 object 内部还是背景；
- Waldo 中 false positives 主要来自哪个区域。

然后做一张 Waldo-specific figure：GT mask、registered primitive coverage、selected mask、false positive/false negative。这个会直接支撑 failure analysis。

### P1：geometry-aware registration 改进

现在是 depth/alpha checks。可以继续加：

- top-k contributing Gaussian accumulation，而不是 center-only；
- 使用 Gaussian projected footprint，而不是只看 center；
- alpha contribution weighted feature aggregation；
- visibility confidence normalization；
- background / low-opacity primitive filtering。

Dr. Splat 的 registration 关键就是沿 pixel-ray 把 embeddings 分配给 dominant Gaussians，而且它也讨论了每条 ray 聚合 Gaussians 数量与 mIoU / memory 的 trade-off。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf?utm_source=chatgpt.com)) 你们可以借这个方向，但要命名成自己的 VPR contribution。

### P2：object/proposal-level pooling

如果目标真是追 mIoU，proposal-level 会比 primitive-level raw threshold 更稳。CAGS 明确指出 isolated per-Gaussian feature learning 缺少 spatial context，会导致 fragmented representations；它通过 local graph、mask-centric contrastive learning 和 context propagation 来减少 fragmentation。([arXiv](https://arxiv.org/abs/2504.11893)) 你们 Waldo 的症状正是 fragmentation，所以可以补一个轻量版本：

- Gaussian kNN graph smoothing；
- registered feature graph propagation；
- SAM/SAM3 proposal pooling；
- selected primitive cluster expansion；
- connected-component cleanup。

但要小心：如果引入 SAM proposal pooling，它会成为新的方法模块，论文复杂度会明显上升。

### P3：text-query calibration

Waldo 这类厨房场景有很多相似物体，text query 容易混。可以试：

- prompt ensemble；
- query aliases；
- background negative prompts；
- score margin：object text vs background text；
- per-scene normalization，但规则必须固定，不能 per-query tuned。

### P4：压缩 registered primitive cache

如果 VPR 存的是高维 SigLIP2 primitive embedding，你的 compact storage claim 会被削弱。必须补 storage table，把 VPR cache 算进去：

$$
S_{\text{total}} = S_{\text{HGCF}} + S_{\text{CTR/VFA/FGC}} + S_{\text{VPR cache}}
$$

如果 VPR cache 太大，可以考虑：

- fp16 cache；
- PCA / low-rank projection；
- product quantization；
- proposal-level descriptor instead of per-Gaussian descriptor；
- only cache visible/semantic Gaussians。

Dr. Splat 用 PQ 来紧凑表示 embeddings，并把 compactness 作为核心贡献之一。([arXiv](https://arxiv.org/html/2502.16652v1)) 你们既然标题和贡献里有 compact，就必须把 VPR 的额外存储算清楚。

---

## 8. Claim 该怎么最终写

我建议最终 claim 改成三层。

### 主 claim

> **CTF-GS compactly distills unified teacher features into 3D Gaussian scenes and supports both rendered-view and registered primitive-level open-vocabulary scene understanding.**

中文：

> **CTF-GS 将 unified teacher features 紧凑蒸馏进 3D Gaussian scene，并同时支持 rendered-view 与 registered primitive-level 的开放词汇场景理解。**

### 机制 claim

> **Rendered-view feature quality is obtained through teacher-space reconstruction and frozen-head geometry consistency; primitive-level querying is enabled by view-to-primitive registration of refined teacher-compatible features.**

中文：

> **rendered-view feature quality 来自 teacher-space reconstruction 和 frozen geometry-head consistency；primitive-level querying 则由 refined teacher-compatible features 的 view-to-primitive registration 实现。**

### 实验 claim

> **CTF-GS achieves strong rendered-view grounding, improves localization over original RADIO features under the same frozen evaluator, supports ScanNet direct point-query, and reaches competitive OpenGaussian-style direct 3D object selection, exceeding OpenGaussian official Acc@0.25 while trailing its mIoU due to Waldo Kitchen fragmentation.**

中文：

> **CTF-GS 在 rendered-view grounding 上表现强，在同一 frozen evaluator 下提升 original RADIO 的定位表现，支持 ScanNet direct point-query，并在 OpenGaussian-style direct 3D object selection 上达到有竞争力结果：Acc@0.25 超过 OpenGaussian official reference，但 mIoU 仍因 Waldo Kitchen fragmentation 落后。**

这就是最诚实、最能打的表述。

---

## 9. 现在论文是否“完整”？

**可以完整投稿，但还需要三类收尾。**

### 必须收尾

1. **VPR protocol card**  
   说明 registration views、是否使用 GT RGB、是否使用 evaluation poses、是否使用 GT masks、threshold 是否固定。

2. **VPR ablation**  
   至少有 center → registered cosine → registered softmax → w/o VFA → w/o visibility check。

3. **VPR storage/runtime**  
   direct selection 的额外 cache 和 precompute cost 必须加入 efficiency/cost table。

4. **Waldo failure analysis**  
   用 qualitative + coverage audit 解释 mIoU 落后，而不是只在文字里说 fragmentation。

5. **External baseline provenance**  
   OpenGaussian official reference 可以放，但标题里要写 official-source；若要 strict SOTA，必须本地跑 OpenGaussian / InstanceGaussian / Dr. Splat 或采用同一个 anchor paper 的整套 baseline。

### 建议收尾

1. **train-only registration result**  
   如果 train-only 和 all-pose 差距小，主表用 train-only，更安全。  
   如果差距大，主表仍可用 fixed all-pose，但必须解释这是 transductive scene-level registration，不是 unseen-pose generalization。

2. **InstanceGaussian / Dr. Splat / CAGS context**  
   至少 related work 要写，因为 2025 相关工作已经有 primitive-level和 instance-aware Gaussian 表示。InstanceGaussian 的 LERF table 已经给出 45.30 / 58.44，不能忽略。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_InstanceGaussian_Appearance-Semantic_Joint_Gaussian_Representation_for_3D_Instance-Level_Perception_CVPR_2025_paper.pdf))

3. **主图更新**  
   framework figure 需要加入 VPR 分支：  
   rendered readout：HGCF → CTR → VFA → 2D heatmap  
   primitive readout：VFA rendered feature → VPR → primitive embeddings → query-select-render

---

## 10. 最终建议

我建议你现在采用这个投稿策略：

**题目：**

> **CTF-GS: Compact Teacher Feature Fields with View-to-Primitive Registration for Open-Vocabulary 3D Gaussian Scene Understanding**

**主线：**

> compact teacher feature distillation + dual readout.

**主结果：**

- LERF rendered-view grounding：主强项；
- teacher-vs-rendered：证明 3D distillation 提升 task-relevant feature usability；
- LERF direct 3D selection：证明 VPR 让 field 进入 primitive-level协议；
- ScanNet direct point query：证明 3D point-level transfer；
- ablation：证明 CTR / FGC / VFA / VPR 必要；
- storage/efficiency：证明 compact claim 没被 VPR cache 破坏；
- failure：Waldo fragmentation。

最后一句话可以这样写进 introduction：

> **CTF-GS is not merely a rendered language field and not merely a direct language-registration method. It first reconstructs compact teacher-compatible features in 3DGS, then uses view-to-primitive registration to turn high-quality rendered features into primitive-level queryable embeddings.**

中文：

> **CTF-GS 既不是单纯的 rendered language field，也不是单纯的 direct language-registration 方法。它先在 3DGS 中重建 compact teacher-compatible features，再通过 view-to-primitive registration 将高质量 rendered features 转化为可直接查询的 primitive-level embeddings。**

这个版本已经具备投稿完整性；继续提高精度的主要突破口是 **Waldo 的 registration coverage / object fragmentation** 和 **VPR cache 的 compactness**，不是继续微调 threshold。

---

**Sources:**

- [https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md)
- [GitHub - yanmin-wu/OpenGaussian: [NeurIPS 2024] OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding · GitHub](https://github.com/yanmin-wu/OpenGaussian)
- [InstanceGaussian: Appearance-Semantic Joint Gaussian Representation for 3D Instance-Level Perception](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_InstanceGaussian_Appearance-Semantic_Joint_Gaussian_Representation_for_3D_Instance-Level_Perception_CVPR_2025_paper.pdf)
- [Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration](https://arxiv.org/html/2502.16652v1)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)