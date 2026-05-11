# 项目总结与优化

**User:** Anonymous  
**Created:** 5/8/2026 13:39:01  
**Updated:** 5/12/2026 0:18:51  
**Exported:** 5/12/2026 0:30:55  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fd76f5-1e00-83ea-956c-f27ef66858cf](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fd76f5-1e00-83ea-956c-f27ef66858cf)  

## Prompt:
2026/5/11 23:24:45

更新进展（git仓库也已更新）：
已补上 **LERF direct 3D object selection**，并接入论文闭环。

主要完成项：
- 新增 evaluator：[eval_lerf_direct_3d_selection.py](/root/RADIO-GS/radio_gs/scripts/eval_lerf_direct_3d_selection.py:1)  
  按 OpenGaussian-style query-select-render：3D Gaussian center 打分、选 primitive、只为 mask evaluation 渲染。
- 新增结果表生成：[build_lerf_direct_3d_selection_table.py](/root/RADIO-GS/radio_gs/scripts/build_lerf_direct_3d_selection_table.py:1)
- 生成报告：[lerf_direct_3d_selection.md](/root/RADIO-GS/output/radio_gs/reports/lerf_direct_3d_selection.md:1)
- 生成 LaTeX 表：[lerf_direct_3d_selection_table.tex](/root/RADIO-GS/paper/lerf_direct_3d_selection_table.tex:1)
- 更新论文初稿：[radio_gs_draft.tex](/root/RADIO-GS/paper/radio_gs_draft.tex:268)，加入 direct 3D readout 方法、实验小节和 limitations。
- 更新主线与投稿状态：[PROJECT_MAINLINE.md](/root/RADIO-GS/docs/PROJECT_MAINLINE.md:1)、[submission_status.md](/root/RADIO-GS/docs/submission_status.md:1)、[final_consistency_audit.md](/root/RADIO-GS/output/radio_gs/reports/final_consistency_audit.md:1)

结果结论比较明确：实验已补齐并对齐到 OpenGaussian-style 评估接口，但当前 direct 3D selection 结果弱。RADIO-GS fixed top0p1 为 **0.0804 macro mIoU / 0.0932 Acc@0.25**，OpenGaussian official context 是 **0.3836 / 0.5143**。所以这部分现在应作为“协议已覆盖但 primitive-level readout 仍不足”的 limitation/stress test，不能作为 dual-readout SOTA 主 claim。

验证已跑：
- `py_compile` 通过。
- 表格、manifest、freeze report、consistency audit 已重建。
- direct-selection helper assertions 通过。
- `git diff --check` 通过。
- LaTeX 环境里没有 `latexmk/pdflatex/tectonic/xelatex`，所以未能实际编译 PDF；已做 `\input` 文件存在性、citation key、duplicate label 静态检查。

协议参考来源：OpenGaussian paper/code（https://arxiv.org/html/2406.02058v2 ，https://github.com/yanmin-wu/OpenGaussian）。当前 GPU 被未知进程占满，我没有强行杀进程；本轮主要完成 CPU 侧整合和验证。
本轮实验验证和论文状态收束已完成。

核心结论：LERF direct 3D object selection 协议已经对齐 OpenGaussian-style query-select-render-evaluate，但当前 RADIO-GS 的 pre-refiner Gaussian-center readout 仍弱。主结果固定 `top0p1` 为 `0.0804` macro mIoU / `0.0932` Acc@0.25；best-by-scene 也只有 `0.0914` mIoU。GPU4/GPU5 上补跑的 KNN readout、semantic/geometry head、scene-softmax、adaptor-promoted checkpoint、voxel aggregation 都没有提升 fixed macro mIoU，所以这不是简单 selector/threshold/calibration 能解决的问题。

我已把这个结论写回：
- [docs/submission_status.md](/root/RADIO-GS/docs/submission_status.md)
- [docs/PROJECT_MAINLINE.md](/root/RADIO-GS/docs/PROJECT_MAINLINE.md)
- [docs/paper_draft_current.md](/root/RADIO-GS/docs/paper_draft_current.md)
- [paper/radio_gs_draft.tex](/root/RADIO-GS/paper/radio_gs_draft.tex)
- [output/radio_gs/reports/lerf_direct_3d_selection.md](/root/RADIO-GS/output/radio_gs/reports/lerf_direct_3d_selection.md)
- [output/radio_gs/reports/submission_freeze_report.md](/root/RADIO-GS/output/radio_gs/reports/submission_freeze_report.md)

新增/完善的实验代码：
- [eval_lerf_direct_3d_selection.py](/root/RADIO-GS/radio_gs/scripts/eval_lerf_direct_3d_selection.py)
- [build_lerf_direct_3d_selection_table.py](/root/RADIO-GS/radio_gs/scripts/build_lerf_direct_3d_selection_table.py)

验证结果：
- `py_compile` 通过新增/修改脚本。
- Direct-selection report、freeze report、paper asset manifest、consistency audit 均已重新生成。
- KNN direct-readout smoke eval 在 GPU4 跑通。
- LaTeX `\input{}` 和 citation key 静态检查通过。
- `git diff --check` 通过。
- 本机未发现 `latexmk/pdflatex/tectonic/xelatex`，所以没有做完整 PDF 编译。
- GPU4/5 当前已释放。

当前投稿判断保持：保守 CTF-GS 路线约 `90%`，可以作为“compact teacher feature field + rendered LERF + ScanNet direct point-query + efficiency/provenance/ablation”的完整论文推进；如果要主打 2025 primitive-level dual-readout SOTA，direct 3D object selection 仍未闭环，下一步必须做 direct 3D language supervision 或 instance/SAM-cluster primitive aggregation，而不是继续微调阈值。

## Response:
2026/5/11 23:25:18

这次更新把 direct 3D selection 从缺失实验变成了明确的 stress test。接下来应调整 claim，保留它揭示 primitive-level 瓶颈，但避免主打 dual-readout SOTA。

## Response:
2026/5/11 23:25:37

> **Considering how to handle user’s repo update inquiry**
>
> The user didn’t explicitly ask for a web-based check, but mentioned that their GitHub repo has been updated, along with local paths. I’m contemplating whether I need to inspect the latest GitHub updates. Since no explicit web call was requested, I’ll proceed carefully to assess the situation.
>
> **Browsing GitHub for updated repo information**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md)
>
> **Considering how to address line count and raw script review**
>
> The line count seems a bit off, possibly due to compression. I’m wondering if I need to directly cite specific lines, though web citations might not be precise enough for that. I might just need to open the script directly and view the raw version for clarity.
>
> **Searching for open draft lines and citations**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/radio_gs/scripts/eval_lerf_direct_3d_selection.py)
>
> **Preparing to respond and gather references**
>
> I need to answer the user's update by reviewing current method names, lines, and relevant details. I’ll cite the repo pages and the uploaded deck. The OpenGaussian paper could also be valuable, so I might need to check for it on Arxiv and gather any updated facts needed. I'll check the web for this.
>
> **Browsing for OpenGaussian references and repo details**
>
> [OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding](https://arxiv.org/html/2406.02058v2)
> [GitHub - yanmin-wu/OpenGaussian: [NeurIPS 2024] OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding · GitHub](https://github.com/yanmin-wu/OpenGaussian)
>
> **Refining the response and title revision**
>
> I need to offer a response with concrete wording. I’m thinking a good suggestion could be something like "收到，最好这样处理" with a revised title. The dual-readout in the title feels risky, so I should go with a conservative main title. Maybe something like: "Compact Teacher Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding." Also, I need to adjust claims about the Direct 3D experiment so it focuses on stress testing, not dual-readout performance.
>
> **Refining phrasing and paper outline**
>
> I’m refining the phrasing to: "supports rendered readout and exposes direct primitive readout, but the current primitive-level selection remains weak." The weak result actually helps highlight a scientific gap, especially with primitive consistency. The paper outline needs an update, with a revised title, focusing on stress tests and clarifying limitations. Contributions should emphasize compact teacher feature distillation, hybrid code fields, and geometry-head consistency, without claiming dual-readout performance. Lastly, "best-by-scene" shouldn't be a main claim.
>
> Thought for 4m 39s

## Response:
2026/5/11 23:29:58

这个进展很有价值，但它**改变了论文策略**：现在不能主打 “dual-readout primitive-level SOTA”，而应该主打 **compact teacher feature field 在 rendered-view / point-query 任务上有效，同时 direct primitive selection 揭示了当前表示的边界**。

我建议你不要把 direct 3D selection 当成失败，而是把它当成一个非常清楚的 stress test：它补齐了和 OpenGaussian-style 2024/2025 primitive-level 工作对话的接口，也证明了当前方法的弱点不是阈值、KNN、softmax、voxel aggregation 这种后处理能解决的，而是 **Gaussian-level object-instance discrimination 不足**。你更新后的主线文档也是这个判断：保守投稿路线约 90%，但如果想做严格 dual-readout / primitive-level CTF-GS paper，则约 78%，除非进一步加入 direct 3D supervision 或 instance aggregation；direct 3D selection 当前 fixed top-10% 只有 0.0804 macro mIoU / 0.0932 Acc@0.25，对比 OpenGaussian official context 0.3836 / 0.5143，应作为 limitation/stress test，而不是主 claim。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md))

---

## 1. 当前论文主张要重新收紧

现在最稳的中心主张是：

> **CTF-GS compactly distills high-dimensional RADIO teacher features into 3D Gaussian scenes and renders teacher-compatible novel-view feature maps for open-vocabulary scene understanding. The same field exposes a direct 3D query interface, but current Gaussian-center readout is not yet sufficient for strong primitive-level object selection.**

中文：

> **CTF-GS 将高维 RADIO teacher features 紧凑蒸馏进 3D Gaussian scene，并在 novel view 渲染 teacher-compatible feature maps，用于开放词汇场景理解。该 feature field 同时暴露了 direct 3D query 接口，但当前 Gaussian-center readout 还不足以支撑强 primitive-level object selection。**

这比 “既适合 2D 又适合 3D” 更准确。

**可以说：同一个 hybrid feature field 具备 2D rendered-view readout 和 3D direct-query readout。**

但不要说：

> hybrid feature field 在 2D 和 3D object selection 上都表现强。

目前证据支持的是：

| 能力 | 当前状态 | 论文表述 |
|---|---|---|
| 2D rendered-view LERF grounding | 强 | main result |
| teacher-vs-rendered feature usability | 强，尤其 LocAcc | main evidence |
| ScanNet direct point-query | 可用 | cross-domain 3D probe |
| LERF direct 3D object selection | 弱 | stress test / limitation |
| primitive-level SOTA | 不支持 | 不 claim |

附件旧 PPT 里的主线还是 “RADIO-GS as reusable foundation-feature memory”，但你现在的更新已经把这个主线修正得更成熟：应该从泛泛的 feature memory 改成 **compact teacher feature field + rendered-feature usability + direct-3D boundary**。`radio_gs_submission_presentation.pptx`

---

## 2. direct 3D selection 应该放在论文哪里？

我建议**放进主文，但不要放在 main result 第一组**。

最好的实验顺序是：

1. **LERF rendered-view open-vocabulary localization**  
   这是主结果，macro LocAcc 0.8712 / mIoU 0.4941。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

2. **Teacher vs rendered feature usability**  
   证明 3D distillation 后的 rendered features 在同一 frozen SigLIP2 evaluator 下提升 localization stability。draft 里现在已经写了 RADIO RGB macro LocAcc 0.7985 vs CTF-GS 0.8712，mIoU 0.4922 vs 0.4941。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

3. **Core ablation + storage footprint**  
   证明 CTR/HCD、FGC/FDH、VFA/refiner、HGCF/hybrid 各自解决什么问题。draft 里现在已经把 CTR 作为最大架构依赖，w/o CTR LocAcc 从 0.858 掉到 0.531，mIoU 从 0.485 掉到 0.260；FGC 是最大训练路线增益。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

4. **ScanNet direct point-query**  
   作为 3D direct feature query 的正向证据。当前结果是 19/15/10 splits mIoU 0.3538 / 0.3573 / 0.4293。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

5. **LERF direct 3D object selection stress test**  
   放在 main paper 或 appendix 都可以；如果放主文，标题必须叫：
   > **Boundary Test: Direct 3D Object Selection on LERF-OVS**

不要叫：

> Dual-readout main result

也不要叫：

> Open-vocabulary 3D object selection SOTA

draft 现在的处理方向是对的：它明确说 direct selection 是 OpenGaussian-style query-select-render stress test，不同于 rendered-view heatmap evaluation；fixed top-10% 结果是 0.080 macro mIoU / 0.093 Acc@0.25，远低于 OpenGaussian reported 0.384 / 0.514，并且 KNN、semantic/geometry heads、scene-softmax、adaptor-promoted checkpoint、voxel aggregation 都没有改善 fixed-protocol macro mIoU。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

---

## 3. 这个弱结果怎么解释才不会伤论文？

关键是**不要让读者以为 direct 3D selection 弱说明整个方法不成立**。它说明的是另一个问题：

> **rendered dense feature usability 和 primitive-level object selection 是两个不同难度的任务。**

OpenGaussian 本身也强调了这点：它指出许多 3DGS open-vocabulary 方法主要关注 2D pixel-level parsing，但 3D point-level tasks 会受到 weak feature expressiveness 和 inaccurate 2D-3D feature association 的限制；直接在 Gaussian feature 上做 text similarity 并选择 primitives 是另一个协议。([arXiv](https://arxiv.org/html/2406.02058v2))

这正好解释为什么你的方法在 rendered-view LERF 很强，但 direct Gaussian-center selection 弱：

- 你的训练目标主要是 **rendered feature map 与 RADIO teacher feature 对齐**；
- splatting 之后的 feature map 可以通过 CTR + VFA + FGC 得到很好的 pixel-level response；
- 但 direct 3D selection 绕过了 VFA，并且直接看 Gaussian center 的 pre-refiner feature；
- Gaussian primitives 本身不是 object instances，一个物体可能由很多 Gaussians 组成，一个 Gaussian 也可能在 alpha blending 下参与多个 pixel；
- 因此 direct selection 需要 instance grouping / clustering / mask-level aggregation，而不是只靠单点 cosine similarity。

这和 OpenGaussian 的设计思路也一致：它不是简单做 Gaussian-center CLIP similarity，而是训练 3D-consistent instance features、做 coarse-to-fine codebook discretization，并通过 instance-level 2D-3D association 把 CLIP features 关联到 3D instances。([arXiv](https://arxiv.org/html/2406.02058v2))

你可以在论文里这样写：

> **This negative result clarifies the boundary of teacher-feature reconstruction. Dense rendered features are sufficient for view-space grounding, but primitive-level object selection requires object-consistent grouping or direct 3D language supervision.**

中文：

> **这个负结果界定了 teacher-feature reconstruction 的边界：dense rendered features 足以支持 view-space grounding，但 primitive-level object selection 还需要 object-consistent grouping 或 direct 3D language supervision。**

这是一个成熟、诚实、能被审稿人接受的说法。

---

## 4. 题目是否需要改？

现在我更坚定建议不要在标题里写 **Dual-Readout**。

最推荐标题仍然是：

> **Compact Teacher Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding**

这个题目是安全的。它覆盖了 rendered-view grounding、ScanNet point query、storage、teacher feature distillation，但不承诺 primitive-level object selection 很强。

不要用：

> Dual-Readout Compact Teacher Feature Fields ...

除非 direct 3D selection 后续至少接近 OpenGaussian / CAGS / Dr. Splat 这类 primitive-level 方法。

方法名仍然可以叫：

> **CTF-GS: Compact Teacher-Feature Gaussian Splatting**

在文中可以写：

> CTF-GS supports rendered-view feature rendering and exposes a direct 3D feature probe.

这里用 **exposes / probes / supports evaluation**，不要用 **excels / achieves / dual-readout superiority**。

---

## 5. Abstract 需要怎么改？

你现在 draft abstract 第三句已经写了 direct selection stress test，但最后一句 “3D Gaussian scenes can serve as reusable foundation-feature memories” 还是略泛。建议改成下面这个版本：

> We study compact teacher-feature distillation for 3D Gaussian scene understanding. Instead of training a scene-specific classifier or storing raw 1280d teacher features on every Gaussian, CTF-GS learns a Hybrid Gaussian Code Field, reconstructs teacher-compatible features through Compact-to-Teacher Reconstruction, and regularizes feature usability with view-space alignment and frozen geometry-head consistency. On LERF-OVS, CTF-GS reaches 0.8712 macro localization accuracy and 0.4941 macro mIoU for rendered-view open-vocabulary grounding, and its rendered features improve localization over original RADIO RGB features under the same frozen evaluator. On ScanNet, direct point-query evaluation supports cross-domain 3D feature usability. An OpenGaussian-style direct 3D object-selection test further reveals that pre-refiner Gaussian-center readout remains weak, indicating that primitive-level selection requires instance aggregation or direct 3D language supervision. These results position compact teacher-feature fields as effective rendered-view feature memories while clarifying the limits of unguided primitive-level readout.

这段的好处是：

- 不夸 direct 3D；
- 保留 2025 primitive-level 工作接口；
- 把 direct selection 弱结果变成 insight；
- 不说 student 全面强于 teacher；
- 不说 dual-readout SOTA。

---

## 6. Contributions 需要怎么改？

建议四条改成：

### Contribution 1：问题定义

> **We formulate compact teacher-feature distillation for queryable 3D Gaussian scene understanding.**  
> The goal is to distill high-dimensional unified teacher features into a compact 3DGS representation that renders teacher-compatible novel-view feature maps and can also be probed by direct 3D queries.

注意这里说 **can be probed by direct 3D queries**，不是 “solves direct 3D object selection”。

### Contribution 2：compact representation

> **We introduce a Hybrid Gaussian Code Field for compact teacher-feature storage.**  
> It combines per-Gaussian codes with a spatial branch, reducing the cost of direct 1280d teacher-feature storage while preserving local detail and scene-level context.

### Contribution 3：teacher-space recovery + geometry-aware feature quality

> **We propose Compact-to-Teacher Reconstruction, View-Space Feature Alignment, and Frozen Geometry-Head Consistency.**  
> These modules recover teacher-compatible rendered features and improve geometry-aware downstream usability without fine-tuning the teacher or downstream heads.

### Contribution 4：evaluation with explicit boundary

> **We evaluate both positive usability and failure boundaries.**  
> CTF-GS is evaluated on LERF rendered-view grounding, teacher-vs-rendered feature comparison, ScanNet direct point-query transfer, storage/efficiency profiling, and an OpenGaussian-style direct 3D object-selection stress test that exposes the need for instance-aware primitive aggregation.

这个 Contribution 4 很重要。它把 direct selection 弱结果变成论文透明度和科学性，而不是硬伤。

---

## 7. 是否还能声称 hybrid feature field 适合 2D 和 3D？

可以，但要换成更精确的三层说法。

### 可以安全说

> **The same compact feature field supports two readout interfaces: rendered dense feature maps for 2D view-space grounding, and direct Gaussian/point-level feature probes for 3D querying.**

中文：

> **同一个 compact feature field 支持两种读出接口：渲染 dense feature map 用于 2D view-space grounding，直接查询 Gaussian / point-level feature 用于 3D probing。**

### 可以谨慎说

> **The field is structurally compatible with both rendered-view and direct-3D protocols, but its current training objective favors rendered feature usability over primitive-level object discrimination.**

中文：

> **该 field 在结构上兼容 rendered-view 和 direct-3D 协议，但当前训练目标更偏向 rendered feature usability，而不是 primitive-level object discrimination。**

### 不要说

> Hybrid feature field is effective for both 2D and 3D object selection.

因为 LERF direct 3D selection 结果不支持这句话。

这一区分很重要：**“结构上适配”是方法属性；“性能上强”是实验结论。** 你现在只能 claim 前者和部分后者。

---

## 8. direct 3D selection 弱结果说明 HGCF 有问题吗？

不是 HGCF 整体有问题，而是说明：

> **HGCF 当前学到的是 renderable teacher feature code，不是 instance-discriminative primitive code。**

这两个目标不同。

你现在的 HGCF + CTR + VFA + FGC 是为下面这个链条优化的：

$$
\text{Gaussian codes} \rightarrow \text{splatting} \rightarrow \text{compact rendered map} \rightarrow \text{CTR} \rightarrow \text{teacher-like 2D feature map}
$$

direct 3D selection 要求的是：

$$
\text{Gaussian center feature} \rightarrow \text{text similarity} \rightarrow \text{object-consistent primitive set}
$$

中间缺了：

- instance grouping；
- same-object Gaussian consistency；
- inter-object Gaussian separation；
- mask / SAM cluster supervision；
- direct text-to-primitive calibration；
- primitive-level language loss。

所以你可以在 Discussion 里写：

> **The failure is not caused by the absence of a direct 3D readout interface; it is caused by the absence of object-level primitive grouping in the current teacher-feature distillation objective.**

中文：

> **失败不是因为没有 direct 3D readout 接口，而是因为当前 teacher-feature distillation 目标没有显式学习 object-level primitive grouping。**

这句话很关键。

---

## 9. 和 OpenGaussian 的关系要怎么写？

OpenGaussian 的协议和你的新增 evaluator 是非常好的参照，但不要让读者觉得你和它正面竞争失败了。应该这样定位：

> OpenGaussian is an instance/primitive-level language Gaussian method; CTF-GS is a compact teacher-feature reconstruction method. The direct-selection protocol is used to test whether reconstructed teacher features alone are sufficient for primitive-level object selection.

也就是说，OpenGaussian 是 **primitive-level method**，你是 **teacher-feature reconstruction method**。direct 3D selection 是用来回答：

> teacher-like feature reconstruction 是否自动带来 strong primitive selection？

现在答案是：

> 还不会。

这其实是一个明确发现。

OpenGaussian paper 本身的设置是：给定 text query，提取 CLIP text feature，与每个 Gaussian 的 language feature 做 cosine similarity，选择相关 3D points，再通过 3DGS 渲染成多视角图像，使用 average IoU 和 accuracy 与 GT object masks 评估；它也明确说这些统一设置下的指标与 comparison methods 的 official metrics 不一致。([arXiv](https://arxiv.org/html/2406.02058v2)) 这意味着你在表格 caption 里必须写清：

> **OpenGaussian official context; not reproduced under our local evaluator; CTF-GS uses SigLIP2/RADIO-compatible readout rather than CLIP Gaussian language features.**

---

## 10. 下一步如果想提升 direct 3D selection，别再调阈值了

你们已经试过 KNN、semantic/geometry heads、scene-softmax、adaptor-promoted checkpoint、voxel aggregation，都没有提高 fixed macro mIoU。主线文档也明确说这不是简单 selector / threshold / calibration 能解决的问题。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md))

下一步只值得做三类方法级改动。

### A. Instance / SAM-cluster primitive aggregation

这是最直接的路线。

流程：

1. 用训练视角 SAM / SAM3 masks；
2. 建立 Gaussian-to-mask soft assignment；
3. 对 same-object Gaussians 做 feature aggregation；
4. direct query 时先选 cluster / instance，再展开到 Gaussians；
5. 渲染 selected cluster。

这会让你接近 OpenGaussian 的核心：它用 SAM masks 训练 3D-consistent instance features，并通过 codebook / instance association 获得更强 object-level primitive selection。([arXiv](https://arxiv.org/html/2406.02058v2))

### B. Direct 3D language supervision

给 Gaussian / cluster 直接加 text-space loss，而不是只依赖 rendered RADIO reconstruction。

例如：

$$
\mathcal{L}_{3D-text}
=
1-\cos(A_{\text{SigLIP2}}(f_i), e_q)
$$

但要小心：不能用 LERF test masks 或 evaluation query labels。最好用训练视角中自动生成的 mask-caption / open-vocabulary pseudo labels，或者 scene-level category bank 做 label-free text heatmap distillation。

### C. Learned compact-to-text adapter

训练一个 small adapter：

$$
a_\phi(\hat{f}_i) \rightarrow \text{text-aligned primitive embedding}
$$

但要写清楚它不使用 evaluation masks，只用 frozen teacher/adaptor pseudo-supervision。否则会被质疑为 task-specific fitting。

如果你只想把论文稳稳投出去，**不要现在再大规模转向这些方法**。这些更像下一篇 primitive-level extension。

---

## 11. 当前稿件还需要立刻改的地方

我建议你现在把 draft 里几处 wording 再降一点。

### Abstract 最后一段

现在 draft 里有一句：

> These results support the central claim that 3D Gaussian scenes can serve as reusable foundation-feature memories...

这句可以保留，但最好改成：

> These results support compact teacher-feature fields as reusable rendered-view feature memories, while clarifying that primitive-level object selection requires additional instance-aware supervision.

这样更精确。

### Introduction contribution 1

现在写的是：

> across rendered-view localization and direct 3D point queries

这没问题。不要改成：

> across rendered-view localization and direct 3D object selection

因为 object selection 目前弱。

### Method direct 3D readout

现在 draft 写得好：direct 3D querying bypasses VFA and decodes pre-refiner compact features at Gaussian centers。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex)) 建议加一句：

> This readout is diagnostic rather than optimized for object-instance consistency.

### Experiment direct selection caption

建议 caption 写：

> **OpenGaussian-style direct 3D object-selection stress test.** The table evaluates whether reconstructed teacher-compatible Gaussian-center features alone are sufficient for primitive-level object selection. They are not yet competitive with instance-aware language Gaussian methods.

---

## 12. 最终投稿路线判断

我现在会把项目分成两条路线：

### 路线 A：保守 CTF-GS，建议推进

主张：

> compact teacher feature distillation for rendered-view and point-query scene understanding.

主结果：

- LERF rendered-view grounding；
- teacher-vs-rendered；
- component ablation；
- storage footprint；
- ScanNet direct point-query；
- direct 3D selection as stress test / limitation。

这条路线是完整论文，逻辑闭合，风险可控。

### 路线 B：primitive-level CTF-GS，不建议现在主打

主张：

> one hybrid field strongly supports both rendered-view grounding and direct 3D object selection.

目前不成立。除非加入：

- direct 3D language supervision；
- instance / SAM-cluster aggregation；
- learned primitive text adapter；
- 或 object-consistent codebook。

否则这条路线会被 OpenGaussian-style baseline 压住。

---

## 13. 最推荐的最终主线一句话

> **CTF-GS shows that compact teacher-feature reconstruction can turn 3D Gaussian scenes into strong rendered-view foundation-feature fields, while the new OpenGaussian-style direct-selection stress test reveals that primitive-level open-vocabulary selection requires additional object-consistent grouping beyond Gaussian-center teacher-feature readout.**

中文：

> **CTF-GS 表明，compact teacher-feature reconstruction 可以把 3D Gaussian scene 变成强 rendered-view foundation-feature field；同时，新增的 OpenGaussian-style direct-selection stress test 说明 primitive-level 开放词汇选择还需要 Gaussian-center teacher-feature readout 之外的 object-consistent grouping。**

这就是你现在最稳、最诚实、也最有论文价值的叙事。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md)
- [OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding](https://arxiv.org/html/2406.02058v2)

## Prompt:
2026/5/12 00:13:33

我如果就是想提高direct 3D object selection的指标达到sota，能否在保留目前的方法框架上加入你提到的几个方法级改动，从像dr.splat类的registration方法中借鉴而来？这样的话是否可以维持甚至提高本文的主要claim：提高特征质量和其他下游任务指标？

## Response:
2026/5/12 00:13:35

我会把这视为一次方法升级而非后处理：先判断能否在现有框架中加入 registration/instance aggregation 冲 SOTA，再界定它是否会改写主 claim。

## Response:
2026/5/12 00:13:57

关键新增点是：若冲 SOTA，应把它命名为独立的 object-aware/direct-3D language-instance grounding 分支，并明确它与原 teacher-feature reconstruction 主干分别贡献哪些结果。

## Response:
2026/5/12 00:18:45

可以做，而且**如果你的目标是把 direct 3D object selection 做到 SOTA，这不是继续调 selector，而是必须把当前方法升级成“teacher-feature reconstruction + object-aware direct registration”的双目标框架**。

但我要先把边界说清楚：这会改变论文主线。原来的 CTF-GS 主线是 **compact teacher feature field / rendered-view feature reconstruction**，证据强在 LERF rendered grounding、teacher-vs-rendered、ScanNet point query 和 HCD/FDH/VFA/HGCF 消融；附件 PPT 里也是这条路线。`radio_gs_submission_presentation.pptx` 现在 direct 3D selection 已经补上，但当前 fixed top0p1 只有 0.0804 macro mIoU / 0.0932 Acc@0.25，对比 OpenGaussian official context 0.3836 / 0.5143，项目文档也明确把它定性为 stress test / limitation，而不是主 claim。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md))

所以问题不是“能不能加”，而是：**加完以后，这篇论文到底还是 teacher-feature distillation paper，还是变成 object-aware Gaussian registration paper？** 我的建议是：可以加，但不要把它做成替代原方法，而要做成一个明确的新模块，服务于 direct primitive-level readout。

---

## 1. 为什么 direct 3D selection 现在弱？

当前做法是：

> pre-refiner Gaussian-center feature → SigLIP2 text similarity → 选 primitive → render binary mask → 评估。

这条链路失败，说明一个事实：

> **rendered feature quality ≠ primitive-level object discriminability。**

你现在的 HGCF + CTR + VFA + FGC 主要优化的是：

$$
\text{Gaussian codes} \rightarrow \text{splatting} \rightarrow \text{rendered feature map} \rightarrow \text{teacher-compatible feature}
$$

它学到的是“渲染后好用的 dense feature”，不是“每个 Gaussian 自己就知道自己属于哪个 object instance”。而 OpenGaussian / Dr. Splat / CAGS / ProFuse 这类方法的强项正是 **primitive-level 或 proposal-level object coherence**。

OpenGaussian 的核心动机就是：已有 3DGS open-vocabulary 方法很多停在 2D pixel-level parsing，3D point-level task 会受 weak feature expressiveness 和 inaccurate 2D-3D feature association 影响；它用 SAM masks 训练 3D-consistent instance features，再做 two-stage codebook 和 instance-level 3D-2D feature association。([arXiv](https://arxiv.org/abs/2406.02058)) Dr. Splat 则直接把 language-aligned CLIP embeddings 注册到 3D Gaussians，绕过 rendering，并用 dominant Gaussians intersected by each pixel-ray 做 language feature registration。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf)) CAGS 进一步指出 isolated per-Gaussian feature learning 缺少 spatial context，会导致 fragmented representations，因此用 local graph、mask-centric contrastive learning 和 context propagation。([arXiv](https://arxiv.org/abs/2504.11893)) ProFuse 的方向也类似：用 dense correspondence 做 cross-view grouping，构造 3D Context Proposals，再把 proposal global feature fused onto Gaussians。([chiou1203.github.io](https://chiou1203.github.io/ProFuse/))

这说明你要补的不是后处理，而是 **object-aware primitive supervision / registration**。

---

## 2. 可以保留现有框架，但要新增一个“object-aware direct branch”

我建议保留现有四个核心模块：

| 当前模块 | 继续保留的角色 |
|---|---|
| HGCF | compact Gaussian code field，仍然是主 representation |
| CTR | compact-to-teacher reconstruction，保证 teacher-space compatibility |
| VFA | 只用于 rendered-view feature alignment |
| FGC | 仍然作为 geometry-aware frozen-head regularization |

然后新增一个模块：

> **OPR: Object-aware Primitive Registration**

或者更完整：

> **OCPR: Object-Consistent Primitive Readout**

我更推荐 **OPR**，因为它简洁，而且和 direct registration 工作能自然对话。

它的作用不是替代 CTR，而是补 direct 3D selection 缺的东西：

> **OPR learns object-consistent primitive/proposal features on top of the compact teacher feature field, so direct 3D queries select coherent object regions rather than isolated high-similarity Gaussians.**

中文：

> **OPR 在 compact teacher feature field 之上学习 object-consistent primitive/proposal features，使 direct 3D query 选出连贯对象区域，而不是零散高相似 Gaussian。**

---

## 3. 具体方法怎么加

### 3.1 从 2D masks 提取 object/proposal descriptors

训练视角上跑 SAM / SAM3，得到 masks：

$$
M_{v,k}
$$

对每个 mask，从 teacher features 中池化一个 descriptor：

$$
d_{v,k} = \text{Pool}(A(T(I_v)), M_{v,k})
$$

这里 $A$ 可以是 SigLIP2 adaptor、CLIP adaptor、RADIO projection、DINO/SAM adaptor，取决于你最终想和谁对比。为了和 OpenGaussian / Dr. Splat 的 CLIP-family baseline 更容易对齐，可以保留一个 **text-aligned branch**；为了维持本文主线，最好还保留一个 **RADIO/SigLIP2 teacher-compatible branch**。

关键原则：**不能用 LERF GT masks、test query labels 或 evaluation masks 做训练**。只能用 training views 上的 automatic masks 和 frozen teacher/adaptor features。

---

### 3.2 用 visibility / dominant-ray 把 mask descriptor 注册到 Gaussians

借鉴 Dr. Splat 的思想：不要只看渲染后的 feature map，而是沿 pixel-ray 找对这个 pixel/mask 贡献最大的 Gaussians。Dr. Splat 的关键是把 CLIP embeddings 分配给每条 pixel-ray 上相交的 dominant Gaussians，并直接在 3D Gaussians 上保存 language-aligned embeddings。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf))

你的版本可以写成：

$$
w_{i,v,k} =
\sum_{p \in M_{v,k}}
\text{contrib}(g_i, p, v)
$$

其中 $\text{contrib}$ 是 Gaussian $g_i$ 对像素 $p$ 的 alpha / transmittance contribution。

然后给 Gaussian $i$ 一个 mask/proposal assignment：

$$
a_i = \arg\max_{v,k} w_{i,v,k}
$$

或者保留 soft assignment：

$$
P(i \in M_{v,k}) \propto w_{i,v,k}
$$

这一步会把你的方法从“render-supervised feature reconstruction”推进到“direct primitive registration”。

---

### 3.3 构造 3D proposals，而不是只训练单个 Gaussian

单个 Gaussian 太碎，direct selection 很难。你需要 proposal / cluster 级别的对象一致性。

可以从三种信号构造 3D proposals：

1. **mask co-assignment**：多个 Gaussians 经常被同一个 2D mask 覆盖；
2. **spatial adjacency**：3D kNN / radius graph；
3. **feature similarity**：teacher/adaptor descriptor similarity。

定义一个 proposal cluster：

$$
C_m = \{g_i \mid a_i = m\}
$$

再做 graph refinement：

$$
E_{ij}=1
\quad \text{if } g_i,g_j \text{ are spatially close and share mask/proposal evidence}
$$

这一步对应 OpenGaussian 的 3D-consistent instance features / codebook 思路，也对应 CAGS 的 local graph context propagation 和 ProFuse 的 3D Context Proposals。([3D AIGC](https://3d-aigc.github.io/OpenGaussian/))

---

### 3.4 新增 OPR head，但不要破坏 CTR

给每个 Gaussian 一个 object-aware primitive feature：

$$
o_i = H_{\text{OPR}}(c_i, \hat{f}_i, \mu_i, \text{geo}_i)
$$

其中：

- $c_i$：HGCF compact code；
- $\hat{f}_i = \text{CTR}(c_i)$：teacher-compatible feature；
- $\mu_i$：Gaussian center；
- $\text{geo}_i$：scale、opacity、normal/depth proxy 等几何属性。

训练损失：

$$
\mathcal{L}_{OPR}
=
\lambda_{\text{mask}}\mathcal{L}_{mask-align}
+
\lambda_{\text{inst}}\mathcal{L}_{instance-contrast}
+
\lambda_{\text{graph}}\mathcal{L}_{graph-smooth}
+
\lambda_{\text{sep}}\mathcal{L}_{inter-object-sep}
$$

其中：

**mask alignment**

$$
\mathcal{L}_{mask-align}
=
\sum_{i,v,k}
w_{i,v,k}
\left(1-\cos(o_i,d_{v,k})\right)
$$

**instance contrastive**

同一个 3D proposal 内的 Gaussians 靠近，不同 proposals 分开。

**graph smoothness**

空间邻近且 co-visible 的 Gaussians 平滑。

**inter-object separation**

同场景不同 proposals 的 feature 要有 margin，避免“整片桌面都像 cup”。

这样做之后 direct 3D selection 的 query 对象变成：

$$
s_i(q)=\cos(o_i, t_q)
$$

而不是原来的：

$$
s_i(q)=\cos(A(\text{CTR}(c_i)), t_q)
$$

这一步才是真正可能把 0.08 拉上去的地方。

---

## 4. 更推荐 proposal-level readout，而不是 primitive-level raw score

为了冲 SOTA，我建议不要只做 Gaussian-level threshold。应该做：

> **proposal-level scoring → select proposals → expand to member Gaussians → render binary mask**

给每个 proposal $C_m$ 一个 pooled feature：

$$
u_m = \text{Pool}_{i \in C_m}(o_i)
$$

text query score：

$$
s_m(q)=\cos(u_m,t_q)
$$

选择 top proposal 或 score-threshold proposals：

$$
\hat{C}(q)=\{C_m \mid s_m(q)>\tau\}
$$

然后渲染 proposal 内所有 Gaussians。

这会比 raw Gaussian selection 稳得多，因为 LERF direct 3D object selection 评估的是 object mask，不是孤立 primitive accuracy。OpenGaussian 的官方代码也把 LERF 评估定义为先渲染 text-selected 3D Gaussians，再计算 object-selection metrics。([GitHub](https://github.com/yanmin-wu/OpenGaussian)) Dr. Splat 的 LERF 3D object selection 也是对 selected 3D Gaussians 的 rendering 计算 2D segmentation accuracy。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf))

---

## 5. 这样还能维持“提高特征质量”的主 claim 吗？

可以，但要改成更准确的版本。

原 claim：

> 提高 teacher foundation feature 的表达能力 / 下游可用性。

加入 OPR 后，建议变成：

> **CTF-GS improves downstream usability of compact teacher features by combining teacher-space reconstruction with object-aware primitive registration.**

中文：

> **CTF-GS 通过 teacher-space reconstruction 和 object-aware primitive registration，提高 compact teacher features 在下游场景理解任务中的可用性。**

这个 claim 比“纯粹提高 teacher feature quality”更稳。因为 direct 3D selection 的提升不是单靠 feature reconstruction，而是靠 **object-level grouping / primitive registration**。

更具体：

| 任务 | 由谁主要支撑 | 能否说提高 feature quality |
|---|---|---|
| LERF rendered-view grounding | HGCF + CTR + VFA + FGC | 可以，属于 rendered teacher-feature usability |
| Teacher-vs-rendered LocAcc | 多视角蒸馏 + FGC | 可以，但限定在 frozen evaluator 下 |
| ScanNet point query | CTR + direct point readout | 可以，属于 point-level feature usability |
| LERF direct 3D object selection | 新增 OPR / proposals | 应说 object-aware primitive usability，而不是单纯 teacher feature fidelity |

也就是说，加 OPR 后，论文主张不会被削弱，反而更完整：

> **teacher-feature reconstruction handles rendered-view feature usability; object-aware registration handles primitive-level 3D selection.**

---

## 6. 但这会带来一个新风险：是否变成“抄 registration 方法”？

会有这个风险。你需要把创新点写清楚。

不要写成：

> 我们也把 CLIP embeddings 注册到 Gaussians。

这会被 Dr. Splat / OpenGaussian / ProFuse 压住。

你应该写成：

> **Existing direct-registration methods attach language embeddings or instance features to Gaussians. In contrast, we start from a compact teacher-compatible feature field and add object-aware primitive registration as a second readout objective, so the same scene representation supports rendered dense teacher features and direct object-level queries.**

中文：

> **已有 direct-registration 方法主要把 language embeddings 或 instance features 挂到 Gaussians 上；本文则从 compact teacher-compatible feature field 出发，将 object-aware primitive registration 作为第二读出目标，使同一场景表示同时支持 rendered dense teacher features 和 direct object-level queries。**

你的差异必须是：

1. **不是只存 language embedding**，而是仍然保留 1280d teacher-compatible reconstruction；
2. **不是只做 direct registration**，而是同时保留 rendered dense feature map；
3. **不是直接每个 Gaussian 存高维 feature**，而是 compact code / proposal code / PQ / low-dimensional object code；
4. **不是用 GT masks 训练**，而是 frozen teacher + automatic masks + 3D visibility registration。

这样才不会变成 Dr. Splat 的弱复刻。

---

## 7. 最推荐的新版方法结构

我建议最终方法叫：

> **Object-Aware Compact Teacher Feature Field**

整体名字可以是：

> **CTF-GS++: Object-Aware Compact Teacher Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding**

如果不想 “++”，论文题目可写：

> **Object-Aware Compact Teacher Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding**

模块：

| 模块 | 作用 |
|---|---|
| HGCF | compact teacher code storage |
| CTR | reconstruct teacher-compatible dense / point features |
| VFA | rendered-view feature alignment |
| FGC | geometry-aware frozen-head regularization |
| **OPR** | object-aware primitive registration for direct 3D selection |
| **CPP** | context/proposal pooling for object-level readout |

其中 OPR / CPP 是新增部分。

---

## 8. 训练流程建议

### Stage 1：保持现有 CTF-GS 主训练

先训练：

$$
\mathcal{L}_{CTF}
=
\mathcal{L}_{RADIO}
+
\mathcal{L}_{FGC}
+
\mathcal{L}_{VFA}
$$

得到强 rendered-view checkpoint。

### Stage 2：冻结或半冻结主干，训练 OPR

为了不伤害 LERF rendered LocAcc，建议先冻结 HGCF + CTR，只训练 OPR / proposal features。

$$
\mathcal{L}_{stage2}
=
\mathcal{L}_{OPR}
+
\lambda_{\text{retain}}\mathcal{L}_{RADIO-retain}
$$

如果 Stage 2 后 rendered metrics 不掉，再试轻微 joint fine-tune。

### Stage 3：proposal-level selection inference

训练完后 direct 3D selection 走 OPR / CPP，不走 VFA：

$$
q \rightarrow t_q \rightarrow s_m(q)=\cos(u_m,t_q)
\rightarrow \text{select proposal}
\rightarrow \text{render selected Gaussians}
$$

### Stage 4：joint checkpoint 选择

最终主结果必须有一条 clear selector：

- 不能按每个 scene 选择最好的分支；
- 不能用 GT masks 调阈值；
- threshold / top-k / proposal selection 规则要全场景固定；
- best-by-scene 只能做 oracle / diagnostic。

---

## 9. 最小可行实验路线

如果你想快速判断这条路能不能冲上去，我建议按这个顺序做。

### P0：proposal oracle diagnostic

先不要训练复杂网络。做一个 diagnostic：

1. 用 SAM/SAM3 masks 生成 proposal；
2. 用 visibility 把 Gaussians 分配到 masks；
3. 用 mask-pooled SigLIP2/RADIO/CLIP descriptor 给 proposal 打分；
4. text query 选 proposal；
5. render proposal mask；
6. 跑 OpenGaussian-style LERF selection。

如果这一步 mIoU 能从 0.08 到 0.25-0.35，说明 object-aware proposal 路线可行。

如果这一步仍然低，说明问题在 descriptor/text alignment 或 mask-to-Gaussian association，不要急着训练 OPR。

### P1：dominant-ray registration

实现 Dr. Splat-style dominant Gaussian assignment：

- 每个 mask 内只注册 top-k alpha-contributing Gaussians；
- 过滤低 opacity / 低 visibility Gaussians；
- 保留 multi-view accumulated confidence。

Dr. Splat 的论文也显示 top-k Gaussians 是性能和内存的关键 trade-off；增加每条 ray 聚合的 Gaussian 数量会影响 localization 和 memory。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf))

### P2：proposal contrastive training

训练 OPR：

- same proposal close；
- different proposals far；
- spatial graph smooth；
- feature retain loss 防止破坏 CTR。

### P3：compact storage

如果 OPR 每个 Gaussian 又存 512d/768d language feature，storage claim 会崩。需要：

- low-dimensional object code；
- proposal-level descriptor；
- codebook / PQ；
- 或者只存 cluster id + small residual。

Dr. Splat 用 PQ compactly represent embeddings，并强调 memory / efficiency，这是你必须正面对齐的点。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf))

---

## 10. 实验表应该怎么设计

如果你加入 OPR，主实验应该重排成：

### Table 1：LERF rendered-view grounding

证明原主线没坏。

| Method | LocAcc | mIoU |
|---|---:|---:|
| CTF-GS | 0.8712 | 0.4941 |
| CTF-GS + OPR | 要求不下降 | 要求不下降或小幅上升 |

### Table 2：LERF direct 3D object selection

这是新增主表。

| Method | Protocol | mIoU | Acc@0.25 |
|---|---|---:|---:|
| OpenGaussian | official / reproduced | - | - |
| Dr. Splat | official / reproduced | - | - |
| CAGS | if available | - | - |
| CTF-GS direct center | current baseline | 0.0804 | 0.0932 |
| CTF-GS + OPR | new | target SOTA | target SOTA |

注意：最好用同一 evaluator 重跑可复现方法。Dr. Splat、OpenGaussian 的官方协议和 numbers 有很多细节，直接混表容易被攻击。OpenGaussian README 也提醒 LERF metrics 可能因优化和样本有限而不稳定。([GitHub](https://github.com/yanmin-wu/OpenGaussian))

### Table 3：ScanNet direct point query

看 OPR 是否提升 3D point-level semantic querying。

### Table 4：Ablation

| Variant | Direct 3D mIoU | Rendered LocAcc | ScanNet mIoU | 说明 |
|---|---:|---:|---:|---|
| CTF-GS baseline | 0.0804 | 0.8712 | 0.3538 |
| + dominant-ray registration | - | - | - |
| + SAM/SAM3 proposal pooling | - | - | - |
| + proposal contrastive | - | - | - |
| + graph context | - | - | - |
| + compact codebook/PQ | - | - | - |
| Full OPR | - | - | - |

这个消融必须证明：不是简单加 SAM mask 后处理，而是 OPR 的 object-aware primitive learning 有效。

---

## 11. 这算不算微调？会不会被质疑？

不算 foundation model fine-tuning，只要你不更新 RADIO / SigLIP2 / DINO / SAM / CLIP。但它是：

> **scene-specific pseudo-supervised primitive registration**

这必须诚实写。

安全表述：

> We do not fine-tune the teacher model or any frozen adaptor. OPR optimizes only scene-specific primitive/proposal features using automatic masks, visibility-based 2D-to-3D assignment, and frozen teacher descriptors. No LERF evaluation masks or test query labels are used for training.

中文：

> 我们不微调 teacher 或 frozen adaptor。OPR 只优化 scene-specific primitive/proposal features，监督来自自动 masks、visibility-based 2D-to-3D assignment 和 frozen teacher descriptors；不使用 LERF evaluation masks 或 test query labels。

如果你用了 LERF 的 GT object masks 或 query labels 训练，那就不能再当 fair open-vocabulary evaluation。

---

## 12. 加 OPR 后的论文 claim 应该怎么写？

不要再写：

> teacher feature reconstruction alone improves all downstream tasks.

应该写：

> **Compact teacher-feature reconstruction provides strong rendered-view features, while object-aware primitive registration converts the same teacher-compatible field into object-consistent 3D readouts.**

中文：

> **compact teacher-feature reconstruction 提供强 rendered-view features；object-aware primitive registration 则把同一个 teacher-compatible field 转化为 object-consistent 3D readout。**

最终中心 claim 可以升级为：

> **We show that compact unified teacher features can be distilled into 3D Gaussian scenes and made useful across rendered-view and primitive-level open-vocabulary understanding by combining teacher-space reconstruction with object-aware direct registration.**

这比原来的 claim 更强，也更能对齐 2025 primitive-level工作。

---

## 13. 我建议你的决策

如果你只是要稳投稿：保留现在的 conservative CTF-GS，不要大改。

如果你明确想冲 direct 3D object selection SOTA：**值得做 OPR，但要接受这是一次方法升级，不是小修。**

我的推荐路线是：

1. **先做 proposal oracle / no-training diagnostic**，确认 automatic mask proposal + visibility assignment 的上限。
2. 如果能到 0.25+ mIoU，再实现 OPR。
3. OPR 先冻结 CTF-GS 主干，避免破坏 rendered LERF。
4. direct 3D selection 用 proposal-level scoring，不再 raw Gaussian-center threshold。
5. 存储上必须用 compact object code / proposal descriptor / PQ，否则 compact claim 会被削弱。
6. 最终论文题目改成：

> **Object-Aware Compact Teacher Feature Fields for Open-Vocabulary 3D Gaussian Scene Understanding**

最关键的一句话是：

> **你可以借鉴 Dr. Splat / OpenGaussian / CAGS / ProFuse 的 registration 与 proposal 思想，但必须把它们变成“teacher-compatible compact field 的 object-aware direct readout”，而不是在 CTF-GS 外面硬贴一个 CLIP-Gaussian registration baseline。**

这样做成功后，本文的 claim 不但能维持，还会更强：从“rendered-view teacher feature field”升级为“rendered-view + primitive-level 都可用的 object-aware compact teacher feature field”。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md)
- [[2406.02058] OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding](https://arxiv.org/abs/2406.02058)
- [Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf)
- [ProFuse: Efficient Cross-View Context Fusion for Open-Vocabulary 3D Gaussian Splatting](https://chiou1203.github.io/ProFuse/)
- [OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding](https://3d-aigc.github.io/OpenGaussian/)
- [GitHub - yanmin-wu/OpenGaussian: [NeurIPS 2024] OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding · GitHub](https://github.com/yanmin-wu/OpenGaussian)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)