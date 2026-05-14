# 项目总结与优化

**User:** Anonymous  
**Created:** 5/8/2026 13:39:01  
**Updated:** 5/13/2026 23:09:59  
**Exported:** 5/13/2026 23:36:58  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fd76f5-1e00-83ea-956c-f27ef66858cf](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fd76f5-1e00-83ea-956c-f27ef66858cf)  

## Prompt:
2026/5/13 23:03:58

按照我们讨论过的最新主线继续推进了一波，git已更新，但感觉结果还是不够好，我还想提升论文质量和实验指标的强支撑性，下一步该如何继续推进。多轮进展总结如下：
1.
已按 `ChatGPT-项目总结与优化 (4).md` 的最后建议继续推进，重点把 DINO/SAM probe 升级成更正式的 downstream task，并做了验证。

本轮主要完成：

- 改进 [eval_lerf_sam_dino_tasks.py](/root/RADIO-GS/radio_gs/scripts/eval_lerf_sam_dino_tasks.py)：加入 prompt-constrained SAM mask 评估、DINO source-to-target mask propagation、source-background contrast 读出。
- 新增/扩展 [test_lerf_sam_dino_tasks.py](/root/RADIO-GS/tests/test_lerf_sam_dino_tasks.py) 覆盖 binary IoU、top-k mask、prompt heatmap constraint、mask propagation、background contrast。
- 用 GPU4/5 跑完四个 LERF scene formal_v4：
  [formal_v4 report](/root/RADIO-GS/output/lerf_sam_dino_tasks/formal_v4_bgcontrast05/lerf_sam_dino_task_report.md)
- 更新论文 [radio_gs_draft.tex](/root/RADIO-GS/paper/radio_gs_draft.tex:562)、项目主线文档和 submission status，并重新生成 [radio_gs_draft.pdf](/root/RADIO-GS/paper/radio_gs_draft.pdf)。

关键指标变化：

| Task | Teacher | Rendered |
|---|---:|---:|
| SAM3 point prompt | 1.0000 / 0.3700 | 1.0000 / 0.4169 |
| SAM3 box prompt | 0.8702 / 0.6560 | 0.8221 / 0.6638 |
| SAM3 mask propagation | 0.7872 / 0.3583 | 0.6667 / 0.3756 |
| DINOv3 dense matching | 0.5895 / 0.8543 | 0.5536 / 0.9048 |
| DINOv3 mask propagation + bg contrast | 0.7163 / 0.3921 | 0.7376 / 0.3684 |

结论：SAM3-adaptor 三个 prompt/mask 任务的 mIoU 已经超过 frame-wise teacher；DINO mask propagation 经 source-background contrast 后大幅提升，Rendered LocAcc 反超 teacher，但 mIoU 仍略低于 teacher。这一版更适合论文主张：“SigLIP2 grounding + SAM3-adaptor region mIoU 提升，DINO propagation gap 显著缩小，但不宣称 DINO 全面超过 teacher。”

验证已跑：

- `pytest tests/test_lerf_sam_dino_tasks.py tests/test_lerf_adaptor_downstream.py -q`：14 passed
- `py_compile`：通过
- `git diff --check`：通过
- `latexmk -pdf`：通过，PDF 已生成；仍有一些旧表格 overfull warning，但新增 SAM/DINO 表已收进版心。
2. ChatGPT-项目总结与优化 (4).md里提到的改进全都落实了吗
没有，全量来说还没有。`ChatGPT-项目总结与优化 (4).md` 里既有“必须修”的投稿闭环项，也有一批更重的 future-method 改进。上一轮主要落实了专家最后强调的最高优先级：把 DINO/SAM downstream probes 升级为正式任务并验证。

**已落实**
- SAM3/DINOv3 formal downstream task：已加入 prompt-constrained SAM3-adaptor segmentation、DINO dense matching、DINO mask propagation + source-background contrast。见 [eval_lerf_sam_dino_tasks.py](/root/RADIO-GS/radio_gs/scripts/eval_lerf_sam_dino_tasks.py:248)。
- SAM3-adaptor 结果已经从 gap 变成 mIoU 优于 teacher：point、box、mask propagation 都赢。见 [PROJECT_MAINLINE.md](/root/RADIO-GS/docs/PROJECT_MAINLINE.md:150)。
- DINO mask propagation 显著提升：Rendered `0.7376 / 0.3684`，teacher `0.7163 / 0.3921`，LocAcc 反超但 mIoU 仍略低。见 [radio_gs_draft.tex](/root/RADIO-GS/paper/radio_gs_draft.tex:583)。
- 论文和状态文档已同步，并且明确不把 SAM3-adaptor 写成 external SAM3 decoder。见 [radio_gs_draft.tex](/root/RADIO-GS/paper/radio_gs_draft.tex:570)。
- 测试和编译已过：`14 passed`，`py_compile`、`git diff --check`、`latexmk` 均通过。

**之前已基本落实**
- CTF-GS + VPR 主线已经确立，direct 3D object selection 主结果为 `0.3850 / 0.6428`。
- VPR ablation 已有 raw center、24/48/96 views、train views only、w/o VFA、w/o depth check、voxel context。见 [lerf_vpr_ablation_table.tex](/root/RADIO-GS/paper/lerf_vpr_ablation_table.tex:9)。
- ScanNet direct point-query 已作为 cross-domain 3D feature probe，而不是 ScanNet leaderboard。

**部分落实但还不够投稿最稳**
- VPR protocol hygiene：已有 protocol metadata 和 ablation，但还缺一个非常清晰的 “VPR Protocol Card” 小节/appendix。
- Storage/runtime：已有 compact storage 和 efficiency 表，但 `(4)` 特别提醒的 VPR registered primitive cache / voxel context cache 还没有单独计入 storage table。当前 storage table 只统计 feature-field、CTR/HCD、VFA/refiner。见 [storage_footprint_table.tex](/root/RADIO-GS/paper/storage_footprint_table.tex:3)。
- Qualitative figures：已有一些可视化，但 `(4)` 要求的 VPR raw → VPR → voxel、VPR ablation、ScanNet direct point-query 这几组投稿主图还没有完整闭环。
- 外部 baseline：仍主要是 official-source context，对 Dr. Splat / CAGS / InstanceGaussian / OpenGaFF 没有本地同协议重跑。

**尚未落实的重方法改动**
- VPR training signal：`L_vpr` 反哺 Gaussian-center feature 还没做。
- Dr. Splat-style contribution-weighted / dominant-ray registration 还没做。
- Component / instance-level VPR readout 还没真正替代 voxel-max。
- Query-adaptive GT-free threshold、negative prompt、prompt ensemble 还没系统验证。
- True frozen SAM3 decoder、mask-logit distillation、Dice/boundary loss、Hungarian instance matching还没接。
- DINO cycle consistency、geometry-aware positive pairs、local contrastive/windowed relation loss还没进入训练。
- ScanNet voxel-context point query、class prompt ensemble、stuff calibration还没完整推进。

所以准确结论是：**专家最后一条最高优先级 DINO/SAM formal task 已经落实并显著改善；VPR 主线已经可投稿；但 `(4)` 里的全部改进没有完全落实。** 当前更像是投稿闭环约 90% 左右，剩下最值得补的是 VPR cache/storage、VPR qualitative、Protocol Card，以及至少一个真正提升 direct 3D 的 component/contribution-weighted VPR 改动。
3.继续推进，直到专家建议的所有改进都落实，并给出验证是否有效的结论
已继续推进并完成一轮收尾修缮。结论是：专家建议中本地可落地的项目已落实或验证为负结果；不能诚实声称已完成的部分也已经在审计里明确标出。

主要新增/修正：

- 加入 VPR protocol card：`paper/vpr_protocol_card.tex`
- 修正 storage 口径：VPR cache 与 compact checkpoint storage 分开报告
- 实现并验证 Dr. Splat-inspired `alpha/alpha_depth` registration weighting：结果下降，不进入主线
- 修正 raw Gaussian-center `top10%` vs VPR `top2%` 指标混淆
- 增加 direct-3D published-context table：Dr. Splat / CAGS / InstanceGaussian / OpenGaFF 只作为 published context，不包装成同评测 SOTA
- 重新导出 VPR direct 3D masks，并生成论文定性图：
  - `paper/figures/lerf_vpr_direct_3d_qualitative.png`
  - `output/radio_gs/reports/lerf_vpr_direct_3d_qualitative_manifest.json`
- 刷新最终审计：
  - `output/radio_gs/reports/expert4_improvement_completion_audit.md`
  - `output/radio_gs/reports/final_consistency_audit.md`

验证结果：

- `pytest tests/test_build_storage_footprint_report.py tests/test_lerf_direct_3d_selection.py tests/test_lerf_sam_dino_tasks.py tests/test_lerf_adaptor_downstream.py -q`
  - `28 passed`
- `py_compile` 覆盖新增/修改脚本：通过
- `latexmk -pdf -interaction=nonstopmode -halt-on-error radio_gs_draft.tex`
  - 通过，生成 12 页 `paper/radio_gs_draft.pdf`
- `git diff --check`
  - 通过
- 无残留训练/评估/LaTeX 进程

有效性结论：

VPR + voxel context 仍是 direct 3D 最有效主线，固定协议为 `0.3850 mIoU / 0.6428 Acc@0.25`。贡献加权 registration 已实现但无效：`alpha` 为 `0.2978 / 0.5389`，`alpha_depth` 为 `0.2967 / 0.5345`，所以应作为 negative ablation/future work。

仍不能声称完成的部分：本地同协议复现 OpenGaussian/Dr. Splat/CAGS、rasterizer-level true contribution assignment、真正 frozen SAM3 decoder/mask-logit distillation。这些需要外部实现/模型或更深 CUDA/rasterizer 改造，当前已在审计中标为未完成，不应写进主 claim。OpenGaFF published-context 数字我也核对了其 arXiv HTML 表格来源：https://arxiv.org/html/2605.06088
4. 还能如何完善改进本项目？可以参考本目录下的以往所有的专家建议，看看有哪些可能有效的但之前没落实的
我重新对照了本目录下几份专家建议和当前审计文档，结论是：项目主线已经比较清楚，真正还值得补的不是继续微调 raw Gaussian-center，而是围绕 **direct 3D object selection 的对象级注册/聚合能力**、**论文可信度闭环**、**跨任务特征质量验证** 做最后一轮收束。

当前最值得推进但尚未完全落实的方向如下。

**P0：对象/组件级 VPR 选择**
这是最有希望继续提高 LERF direct 3D 的方向。当前 VPR 已把结果推到约 `0.3850 mIoU / 0.6428 Acc@0.25`，但本质仍是 primitive-level top-p 选择，容易碎片化。专家建议里多次提到的未充分落实方向是：

- 用 VPR 注册特征构建 kNN / voxel / feature graph；
- 聚成 object/component proposals；
- 先按文本打分 component，再展开到 Gaussians 渲染；
- 用 GT-free 的面积、连通性、score margin 做筛选；
- 特别针对 Waldo / Teatime 小物体和碎片化问题。

这比继续调 top2% / top5% 更像方法级升级，也更贴近 OpenGaussian / CAGS / InstanceGaussian / Dr. Splat 这条 direct 3D object selection 主线。

**P0：Waldo / Teatime query-level 失败诊断**
目前 direct 3D 的均值已经可用，但 Waldo 场景仍是短板。应该补：

- 每个 query 的 IoU histogram；
- GT mask 覆盖率和注册 view count；
- selected primitives 在前景/背景中的比例；
- false positive 来源区域；
- component 数量和面积分布；
- top-p / adaptive threshold 对每个 query 的影响。

这个工作不一定直接涨点，但能判断下一步是“召回不足”“背景混入”“组件碎裂”还是“文本对齐错误”。它也是论文 failure analysis 的素材。

**P0：外部 baseline 同协议风险处理**
严格 SOTA claim 仍有风险。当前还没有完整本地重跑 OpenGaussian / Dr. Splat / CAGS。建议至少做一个层级：

- 主文表格清楚标注：official / same-evaluator / reproduced；
- 如果时间允许，优先接 OpenGaussian-style evaluator；
- 若不能完整复现，则不要把不同论文的数字混成“同协议 SOTA”，而是写成 protocol-aligned comparison with reported numbers。

这对审稿风险比小幅涨点更关键。

**P1：VPR consistency training**
当前 VPR 是强 readout，但 raw Gaussian-center 仍弱。可以把 VPR 从后处理变成训练信号：

```text
L_vpr = 1 - cos(f_center, stopgrad(f_registered))
```

再配合 voxel/component context consistency，让 Gaussian 自身特征靠近注册后的多视角特征。这样可以增强“feature field 本体质量”，而不只是 readout 技巧。风险是可能影响 rendered-view 指标，所以应单独开分支，只在 LERF direct 3D 和 ScanNet 同时验证有效后再纳入主方法。

**P1：ScanNet direct point-query 增强**
专家建议里还有几个较稳的 ScanNet 提升点：

- point-query voxel context pooling；
- class prompt ensemble；
- background/stuff 类校准；
- DINO cross-view weight sweep；
- 2D-to-3D pseudo-logit consistency。

这些不会改变主线，但能增强“同一 compact teacher field 支持 3D semantic querying”的 claim。

**P1：GT-free adaptive threshold / negative prompts**
当前 fixed top2% 是强 baseline，但对象尺寸差异大。可以试：

- score margin threshold；
- entropy threshold；
- positive prompt vs background negative prompt；
- prompt ensemble；
- validation-fixed threshold policy。

注意不能 per-query 用 GT 调阈值，否则投稿时站不住。

**P2：true rasterization-contribution registration**
之前做过 `alpha` / `alpha_depth`，但那是中心采样近似，结果反而下降。真正 Dr. Splat 类 registration 更接近：

- 记录每个像素/ray 上真实 top-k alpha compositing contributors；
- 将 SAM/object mask 特征注册到这些 contributor；
- 多视角累积 confidence；
- 过滤低可见性和低 opacity primitive。

这个方向潜力高，但工程成本也高，需要改 rasterizer 或增加贡献记录路径。

**P2：SAM3 / mask-logit distillation 真正落地**
目前 DINO/SAM adaptor 更像 frozen-feature probe，不宜强写成完整 SAM3 downstream。若要升级，需要接 frozen SAM3 decoder 或 mask-logit supervision：

- mask logits / IoU token / presence score；
- soft Dice / boundary loss；
- promptable segmentation evaluation；
- 不使用 LERF test GT mask 训练。

否则论文中应保守命名为 “SAM adaptor probe”。

**P2：论文形态收束**
投稿完成度还差的不是主线，而是呈现：

- CVPR 模板迁移；
- main / appendix 表格拆分；
- direct 3D、rendered-view、ScanNet 三套协议明确隔离；
- storage table 明确 persistent compact checkpoint vs ephemeral VPR cache；
- bootstrap CI 或 seed robustness；
- qualitative comparison 按同一 query / view / mask 协议展示。

我建议下一步按这个顺序推进：

1. 先做 **Waldo/Teatime query-level audit + bootstrap CI**，定位 direct 3D 还能涨在哪里。
2. 然后实现 **component/object-aware VPR selection**，这是当前最可能有效且最符合投稿主线的方法升级。
3. 并行补 **ScanNet prompt/context enhancement** 和 **adaptive threshold / negative prompt calibration**。
4. 最后根据效果决定是否上更重的 **VPR consistency training** 或 true rasterization-contribution registration。

如果要我继续直接推进，我会优先从第 1 和第 2 项开始，因为它们最可能在不破坏现有主线的情况下提升 direct 3D 指标。
5.
已继续推进并完成一轮有效改进。

核心结果：direct 3D object selection 从原主线 `top0p02` 的 `0.3850 mIoU / 0.6428 Acc@0.25`，提升到新的 GT-free `meanstd2p5` selector：`0.3934 mIoU / 0.6594 Acc@0.25`。这是固定规则，不用 GT mask、不做 per-query oracle threshold。更新后的表在 [paper/lerf_direct_3d_selection_table.tex](/root/RADIO-GS/paper/lerf_direct_3d_selection_table.tex)，报告在 `output/radio_gs/reports/lerf_direct_3d_selection.md`。

这轮做了几件事：

- 在 [eval_lerf_direct_3d_selection.py](/root/RADIO-GS/radio_gs/scripts/eval_lerf_direct_3d_selection.py) 加入 `seed_expand_components` 和 query-level audit/Bootstrap CI。
- 实验发现 seed-expand component selector 不适合作为主方法：Teatime/Waldo 都下降，因此只保留为负向诊断。
- 跑完四场景 `mean_std` sweep，确定 `meanstd2p5` 是当前最强 paper-facing fixed selector。
- 新增 query-level audit 脚本 [build_lerf_direct_3d_query_audit.py](/root/RADIO-GS/radio_gs/scripts/build_lerf_direct_3d_query_audit.py)，输出 `output/radio_gs/reports/lerf_direct_3d_query_audit.md` 和 [paper/lerf_direct_3d_query_audit_table.tex](/root/RADIO-GS/paper/lerf_direct_3d_query_audit_table.tex)。
- 同步更新了 [paper/radio_gs_draft.tex](/root/RADIO-GS/paper/radio_gs_draft.tex)、[docs/PROJECT_MAINLINE.md](/root/RADIO-GS/docs/PROJECT_MAINLINE.md)、[docs/submission_status.md](/root/RADIO-GS/docs/submission_status.md)、VPR protocol card 和 consistency audit。

验证结果：

- `pytest tests/test_lerf_direct_3d_selection.py tests/test_build_lerf_direct_3d_selection_table.py tests/test_build_lerf_direct_3d_query_audit.py -q`：`18 passed`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error radio_gs_draft.tex`：编译成功，生成 12 页 PDF；仍有一些既有 overfull/underfull warnings，但无 fatal error。

补充说明：GPU4 当时显存已满，Waldo 改用 GPU2 跑；GPU5 用于 Teatime/Ramen。当前最合理的论文表述是：`meanstd2p5` 作为 GT-free adaptive selector 提升 VPR primitive readout，但仍不应声称全局 direct-3D SOTA，因为 CAGS/OpenGaFF 等 published-context 仍更高，且外部 baseline 未本地同协议复现。
6.继续推进优化，直到获得更强实验结果支撑
已继续推进并拿到更强 direct 3D 结果。

核心改动：在 [eval_lerf_direct_3d_selection.py](/root/RADIO-GS/radio_gs/scripts/eval_lerf_direct_3d_selection.py) 中加入 GT-free selector ratio bounds：`selection_min_ratio=0.005`、`selection_max_ratio=0.02`。它解决了 `mean+std` 阈值过严导致的空预测，同时保留 2% cap 控制背景泄漏。

新的 LERF direct 3D object selection 固定主结果：

| Scene | mIoU | Acc@0.25 |
|---|---:|---:|
| Figurines | 0.4879 | 0.8036 |
| Ramen | 0.4536 | 0.7324 |
| Teatime | 0.5111 | 0.7966 |
| Waldo Kitchen | 0.2008 | 0.3636 |
| Macro | 0.4133 | 0.6741 |

对比之前主线：
- `top0p02`: `0.3850 / 0.6428`
- `meanstd2p5`: `0.3934 / 0.6594`
- `meanstd2p5 + cap0.02`: `0.4072 / 0.6607`
- 当前 `meanstd2p5 + floor0.005 + cap0.02`: `0.4133 / 0.6741`

已更新：
- [lerf_direct_3d_selection.md](/root/RADIO-GS/output/radio_gs/reports/lerf_direct_3d_selection.md)
- [lerf_direct_3d_selection_table.tex](/root/RADIO-GS/paper/lerf_direct_3d_selection_table.tex)
- [radio_gs_draft.tex](/root/RADIO-GS/paper/radio_gs_draft.tex)
- [PROJECT_MAINLINE.md](/root/RADIO-GS/docs/PROJECT_MAINLINE.md)
- [submission_status.md](/root/RADIO-GS/docs/submission_status.md)
- query-level audit 和 submission freeze report 也已重建。

验证：
- `27 passed in 3.51s`
- `latexmk -pdf ... radio_gs_draft.tex` 成功，生成 12 页 PDF；只有已有的 overfull/underfull 排版警告。
- `git diff --check` 通过。

结论：这轮改动是有效的跨场景提升，主要提升来自 Figurines 和 Teatime 的空预测修复，Ramen/Waldo 没有回退。论文表述仍应保持克制：该结果高于 OpenGaussian official macro reference，但 Waldo 仍弱，且对 CAGS/OpenGaFF 等更新方法不能声称全局 SOTA。

## Response:
2026/5/13 23:04:02

我会把最新结果先重新定位：哪些已经能支撑主 claim，哪些只是接近但不能包装成 SOTA。然后按任务给出下一轮最可能涨点、同时增强论文可信度的优先级。

## Response:
2026/5/13 23:04:25

我会把主线进一步收束为“2D teacher-vs-student 下游可用性 + 3D beyond-teacher 能力”，并明确哪些结果进主文、哪些只能进附录。下一轮重点不是无限加实验，而是补强 DINO mIoU、Waldo direct 3D 和 ScanNet 15/10 splits。

## Response:
2026/5/13 23:09:58

你现在已经不是“结果不够好所以论文站不住”的状态了。更准确地说：**论文主线已经能站住，但如果你想把 claim 从“strong / competitive”推到“多个任务 SOTA + student foundation feature field 明显强于 teacher”，还差两类东西：DINO/SAM/ScanNet 的进一步闭合，以及外部 baseline 的同协议可信度。**

旧附件 PPT 里的叙事还是 “RADIO-GS as reusable foundation-feature memory”，而且 DINO/SAM 还只是 weak diagnostic；现在项目已经明显升级到 **CTF-GS + VPR + frozen-adaptor downstream task evaluation**，旧 PPT 不能再作为论文结构依据，只能作为历史阶段总结。`radio_gs_submission_presentation.pptx`

---

## 1. 目前最合理的论文判断

现在可以写成：

> **CTF-GS learns a compact multiview student foundation-feature field. It improves SigLIP2 grounding and SAM3-adaptor region mIoU over frame-wise teacher features, narrows the DINOv3 propagation gap, and adds 3D scene-understanding readouts through VPR and ScanNet direct point queries.**

这条主张是成立的。仓库当前主线也已经把方法定义为：HGCF 存 compact code，CTR 恢复到 1280d RADIO space，VFA 做 view-space alignment，FGC 做 frozen geometry-head consistency，VPR 把 rendered features 注册回 Gaussian primitives 做 direct 3D querying。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

但现在仍然**不要写**：

> student feature field 全面强于 teacher；  
> CTF-GS 在 2D / 3D 所有任务上全面 SOTA；  
> SAM3 official instance segmentation 已经完成；  
> VPR 直接证明 feature field 本体质量提升。

现在更稳的说法是：**student 在 SigLIP2 和 SAM3-adaptor region tasks 上已经有明确胜出；DINOv3 还有 mIoU gap；VPR 证明 rendered feature 可以转化为强 primitive-level readout，但不是 raw Gaussian-center feature 本身已经强。**

---

## 2. 当前实验强度：哪些已经够强，哪些还不够

### 2.1 2D novel-view teacher-vs-student

这一组现在比之前强很多，已经可以成为主线核心。

| Frozen head / adaptor | 当前结论 | 论文位置 |
|---|---|---|
| **SigLIP2 grounding** | student LocAcc 0.8712 > teacher 0.7985，mIoU 也略高 | 主结果 |
| **SAM3-adaptor prompt tasks** | point / box / mask propagation 的 rendered mIoU 全部高于 teacher | 主结果或强辅助结果 |
| **DINOv3 dense / propagation** | dense matching score更高，mask propagation LocAcc 更高，但 mIoU 低于 teacher | 辅助结果 / 待优化点 |

仓库当前结果显示，SAM3-adaptor point prompt mIoU 从 teacher 0.3700 到 rendered 0.4169，box prompt 从 0.6560 到 0.6638，mask propagation 从 0.3583 到 0.3756；DINO mask propagation 的 LocAcc 从 teacher 0.7163 到 rendered 0.7376，但 mIoU 从 0.3921 降到 0.3684。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

所以你的论文可以说：

> **The student improves selected frozen-head downstream tasks over the frame-wise teacher, especially SigLIP2 grounding and SAM3-adaptor region segmentation.**

不要说：

> **The student uniformly outperforms the teacher across all adaptor spaces.**

DINOv3 本来就适合 dense feature / correspondence / 3D matching 这类任务；DINOv3 论文也强调其 dense features 可用于 geometric tasks such as depth estimation and 3D matching。([arXiv](https://arxiv.org/html/2508.10104v1?utm_source=chatgpt.com)) 你现在用 DINO mask propagation / dense matching 是合理的，但下一步必须把 DINO mIoU 追上 teacher，才能让“三头 frozen downstream tasks”这个主张更硬。

---

### 2.2 3D beyond-teacher ability

这一组现在也已经很有说服力，但不要过度包装。

**LERF direct 3D object selection** 已经从 raw Gaussian-center 的 0.0804 / 0.0932 提升到 VPR + voxel + meanstd2p5 + floor/cap 的 **0.4133 mIoU / 0.6741 Acc@0.25**。这超过 OpenGaussian official macro reference 的 0.3836 / 0.5143，但仓库也明确写了：Dr. Splat、CAGS、InstanceGaussian、OpenGaFF 的 published-context 数字仍更高，因此不能声称 global direct-3D SOTA。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

这个定位很正确。OpenGaussian 的协议本来就是 text-selected Gaussians → render multi-view masks → compute metrics。([GitHub](https://github.com/yanmin-wu/OpenGaussian?utm_source=chatgpt.com)) CAGS、Dr. Splat 等近作也都围绕 primitive / instance-level 3D Gaussian understanding 展开；CAGS 特别指出 isolated per-Gaussian feature learning 缺少 spatial context，会导致 fragmented representations。([arXiv](https://arxiv.org/abs/2504.11893?utm_source=chatgpt.com))

**ScanNet direct point-query** 当前 base 是 19/15/10 split mIoU 0.3538 / 0.3573 / 0.4293；DINO-CV 10-scene ablation 提升到 0.3640 / 0.3662 / 0.4308。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md)) 这说明 3D point-query 任务还有可提升空间，而且 DINO cross-view branch 对 ScanNet 是正向的。

---

## 3. 目前论文是否完整？

**完整，但还不是最强版本。**

现在已经有：

- 2D rendered-view grounding；
- teacher-vs-student；
- SAM3 / DINOv3 downstream tasks；
- LERF direct 3D object selection；
- ScanNet direct point query；
- component ablation；
- VPR ablation；
- storage / efficiency；
- qualitative figures；
- protocol card；
- LaTeX draft。

仓库的 submission status 也把当前 conservative paper package 评为约 92%，并明确说 VPR-backed primitive-level paper 已经 credible，但还要谨慎处理 Waldo Kitchen 和 protocol provenance。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md))

现在真正缺的不是“再补一个实验类型”，而是下面四件事：

1. **DINO mIoU 追上 teacher**，否则“三头 frozen downstream task”里始终有一个短板。
2. **VPR direct 3D 再提升 Waldo / mIoU**，否则 direct 3D 只能写 competitive，不能写 global SOTA。
3. **ScanNet 加强到更像 3D semantic result**，而不是仅仅是 supportive probe。
4. **外部 baseline 至少本地复现一个强 anchor**，否则所有 SOTA 话术都必须降级。

---

## 4. 下一步最值得做的 P0：DINOv3 mIoU gap

你现在最该优先补的是 DINOv3 mask propagation mIoU。原因很简单：你想主打 “student feature field 强于 teacher”，那三头里 DINO 不能明显落后。

当前 DINO 结果是：

| Task | Teacher | Rendered |
|---|---:|---:|
| DINO dense matching score | 0.8543 | **0.9048** |
| DINO mask propagation LocAcc | 0.7163 | **0.7376** |
| DINO mask propagation mIoU | **0.3921** | 0.3684 |

这说明 rendered feature 的相似度/匹配峰不差，甚至更强，但 mask region 不够准。问题大概率不是 feature 完全不对，而是 **propagation mask 过窄、过碎或被相似背景吸走**。

建议做三个改动。

### 4.1 DINO local contrastive propagation loss

不要只做全局 DINO relation loss。全局 relation 很容易把相似区域一起平滑掉。改成局部 positive / negative：

- positive：通过 3DGS depth + pose 找到 source-target 中同一 3D 点或同一小区域；
- hard negative：同一视图内靠近但在 mask 外的区域；
- loss：InfoNCE 或 margin contrast；
- 目标：提高 mask propagation mIoU，而不是只提高 dense matching score。

论文里可以叫：

> **Geometry-Aware DINO Correspondence Consistency**

### 4.2 DINO propagation region expansion with background suppression

你已经加了 source-background contrast，LocAcc 反超 teacher。下一步要让 region 更完整：

$$
S(x)=\cos(f_x, f_{\text{fg}})-\lambda \cos(f_x, f_{\text{bg}})
$$

继续 sweep 的不是 $\lambda$ 本身，而是：

- foreground prototype 用 top-k robust mean，不用单一 source mask mean；
- background prototype 分成 near-boundary bg 和 global bg；
- 对 target heatmap 做 connected component 约束；
- 小物体用更低 area floor，大物体用更高 cap。

目标指标很明确：**DINO mask propagation mIoU ≥ 0.3921，同时 LocAcc 不低于 0.7376。**

### 4.3 DINO cycle consistency

加入 source → target → source 的 mask cycle：

$$
M_s \rightarrow \hat{M}_t \rightarrow \hat{M}_s
$$

让 $\hat{M}_s$ 回到原 source mask。这个对 DINO correspondence 很自然，也比单纯 feature cosine 更能修正过平滑。DINOv3 官方 repo 本身就把 dense/sparse matching 作为示例用途，采用这种 downstream-style verification 很合理。([GitHub](https://github.com/facebookresearch/dinov3/blob/main/notebooks/dense_sparse_matching.ipynb?utm_source=chatgpt.com))

**P0 成功标准：**  
DINO rendered mIoU 从 0.3684 提到 ≥0.3921；如果能到 0.40+，你就可以写：

> CTF-GS improves or matches the frame-wise teacher across SigLIP2, SAM3-adaptor, and DINOv3 downstream tasks.

这会显著提高论文主线强度。

---

## 5. P0：VPR direct 3D 继续涨点，重点不是 selector，而是 component-level readout

你已经把 selector 从 top2% 推到 meanstd2p5 + floor/cap，direct 3D 到了 0.4133 / 0.6741。继续调阈值的收益会变小。下一步要改 **object/component readout**。

当前 direct 3D 的主要短板仍是 Waldo Kitchen：mIoU 0.2008，低于 OpenGaussian 0.2270。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md)) Waldo 是 clutter + object fragmentation 的典型场景。CAGS、InstanceGaussian、OpenGaFF 这类方法之所以 mIoU 更高，本质都在处理 spatial context / instance grouping / fragmented Gaussian semantics。CAGS 明确说孤立 per-Gaussian feature learning 会导致 fragmented representations，并用 local graph 和 mask-centric contrast 来缓解。([arXiv](https://arxiv.org/abs/2504.11893?utm_source=chatgpt.com))

建议下一步不要再叫 seed-expand，而是做一个正式模块：

> **Component-Aware VPR Selection**

实现方式：

1. 用 VPR registered primitive scores 得到 seed primitives；
2. 在 voxel / kNN graph 上做 connected components；
3. component feature 用 mean / max / top-k pooled score；
4. query 先选 component，再展开到 primitives；
5. 用 GT-free rules 控制 component size：
   - min area ratio；
   - max area ratio；
   - score margin；
   - component compactness；
   - view coverage count。

你上一轮 seed-expand 下降，不代表 component-level 不行。seed-expand 常见问题是把背景一起扩进去。新的 component selection 要避免“从 seed 向外无脑扩张”，而是 **先形成 GT-free candidate components，再按文本和形状规则筛选**。

**P0 成功标准：**

- Macro mIoU ≥ 0.43；
- Waldo ≥ 0.23；
- Acc@0.25 保持 ≥ 0.67；
- 不使用 GT masks / per-query oracle threshold。

只要这组达成，direct 3D 就会从“competitive”更接近“strong”。

---

## 6. P1：ScanNet 不要只保留 base，应该把 DINO-CV 升级成 candidate main

ScanNet 现在 base 和 DINO-CV 都有。DINO-CV 在 19/15/10 split mIoU 上分别 +0.0102、+0.0089、+0.0014，mAcc 也全提升。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md)) 这说明它不是偶然负优化，可以考虑作为 **ScanNet candidate main**，但要确认它不会伤 LERF main table。

下一步建议做：

### 6.1 ScanNet prompt ensemble

每个类别不要只用一个 prompt。做固定 prompt bank：

- `{class}`
- `a {class}`
- `a photo of a {class}`
- `indoor {class}`
- 对 stuff 类单独加 `wall surface`, `floor surface`, `cabinet furniture` 等 aliases。

输出用 mean 或 max，但规则固定。这个不需要训练，成本低，可能对 15/10 split mIoU 提升明显。

### 6.2 Point-query voxel context pooling

当前 point query 如果是单点 / label point decode，容易噪声大。可以模仿 VPR voxel context：

$$
f(p)=\text{pool}_{g_i \in \mathcal{N}(p)} w_i f_i
$$

权重用 distance、opacity、registered confidence。不要只 nearest Gaussian。

### 6.3 Stuff-class calibration

ScanNet 的 15/10 split 往往 stuff 类和 large object 类混淆。做固定 calibration：

- class prior 只从 train scenes / class names 估计；
- 或用 background text margin；
- 不用 GT test labels调参。

**P1 成功标准：**

- split19 ≥ 0.37；
- split15 ≥ 0.39；
- split10 ≥ 0.45；
- mAcc 不下降。

如果 ScanNet 上也能明显涨，你的“3D scene understanding beyond teacher”会更有支撑。

---

## 7. P1：SigLIP2 rendered-view mIoU 仍然可以涨

你现在 SigLIP2 LocAcc 很强，但 mIoU 0.4941 / adaptor candidate 0.4979 不算高。想让 2D rendered-view comparison 更像强 baseline，就要把 mIoU推到 0.52+，同时 LocAcc 不掉。

建议：

### 7.1 Heatmap peak + region 双目标

你们已经知道 mIoU 和 LocAcc 有 tradeoff。下一步训练目标要显式分两路：

- peak loss：teacher/student heatmap argmax neighborhood KL；
- region loss：soft mask / top-percentile region consistency；
- boundary loss：alpha/depth edge guided smoothness。

目标不是单纯让热图变宽，而是 **object内变宽，object外压低**。

### 7.2 Negative prompt margin

对于每个 query：

$$
S = S_{\text{pos}} - \beta \max_j S_{\text{neg},j}
$$

negatives 选：

- background；
- table / shelf / wall / hand；
- same-scene distractor object；
- “not {query}”。

这个对 Waldo、Figurines尤其有用。

### 7.3 Small-object weighted VFA

Figurines 和 Waldo里小物体 / clutter最容易 peak shift。训练 VFA 时对小 mask、边界附近、high entropy heatmap区域加权。

**P1 成功标准：**

- LERF rendered mIoU ≥ 0.52；
- LocAcc 仍 ≥ 0.8712；
- Figurines mIoU ≥ 0.45；
- Waldo mIoU ≥ 0.43。

---

## 8. P1：SAM3-adaptor 已经能主表，但不要叫 official SAM3 instance segmentation

你现在 SAM3-adaptor 三个 prompt任务 mIoU 都超过 teacher，这是好结果。SAM3 官方是 promptable segmentation / promptable concept segmentation，支持 text、points、boxes、masks等 prompts；SA-Co/Gold 的官方指标是 cgF1。([GitHub](https://github.com/facebookresearch/sam3?utm_source=chatgpt.com)) ([GitHub](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md?utm_source=chatgpt.com))

但你现在没有调用 external frozen SAM3 mask decoder，paper draft 也写了 “no external SAM3 mask decoder is called”。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex)) 所以论文里必须叫：

> **SAM3-adaptor promptable region probes**

不要叫：

> **SAM3 instance segmentation benchmark**

如果你想继续提升 SAM3：

- 对 box prompt，主要提升 Hit/LocAcc，从 0.8221 追 teacher 0.8702；
- 对 mask propagation，主要提升 LocAcc，从 0.6667 追 teacher 0.7872；
- mIoU 已经赢 teacher，别牺牲它。

建议加一个 **prompt confidence calibration**：

- point prompt 用局部 Gaussian peak；
- box prompt 用 box内 top-k foreground + box外 hard negative；
- mask propagation 用 source mask erosion/dilation 双 prototype，减少边界错配。

---

## 9. 论文质量方面，下一轮最该改的不是代码，而是结构

你现在论文应该从“方法组件平铺”改成“能力证明”结构。

建议实验章节按这个顺序：

### 5.1 Student vs frame-wise teacher on 2D novel-view downstream tasks

这张表是你主线核心：

| Head | Task | Teacher | Student | Takeaway |
|---|---|---:|---:|---|
| SigLIP2 | grounding | 0.7985 / 0.4922 | 0.8712 / 0.4941 | clear win |
| SAM3-adaptor | point / box / mask | teacher | student | mIoU win |
| DINOv3 | dense / propagation | teacher | student | LocAcc/score win, mIoU gap |

### 5.2 3D abilities beyond the 2D teacher

放：

- LERF direct 3D object selection via VPR；
- ScanNet direct point query。

### 5.3 External method context

分开写：

- rendered-view LERF：不要强 SOTA；
- direct 3D LERF：高于 OpenGaussian official，但低于部分近作 published context；
- ScanNet：supportive，不写 leaderboard SOTA。

OpenGaFF 最近同时报告 LERF 2D 和 3D，并采用 2D rendered feature map mIoU/Acc、3D per-Gaussian query 后 render selected Gaussians 的 mIoU/Acc@0.25，这说明你们的两个 LERF协议划分是合理的。([arXiv](https://arxiv.org/html/2605.06088v1?utm_source=chatgpt.com))

### 5.4 What makes the field work?

放 ablation：

- HGCF；
- CTR；
- VFA；
- FGC；
- FAC；
- VPR；
- selector / voxel context。

### 5.5 Cost, storage, and limitations

这里一定写清：persistent compact checkpoint 是 compact 的，但可选 VPR cache 如果 materialize 会大于 direct 1280-D feature storage。仓库主线里已经把这个点写得很清楚：compactness claim 只适用于 persistent scene/checkpoint storage；VPR cache 是 streamed inference-time readout，若持久化要单独报告。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

---

## 10. 下一步具体推进顺序

我建议你不要同时开太多新分支。按这个顺序推进：

### Step 1：DINO mIoU 冲刺

目标：DINO rendered mask propagation mIoU ≥ teacher 0.3921。

做：

- source-background contrast prototype改进；
- geometry-aware positive pairs；
- local contrastive / cycle consistency；
- target connected component cleanup。

这是最能强化“student 强于 teacher”的一步。

### Step 2：Component-aware VPR

目标：direct 3D macro mIoU ≥ 0.43，Waldo ≥ 0.23。

做：

- component proposal scoring；
- score-margin selection；
- component size prior；
- background negative prompts；
- 不用 GT mask调阈值。

这是最能强化“3D beyond teacher”的一步。

### Step 3：ScanNet prompt ensemble + voxel pooling

目标：split19 ≥ 0.37，split15 ≥ 0.39，split10 ≥ 0.45。

做：

- class prompt ensemble；
- point-query neighborhood pooling；
- stuff calibration；
- DINO-CV candidate checkpoint作为主候选。

### Step 4：SigLIP2 mIoU提升

目标：mIoU ≥ 0.52，LocAcc不掉。

做：

- negative prompt margin；
- boundary-aware VFA；
- small-object weighting；
- heatmap peak + region 双目标。

### Step 5：只本地复现一个强 external baseline

优先 OpenGaussian。原因是你现在 VPR 结果已经超过 OpenGaussian official reference；如果本地同 evaluator 复现 OpenGaussian，并且 CTF-GS + VPR 仍超过它，你就能把 direct 3D 这张表从 “official-source context” 升级成 “same-evaluator comparison”。

Dr. Splat / CAGS / OpenGaFF 可以继续作为 published context，不必全部重跑。Dr. Splat 的核心是把 CLIP embeddings 直接注册到 pixel-ray dominant Gaussians，工程对齐成本更高。([arXiv](https://arxiv.org/abs/2502.16652?utm_source=chatgpt.com))

---

## 11. 最终建议：现在不要再追“所有任务 SOTA”，先把主线打硬

你现在最强的论文形态是：

> **A multiview student foundation-feature field that improves selected frozen-head novel-view tasks over the frame-wise teacher and enables additional 3D scene-understanding readouts through VPR.**

这条线足够好，而且比“又一个 language Gaussian field”更有特点。

下一轮最有价值的成果排序是：

1. **DINO mIoU 追上 teacher**：最能强化 student > teacher。
2. **VPR component-level readout 提高 Waldo / mIoU**：最能强化 3D direct selection。
3. **ScanNet prompt/context enhancement**：最能强化 3D scene understanding。
4. **本地复现 OpenGaussian**：最能降低审稿风险。
5. **重新做主图和 qualitative**：最能提高论文观感。

如果只能做一件事，我会先做 **DINO mIoU 冲刺**。因为你现在的核心主张已经不是单纯 SOTA，而是 **student foundation-feature field 的下游可用性强于 frame-wise teacher**。这个主张一旦在 SigLIP2、SAM3-adaptor、DINOv3 三个 frozen-head tasks 上都站住，整篇论文的逻辑会明显更硬。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md)
- [DINOv3](https://arxiv.org/html/2508.10104v1?utm_source=chatgpt.com)
- [[NeurIPS 2024] OpenGaussian: Towards Point-Level 3D ...](https://github.com/yanmin-wu/OpenGaussian?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)