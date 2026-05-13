# 项目总结与优化

**User:** Anonymous  
**Created:** 5/8/2026 13:39:01  
**Updated:** 5/12/2026 23:53:58  
**Exported:** 5/13/2026 0:09:25  
**Link:** [https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fd76f5-1e00-83ea-956c-f27ef66858cf](https://chatgpt.com/g/g-p-69fa063684d4819197d1970b1d8a1e00/c/69fd76f5-1e00-83ea-956c-f27ef66858cf)  

## Prompt:
2026/5/12 22:13:11

最新按计划推进的进展如下（git仓库已同步更新）：
已按 `(3)` 专家建议把主线纠正为 **CTF-GS + VPR(View-to-Primitive Registration)**，并完成一轮方法升级、GPU4/5 实验验证、报告/LaTeX 草稿同步。

核心进展：

- **Direct 3D object selection 主结果升级**
  - 原 raw Gaussian-center：`0.0804 mIoU / 0.0932 Acc@0.25`
  - 24-view VPR：`0.3421 / 0.5547`
  - 48-view VPR + voxel：`0.3802 / 0.6532`
  - **96-view VPR + voxel 主表：`0.3850 / 0.6428`**
  - OpenGaussian official reference：`0.3836 / 0.5143`
  - 另测 lower blend：`0.3852 / 0.6313`，mIoU 只高 `0.0002`，Acc 更低，所以没有替换主表。
  - 96-view component refinement：`0.3849 / 0.6428`，未提升，保留为诊断。

- **方法和论文主线已修正**
  - 不再 claim “raw dual-readout 全面强”，而是准确写成：rendered feature 通过 **VPR 注册回 Gaussian primitives**，支撑 primitive-level direct querying。
  - 论文标题/abstract/method/limitations 已改为 VPR 版本。
  - 新增 `paper/lerf_vpr_ablation_table.tex`，系统展示 VPR、view coverage、VFA、depth check、voxel context 的贡献。

- **已更新关键文件**
  - `paper/radio_gs_draft.tex`
  - `paper/lerf_direct_3d_selection_table.tex`
  - `paper/lerf_vpr_ablation_table.tex`
  - `docs/PROJECT_MAINLINE.md`
  - `docs/paper_draft_current.md`
  - `docs/submission_status.md`
  - `output/radio_gs/reports/lerf_direct_3d_selection.md`
  - `output/radio_gs/reports/submission_freeze_report.md`
  - `output/radio_gs/reports/final_consistency_audit.md`

- **代码侧补强**
  - `eval_lerf_direct_3d_selection.py` 增加 train/val registration split、VFA ablation、VPR protocol metadata。
  - 新增/扩展单测覆盖 frame selection、VFA disable、voxel aggregation、component refinement。

验证结果：

- `pytest tests/test_lerf_direct_3d_selection.py tests/test_direct_point_supervision.py -q`：`33 passed`
- `py_compile` 覆盖修改脚本：通过
- `git diff --check`：通过
- LaTeX 输入文件检查：`missing_inputs=[]`
- 环境里没有 `pdflatex/latexmk/chktex`，所以未做 PDF 编译。

当前完成度判断：按“完整投稿大纲”约 **90-92%**。主线、方法、LERF/ScanNet/direct 3D/消融/效率/草稿已经闭环；剩余最硬缺口是外部 baseline 本地同协议复现和最终 CVPR 模板排版压缩。严格 SOTA 仍需谨慎，因为我们现在是 official-source context 对比，不是所有 baseline 同 evaluator 重跑。

相关工作调研依据包括：OpenGaussian、Dr. Splat、CAGS、InstanceGaussian：
https://arxiv.org/abs/2406.02058  
https://arxiv.org/abs/2502.16652  
https://arxiv.org/abs/2504.11893  
https://openaccess.thecvf.com/content/CVPR2025/html/Li_InstanceGaussian_Appearance-Semantic_Joint_Gaussian_Representation_for_3D_Instance-Level_Perception_CVPR_2025_paper.html
---------
目前看起来是否已经实现各项实验评估的指标达到sota了呢？本项目从论文投稿角度来看是否合理？包括从动机、综述、主要claim和贡献、方法描述包括一些实现细节以及实验等所有的部分，是否都逻辑自洽？我最关心实验，实验设置和评估是否都已经完整无误了呢？除了各种定量评估外，目前还缺少定性结果的可视化，之前生成的不够全，也大多仍存在问题（包括论文草稿里放的）。请完整带我梳理一遍，下一步应该如何修改？

## Response:
2026/5/12 22:13:13

我会先按投稿完整性核对 SOTA 与协议边界，再系统审查动机、claim、方法、实验和可视化缺口。重点判断哪些结果能强说，哪些还需同协议复现支撑。

## Response:
2026/5/12 22:13:57

我已核对当前主线与外部结果：VPR 已能支撑强 dual-readout 论文，但只能相对 OpenGaussian official context 局部称强。下一步会重点区分严格 SOTA、协议风险和还缺的定性图。

## Response:
2026/5/12 22:17:33

> **Addressing issues in paper draft and final report structure**
>
> I spotted a potential typo in the paper draft, specifically in line 17, where it says "original..." but contradicts the current VPR result, which states the VPR method is best. This needs correction. I'll also need to check the test response and finalize the structure for the audit with all necessary details and comparisons.
>
> Thought for 8m 14s

## Response:
2026/5/12 22:21:28

结论先说清楚：**现在项目已经从“保守 feature reconstruction paper”升级成了一个可以投稿的 CTF-GS + VPR 双读出论文，但还不能稳妥地写成“全面 SOTA”。** 更准确的判断是：

**CTF-GS 在 rendered-view LERF grounding 上有强主结果；VPR 让 LERF direct 3D object selection 达到 OpenGaussian 官方参考附近并略超其 macro mIoU / Acc@0.25；ScanNet direct point-query 是有效的 3D feature probe。但如果把 Dr. Splat、CAGS、InstanceGaussian、OpenGaFF 等更近工作都纳入同一语境，当前 direct 3D selection 不能称全局 SOTA。**

你们现在的论文主线是合理的，但要把 claim 写得足够精确：**不是 raw Gaussian-center dual-readout，而是 compact teacher feature reconstruction + view-to-primitive registration。**

---

## 1. 现在是否达到 SOTA？

### 1.1 LERF rendered-view 2D grounding：强，但不要直接称 SOTA

你们当前 rendered-view LERF 主表是：

| Metric | Macro |
|---|---:|
| LocAcc | **0.8712** |
| mIoU | **0.4941** |

这个结果可以作为 CTF-GS 的主结果。它证明 rendered teacher-compatible features 在 LERF-OVS text grounding 上可用，并且 teacher-vs-rendered 对比显示 original RADIO RGB 的 macro LocAcc 是 0.7985，而 CTF-GS rendered features 是 0.8712。这个结论支撑的是 **task-relevant localization usability 提升**，不是“student 全面强于 RADIO teacher”。仓库主线也把这个结果列为当前 active quantitative claim。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

但 rendered-view 2D OVS 目前不能写成严格 SOTA，因为你们的外部 rows 不是同 evaluator 复现。更近的 OpenGaFF 报告了 LERF-OVS 2D mIoU 64.98 / Acc 82.68，而你们是 mIoU 49.41 / LocAcc 87.12；这说明你们 LocAcc 很强，但 mIoU 不是最高，而且协议、text head、normalization、temperature 都可能不同。([arXiv](https://arxiv.org/html/2605.06088v1))

**建议写法：**

> CTF-GS achieves strong rendered-view open-vocabulary localization on LERF-OVS and improves localization over frame-wise RADIO features under the same frozen evaluator.

不要写：

> CTF-GS is SOTA on LERF-OVS 2D OVS.

---

### 1.2 LERF direct 3D object selection：超过 OpenGaussian 官方参考，但不是全局 SOTA

你们最新 direct 3D selection 主表是：

| Method | mIoU | Acc@0.25 |
|---|---:|---:|
| Raw Gaussian-center | 0.0804 | 0.0932 |
| 24-view VPR | 0.3421 | 0.5547 |
| 48-view VPR + voxel | 0.3802 | 0.6532 |
| **96-view VPR + voxel** | **0.3850** | **0.6428** |
| OpenGaussian official reference | 0.3836 | 0.5143 |

相对 **OpenGaussian official reference**，你们确实略超 macro mIoU，并明显超过 Acc@0.25。仓库主线也记录：VPR + voxel context fixed top0p02 达到 0.3850 / 0.6428，OpenGaussian official context 是 0.3836 / 0.5143。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

但是，不能据此写“direct 3D object selection SOTA”。原因有三点。

第一，OpenGaussian 官方 GitHub 自己说明 LERF object selection 是先渲染 text-selected 3D Gaussians，再计算 metrics，并且提醒由于 LERF evaluation samples 有限，metrics 可能不稳定；这意味着 0.3850 vs 0.3836 这种 0.0014 的 mIoU 差距不能作为强 SOTA 证据。([GitHub](https://github.com/yanmin-wu/OpenGaussian))

第二，CAGS 报告的 LERF direct 3D object selection 是 **50.79 mIoU / 69.62 mAcc@0.25**，明显高于你们当前 38.50 / 64.28；它的表里 Dr. Splat 也是 43.29 / 64.30，高于你们的 mIoU。([arXiv](https://arxiv.org/html/2504.11893v1))

第三，InstanceGaussian 报告 LERF open-vocabulary 3D object selection 是 **45.30 mIoU / 58.44 mAcc@0.25**，OpenGaFF 报告 3D LERF 是 **54.36 mIoU / 80.84 Acc**。这两者也说明你们当前不能称全局 SOTA。([arXiv](https://arxiv.org/html/2411.19235v1)) ([arXiv](https://arxiv.org/html/2605.06088v1))

**建议写法：**

> CTF-GS + VPR reaches competitive OpenGaussian-style direct 3D object selection, slightly exceeding the OpenGaussian official macro reference and strongly improving over raw Gaussian-center readout. However, we do not claim universal primitive-level SOTA without locally rerunning recent instance-aware baselines.

这句话是稳的。

---

### 1.3 ScanNet direct point-query：有效，但不是 ScanNet SOTA

你们当前 ScanNet v67 direct point-query 是：

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3538 | 0.6076 |
| 15 classes | 0.3573 | 0.6203 |
| 10 classes | 0.4293 | 0.7051 |

这个结果能证明 feature field 可被 3D point-level 直接查询，不只是 LERF rendered heatmap。仓库主线也是这么定位的：ScanNet v67 direct point query 是 cross-domain feature-usability evidence，而不是完整 ScanNet leaderboard。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

如果和 CAGS 的 ScanNet 表比较，CAGS 是 19-class 32.6 / 48.9，15-class 41.1 / 62.0，10-class 54.8 / 75.9；你们 19-class mIoU 和 mAcc 很强，但 15/10 mIoU 明显低于 CAGS。([arXiv](https://arxiv.org/html/2504.11893v1))

**建议写法：**

> ScanNet direct point-query supports cross-domain 3D feature usability, but it is not presented as a full standardized ScanNet semantic segmentation leaderboard.

---

## 2. 论文主线是否自洽？

现在是自洽的，但要继续压住一个核心边界：

**CTF-GS 的本体贡献是 compact teacher feature reconstruction；VPR 是把 rendered-view feature quality 转化为 primitive-level 3D query ability 的桥梁。**

这和旧 PPT 里的“3DGS as reusable foundation-feature memory”相比更准确。旧 PPT 还停留在 RADIO-GS / HCD / FDH / refiner 的表述，没有纳入最新 VPR 结果，所以后续汇报和论文必须彻底替换旧叙事。`radio_gs_submission_presentation.pptx`

建议最终中心 claim 写成：

> **CTF-GS compactly distills high-dimensional RADIO teacher features into 3D Gaussian scenes. It renders teacher-compatible novel-view feature maps for 2D open-vocabulary grounding, and uses View-to-Primitive Registration to convert refined rendered features into primitive-level embeddings for direct 3D object selection.**

中文：

> **CTF-GS 将高维 RADIO teacher features 紧凑蒸馏进 3D Gaussian scene；它既能渲染 teacher-compatible novel-view feature map 用于 2D 开放词汇定位，也能通过 VPR 将 refined rendered features 注册回 Gaussian primitives，用于 direct 3D object selection。**

这条主线现在能把所有模块串起来：

| 模块 | 论文名 | 解决的问题 |
|---|---|---|
| Hybrid field | **HGCF: Hybrid Gaussian Code Field** | 高维 teacher feature 的 compact storage |
| HCD | **CTR: Compact-to-Teacher Reconstruction** | compact code 回到 1280d RADIO teacher space |
| Refiner | **VFA: View-Space Feature Aligner** | rendered feature 的 boundary / peak 修正 |
| FDH | **FGC: Frozen Geometry-Head Consistency** | 几何感知 frozen-head feature regularization |
| registered_view | **VPR: View-to-Primitive Registration** | 将 rendered feature 注册回 primitives，支撑 direct 3D selection |
| voxel aggregation | **Voxel context aggregation** | 缓解 primitive fragmentation |

仓库文档也已经把当前主线整理为 HGCF、CTR、VFA、FGC、VPR，并将 RADIO-GS 保留为实现名、CTF-GS 作为论文方法名。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md))

---

## 3. 动机和 related work 是否合理？

整体合理，但 related work 需要按你现在的最终 claim 重新梳理，而不是继续围绕“foundation feature memory”泛泛展开。

### 建议 related work 组织

**第一类：3D feature fields / language fields。**  
讲 LERF、LangSplat、LEGaussians、Feature3DGS、FMGS。目的：承认已有工作已经做 feature lifting / language field，不要 claim first。

**第二类：primitive-level / instance-aware Gaussian open-vocabulary understanding。**  
讲 OpenGaussian、Dr. Splat、CAGS、InstanceGaussian、OpenGaFF。目的：解释为什么你们需要 VPR。OpenGaussian 明确指出很多方法停留在 2D pixel-level parsing，而 3D point-level task 会受 feature expressiveness 和 2D-3D association 不准确影响；这正好是 VPR 的动机。([arXiv](https://arxiv.org/html/2406.02058v2))

**第三类：unified teacher foundation features。**  
讲 RADIO / C-RADIO、multi-teacher distillation、SigLIP / DINO / SAM adaptor。目的：解释你们不是单纯存 CLIP language embedding，而是重建 unified teacher feature space。

**第四类：context / aggregation / registration。**  
讲 Dr. Splat dominant-ray registration、CAGS context propagation、InstanceGaussian instance aggregation、OpenGaFF codebook attention。目的：说明 VPR 和 voxel context 在这条线上的位置。CAGS 明确将碎片化归因于 isolated per-Gaussian feature learning 缺乏 spatial context，并用 local graph / mask-centric contrastive learning 缓解 fragmentation。([arXiv](https://arxiv.org/abs/2504.11893?utm_source=chatgpt.com))

### 需要避免的 related work 表述

不要写：

> Existing methods cannot do 3D object selection.

这是错的。OpenGaussian、Dr. Splat、CAGS、InstanceGaussian、OpenGaFF 都已经在做。

要写：

> Existing primitive-level methods often attach language / instance embeddings directly to Gaussians. CTF-GS instead starts from compact teacher-feature reconstruction and uses VPR to expose a primitive-level readout from refined rendered features.

这才是你的差异点。

---

## 4. 方法描述还需要补哪些实现细节？

现在方法大体完整，但 VPR 的可审查性还不够。你需要在正文或 appendix 加一个 **VPR Protocol Card**，否则 reviewer 会重点质疑 fair evaluation。

建议写成一张表：

| Item | 当前设置 | 为什么重要 |
|---|---|---|
| Registration feature | VFA-refined rendered RADIO feature projected by frozen SigLIP2 | 说明不是 raw Gaussian-center |
| Registration views | 96 views main；24 train-view-only audit | 防止 view leakage 质疑 |
| Visibility filter | depth / alpha check | 防止错误注册到不可见 Gaussian |
| Context aggregation | GT-free voxel-max | 缓解 fragmentation |
| Selector | fixed top-2% primitive selector | 避免 per-query threshold tuning |
| GT masks | only final metric stage | 排除用 GT 调 scoring |
| Text head | frozen SigLIP2 | 与 CLIP baselines 区分 |
| External baselines | official-source context, not local rerun | 避免伪 SOTA |

你们 draft 已经写到：VPR renders teacher-space features, applies VFA and frozen SigLIP2, registers back to visible Gaussian centers with depth/alpha checks, and uses LERF masks only for final evaluation。这个方向是对的。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

但还需要补四个关键实现细节：

**第一，96-view all-poses 是否包含 evaluation annotated frames。**  
如果 VFA 使用真实 RGB guide，那么使用 eval frames 会有 leakage 风险。若 VFA 只用 rendered signals / alpha / depth / predicted feature，不用 GT RGB，也要明确写。若不确定，主表建议用 train-view-only 或同时报告 train-view-only 主结果。

**第二，top-2% 是否通过 test masks 调参。**  
如果 top-2% 是看 LERF GT 后选的，不能作为 clean protocol。建议补一个 selector sensitivity curve：top1%、top2%、top5%、top10%，并说明主值是预先固定或通过 validation chosen。

**第三，raw Gaussian-center 指标前后不一致。**  
你在叙述里说 raw Gaussian-center 是 0.0804 / 0.0932，但 VPR ablation 表里 raw Gaussian-center under fixed top-2% 是 0.012 / 0.009。这个不是错误，但必须写清：0.0804 是旧 fixed top0p1；0.012 是同 top2% selector 下的 raw center。当前表 caption 还不够明显。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/lerf_vpr_ablation_table.tex))

建议表里加两行：

| Variant | Selector | mIoU | Acc |
|---|---|---:|---:|
| Raw center | top10%, previous protocol | 0.0804 | 0.0932 |
| Raw center | top2%, VPR protocol | 0.012 | 0.009 |

**第四，VPR cache / voxel context 的存储成本。**  
现在 storage table 报告 direct 1280-D fp16 vs compact total，节省 1.74× 到 4.04×。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md)) 但 VPR 如果额外缓存 per-Gaussian SigLIP2 embeddings 或 voxel scores，这部分必须进入 storage / runtime table，否则 compact claim 会被质疑。

---

## 5. 实验设置是否完整无误？

我不会说“已经完整无误”。更准确是：**核心实验已经闭环，但还有几项必须补强，尤其是 protocol hygiene 和 baseline provenance。**

### 已经完整的部分

**LERF rendered-view grounding** 已经完整：四场景、LocAcc/mIoU、temperature、teacher-vs-rendered、组件消融都有。

**LERF direct 3D object selection** 已经从之前的 limitation 变成主实验：OpenGaussian-style query-select-render、VPR main result、VPR ablation、train-view audit、VFA/depth/voxel ablation都有。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/lerf_direct_3d_selection_table.tex))

**ScanNet direct point-query** 已经能作为 cross-domain feature probe：19/15/10 class splits、mIoU/mAcc、DINO cross-view ablation都有。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

**Storage / efficiency** 已经有初版：eval wall time、peak VRAM、storage footprint。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

### 仍然需要补强的部分

**第一，外部 baseline 仍不是 strict same-evaluator。**  
你现在能说 “official-source context”，不能说 strict SOTA。若要 SOTA，至少要本地同协议复现 OpenGaussian、Dr. Splat、CAGS 中的一个强 baseline。OpenGaussian 的官方 repo 提供了 LERF evaluation 流程，但也说明最新 code / hyperparameter 可能使指标高于 paper，且 LERF metrics 可能不稳定。([GitHub](https://github.com/yanmin-wu/OpenGaussian))

**第二，VPR 96-view main protocol 要防 leakage。**  
建议把 train-view-only row 放进主文，而不是只放 appendix。现在 train-view-only 24 views + voxel 是 0.349 / 0.595，虽然低于 96-view，但已经证明 VPR 不是靠 annotated frames 才有效。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/lerf_vpr_ablation_table.tex))

**第三，direct 3D selection 的 margin 太小。**  
0.3850 vs OpenGaussian 0.3836 的 mIoU margin 太小，建议加 query-level bootstrap confidence interval。否则“slightly exceeds”可以写，但不能强调。Acc@0.25 的 margin较大，更适合作为亮点。

**第四，ScanNet 仍要确认协议细节。**  
如果要和 OpenGaussian / CAGS / Dr. Splat 的 ScanNet numbers比较，需要核对：10 scenes 是否一致、label PLY 是否一致、是否 every-20-frame training、是否 freeze coordinates / disable densification、text head 是否同源。CAGS 的 ScanNet protocol 明确提到使用与 Dr. Splat 相同 10 scenes、every 20 frames、freeze coordinates / deactivate densification 以对齐输入点云和 GT 点云。([arXiv](https://arxiv.org/html/2504.11893v1))

---

## 6. 目前最该改的 quantitative tables

### 表 1：LERF rendered-view grounding

保留现在主表，但标题改成：

> **Rendered-view open-vocabulary grounding on LERF-OVS.**

不要叫 overall LERF OVS SOTA。

### 表 2：LERF direct 3D object selection

现在表太少，只放 OpenGaussian 和 CTF-GS 容易被认为 cherry-pick。建议改成两版：

**主文版：**

| Method | Source | Text head | mIoU | Acc@0.25 |
|---|---|---|---:|---:|
| OpenGaussian | official paper | CLIP | 0.3836 | 0.5143 |
| CTF-GS raw center | local | SigLIP2 | 0.0804 | 0.0932 |
| CTF-GS + VPR | local | SigLIP2 | 0.3850 | 0.6428 |

**appendix context版：**

| Method | mIoU | Acc@0.25 |
|---|---:|---:|
| OpenGaussian | 38.36 | 51.43 |
| Dr. Splat | 43.29 | 64.30 |
| CAGS | 50.79 | 69.62 |
| InstanceGaussian | 45.30 | 58.44 |
| OpenGaFF | 54.36 | 80.84 |
| CTF-GS + VPR | 38.50 | 64.28 |

这个 context 表能避免 reviewer 说你忽略近作。CAGS、InstanceGaussian、OpenGaFF 的 published numbers 都高于你们 mIoU，所以这张表应放在 “context / not same evaluator” 里，而不是主 SOTA claim 表。([arXiv](https://arxiv.org/html/2504.11893v1))

### 表 3：VPR ablation

现在表很有价值，但要补两点：

1. 加 `registered views source`：train-only / all-pose。
2. 加 `selector`：top2% / previous top10%。

目前 “w/o depth check” mIoU 0.347 / Acc 0.615，不差于 24-view VPR，这说明 depth check 不是主提升项，甚至可能牺牲 coverage；文字里不要说 depth check 是核心必要项，而应说它是 conservative visibility hygiene。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/lerf_vpr_ablation_table.tex))

### 表 4：Storage + VPR runtime

需要新增：

| Component | Storage / Runtime |
|---|---:|
| HGCF checkpoint |
| CTR / VFA params |
| VPR registered primitive cache |
| voxel context cache |
| 24-view registration time |
| 96-view registration time |
| direct selection query time |

否则标题里 “compact” 与 VPR 的额外代价之间会有空洞。

---

## 7. 定性可视化现在最缺什么？

你说得对：**目前 qualitative 还不够完整，而且旧图很可能跟不上 VPR 主线。** 论文里至少需要 4 组主图。

### Figure 1：Framework figure

必须重画。旧方法图只到 rendered features，不够了。

建议结构：

**Left：Training**

RGB / poses / frozen 3DGS  
→ RADIO teacher features  
→ HGCF  
→ CTR  
→ VFA  
→ losses：feature reconstruction + FGC + optional FAC

**Right：Inference**

Rendered readout：HGCF → CTR → VFA → 2D heatmap  
VPR readout：VFA rendered feature → SigLIP2 → depth/alpha registration → primitive embedding → query-select-render  
Point readout：HGCF / CTR → ScanNet vertex query

这张图要明确：**VPR 不是另一个 teacher rerun，也不是 GT mask registration。**

---

### Figure 2：LERF rendered-view qualitative

每个 scene 一行，四列或五列：

1. RGB
2. GT mask
3. original RADIO RGB heatmap
4. CTF-GS rendered heatmap
5. error / overlay

每个 scene 选一个代表 query，不要只选最好看的。建议固定：

- Figurines：一个小目标成功或失败案例；
- Ramen：强成功案例；
- Teatime：强成功案例；
- Waldo：LocAcc 提升但 mIoU 可能下降的案例。

现有 draft 已经引用 `submission_freeze_lerf_qualitative_comparison.png`，但这张图需要检查是否包含 teacher vs rendered、GT mask、query 名称、统一 colormap 和色条。draft 目前只写了 qualitative shortlist 和几个 frame id。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

---

### Figure 3：VPR direct 3D object selection qualitative

这是现在最重要的新图。建议每个 scene 一行：

1. Reference RGB
2. GT object mask
3. Raw Gaussian-center selection render
4. 24-view VPR
5. 96-view VPR + voxel
6. FP / FN error map

特别要展示 Waldo Kitchen。因为 Waldo 是你们落后 OpenGaussian 的主要场景，必须主动解释。

这张图的 caption 要写：

> VPR converts high-quality rendered features into primitive-level embeddings. Raw Gaussian-center readout is fragmented; VPR improves object coverage, while Waldo remains difficult due to object fragmentation and clutter.

---

### Figure 4：VPR ablation qualitative

挑一个 Ramen / Teatime 成功案例和一个 Waldo 失败案例，展示：

- VPR w/o VFA
- VPR w/o depth check
- VPR 24 views
- VPR 96 views + voxel

这张图直接对应 ablation 表。现在 w/o VFA 从 0.342/0.555 级别掉到 0.070/0.099，视觉上应该非常明显。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/lerf_vpr_ablation_table.tex))

---

### Figure 5：ScanNet direct point-query qualitative

至少放 2 个 scene：

1. GT point labels
2. CTF-GS prediction
3. error map
4. optional DINO cross-view variant

这张图不是为了 SOTA，而是为了证明 feature field 能 direct 3D query。

---

### Figure 6：Failure analysis

不要只放失败截图，要有诊断图：

- small object size vs LocAcc failure；
- Waldo registration coverage map；
- selected primitives spatial distribution；
- mIoU high but LocAcc low 的 heatmap例子；
- voxel context 前后 fragmentation 对比。

---

## 8. 定性图生成的具体规范

为了避免 reviewer 觉得 cherry-pick，建议写一个 `figure_manifest.json`，每张图记录：

| Field | 内容 |
|---|---|
| scene | figurines / ramen / teatime / waldo |
| frame id | 固定 |
| query | text |
| metric | LocAcc / IoU |
| selection rule | best / median / failure |
| feature source | teacher / rendered / VPR |
| checkpoint | exact path |
| temperature / top-p | exact value |

定性选择规则建议：

- 主图：每个 scene 选一个 **median-quality** query，而不是 top query；
- failure图：每个 scene 选一个 worst query；
- VPR图：每个 scene 选同一 query 展示 raw → VPR → voxel；
- 所有 heatmap 用同一 colormap；
- 同一 row 中 heatmap normalize 规则固定，最好标明 per-query normalization；
- binary mask 用一致颜色，FP/FN 分别用红/蓝；
- figure caption 写清是否使用 teacher heatmap、rendered heatmap、registered primitive mask。

---

## 9. 还可以继续提升哪些指标？

如果你还想提高精度，优先级如下。

### P0：Waldo Kitchen targeted audit

Waldo 是 direct 3D mIoU 的主要瓶颈：你们 0.1991，OpenGaussian 0.227，OpenGaFF 0.4861，CAGS 0.3762。仓库主线也记录 Waldo remains weak。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

下一步不要盲目调 top-p，而要做：

- query-level breakdown；
- registration coverage；
- selected component count；
- false positive region source；
- voxel context 前后 selected primitive 数；
- per-query IoU histogram。

如果 Waldo 是 object fragmentation，应该加 instance/proposal aggregation；如果是 text ambiguity，应该加 prompt ensemble / background negatives。

---

### P1：instance / component-aware VPR

现在 voxel-max 是 GT-free context aggregation，但仍不是 instance grouping。CAGS、InstanceGaussian、OpenGaFF 的高 mIoU都说明：**direct 3D mIoU 想继续涨，必须从 primitive-level score 转向 instance/proposal-level readout。**

可做的轻量升级：

1. VPR registered embeddings 建 kNN graph；
2. 按 voxel + feature similarity 做 connected components；
3. query 时先给 component 打分，再展开到 Gaussians；
4. 对小 component 做 merge / suppress；
5. 保留 GT-free，不用 LERF mask。

这会直接针对 Waldo / Teatime fragmentation。

---

### P2：contribution-weighted registration

现在 VPR 是 depth/alpha visible center registration。Dr. Splat 的关键思路是沿 pixel ray 给 dominant Gaussians 注册 language embedding，而不是简单 center visibility。Dr. Splat 论文把方法定位为 direct language embedding registration，并将 embeddings 分配给 pixel-ray 上的 dominant Gaussians。([arXiv](https://arxiv.org/html/2502.16652v1))

你们可以进一步做：

- top-k alpha contribution registration；
- Gaussian footprint overlap registration；
- contribution-weighted feature average；
- visibility confidence normalization；
- low-opacity primitive suppression。

这比继续加 views 更可能提升 Waldo mIoU。

---

### P3：VPR calibration

目前 fixed top2% 是简单规则。可以补：

- fixed top-p sensitivity；
- query-adaptive but GT-free threshold，例如 score entropy / margin；
- objectness prior；
- background negative prompt；
- prompt ensemble。

但一定不能用 GT masks 调 threshold。

---

## 10. 论文下一步修改清单

### P0：必须改

1. **标题和 abstract 保留 VPR，但去掉 universal SOTA 暗示。**  
   推荐标题：
   > CTF-GS: Compact Teacher Feature Fields with View-to-Primitive Registration for Open-Vocabulary 3D Gaussian Scene Understanding

2. **direct 3D table 加 Dr. Splat / CAGS / InstanceGaussian / OpenGaFF context。**  
   主文可以只放 OpenGaussian，但 appendix 必须给近作 context，避免显得 selective。

3. **补 VPR Protocol Card。**  
   特别说明 96 views、train-only audit、GT mask usage、VFA 是否用 observed RGB、top2% 如何确定。

4. **修正 raw center 数字的表述。**  
   区分旧 top10% 0.0804 和同 top2% 0.012。

5. **加 confidence interval / bootstrap。**  
   因为 0.3850 vs 0.3836 margin 很小。Acc@0.25 可以作为更稳亮点。

6. **把 VPR cache 加入 storage / runtime。**

---

### P1：强烈建议改

1. **主图重画为 CTF-GS + VPR。**
2. **补 VPR qualitative figure。**
3. **补 Waldo failure figure。**
4. **补 ScanNet qualitative。**
5. **把 adaptor内容移到 appendix 或 diagnostic section。**  
   DINO/SAM 现在不要抢主线。

---

### P2：可以继续冲指标

1. Waldo targeted coverage audit。
2. contribution-weighted VPR。
3. component / instance-level VPR readout。
4. GT-free prompt ensemble / negative prompts。
5. local baseline rerun：至少 OpenGaussian 或 Dr. Splat。

---

## 11. 最终投稿判断

现在项目从论文角度是合理的，主线也已经自洽：

**动机合理**：3D scene understanding 需要 queryable feature fields；2D teacher features 强但 view-bound；3DGS 可渲染、显式、高效，但直接存高维 feature 代价高。

**综述合理**：能和 LERF / LangSplat / FMGS 的 rendered feature field、OpenGaussian / Dr. Splat / CAGS 的 primitive-level method、RADIO 的 unified teacher feature 串起来。

**claim 合理**：只要写成 compact teacher feature field + VPR registered primitive-level readout，而不是 raw dual-readout SOTA。

**方法合理**：HGCF、CTR、VFA、FGC、VPR 分工清晰。

**实验基本完整**：LERF rendered-view、teacher-vs-rendered、direct 3D selection、ScanNet、ablation、storage/efficiency 都有。主要缺口是 strict same-evaluator baseline、VPR protocol hygiene、定性图和 confidence interval。

最稳的最终主张是：

> **CTF-GS shows that high-dimensional RADIO teacher features can be compactly distilled into 3D Gaussian scenes. Through teacher-space reconstruction and frozen geometry-head consistency, it produces strong rendered-view features; through View-to-Primitive Registration, those features can be converted into competitive primitive-level embeddings for direct 3D object selection.**

中文：

> **CTF-GS 表明，高维 RADIO teacher features 可以被紧凑蒸馏进 3D Gaussian scene。通过 teacher-space reconstruction 和 frozen geometry-head consistency，它得到强 rendered-view features；通过 VPR，这些 rendered features 可以被注册为 primitive-level embeddings，从而支持有竞争力的 direct 3D object selection。**

这篇现在可以推进投稿，但请不要写“全面 SOTA”。更好的策略是：**claim 新问题和新机制，结果写 strong / competitive / official-context above OpenGaussian，而不是 universal SOTA。**

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md)
- [OpenGaFF: Open-Vocabulary Gaussian Feature Field with Codebook Attention](https://arxiv.org/html/2605.06088v1)
- [GitHub - yanmin-wu/OpenGaussian: [NeurIPS 2024] OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding · GitHub](https://github.com/yanmin-wu/OpenGaussian)

## Prompt:
2026/5/12 23:21:51

有了VPR之后，除了原先的LERF rendered-view grounding和scannet 3d segmentation指标是否有提高？或者说面向direct 3d object selection的结果升级是否带动了整体特征场的质量提升？我还是觉得student 特征场强于teacher 特征才是主线中的重点，实验论证这应该主要聚焦在2d 新视角基础特征的三个下游任务上（变换到sam3/siglip2/dinov3特征后的下游任务，而且由于teacher没有3d能力所以只能在2d做，这个主要是和自己比），然后才是比teacher多出来的3d能力（主要是场景理解能力），包括Direct 3D object selection和3d semantic segmentation，展现一下这两项3d场景理解任务都达到多个对比方法中的sota水准，顺便再展示一下2d场景理解也就是render-grounding（本身属于2d 新视角基础特征的下游任务）也达到多个对比方法中的sota水准，你觉得这样是不是逻辑更好？

## Response:
2026/5/12 23:21:54

我会先区分“VPR提升读出接口”与“student特征整体强于teacher”，再判断新主线能否由现有多头2D和3D实验共同支撑。

## Response:
2026/5/12 23:22:22

现有证据更适合写成“SigLIP2 任务上 student 增强，DINO/SAM 仍需干净 frozen-head benchmark 验证”。VPR 提升的是 direct 3D readout，不应反推整体特征场质量提升。

## Response:
2026/5/12 23:28:52

你这个新逻辑**方向更好**，但现在需要非常精确地分清三件事：

1. **VPR 是否提升了“特征场质量”？**  
   目前看，VPR 主要提升的是 **direct 3D readout 能力**，不是直接提升原始 rendered-view feature field 本身。

2. **student 是否强于 teacher？**  
   目前只在 **SigLIP2 / LERF text grounding** 上有清楚证据；在 DINOv3 / SAM3 下游探针上，当前 aggregate 还没有强于 teacher。

3. **3D 任务是否达到 SOTA？**  
   现在 direct 3D object selection 已经非常接近并略超 OpenGaussian official reference，但如果把 CAGS、Dr. Splat、InstanceGaussian、OpenGaFF 等近作都纳入，不应写成全面 SOTA。

所以我建议把主线升级为：

> **CTF-GS learns a multiview student foundation-feature field that can outperform the frame-wise teacher on selected novel-view downstream tasks, while also enabling 3D scene-understanding readouts unavailable to the 2D teacher.**

中文：

> **CTF-GS 学习一个多视角 student foundation-feature field。它在部分 novel-view 下游任务上优于逐帧 teacher，同时额外支持 2D teacher 本身不具备的 3D 场景理解读出。**

这个逻辑是对的，但要把“部分”写清楚，不能直接说 “student 全面强于 teacher”。

---

## 1. VPR 有没有提高 LERF rendered-view grounding 和 ScanNet？

**严格说，没有证据表明 VPR 已经提高了原先的 LERF rendered-view grounding 或 ScanNet direct point-query 主指标。**

原因很简单：VPR 当前是一个 **readout / registration module**，它把已经渲染出的 VFA-refined RADIO/SigLIP2 features 注册回 Gaussian primitives，用于 direct 3D object selection。仓库主线也把 VPR 定义为 rendered SigLIP2-aligned features → visible Gaussian primitives → 3D query-select-render 的流程。它提升的是 direct 3D object selection：raw Gaussian-center 从 0.0804 / 0.0932 提升到 VPR + voxel 的 0.3850 / 0.6428。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

但 LERF rendered-view grounding 的主结果仍是 0.8712 LocAcc / 0.4941 mIoU；adaptor-enhanced candidate 是 0.8712 / 0.4979。这个提升来自 DINO/SAM/cross-view adaptor branch 的 selector，不是 VPR。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

ScanNet 也是类似。当前主结果仍是 19/15/10 split mIoU 0.3538 / 0.3573 / 0.4293；DINO cross-view 10-scene ablation 有小幅提升到 0.3640 / 0.3662 / 0.4308，但这也是 cross-view DINO branch，不是 VPR。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md))

所以现在最准确的结论是：

> **VPR 没有直接证明整体 feature field 质量提升；它证明高质量 rendered features 可以通过 view-to-primitive registration 转化成强 direct 3D primitive readout。**

换句话说，VPR 证明的是：

> **student feature field 的 2D rendered feature 足够好，以至于可以被注册回 3D 并支撑 direct object selection。**

这很有价值，但它不是“VPR 训练后 feature field 更好”的证据。

如果你想证明 VPR 带动整体特征场质量，需要做新的实验：用 VPR 产生的 primitive consistency / object consistency 反向训练 HGCF/CTR/VFA，然后重新报告 LERF rendered-view、DINO/SAM/SigLIP2 probes 和 ScanNet。现在还不能这么 claim。

---

## 2. “student feature field 强于 teacher”作为主线是否更好？

**作为科学问题，它更好；作为当前论文 claim，需要收紧。**

你想要的主线可以这样写：

> **Does multiview 3D distillation produce a student foundation-feature field that is more useful than frame-wise teacher features?**

这个比“3DGS as reusable feature memory”更有锋芒。它能解释为什么我们不仅做 reconstruction loss，还要做 frozen-head consistency、VFA、DINO/SAM/SigLIP adaptor probes，以及 teacher-vs-rendered 对比。

但是当前数据只支持一个保守版本：

> **CTF-GS improves task-relevant usability over the frame-wise teacher in SigLIP2 text grounding, while DINOv3/SAM3 adaptor probes show partial compatibility but not yet aggregate superiority.**

原因在你们自己的主线文档里已经很清楚：SigLIP2 grounding 下，rendered feature 的 macro LocAcc 从 teacher 的 0.7985 提升到 0.8712，mIoU 从 0.4922 到 0.4941。这个是强证据。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md))

但 DINOv3 / SAM3 下游探针目前是 mixed：DINOv3 rendered mIoU 接近 teacher，但 LocAcc 下降；SAM3 还有明显 teacher-rendered gap。具体表里，DINOv3 prototype segmentation teacher/rendered 是 0.6543/0.6277 LocAcc、0.0945/0.0937 mIoU；SAM3 prototype segmentation teacher/rendered 是 0.8404/0.6649 LocAcc、0.0757/0.0564 mIoU。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

所以你不能现在写：

> student feature field is stronger than the teacher across SigLIP2, DINOv3, and SAM3.

你可以写：

> student feature field is stronger than the frame-wise teacher on SigLIP2 open-vocabulary localization, and remains partially compatible with DINOv3/SAM3 adaptor spaces, exposing the next optimization target.

---

## 3. 你提出的新实验逻辑是否更好？

我认为**整体逻辑更好**，但要改成下面这种层级。

### 第一层：2D novel-view foundation-feature usability

这层是和 teacher 比，因为 teacher 是 2D model，可以在 held-out RGB frame 上抽 feature，但它没有 persistent 3D feature field。

这一层应该回答：

> **student rendered feature 是否比 frame-wise teacher feature 更适合 novel-view 下游任务？**

推荐放三个 frozen-head downstream tasks：

| 下游空间 | 任务 | 当前状态 | 论文角色 |
|---|---|---|---|
| SigLIP2 | open-vocabulary grounding / localization | student 明显优于 teacher LocAcc，mIoU 略优 | **主证据** |
| DINOv3 | matching / prototype segmentation / relation consistency | student 接近 teacher mIoU，但 aggregate LocAcc 低 | 兼容性证据 / 待优化 |
| SAM3 | prompt / prototype segmentation / region probe | student 明显弱于 teacher | limitation / future work |

这层应该是论文的核心，因为它直接回答“student 特征场是否强于 teacher 特征”。

但当前只能把 SigLIP2 写成 strong win，DINO/SAM 写成 compatibility probes。你们旧 PPT 里也已经把 SAM/DINO 定义成 diagnostic，而不是主结果；这点现在仍然成立。`radio_gs_submission_presentation.pptx`

### 第二层：teacher 没有的 3D scene-understanding ability

这层不能再和 teacher 比，因为 2D teacher 没有 Gaussian primitive、3D point cloud 或 persistent scene memory。这里应该和 3D open-vocabulary 方法比。

包括：

| 3D 任务 | 当前结果 | 论文角色 |
|---|---:|---|
| LERF direct 3D object selection | 0.3850 mIoU / 0.6428 Acc@0.25 | VPR 让 CTF-GS 进入 primitive-level 3D evaluation |
| ScanNet direct point-query segmentation | 19/15/10 mIoU 0.3538 / 0.3573 / 0.4293 | cross-domain 3D feature usability |

这层回答：

> **student feature field 不只是 2D teacher 的替代品，而是一个 3D scene representation。**

这个逻辑非常好。

### 第三层：和外部方法比

这一层要谨慎。它不是证明“student 强于 teacher”，而是证明“我们的 3D representation 在公开任务上有竞争力”。

你可以比较：

- LERF rendered-view OVS；
- LERF direct 3D object selection；
- ScanNet direct 3D semantic segmentation / point query。

但这里不能轻易写 SOTA。

---

## 4. 现在能不能说 2D / 3D 都 SOTA？

### 4.1 LERF rendered-view 2D grounding

你们的 LocAcc 很强：0.8712。如果换成百分比是 87.12%。OpenGaFF 最近表中 2D Acc 是 82.68，但它的 2D mIoU 是 64.98，高于你们的 49.41。([arXiv](https://arxiv.org/html/2605.06088v1))

所以你可以说：

> **strong localization performance, competitive 2D rendered-view grounding, and higher localization than several reported references.**

但不建议写：

> SOTA 2D rendered-view OVS.

因为 mIoU 不够。

### 4.2 LERF direct 3D object selection

你们 0.3850 / 0.6428 确实略超 OpenGaussian official 0.3836 / 0.5143。仓库主线也明确记录这个结果，并注明仍是 official-source context，不是本地同 evaluator SOTA。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

但是 CAGS 把 fragmentation 作为核心问题，并报告明显更高的 LERF-OVS/ScanNet表现；Dr. Splat 是 CVPR 2025 direct registration 方法；InstanceGaussian 在 LERF 3D object selection 表中报告 45.30 mIoU / 58.44 mAcc@0.25；OpenGaFF 最近表中 3D LERF 是 54.36 mIoU / 80.84 Acc。([arXiv](https://arxiv.org/abs/2504.11893?utm_source=chatgpt.com))

所以不能写：

> direct 3D object selection SOTA.

可以写：

> **competitive OpenGaussian-style direct 3D object selection, exceeding OpenGaussian official reference in macro mIoU and Acc@0.25, while still below recent instance/context-aware methods in mIoU.**

### 4.3 ScanNet 3D semantic segmentation

你们 ScanNet v67 是有效 3D point-query evidence，但不是 SOTA。OpenGaFF 在 ScanNet-v2 19/15/10 class splits 报告 36.55 / 42.78 / 57.85 mIoU，明显高于你们的 35.38 / 35.73 / 42.93 在 15/10 splits 上的结果；它也报告了更高的 15/10 mAcc。([arXiv](https://arxiv.org/html/2605.06088v1))

所以这里建议写：

> **cross-domain 3D direct point-query support**

不要写：

> ScanNet SOTA.

---

## 5. 你的新主线怎么改最稳？

我建议把文章主线改成“两阶段能力论证”。

### 主 claim

> **CTF-GS learns a multiview student foundation-feature field from a frozen 2D teacher. The student improves selected novel-view downstream usability over frame-wise teacher features and, unlike the 2D teacher, exposes 3D scene-level readouts through point queries and VPR primitive registration.**

中文：

> **CTF-GS 从冻结 2D teacher 中学习一个多视角 student foundation-feature field。该 student 在部分 novel-view 下游任务上提升了逐帧 teacher 的可用性，并且相比 2D teacher 额外提供了 3D point query 和 VPR primitive registration 等场景级读出能力。**

这个主 claim 比“student 全面强于 teacher”更稳，也比“3DGS as feature memory”更有论文味。

### 贡献点建议改成四条

**Contribution 1：问题定义**

> We formulate multiview student foundation-feature fields for 3D Gaussian scenes: the goal is not only to reconstruct teacher features, but to improve novel-view downstream usability and expose additional 3D scene-understanding readouts.

中文：

> 我们提出 3D Gaussian scene 中的多视角 student foundation-feature field 问题：目标不只是重建 teacher feature，而是提升 novel-view 下游可用性，并提供额外的 3D 场景理解读出。

**Contribution 2：compact teacher-feature field**

> HGCF + CTR compress high-dimensional RADIO features into a compact 3D Gaussian feature field and decode them back to teacher-compatible feature space.

**Contribution 3：feature-quality and adaptor-space supervision**

> VFA, FGC, and optional FAC improve rendered feature usability under frozen downstream heads, including SigLIP2, DINOv3, and SAM3.

注意：这里不要写 “all three outperform teacher”，而是写 “evaluate / regularize under”。

**Contribution 4：VPR for 3D readout**

> VPR registers refined rendered features back to Gaussian primitives, enabling direct 3D object selection from the same student field.

---

## 6. 实验章节应该重排

我建议按下面顺序写，而不是按数据集平铺。

### 6.1 Does the student improve over the frame-wise teacher?

这应该成为第一组核心实验。

#### Table A：SigLIP2 text grounding teacher vs student

已有，强推：

| Source | Macro LocAcc | Macro mIoU |
|---|---:|---:|
| RADIO RGB teacher | 0.7985 | 0.4922 |
| CTF-GS rendered student | 0.8712 | 0.4941 |

结论：

> student improves SigLIP2 localization stability under the same evaluator.

#### Table B：DINOv3 downstream probes teacher vs student

当前不要写 win。写：

> DINOv3 probes show partial preservation; rendered student is close in mIoU but lower in localization.

如果你想让主线更强，下一步要优化这张表。目标至少是：

- DINO prototype segmentation mIoU ≥ teacher；
- DINO matching LocAcc 接近或超过 teacher；
- 保证 SigLIP2 不掉。

#### Table C：SAM3 downstream probes teacher vs student

当前是 gap。建议作为 limitation 或 appendix。如果你坚持把 SAM3 放进主线，就必须进一步提升，否则会削弱“student 强于 teacher”。

目标至少是：

- SAM3 point/box prompt mIoU 缩小 gap；
- 或者选择一个更合理的 SAM3 probe，使 teacher/student 对比不是天然偏 teacher。

### 6.2 What 3D abilities does the student gain beyond the teacher?

这一节放：

#### Table D：LERF direct 3D object selection

主表放：

| Method | mIoU | Acc@0.25 |
|---|---:|---:|
| OpenGaussian official | 0.3836 | 0.5143 |
| CTF-GS raw center | 0.0804 | 0.0932 |
| CTF-GS + VPR + voxel | 0.3850 | 0.6428 |

同时加 context：

> not universal SOTA; recent context-aware/instance-aware methods report higher mIoU.

#### Table E：ScanNet direct point-query

放当前 19/15/10 split，说明是 3D scene understanding support。

### 6.3 How does it compare with external methods?

这一节放外部 comparison，但 caption 必须写 protocol caveat。

- 2D LERF rendered-view：strong localization, not mIoU SOTA。
- 3D LERF direct selection：above OpenGaussian official, below some recent methods。
- ScanNet：competitive / support evidence, not SOTA。

### 6.4 What components matter?

最后放 ablation：

- CTR / HCD；
- FGC；
- VFA；
- HGCF；
- FAC；
- VPR；
- voxel context。

这样逻辑比“LERF → ScanNet → ablation”更好，因为它先回答你的核心科学问题。

---

## 7. 如果你坚持“student feature field 强于 teacher”作为主线，下一步必须补什么？

现在还差两类证据。

### 7.1 多头 teacher-vs-student 不能只靠 SigLIP2

当前 SigLIP2 是 win，DINO/SAM 是 mixed/弱。你需要把 DINO/SAM 提上来。

建议训练一个 **FAC-main checkpoint**：

$$
\mathcal{L}
=
\mathcal{L}_{RADIO}
+
\lambda_{sig}\mathcal{L}_{SigLIP2}
+
\lambda_{dino}\mathcal{L}_{DINO-rel}
+
\lambda_{sam}\mathcal{L}_{SAM-region}
+
\lambda_{fgc}\mathcal{L}_{FGC}
+
\lambda_{peak}\mathcal{L}_{text-peak}
$$

关键是加 **peak-preservation**，否则 DINO/SAM 会提高 region overlap 但伤 LocAcc。你们已有经验正是：DINO/SAM 分支经常让 mIoU 上去，但移动 argmax peak。草稿也明确写了这个 failure mode。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex))

训练目标不是让所有指标都最大，而是：

- SigLIP2 LocAcc 不掉；
- DINO/SAM mIoU 接近或超过 teacher；
- SAM gap 缩小；
- ScanNet 不掉。

### 7.2 “student 强于 teacher”要用 held-out novel-view protocol

必须确保 teacher 和 student 比的是同一批 held-out RGB frames：

- teacher：直接对 held-out RGB 跑 RADIO；
- student：只用 3D field 渲染 feature；
- evaluator：同一个 frozen head；
- no GT masks used for training；
- no evaluation view leakage。

这样才是干净的 teacher-vs-student。

---

## 8. 这会不会削弱 VPR 的地位？

不会。VPR 在新逻辑里更自然。

它不再被解释为“整体 feature field 更强”的证据，而是解释为：

> **如果 student rendered features 已经具备高质量 SigLIP2/RADIO usability，那么 VPR 可以把这种 2D feature quality 转化成 3D primitive-level usability。**

也就是：

1. **2D teacher-vs-student** 证明 student rendered features 好；
2. **VPR direct selection** 证明这些 rendered features 可以回注到 3D primitives；
3. **ScanNet point-query** 证明 feature field 本身也能被 3D points 查询。

这个链条很顺。

---

## 9. 最终建议：采用你的逻辑，但不要过度 claim SOTA

我建议最终故事线这样写：

### 标题

> **CTF-GS: Multiview Student Foundation-Feature Fields with View-to-Primitive Registration for Open-Vocabulary 3D Scene Understanding**

比现在的标题更强调 student-vs-teacher 和 VPR。

### 摘要主句

> **We ask whether multiview 3D distillation can produce a student foundation-feature field that is more useful than frame-wise teacher features at novel views, while also enabling 3D scene-level readouts unavailable to the 2D teacher.**

### 实验主线

1. **2D novel-view teacher-vs-student**
   - SigLIP2 grounding：student wins；
   - DINOv3：student close / needs improvement；
   - SAM3：student gap / limitation or improve before main claim。

2. **3D abilities beyond teacher**
   - LERF direct 3D selection via VPR；
   - ScanNet direct point-query。

3. **External method comparison**
   - 2D LERF：strong LocAcc, not mIoU SOTA；
   - 3D LERF：above OpenGaussian official, not universal SOTA；
   - ScanNet：supportive, not SOTA。

4. **Ablation**
   - CTR / FGC / VFA / HGCF / FAC / VPR / voxel context。

### 最稳结论

> **CTF-GS does not yet show that the student universally dominates the teacher across all adaptor spaces. It does show that multiview distillation can improve SigLIP2 novel-view localization over the frame-wise teacher, preserve useful DINO/SAM compatibility to varying degrees, and add 3D scene-understanding abilities through VPR and direct point queries.**

中文：

> **CTF-GS 目前还不能证明 student 在所有 adaptor 空间都全面强于 teacher。它已经证明：多视角蒸馏可以在 SigLIP2 novel-view localization 上超过逐帧 teacher，在 DINO/SAM 空间保留一定兼容性，并通过 VPR 和 direct point query 获得 2D teacher 不具备的 3D 场景理解能力。**

这就是我建议的最终逻辑。它更清晰，也更能经受审稿人追问。

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md)
- [OpenGaFF: Open-Vocabulary Gaussian Feature Field with Codebook Attention](https://arxiv.org/html/2605.06088v1)

## Prompt:
2026/5/12 23:45:40

和SigLIP2下游任务用grounding来验证一样，DINOv3 / SAM3也应该用某个下游应用来验证，SAM3就应该是实例分割？dinov3是不是可以用某个下游任务头验证就行？只要和teacher保持协议一样就行，无论是radio的adaptor还是DINOv3 / SAM3的官方下游head都是冻住的。这里难道还需要直接对比特征本体进行评估吗？
此外，每个需要做实验验证的任务，你帮我选一个论文里对比方法实验结果相对最全的开源对比方法，我直接用他的协议去评估。
最后，我就是想要进一步提升每个任务的指标，该如何继续优化方法架构或者训练策略等等？

## Response:
2026/5/12 23:45:43

我会把这拆成三层：先定“用什么任务证明 student 比 teacher 更有用”，再为每个任务选协议锚点，最后给出针对性提指标路线。

## Response:
2026/5/12 23:45:48

这里需要进一步避免把 adaptor-space 兼容性、下游任务提升和 3D readout 提升混成一句大 claim。DINO/SAM 应绑定明确应用任务，特征本体指标只作 sanity check。

## Response:
2026/5/12 23:50:27

> **Choosing method options and structuring tasks**
>
> For tasks related to ScanNet, CAGS should be used since it offers reproducible results with OpenGaussian data and more baseline comparisons like DrSplat. For DINOv3 tasks, use its official dense/sparse matching and compare Teacher vs. CTF-GS rendered. For SAM3 instance segmentation, use SAM3's official promptable segmentation protocol with point/box/text prompts.
>
> Thought for 8m 13s

## Response:
2026/5/12 23:53:56

你的判断是对的：**DINOv3 / SAM3 不应该只做“特征相似度”评估，最好都落到具体下游任务。** 论文主线也可以改得更锋利：

> **CTF-GS 学到的不是一个数值上贴近 RADIO teacher 的 feature copy，而是一个 multi-view student foundation-feature field。它应该在 2D novel-view frozen-head downstream tasks 上和 frame-wise teacher 对比，并在 teacher 不具备的 3D scene-understanding tasks 上和 3D 方法对比。**

但有一个边界必须守住：**目前数据还不能说 student 在 SigLIP2 / DINOv3 / SAM3 三个空间都强于 teacher。** 仓库当前结果显示 SigLIP2 grounding 是明确 win；DINOv3 接近但 LocAcc 有下降；SAM3 仍有明显 teacher-rendered gap。项目主线文档也已经把 DINOv3/SAM3 probes 定义为 diagnostic rather than main-claim improvements。([raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md)) 旧 PPT 里也把 DINO/SAM 放在 adaptor ablation / diagnostic，而不是主贡献，这个判断目前仍然成立。`radio_gs_submission_presentation.pptx`

---

## 1. 是否还需要直接评估“特征本体”？

**主论文不需要把 feature intrinsic metric 作为核心证据。**

你现在这篇文章真正要证明的是：

> **student rendered features 在 frozen downstream heads 下是否更有用。**

所以主实验应该是下游任务，而不是 cosine / L2 / CKA 这类特征本体指标。特征本体评估可以放 appendix，用来解释失败和消融，比如：

| 指标 | 作用 | 是否主表 |
|---|---|---|
| RADIO-space cosine / L1 | 证明 CTR 没崩 | Appendix / ablation |
| adaptor-space cosine | 诊断 DINO/SAM gap | Appendix |
| CKA / Gram similarity | 解释 DINO relation preservation | Appendix |
| feature norm / entropy | 诊断 heatmap peak shift | Appendix |
| frozen-head downstream metrics | 证明 feature usability | **主表** |

因此论文里应写：

> We evaluate reconstructed features primarily by downstream frozen-head usability, not by feature reconstruction error alone.

---

## 2. 三个 2D novel-view foundation-feature downstream tasks 怎么定义？

### 2.1 SigLIP2：open-vocabulary grounding

这个已经是你最强的任务。

**任务定义：**

- teacher：对 held-out RGB frame 跑 RADIO/C-RADIO，经过 frozen SigLIP2 adaptor；
- student：从 3DGS novel view 渲染 CTF-GS feature，经过同一个 frozen SigLIP2 adaptor；
- query：LERF text query；
- 输出：2D heatmap；
- 指标：LocAcc / mIoU。

你们现在的结果已经能支持 “student 在 SigLIP2 grounding 上强于 frame-wise teacher”：RADIO RGB teacher macro LocAcc 0.7985 / mIoU 0.4922，CTF-GS rendered student 0.8712 / 0.4941。([raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md))

**推荐协议锚点：LangSplat / LERF rendered-view protocol。**

LangSplat 是开源的 CVPR 2024 Highlight，提供 LERF dataset、pretrained models 和 evaluation 流程；它的 LERF evaluation 是先 render language feature map，再做 LERF-style localization / segmentation evaluation。([github.com](https://github.com/minghanqin/LangSplat)) 你的 rendered-view grounding 表可以以 LangSplat-style LERF 协议为外部比较锚点，但 teacher-vs-student 表必须使用你自己的 same frozen SigLIP2 evaluator。

---

### 2.2 DINOv3：dense correspondence / mask propagation

DINOv3 最适合验证的不是普通语义分割，而是 **dense correspondence / cross-view matching / mask propagation**。

原因是 DINOv3 官方定位就是强 dense feature map，适合 geometric tasks such as depth estimation or 3D matching；官方 repo 也提供 dense/sparse matching、segmentation tracking、DINOv3-based zero-shot segmentation 等 notebooks。([arxiv.org](https://arxiv.org/html/2508.10104v1)) ([github.com](https://github.com/facebookresearch/dinov3))

我建议把 DINOv3 任务正式定义为：

> **DINOv3 Cross-View Mask Propagation**

协议如下：

1. 在同一 LERF scene 中选 source view 和 target novel view；
2. source view 有 GT object mask；
3. teacher 和 student 分别输出 DINOv3 adaptor-space dense features；
4. 用 DINO feature nearest-neighbor / soft correspondence 把 source mask propagate 到 target view；
5. 与 target GT mask 比较；
6. 指标：target mIoU、pointing accuracy / LocAcc、PCK-style correspondence accuracy、cycle consistency error。

这个任务比你现在的 “prototype segmentation / source-target matching probe” 更像正式下游应用。你现在已有的 DINO probe 可以改造成这个协议：当前文档里 DINOv3 prototype segmentation 和 source-target matching 已经完成，但结果是 teacher 略强或接近，rendered DINO mIoU 接近 teacher、LocAcc 下降。([raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

**推荐协议锚点：DINOv3 official dense/sparse matching + segmentation tracking protocol。**

这不是 3DGS 对比方法，而是 DINOv3 下游评估本身的官方开源协议。因为这个实验的核心是 teacher-vs-student，不是和外部 3DGS 方法比。你可以在论文里说：

> We follow the DINOv3-style dense matching / mask propagation setup and use the same frozen DINOv3 adaptor for teacher and student features.

---

### 2.3 SAM3：promptable instance / concept segmentation

SAM3 应该验证 **segmentation**，这个判断对。

但这里要特别小心：**只有当你的 RADIO SAM3 adaptor 能真正接入 frozen SAM3 mask decoder / task head 时，才能把实验叫做 SAM3 instance segmentation。** 如果你只是把 RADIO feature 投到 SAM3 adaptor space，然后做 prototype segmentation，那论文里应叫：

> SAM3-adaptor region segmentation probe

而不是：

> SAM3 instance segmentation.

SAM3 官方任务是 promptable segmentation：它可以用 text、point、box、mask、exemplar 等 prompts 做图像和视频中的检测、分割和跟踪；官方 SA-Co/Gold benchmark 是 promptable concept segmentation，图像和 noun phrase 配对并有 exhaustive instance masks，官方指标是 cgF1。([github.com](https://github.com/facebookresearch/sam3)) ([github.com](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/gold/README.md))

对你的论文，我建议用两个版本：

#### Version A：最适合当前项目的 teacher-vs-student 版本

**SAM3 Prompted Instance Segmentation on LERF-OVS**

- teacher：held-out RGB → RADIO teacher feature → frozen SAM3 adaptor/head；
- student：CTF-GS rendered feature → same frozen SAM3 adaptor/head；
- prompt：GT box / center point / text phrase 三种可以选一到两种；
- 输出：instance mask；
- 指标：mask mIoU、Dice、boundary F、Acc@0.5，若做 text concept segmentation可加 cgF1-style presence + mask score；
- GT mask 只用于 prompt construction 和 evaluation；如果用 box/point prompt，这在 SAM 系列评估里是常规做法，但要写清。

如果你能做 text-prompt all-instance segmentation，就更强；如果做不了，就先做 box/point prompt，因为它更稳定地测试 SAM-style region quality。

#### Version B：更标准但不一定适合你的 3D论文

**SA-Co/Gold official SAM3 benchmark**

这个协议最标准，但它是 generic image benchmark，不是你的 3D scene novel-view benchmark。除非你能为 SA-Co 图像构建 3D scene 并渲染 student features，否则它不能直接用于 CTF-GS。

**推荐协议锚点：SAM3 official promptable concept segmentation protocol；若转到 3DGS promptable segmentation，则用 SAGA/Gaussian Grouping 作 3D协议参考。**

SAGA 是开源的 3D Gaussian promptable segmentation方法，它把 SAM segmentation capability distill 到 Gaussian affinity features，并支持 promptable 3D segmentation。([github.com](https://github.com/Jumpat/SegAnyGAussians)) 但 SAGA 是 3D promptable segmentation方法，不是你的 2D novel-view teacher-vs-student协议。不要混淆。

---

## 3. 每个实验该选哪个开源协议锚点？

我建议你按下面这张表定稿。

| 论文证据层 | 任务 | 你应该跑什么 | 推荐协议锚点 |
|---|---|---|---|
| 2D teacher-vs-student | SigLIP2 grounding | LERF rendered-view heatmap LocAcc / mIoU | **LangSplat / LERF rendered-view protocol** |
| 2D teacher-vs-student | DINOv3 correspondence | LERF source-target mask propagation / dense matching | **DINOv3 official dense/sparse matching + segmentation tracking setup** |
| 2D teacher-vs-student | SAM3 segmentation | LERF prompted instance segmentation，box/point/text prompt | **SAM3 official promptable segmentation task; cgF1 only if doing text concept segmentation** |
| 3D beyond teacher | Direct 3D object selection | LERF query-select-render mIoU / Acc@0.25 | **OpenGaussian protocol** |
| 3D beyond teacher | 3D semantic segmentation | ScanNet 19/15/10 point-level mIoU / mAcc | **OpenGaussian ScanNet protocol** |
| Optional 3D promptable segmentation | Prompted 3D Gaussian segmentation | click/box/text prompt to segment Gaussians | **SAGA / Gaussian Grouping** |

为什么我建议 3D 任务继续用 OpenGaussian？因为 OpenGaussian 官方 repo 同时覆盖 ScanNet 19/15/10 evaluation 和 LERF open-vocabulary object selection in 3D space，并明确流程是 text-selected Gaussians → render multi-view masks → compute metrics。([github.com](https://github.com/yanmin-wu/OpenGaussian)) Dr. Splat 是更近的 CVPR 2025 Highlight，而且提出 direct language embedding registration 和 PQ compact representation，很适合做强相关工作和对比上下文；但它的 volume-aware 3DGS pseudo-label protocol 会改变你的评估定义，切换成本更高。([arxiv.org](https://arxiv.org/html/2502.16652v1)) CAGS 也很相关，但它的 GitHub README 明确说数据准备沿用 OpenGaussian，并且 ScanNet evaluation 部分仍是 TODO，所以不适合作为唯一协议锚点。([github.com](https://github.com/Wistzz/CAGS))

---

## 4. 论文主线可以怎样改得更顺？

你现在想强调：

> student 特征场强于 teacher 特征。

我建议改成更稳、更科学的版本：

> **Multiview 3D distillation can produce a student foundation-feature field that improves selected novel-view downstream usability over frame-wise teacher features, while enabling 3D scene-understanding tasks unavailable to the 2D teacher.**

中文：

> **多视角 3D 蒸馏可以得到一个 student foundation-feature field。它在部分 novel-view 下游任务上优于逐帧 teacher，并额外支持 2D teacher 不具备的 3D 场景理解任务。**

这样可以把实验自然分成两组。

### Group A：2D novel-view downstream tasks，和 teacher 比

| Task | 比较对象 | 目标 |
|---|---|---|
| SigLIP2 grounding | teacher RGB feature vs student rendered feature | 证明 text alignment / localization usability |
| DINOv3 mask propagation | teacher DINO-space feature vs student DINO-space feature | 证明 correspondence / structure preservation |
| SAM3 prompted segmentation | teacher SAM-space feature vs student SAM-space feature | 证明 region / instance mask usability |

### Group B：3D scene understanding，和 3D方法比

| Task | 比较对象 | 目标 |
|---|---|---|
| LERF direct 3D object selection | OpenGaussian / Dr. Splat / CAGS / InstanceGaussian context | 证明 VPR primitive-level readout |
| ScanNet 3D semantic segmentation | OpenGaussian-style 19/15/10 | 证明 point-level scene semantics |

这套逻辑比“所有实验都混在 LERF/ScanNet下面”更好。

---

## 5. 如果要进一步提升每个任务，具体怎么改？

下面我按任务给你可执行的优化方向。

---

### 5.1 SigLIP2 grounding：继续提升 LocAcc 和 mIoU

当前 SigLIP2 是你最强任务。下一步应该提高 mIoU，同时不损失 LocAcc。

**方法改动：**

1. **Peak-preserving text heatmap distillation**  
   你已经有 spatial text-heatmap KL。继续强化，但要用固定 text bank，不能用 test GT masks。目标是保住 argmax peak。

2. **Multi-temperature training / calibration**  
   当前不同 scene 用不同 temperature。可以训练时随机化 temperature 或加入 entropy regularization，让 heatmap 不依赖单一温度。

3. **Boundary-aware VFA**  
   VFA 输入里加入 rendered depth edge、alpha edge、RGB edge，一起预测 feature residual。目标是提高 mIoU，不让 heatmap 泄漏到背景。

4. **Small-object oversampling**  
   Figurines 是小目标瓶颈。训练时对小 mask / high-frequency views 加权，或者增加 feature render resolution。

5. **Negative prompt contrast**  
   对同场景 query 加 background / distractor negatives，例如 “not cup / background / table / hand”。这可以减少 Waldo / kitchen 类场景混淆。

---

### 5.2 DINOv3：把 probe 升级为正式 mask propagation，并优化 correspondence

你现在 DINO 结果不够强，说明 DINO relation loss 可能让区域更平滑，但没有保住 target peak。

**建议正式任务：**

> Source-view mask propagation to target novel view.

**方法改动：**

1. **DINO cross-view cycle consistency**  
   source → target → source，要求 mask 或 keypoint 能 cycle back。

2. **3D geometry-aware positive pairs**  
   用 3DGS depth / camera pose 找跨视角同一 3D 点，构造 positive feature pairs；不同物体/远距离点作 negative。

3. **Local contrastive loss**  
   对同一 object 内 features 拉近，对 object boundary 两侧 features 拉开。不要只做全图 Gram loss，否则 peak 会被平滑。

4. **Relation loss 局部化**  
   当前全局 token affinity 可能过强。改成 windowed relation / object-mask pseudo-region relation，避免把整个场景拉平。

5. **DINO branch 的 loss schedule 后置**  
   先让 CTR 学好 RADIO space，再开启 DINO relation；权重要小，防止覆盖 SigLIP peak。

---

### 5.3 SAM3：从 prototype probe 改成 promptable instance segmentation

当前 SAM3 gap 明显。要提升，不能只用 prototype segmentation，要训练更像 SAM 的任务。

**建议正式任务：**

> Prompted instance segmentation on LERF held-out views.

**方法改动：**

1. **接入真正 frozen SAM3 decoder / mask head**  
   如果 RADIO SAM3 adaptor 只能给 token，但没有接 SAM3 mask decoder，那论文里不能叫 SAM3 instance segmentation。优先确认官方 adaptor 输出是否能喂给 frozen SAM3 head。

2. **Teacher-mask pseudo supervision**  
   在训练视角上，用 frozen SAM3 teacher 对 RGB / teacher features 产生 masks；student rendered features 经过同一 head，匹配 teacher masks。不要用 LERF GT test masks训练。

3. **Mask-logit distillation，而不是 feature cosine**  
   对 SAM 这种任务，直接匹配 feature 往往不够。更好的是匹配 mask logits、IoU score、presence token、mask embedding。

4. **Boundary / Dice loss**  
   加 soft Dice、boundary F loss，专门补 SAM mask gap。

5. **Prompt augmentation**  
   point、box、text、mask prompt 混合训练。评估时固定一种或两种 prompt，不能按 query 选最优。

6. **Instance-level Hungarian matching**  
   如果 SAM3 text prompt输出多个 instances，用 Hungarian matching 对预测 masks 和 GT instances 匹配，再算 AP / mIoU / cgF1-style metric。

---

### 5.4 VPR / Direct 3D object selection：从 VPR readout 升级成 VPR training signal

现在 VPR 已经很强，但它主要是 readout，不一定提升底层 feature field。要让 direct 3D 进一步涨，必须把 VPR 纳入训练。

**方法改动：**

1. **VPR consistency loss**  
   用 registered primitive embedding 反过来监督 Gaussian-center embedding：

   $$
   \mathcal{L}_{vpr} = 1 - \cos(f_i^{center}, f_i^{registered})
   $$

   这样 raw Gaussian-center readout 会变强，不只是 VPR readout强。

2. **Contribution-weighted registration**  
   不只用 Gaussian center depth check，改用 alpha contribution / projected footprint top-k registration。Dr. Splat 的核心就是沿 pixel ray 把 language embeddings 注册到 dominant Gaussians，而不是依赖 rendered-map threshold。([arxiv.org](https://arxiv.org/html/2502.16652v1))

3. **Component / instance-level selection**  
   现在 voxel context 是好方向，但还不是 instance grouping。可以建 3D kNN graph，用 feature similarity + voxel adjacency 做 connected components，然后 query component，而不是 query isolated primitives。

4. **Query-adaptive but GT-free threshold**  
   fixed top2% 已经可用，但不同 scene / object size 差异大。可以用 entropy、score margin、component size prior 自适应选阈值，但不要用 GT masks调。

5. **Waldo-targeted fragmentation audit**  
   继续拆 Waldo：registration coverage、selected component count、FP/FN map、per-query IoU histogram。它是 direct 3D mIoU 的主要瓶颈。

---

### 5.5 ScanNet 3D semantic segmentation：提高 point-query mIoU

ScanNet 当前是有效证据，但还不是最强。要提升，应该把 VPR/voxel/graph思想迁移到 point query。

**方法改动：**

1. **Point-query voxel context**  
   对 queried vertices 周围的 Gaussian / voxel features 做 weighted pooling，而不是单点 nearest / direct decode。

2. **Class prompt ensemble**  
   每个类别用多个 prompt，例如 “chair”, “a chair”, “furniture chair”, “seat”。最后平均或 max。

3. **Background / stuff calibration**  
   ScanNet stuff 类容易混。对 wall/floor/table/chair 等高频类做 temperature / prior calibration，但必须在 validation 上固定。

4. **DINO cross-view full sweep**  
   文档里 DINO cross-view 10-scene ablation已经有小幅提升：split19 从 0.3538 到 0.3640，split15 从 0.3573 到 0.3662，split10 从 0.4293 到 0.4308。([raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md)) 可以继续做权重 sweep，但目标是稳定提升，不要只挑个别scene。

5. **2D-to-3D semantic pseudo-label consistency**  
   用 rendered feature 在训练视角生成 semantic pseudo logits，再投到 3D vertices / Gaussians 做 consistency。这类似 VPR，但针对类别而不是 object query。

---

## 6. 训练策略建议：从单一 reconstruction 变成 staged multi-head

如果你要让每个任务都涨，我建议训练流程改成 5 阶段。

### Stage 1：RADIO reconstruction warmup

只训练 HGCF + CTR，确保 RADIO-space base feature稳定。

### Stage 2：FGC + VFA

加入 FGC 和 VFA，优化几何一致性和 rendered-view alignment。

### Stage 3：SigLIP2 peak-preserving grounding

加入 text heatmap distillation、negative prompts、spatial KL，目标是 LERF LocAcc 和 mIoU。

### Stage 4：DINO/SAM frozen-head task losses

不要只做 adaptor cosine。改成：

- DINO：mask propagation / correspondence loss；
- SAM：mask-logit / promptable segmentation loss。

### Stage 5：VPR consistency + 3D context

把 VPR registered embeddings 作为 primitive-level pseudo target，加入 graph / voxel / component context，让 3D readout反哺 feature field。

训练时要保留一个 retain loss：

$$
\mathcal{L}_{retain} =
\cos(\hat{F}, F^{teacher}_{RADIO})
+
\cos(A_{sig}(\hat{F}), A_{sig}(F^{teacher}))
$$

否则 DINO/SAM/VPR 很容易把 SigLIP2 grounding 搞坏。

---

## 7. 最终实验矩阵建议

论文主表可以这样安排：

### Table 1：2D novel-view frozen-head downstream tasks

| Head | Task | Teacher | Student | Claim |
|---|---|---:|---:|---|
| SigLIP2 | LERF grounding | 0.7985 / 0.4922 | 0.8712 / 0.4941 | student win |
| DINOv3 | mask propagation | 待重跑 | 待重跑 | target: close or win |
| SAM3 | prompted segmentation | 待重跑 | 待重跑 | target: narrow gap or win |

这张表是你“student 特征场强于 teacher”的核心证据。

### Table 2：3D scene understanding beyond teacher

| Task | Protocol | CTF-GS |
|---|---|---:|
| LERF direct 3D object selection | OpenGaussian-style | 0.3850 / 0.6428 |
| ScanNet direct point-query | OpenGaussian-style 19/15/10 | 0.3538 / 0.3573 / 0.4293 |

这张表是“teacher 不具备的 3D 能力”。

### Table 3：External comparison

按协议分开，不混表：

- rendered-view LERF：LangSplat-style；
- direct 3D LERF：OpenGaussian-style；
- ScanNet：OpenGaussian-style；
- SAM promptable 3D segmentation：SAGA-style，如果你决定做。

---

## 8. 最后一条关键建议

你现在最该优先做的不是再加外部 baseline，而是把 **DINOv3 和 SAM3 的 downstream probes 升级成正式任务**：

1. DINOv3：**source-target mask propagation / dense matching**；
2. SAM3：**prompted instance segmentation**；
3. 两者都做 teacher-vs-student，同一 frozen head，同一 held-out views；
4. 再决定它们能不能进主表。

如果这两项能从现在的 diagnostic 变成至少 “student close to teacher”，你的主线就很强；如果其中一项能超过 teacher，就可以写：

> CTF-GS improves SigLIP2 grounding and preserves or improves DINO/SAM downstream usability, while adding 3D scene-level readouts through VPR and ScanNet point querying.

如果 DINO/SAM 仍然弱，就要诚实写成：

> CTF-GS currently improves SigLIP2 grounding, but DINO/SAM adaptor tasks reveal remaining gaps in correspondence and instance-mask preservation.

这不是坏事，但主 claim 就必须落在 SigLIP2 + 3D能力上，而不能写成“三个 foundation head 全面超过 teacher”。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)