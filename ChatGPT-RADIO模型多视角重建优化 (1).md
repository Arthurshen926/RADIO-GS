# RADIO模型多视角重建优化

**User:** Anonymous  
**Created:** 5/16/2026 19:05:44  
**Updated:** 5/17/2026 21:13:59  
**Exported:** 5/17/2026 21:20:42  
**Link:** [https://chatgpt.com/c/6a084f88-5b88-83ea-941d-2478e7835cb5](https://chatgpt.com/c/6a084f88-5b88-83ea-941d-2478e7835cb5)  

## Prompt:
2026/5/17 15:03:58

按照上面讨论的内容，我进行了一波推进，git仓库已更新，主要是
上一轮 goal 主要把项目从“强原型 + 很多结果”整理成了“保守但可投稿的证据包”。核心不是继续堆模块，而是补齐投稿最容易被审稿人质疑的几件事：协议、对照、失败分析、可复现性和代码可审计性。

  投稿角度的主要优化

  1. 锁定 claim 和协议
      - 明确主 claim 是 compact teacher-compatible 3D scene memory，而不是泛化式“全面 3D 理解 SOTA”。
      - 把 rendered-view grounding、primitive-level direct 3D selection、SAM3 boundary readout 分开写。
      - 固定 global threshold 作为主协议，scene-locked / best-by-scene 只作为 diagnostic。
  2. 补齐 controlled evidence table
      - 新增 nearest-view RADIO cache baseline。
      - 新增 per-Gaussian 1280-D explicit RADIO memory baseline。
      - 现在可以证明 full CTF-GS 不只是“缓存 teacher feature”或“把 1280-D 原始特征挂到 Gaussian 上”。
  3. 补机制分析和失败解释
      - strict pad16 SAM3-box row 重新生成 alpha/depth geometry maps。
      - 得到 208/208 query-level geometry records 和 overlays。
      - 生成 paper/figures/alpha_depth_boundary_cases.png，用于 Waldo / boundary failure 机制讨论。
      - 结论更稳：alpha/depth discontinuity 与 boundary error 相关性较弱，只作为 mechanism context，不夸成因果证明。
  4. 提升可复现性和 artifact 可信度
      - 更新 submission_freeze_manifest、paper_assets_manifest、final_consistency_audit。
      - 每个关键表格数字都能追到 source JSON / evaluator / config / checkpoint / feature path。
      - verify_submission_provenance.py --check_paths 已通过。
  5. 代码可审计性从 risk 变为 pass
      - train_feature_field.py 从 6122 行拆到 3735 行。
      - 新增 radio_gs/training/ 模块拆分：
          - tensor cache IO
          - feature/data utilities
          - artifact/checkpoint mixin
          - feature supervision mixin
      - raw torch.load 统一收敛到 load_training_tensor_cache。
      - train_feature_field_audit 现在 overall status 是 pass，open items 为空。

  实验指标提升 / 关键结果
  | 对比 | LocAcc | mIoU | 说明 |
  |---|---:|---:|---|
  | Frame-wise RADIO teacher | 0.7985 | 0.4634 | 2D teacher baseline |
  | Nearest-view RADIO cache | 0.2722 | 0.1545 | 证明简单缓存远不够 |
  | Per-Gaussian 1280-D explicit memory | 0.5642 | 0.3182 | 原始 1280-D feature 挂 Gaussian 的强对照 |
  | Full CTF-GS rendered | 0.8712 | 0.5243 | 主方法 |

  相对提升：

  - Full CTF-GS vs frame-wise RADIO teacher：+0.0727 LocAcc / +0.0609 mIoU
  - Full CTF-GS vs nearest-view cache：+0.5990 LocAcc / +0.3698 mIoU
  - Full CTF-GS vs per-Gaussian 1280-D memory：+0.3070 LocAcc / +0.2061 mIoU

  Direct 3D 侧：
  | 方法 | Macro mIoU | Acc@0.25 | 说明 |
  |---|---:|---:|---|
  | OpenGaussian official context | 0.3836 | 0.5143 | 外部官方数值，仅作 context |
  | CTF-GS VPR fixed thr0p25 + RGB snap | 0.4801 | 0.6760 | SAM3-independent direct 3D row |
  | CTF-GS + official SAM3 box fixed pad16 | 0.5705 | 0.6835 | 主 strict SAM3-box readout |
  | CTF-GS + SAM3 box scene-locked diagnostic | 0.5972 | 0.7009 | diagnostic，不进主协议 claim |

  相对 OpenGaussian official context：

  - VPR fixed row：+0.0965 mIoU / +0.1617 Acc@0.25
  - SAM3-box fixed row：+0.1869 mIoU / +0.1692 Acc@0.25

  最终验证

  - pytest -q tests: 344 passed, 14 warnings
  - compileall: passed
  - provenance check: passed
  - paper asset manifest path scan: passed
  - latexmk 论文编译: passed
  - git diff --check: passed

  整体上，上一轮把项目推进成了一个更适合投稿的“保守强证据包”：主结果更有对照，direct-3D claim 更干净，SAM3 贡献被拆开，失败案例有机制分析，代码和 artifact 也更容易审计。
你觉得是否达到预期目标了呢？帮我梳理一下目前的投稿状态，并再分析一下存在哪些问题需要改进

## Response:
2026/5/17 15:03:59

我会按投稿审稿视角重新审视更新后的 claim、证据链、协议可信度、代码可审计性和剩余风险，并给出当前 readiness 判断与优先修改清单。

## Response:
2026/5/17 15:04:11

初步看，你这轮已经达到“保守可投稿证据包”的目标，但还不能安全支撑强 SOTA/顶刊无硬伤主张。当前最需优先处理的是公开 artifact 可见性、SAM3-box 数字同步和外部 baseline 协议边界。

## Response:
2026/5/17 15:10:27

我的判断是：**基本达到了上一轮预期目标**。现在这个项目已经从“强原型 + 很多结果”变成了一个**保守但可投稿的证据包**。但它还没有到“可以无顾虑地打全局 SOTA / 顶刊强 claim”的状态。更准确地说：

> **作为一个 framed 很清楚、claim 保守、实验链条可审计的投稿版本：已经接近 ready。**  
> **作为一个声称全面领先现有 open-vocabulary 3D scene understanding 方法的顶刊版本：仍然需要补强。**

你这一轮最大的进步不是某个数字涨了，而是把审稿人最容易攻击的几个点补上了：**缓存式 baseline、naive explicit-memory baseline、direct-3D 与 SAM3 readout 拆分、失败案例机制分析、artifact provenance、代码审计**。仓库当前主线也已经把论文定位明确为 CTF-GS：一种 compact teacher-compatible Gaussian feature memory，用于把 frozen RADIO teacher features 蒸馏进 3D Gaussian scene，并支持 rendered teacher-compatible feature maps 与 VPR Gaussian-level direct 3D querying。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/PROJECT_MAINLINE.md))

---

## 1. 当前投稿状态：已经是“保守强证据包”

我会把现在的状态评为：

| 维度 | 当前状态 | 判断 |
|---|---|---|
| **论文目标 / claim** | 从“全面 3D 理解 SOTA”收敛到 compact teacher-compatible 3D scene memory | **达标** |
| **主实验链条** | LERF rendered grounding、teacher-vs-rendered、cache baseline、1280-D explicit baseline、VPR direct 3D、SAM3-box readout、ScanNet point-query | **基本达标** |
| **direct-3D claim** | 已拆成 VPR primitive selection、RGB snap、SAM3 boundary readout | **明显变强** |
| **失败分析** | Waldo、query-level audit、alpha/depth boundary overlays、弱相关性谨慎表述 | **可投稿，但还可深化** |
| **可复现性** | provenance check、paper asset manifest、source JSON / evaluator / config / checkpoint path 追踪 | **内部达标，公开 artifact 仍有问题** |
| **代码审计** | 训练脚本从 6122 行拆到 3735 行，并新增 training modules；audit pass | **从 risk 变为 pass-minus** |
| **顶刊级说服力** | 有系统性创新和强 controlled evidence，但外部 SOTA 与跨域泛化仍需谨慎 | **接近，但不宜过度 claim** |

目前 `submission_status.md` 本身也把状态描述为“mature research prototype, conservative paper package”，并估计 conservative package 约 92%，更严格的 VPR-backed primitive-level paper 约 87%；剩余主要是 presentation、related work、diagnostic 决策等问题。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/submission_status.md)) 这个判断我基本认同。

---

## 2. 这一轮最重要的进步

### 2.1 你补上了最关键的 controlled baselines

新增的两个 baseline 非常关键：

| 方法 | LocAcc | mIoU | 作用 |
|---|---:|---:|---|
| Frame-wise RADIO teacher | 0.7985 | 0.4634 | 2D teacher baseline |
| Nearest-view RADIO cache | 0.2722 | 0.1545 | 排除“只是缓存最近视角 teacher feature” |
| Per-Gaussian 1280-D explicit memory | 0.5642 | 0.3182 | 排除“只是把原始 1280-D feature 挂到 Gaussian 上” |
| Full CTF-GS rendered | 0.8712 | 0.5243 | 主方法 |

这组表的意义很大。上一轮最大的潜在质疑是：**CTF-GS 到底是不是一个复杂 feature cache？** 现在 nearest-view cache 的 0.2722 / 0.1545 和 per-Gaussian 1280-D explicit memory 的 0.5642 / 0.3182 都明显低于 full CTF-GS 的 0.8712 / 0.5243，这说明主方法不是简单缓存，也不是 naive explicit high-dimensional storage。论文草稿里也记录了 per-Gaussian explicit baseline 的 registered fraction 只有 0.2020，存储约 1039.7 MiB，而 compact full model 同时更准、更紧凑。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

这是目前最能支撑“方法必要性”的证据之一。

### 2.2 主 claim 现在更安全

现在最稳的主 claim 应该是：

> **CTF-GS 是一种 compact teacher-compatible 3D Gaussian feature memory。它在固定协议下能渲染 novel-view RADIO-compatible features，并且这些 rendered features 在 LERF-OVS 上优于 frame-wise RADIO teacher 和两个 controlled memory baselines。**

这个 claim 是有实验证据支撑的。主线文档记录，full CTF-GS rendered 在四个 LERF scenes 上达到 Macro LocAcc 0.8712、mIoU 0.5243；同协议下 frame-wise RADIO RGB baseline 是 0.7985 / 0.4634。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/PROJECT_MAINLINE.md))

更重要的是，你没有把它写成“CTF-GS 全面比 RADIO 更强”，而是限定在 **same frozen evaluator / same query protocol / rendered feature grounding** 下。这种表述更抗审稿。

### 2.3 direct-3D claim 变干净了

你现在把 direct-3D 侧拆成：

1. **VPR fixed row + RGB snap**：SAM3-independent direct 3D row；
2. **SAM3-box fixed pad16**：3D selection + frozen SAM3 boundary readout；
3. **scene-locked diagnostic**：只作为 diagnostic，不进主协议 claim。

这比上一版强很多。论文草稿当前也明确说明：SAM3 candidate 不是由 GT mask 选择，而是由 rendered prediction overlap 选择；OpenGaussian 仍作为 official-source context，不是 locally rerun apples-to-apples baseline。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

目前比较稳的写法是：

> **VPR enables mask-free primitive-level 3D object selection. When combined with a frozen SAM3 box readout, it provides boundary-completed object masks, but SAM3-box results should be interpreted as 3D selection plus 2D foundation-model boundary readout, not pure 3D segmentation.**

这个区分非常重要。

### 2.4 失败分析现在更可信

你没有强行把 alpha/depth discontinuity 和 boundary error 写成因果证明，而是把它作为 mechanism context。这是正确的。论文草稿当前记录了 208/208 query-level overlays，并明确说 alpha/depth correlation 较弱，因此只能作为 diagnostic context，而不是 causal explanation。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

这会让论文更可信。顶刊审稿人不怕你有 failure；他们怕你装作没有 failure，或者把弱证据讲成强因果。

### 2.5 代码审计风险下降明显

代码拆分也是真实进步。仓库现在有 `radio_gs/training/` 模块，包括 artifact mixin、feature supervision mixin、feature training utilities、tensor cache IO 等；`train_feature_field.py` 当前约 3735 行，比之前 6000+ 行明显可审计。([GitHub](https://github.com/Arthurshen926/RADIO-GS/tree/main/radio_gs/training))

从投稿角度看，这已经足够从“reviewer 一看就害怕的 monolith”降到“虽然还有大脚本，但主功能已经拆分”的状态。测试目录也已经覆盖 baselines、VPR、ScanNet、provenance、report builders 等很多关键路径。([GitHub](https://github.com/Arthurshen926/RADIO-GS/tree/main/tests))

---

## 3. 现在最适合的论文定位

我建议当前论文定位定为：

> **Compact Teacher-Compatible Gaussian Feature Fields for Open-Vocabulary 3D Scene Memory**

不要主打：

> “Open-vocabulary 3D scene understanding SOTA”

也不要主打：

> “RADIO-GS solves general 3D foundation models”

最安全、最有贡献感的版本是：

> **2D foundation features are useful but image-plane-bound. CTF-GS turns frozen RADIO/C-RADIO dense features into a compact, renderable, queryable 3D scene memory. This scene memory preserves teacher-compatible downstream usability, improves rendered open-vocabulary grounding under a fixed protocol, and supports primitive-level VPR selection with optional frozen SAM3 boundary readout.**

这个 framing 能同时解释三个东西：

1. 为什么 RADIO 重要；
2. 为什么 3DGS 重要；
3. 为什么你的方法不是 feature cache 或 naive high-dimensional Gaussian storage。

---

## 4. 仍然存在的主要问题

下面这些不是“项目失败点”，而是**投稿前最好解决的问题**。我按优先级排序。

---

# P0：paper table / protocol 仍有不一致，需要立即清理

这是我现在看到的最大短期风险。

你的文字里当前主推的是：

- **VPR fixed thr0p25 + RGB snap**：0.4801 mIoU / 0.6760 Acc@0.25；
- **SAM3 box fixed pad16**：0.5705 / 0.6835；
- **scene-locked diagnostic**：0.5972 / 0.7009。

但是仓库里的部分 table / protocol card 仍然显示旧口径。例如 direct selection table 里有 fixed thr0p25 row 和 SAM3 fixed row，但另一个 published-context table 仍然报告了 `mean+2.5std` VPR row，而 VPR protocol card 也还写着 fixed mean+2.5std、floor/cap、context aggregation voxel-max 等设置。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/lerf_direct_3d_selection_table.tex))

这会被 reviewer 或 artifact evaluator 抓住，因为他们会问：

> 你的主协议到底是 fixed global threshold 0.25，还是 mean+2.5std threshold？  
> 你的 VPR row 到底是哪一个？  
> paper text、table caption、protocol card、JSON source 是否完全一致？

投稿前建议你做一个 canonical registry，例如：

```yaml
main_rows:
  lerf_rendered:
    threshold: 0.60
    locacc: 0.8712
    miou: 0.5243

  direct_vpr_primary:
    name: "CTF-GS VPR fixed thr0p25 + RGB snap"
    threshold: 0.25
    miou: 0.4801
    acc25: 0.6760
    uses_sam3: false
    protocol: "fixed_global"

  direct_sam3_primary:
    name: "CTF-GS + official SAM3 box fixed pad16"
    threshold: 0.25
    pad: 16
    miou: 0.5705
    acc25: 0.6835
    uses_sam3: true
    protocol: "fixed_global"

  direct_sam3_diagnostic:
    name: "CTF-GS + SAM3 box scene-locked"
    miou: 0.5972
    acc25: 0.7009
    protocol: "diagnostic_only"
```

然后所有 LaTeX table、protocol card、abstract、README、submission status 都从这个 registry 生成或至少人工核对。这个问题必须在投稿前解决。

---

# P0：公开 artifact 还有一个明显 reproducibility 问题

你本地 provenance check 已经通过，这是好事。但从公开仓库看，`output` 是一个指向本地绝对路径的 symlink：`/mnt/pool/sqy/results/RADIO-GS/output`。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/output)) 同时 `.gitignore` 里也忽略了 `output/`、datasets、generated configs、`.pt/.pth/.ply/.npy/.npz` 等大部分生成物。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/.gitignore))

这意味着：**你本地 artifact 是可追踪的，但公开 GitHub 上某些 source-of-truth report / generated JSON / freeze report 可能无法直接访问。**

这不是科学问题，但会影响 artifact credibility。建议投稿前做一个轻量、可公开的 artifact snapshot：

```text
paper/artifacts/
  final_rows.yaml
  submission_freeze_manifest.json
  paper_assets_manifest.json
  final_consistency_audit.json
  lerf_main_results.json
  direct3d_main_results.json
  scannet_point_query_results.json
  provenance_report.md
  checksums.txt
```

大文件 checkpoint 可以不提交 GitHub，但关键 result JSON、manifest、report snapshot 应该放在 repo 或 release/Zenodo。不要让 paper 里的 source-of-truth 路径指向一个本地 symlink。

---

# P1：外部 baseline 仍然只能支持 context，不支持 strict SOTA

你现在对 OpenGaussian / LangSplat / LEGaussians / Dr. Splat / CAGS / InstanceGaussian / OpenGaFF 等外部方法的处理更谨慎了，这是正确的。paper draft 也说明 external baseline rows 是 official-source / published context，不是完全同协议复现。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

但这也意味着：

> **目前不能强 claim “CTF-GS is state-of-the-art among all open-vocabulary 3D Gaussian methods.”**

尤其是 2025-2026 这一方向很拥挤。CAGS 明确研究 cross-view granularity inconsistency，并在 LERF-OVS / ScanNet 等开放词汇 3D 场景理解任务上做了 context-aware Gaussian modeling。([arXiv](https://arxiv.org/abs/2504.11893)) OpenGaFF 是 2026 年 5 月提交的 Gaussian Feature Field 方法，用 codebook attention 处理 open-vocabulary 3D scene understanding。([arXiv](https://arxiv.org/abs/2605.06088)) InstanceGaussian 也已经在 CVPR 2025 讨论 instance-level Gaussian representation、boundary delineation 和 category-agnostic perception。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Li_InstanceGaussian_Appearance-Semantic_Joint_Gaussian_Representation_for_3D_Instance-Level_Perception_CVPR_2025_paper.html)) Dr. Splat 则直接把 language-aligned CLIP embeddings 与 3D Gaussians 关联，用于 direct 3D perception。([Dr. Splat](https://drsplat.github.io/?utm_source=chatgpt.com))

所以你的相关工作和 claim 要非常精确：

- 你可以说：**under our fixed RADIO/SigLIP2 evaluator and controlled baselines, CTF-GS shows stronger teacher-compatible scene-memory behavior**。
- 不要说：**universally best open-vocabulary 3D method**。
- 除非你愿意把这些 recent methods 在同一 scene、query、evaluator、threshold、mask protocol 下重新跑一遍。

当前版本中，外部 baseline 最适合作为 contextual reference，不适合作为主结论核心。

---

# P1：统计证据还偏窄

你已经做了 n=3 seeds 和 bootstrap / audit，这是好方向。但从顶刊角度，四个 LERF scenes 仍然偏窄。submission status 也承认 seed stats improved but still narrow。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/submission_status.md))

建议至少补到 appendix：

1. **per-query bootstrap CI**：尤其是 LERF mIoU、LocAcc；
2. **leave-one-scene-out threshold diagnostic**：证明 fixed threshold 不是某个 scene tuning；
3. **query category breakdown**：
   - small object；
   - reflective / transparent；
   - texture-like object；
   - container / part-object；
   - occluded object；
   - multi-instance object；
4. **object footprint vs performance**；
5. **view coverage vs VPR accuracy**；
6. **teacher confidence / multi-view agreement vs failure rate**。

这些不一定都要放主文，但至少 appendix 有，会显著降低 reviewer 对“结果是不是偶然”的担心。

---

# P1：SAM3-box row 需要继续严格措辞

当前 SAM3-box fixed pad16 的 0.5705 / 0.6835 很有吸引力，但它也最容易被质疑。

最稳的说法是：

> **VPR supplies 3D primitive/object localization. Frozen SAM3 supplies image-space boundary completion from the selected box.**

不要说：

> “CTF-GS directly segments 3D objects with mIoU 0.5705.”

因为这个 row 不是纯 3D primitive segmentation。它是 **3D selection + 2D frozen boundary readout**。你现在已经拆开了，这是正确方向。建议在主表 caption 里明确写：

```text
The SAM3-box rows evaluate whether VPR-selected 3D evidence can drive a frozen 2D boundary readout.
They are not claimed as purely Gaussian-level segmentation.
```

这样可以提前堵住审稿意见。

---

# P1：ScanNet 结果只能作为 cross-domain utility，不要当 leaderboard

ScanNet v67 direct point query 是很好的 cross-domain probe。主线文档里也记录了 split 19/15/10 的 mIoU，以及 contextual kNN 的提升。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/PROJECT_MAINLINE.md))

但它不是标准 ScanNet semantic segmentation leaderboard。paper draft 目前也把它写成 feature usability / point-query support evidence，这是对的。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/paper_draft_current.md))

建议论文里固定表述：

> **ScanNet is used as a cross-domain point-query probe for the learned teacher-compatible feature memory, not as a claim of full supervised ScanNet semantic segmentation SOTA.**

这句话最好在 main text 和 table caption 都出现一次。

---

# P2：代码已经过 pass，但还不是“漂亮 release”

现在代码风险已经从“monolith 不可审计”降到了“可接受但仍有大入口脚本”。`train_feature_field.py` 仍然有 3735 行。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/radio_gs/scripts/train_feature_field.py)) 这不再是投稿 blocker，但对 artifact review 仍然不是最优。

建议后续继续做两件事：

1. 提供 paper-specific thin wrappers：

```bash
python scripts/paper/extract_features.py --config configs/paper/...
python scripts/paper/train_ctfgs.py --config configs/paper/...
python scripts/paper/eval_lerf.py --config configs/paper/...
python scripts/paper/eval_vpr_direct3d.py --config configs/paper/...
python scripts/paper/build_tables.py --registry paper/artifacts/final_rows.yaml
```

2. 对 tensor cache loading 再做一点安全清理。现在 `load_training_tensor_cache` 已经统一封装了 `torch.load`，并做了路径存在性检查。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/radio_gs/training/tensor_cache_io.py)) 如果未来开放第三方下载的 tensor cache，建议尽量使用 `weights_only=True` 或 safetensors / manifest checksum，避免 reviewer 对 pickle-based torch load 安全性的担心。

这个不是科学问题，是工程 hygiene。

---

# P2：abstract 和主文叙事还需要“减法”

我看现在 paper abstract 有点 overloaded：结果、caveat、direct 3D、SAM3、ScanNet、diagnostic 都想塞进去。LaTeX draft 的 abstract 段落目前承载了很多信息。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

顶刊/顶会 abstract 应该更聚焦。建议 abstract 只保留三组数字：

1. LERF rendered main：0.8712 / 0.5243；
2. teacher / cache / explicit baseline 对比中的一个或两个关键点；
3. direct-3D VPR + SAM3-box 的主协议结果。

scene-locked diagnostic、alpha/depth correlation、ScanNet split details、external baseline provenance 都放正文或 appendix。

一个更干净的 abstract 逻辑是：

> 2D VFMs are image-bound → we introduce CTF-GS as compact teacher-compatible 3D Gaussian scene memory → it reconstructs RADIO-compatible novel-view dense features and supports primitive-level VPR → under fixed LERF-OVS protocol it improves over frame-wise RADIO and controlled cache/explicit baselines → VPR enables direct 3D selection and frozen SAM3 boundary readout → limitations include small objects, multi-instance ambiguity, and dependence on geometry.

---

## 5. 当前最稳的 claim 版本

我建议你在论文里固定使用以下 claim 层级。

### Claim A：主 claim，可以强写

> **CTF-GS learns a compact teacher-compatible Gaussian feature memory that renders novel-view RADIO-compatible dense features and outperforms frame-wise RADIO, nearest-view cache, and per-Gaussian explicit 1280-D memory under a fixed LERF-OVS protocol.**

这个 claim 现在有强支撑。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/PROJECT_MAINLINE.md))

### Claim B：direct-3D claim，可以中等强度写

> **VPR registers rendered teacher-compatible evidence to Gaussian primitives and enables SAM3-independent direct 3D selection; a frozen SAM3 box readout can further complete image-space boundaries.**

这个 claim 有支撑，但要保留“not pure 3D segmentation”的边界。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/paper/radio_gs_draft.tex))

### Claim C：cross-domain utility，可以保守写

> **ScanNet point-query and adaptor probes suggest that the learned feature memory retains useful downstream structure beyond LERF, but these are used as transfer probes rather than standardized leaderboard claims.**

这个 claim 安全。([GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/paper_draft_current.md))

### Claim D：global SOTA，不建议写

不建议写：

> “CTF-GS achieves state-of-the-art open-vocabulary 3D scene understanding.”

除非你补齐同协议复现外部 recent methods。

---

## 6. 我建议投稿前最后做的 checklist

### 必须做

1. **统一 VPR 主协议**  
   修正所有 table、caption、protocol card、abstract、README 中关于 `thr0p25` vs `mean+2.5std` 的不一致。

2. **公开 artifact snapshot**  
   不要让 paper source-of-truth 指向本地 symlink。把 final JSON、manifest、audit report、checksums 放到 `paper/artifacts/` 或 release archive。

3. **精简 abstract**  
   从“结果罗列”改成“问题-方法-核心结果-边界”。

4. **所有外部 baseline caption 加 protocol caveat**  
   明确 official-source / published-context，不是 locally rerun apples-to-apples。

5. **SAM3-box caption 加边界说明**  
   明确这是 3D selection + frozen 2D boundary readout。

### 强烈建议做

6. **加 CI / bootstrap appendix table**  
   尤其是 LERF per-query 和 per-scene confidence interval。

7. **加 query category breakdown**  
   小物体、多实例、遮挡、边界复杂 object 分开分析。

8. **加一张 claim map figure**  
   图里明确区分：
   - rendered feature grounding；
   - VPR primitive selection；
   - SAM3 boundary readout；
   - ScanNet point-query probe。

9. **加一段“why CTF-GS beats frame-wise RADIO”机制解释**  
   不要只说数字高。解释为 multi-view aggregation、geometry-regularized denoising、compact teacher-compatible reconstruction、screen-space correction 共同作用，并用 ablation 支撑。

---

## 7. 最终判断

我的结论是：

> **这一轮已经达到“保守可投稿证据包”的预期目标。**

尤其是以下四点已经从弱点变成了优点：

1. **controlled baselines 补强**：nearest-view cache 和 per-Gaussian 1280-D memory 让方法必要性更可信；
2. **direct-3D claim 变清楚**：VPR、RGB snap、SAM3-box readout 被拆开；
3. **失败分析更诚实**：Waldo 和 alpha/depth analysis 没有被过度解释；
4. **可复现性和代码审计明显改善**：provenance、manifest、tests、training module split 都到位。

但投稿前仍要修三个关键点：

1. **协议/表格不一致**，尤其是 VPR threshold 口径；
2. **公开 artifact 链接 / symlink 问题**；
3. **外部 baseline 只能做 context，不能支撑 global SOTA**。

所以我会给当前状态一个比较明确的评级：

| 目标 | 是否达到 |
|---|---|
| 保守投稿包 | **基本达到** |
| 顶会 workshop / conference-style paper | **接近 ready** |
| 顶刊长文强证据 | **还需补统计、外部 baseline、跨域机制分析** |
| 全局 open-vocabulary 3D SOTA claim | **目前不建议** |

一句话总结：

> **现在的 RADIO-GS / CTF-GS 已经不是“有趣但散”的项目，而是一个 claim 边界清楚、主实验链条完整、代码和 artifact 可审计的投稿包；剩下最重要的不是继续加模块，而是把协议、表格、artifact 和措辞彻底收敛到一个不可被误读的版本。**

---

**Sources:**

- [RADIO-GS/docs/PROJECT_MAINLINE.md at main · Arthurshen926/RADIO-GS · GitHub](https://github.com/Arthurshen926/RADIO-GS/blob/main/docs/PROJECT_MAINLINE.md)
- [[2504.11893] CAGS: Open-Vocabulary 3D Scene Understanding with Context-Aware Gaussian Splatting](https://arxiv.org/abs/2504.11893)
- [CVPR 2025 Open Access Repository](https://openaccess.thecvf.com/content/CVPR2025/html/Li_InstanceGaussian_Appearance-Semantic_Joint_Gaussian_Representation_for_3D_Instance-Level_Perception_CVPR_2025_paper.html)
- [Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct ...](https://drsplat.github.io/?utm_source=chatgpt.com)

## Prompt:
2026/5/17 20:35:05

现在本论文里的整个逻辑流程还是不够简洁直观，不管是从方法上还是实验配置上都很绕，很多细节都可以作为补充材料，但是也要简洁直观，梳理的同时顺便看下很多模块词汇或者用词是否可以优化一下，感觉还不是很到位。
本文使用的实验评估协议或者说任务是否就三个：lerf上的2d grounding、3d object selection和scannet上的3d segmentation？
我们的实验协议难道和其他对比的sota方法不一样吗，不都是使用的lerf 官方协议吗？不过好像大多数论文里的同一个方法同一个任务的数据确实都不一样，那本论文最好也是把对比方法重新复现一遍以示公平，包括lerf上的2d grounding、3d object selection和scannet上的3d segmentation，请你列一个我们可以复现的已开源了的方法，以及每一个的复现计划和工作量，我们的目标肯定是在同样协议下本文指标超过其他复现方法达到SOTA。并且检查我们的实验协议，如果存在问题请及时参考其他论文修复

## Response:
2026/5/17 20:35:09

我会先压缩论文主线和术语，再核对 LERF、direct-3D、ScanNet 三类任务协议是否公平，最后列出可同协议复现的开源 SOTA/baseline、优先级和工作量。

## Response:
2026/5/17 20:35:22

初步看下来，主文应强制收敛为三条主协议，其余 adaptor、diagnostic、consistency 分析放补充材料；同时 VPR 阈值口径仍需立刻统一。

## Response:
2026/5/17 20:44:11

我觉得你这个判断是对的：**现在论文的逻辑仍然偏绕**。不是因为工作本身不成立，而是因为主文里同时塞了太多“方法组件名、诊断分支、readout 变体、协议 caveat 和辅助任务”。顶刊论文不能让读者先理解十几个 acronym 才知道你在解决什么问题。

我建议现在强制把论文主线压缩为：

> **一个方法：Compact teacher-feature Gaussian scene memory。**  
> **三个主任务：LERF rendered-view OVS、LERF direct 3D object selection、ScanNet open-vocabulary 3D semantic segmentation。**  
> **两类附录：controlled evidence / diagnostics。**

---

# 1. 三个主任务的划分是对的，但名称要改得更精确

你说的“三个任务”基本正确，但建议不要写成“LERF 上的 2D grounding、3D object selection 和 ScanNet 上的 3D segmentation”。更标准、更不容易被误解的写法是：

| 当前说法 | 建议论文用语 | 说明 |
|---|---|---|
| LERF 上的 2D grounding | **LERF-OVS rendered-view open-vocabulary segmentation / localization** | 不是普通 2D grounding，而是从 3D scene memory 渲染 feature map 后做 2D open-vocabulary mask / localization |
| 3D object selection | **LERF-OVS text-to-Gaussian object selection** 或 **direct 3D object selection** | query 在 3D Gaussian primitives 上发生，render 只用于和 LERF masks 计算 mIoU / Acc@0.25 |
| ScanNet 上的 3D segmentation | **ScanNet-v2 open-vocabulary point-cloud semantic segmentation** | 如果想和 SOTA 公平比较，必须是对 GT semantic point cloud / vertices 做 per-point label prediction，而不只是当前的 direct point-query probe |

这个三任务划分和近期文献的主流协议是一致的。OpenGaFF 这类近期论文明确把 LERF-OVS 分成 2D rendered feature map evaluation 和 3D per-Gaussian querying evaluation，并在 ScanNet-v2 上做 ground-truth semantic point cloud 的 open-vocabulary 3D segmentation，指标是 mIoU 和 Acc/mAcc。([arXiv](https://arxiv.org/html/2605.06088v1))

所以主文建议只保留这三个任务：

1. **T1: LERF-OVS Rendered-view OVS**  
   输入 text query，方法输出 rendered heatmap / mask。指标：mIoU、LocAcc。

2. **T2: LERF-OVS Direct 3D Selection**  
   输入 text query，方法在 3D Gaussians 上打分/选择，再把 selected Gaussians render 成 mask。指标：mIoU、Acc@0.25。

3. **T3: ScanNet-v2 OV 3D Semantic Segmentation**  
   输入类别名称集合，方法对 ScanNet point cloud / vertices 输出开放词汇语义标签。指标：mIoU、mAcc / Acc。

其他内容，比如 SAM3 point/box prompt、DINO matching、VPR-to-field consistency、alpha/depth boundary analysis、threshold sweeps、RGB snap variants、rasterizer hit variants，都应该进 appendix 或 diagnostics。

---

# 2. 你们现在的协议“不完全等价于其他 SOTA”，需要收敛成 strict local benchmark

你问：“我们的实验协议难道和其他 SOTA 方法不一样吗，不都是 LERF 官方协议吗？”

结论是：**大方向一样，但细节还没有严格一样。**

“使用 LERF-OVS”并不自动等于“公平同协议”。近两年的论文虽然都说 follow LERF-OVS / OpenGaussian-style protocol，但实际会在以下地方变化：

1. 使用的 scene 数不同：4 scenes、5 scenes、是否包含 bouquet、是否把 kitchen 写作 waldo_kitchen；
2. query list / annotation masks 版本不同；
3. text encoder 不同：CLIP、OpenCLIP、SigLIP、SigLIP2；
4. prompt template 不同；
5. threshold / calibration 不同；
6. 是否 query 2D rendered feature map，还是 query 3D Gaussians；
7. 是否有 SAM / mask proposal / boundary refinement；
8. ScanNet 是不是标准 point-cloud semantic segmentation，还是自定义 point-query probe；
9. 是否使用 test/annotated frames 参与 scene optimization；
10. 是否用每个 scene 的 best threshold 或 validation-free fixed threshold。

你们当前论文草稿已经承认外部 baseline rows 是 official-source context，而不是 locally rerun same-evaluator results；并且 current ScanNet row 也写明不是 full standardized segmentation leaderboard，而是 direct point-query feature probe。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex)) 这说明现在还不能直接写“same protocol SOTA”。

如果目标是“同协议 SOTA”，我建议新增一个独立 benchmark package，命名为：

> **Unified RADIO-GS Open-Vocabulary 3D Benchmark**

核心是：所有方法都只负责输出 prediction，最终指标全部由你们统一 evaluator 计算。

---

# 3. 当前结果离“全任务 SOTA”还有差距，要诚实判断

这是非常重要的一点。

你们当前 T1 rendered-view LERF 结果是 0.8712 LocAcc / 0.5243 mIoU；same-evaluator 下超过 frame-wise RADIO teacher、nearest-view cache 和 per-Gaussian 1280D memory，这个 controlled evidence 很强。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md))

但如果看近期 published context，OpenGaFF 的 LERF-OVS 2D table 报告 OccamLGS 61.30 mIoU、LangSplatV2 59.90 mIoU、OpenGaFF 64.98 mIoU，而你们目前是 52.43 mIoU。你们的 LocAcc 很强，但 mIoU 暂时不是 published SOTA。([arXiv](https://arxiv.org/html/2605.06088v1))

T2 direct 3D 也类似。你们 pure VPR row 是 0.4801 mIoU / 0.6760 Acc@0.25，SAM3-box fixed row 是 0.5705 / 0.6835。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/PROJECT_MAINLINE.md)) 但 OpenGaFF reported 3D row 是 54.36 mIoU / 80.84 Acc@0.25；OccamLGS 是 47.22 / 74.84。([arXiv](https://arxiv.org/html/2605.06088v1)) 也就是说，SAM3-box row 的 mIoU 有竞争力，但 Acc@0.25 不够高，而且它不是 pure Gaussian selection。

T3 ScanNet 现在也不能直接叫 SOTA。你们 current contextual kNN row 是 0.3637 / 0.3708 / 0.4512 mIoU on 19/15/10 splits。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/docs/submission_status.md)) OpenGaFF reported 36.55 / 42.78 / 57.85。([arXiv](https://arxiv.org/html/2605.06088v1)) 所以至少在 published context 下，ScanNet 15/10-class 还明显不够。

所以现在正确定位是：

> **我们有强 controlled evidence，证明 CTF-GS 不是 cache，也不是 naive raw-feature storage；但如果目标是同协议全任务 SOTA，还需要重新复现 baseline，并且提升 T1 mIoU、T2 Acc@0.25、T3 ScanNet 15/10 split。**

---

# 4. 主文逻辑应该重写成“一图三读出”

我建议把方法主线改成下面这个结构。

## 4.1 一句话问题定义

> 2D foundation models produce powerful dense features, but these features are image-bound. We ask whether a 3D Gaussian scene can store a compact, teacher-compatible foundation-feature memory that supports both rendered-view and primitive-level open-vocabulary querying.

中文理解就是：

> **把 2D foundation feature 从图像平面搬进 3D Gaussian scene，使其成为可渲染、可查询、可复用的 3D scene memory。**

## 4.2 主方法只保留三个概念

现在 HGCF、CTR、VFA、FGC、VPR、VPR-to-field、SAM3 box、RGB snap 全部写在主文里，会显得太绕。建议主文只讲：

### Module A：Compact Gaussian Feature Field

对应现在的 HGCF + compact latent storage。

写法：

> We attach compact latent features to the frozen 3DGS geometry and combine per-Gaussian latent splatting with a coarse spatial context branch.

不要在 introduction / abstract 里先抛 HGCF 这个 acronym。

### Module B：Teacher-Space Decoder

对应现在的 CTR / HCD。

写法：

> A lightweight decoder reconstructs RADIO-compatible teacher-space features from the compact rendered feature map.

“Compact-to-Teacher Reconstruction”是准确的，但“Teacher-Space Decoder”更直观。CTR/HCD 可以只在 method detail 或 appendix 里出现。

### Module C：View-to-Primitive Registration

VPR 是最值得保留的 acronym，因为它真正解释了 2D rendered features 如何转成 3D primitive readout。

写法：

> VPR registers rendered teacher-compatible evidence back to visible Gaussians, enabling text-to-Gaussian object selection.

这比“VPR rendered-feature-to-primitive registration with RGB snap and floor/cap threshold”清楚得多。

---

# 5. 术语建议：减少 acronym，保留最关键的两个

我建议 paper-facing 术语这样改：

| 当前术语 | 建议术语 | 是否保留 acronym | 理由 |
|---|---|---:|---|
| RADIO-GS | CTF-GS | 是 | RADIO-GS 作为 repo 名；paper method 用 CTF-GS |
| Compact Teacher Feature Fields with View-to-Primitive Registration... | **Compact Teacher-Feature Gaussian Splatting** | 是 | 标题更短 |
| HGCF: Hybrid Gaussian Code Field | **Compact Gaussian Feature Field** | 不建议 | HGCF 不够直观，像工程模块 |
| HCD codec / CTR | **Teacher-Space Decoder** 或 **Teacher Feature Decoder** | 不建议 | “decode to teacher feature space”更直观 |
| VFA: View-Space Feature Alignment | **View Refiner** / **screen-space feature refinement** | 不建议 | VFA 可以放 appendix |
| FGC: Frozen Geometry-Head Consistency | **Frozen-head geometry consistency** | 不建议 | 解释即可，不需要 acronym |
| VPR: View-to-Primitive Registration | **VPR** | 建议保留 | 这是 direct-3D bridge，最有辨识度 |
| VPR-to-field consistency | **Registry distillation** | 放 appendix | 主文不需要 |
| RGB snap | **boundary snap** | appendix / implementation | 主结果可以一句话 |
| SAM3 box fixed pad16 | **frozen SAM3 boundary readout** | appendix / secondary row | 不能和 pure 3D selection 混为一个主方法 |
| 3D object selection | **text-to-Gaussian object selection** | - | 更具体 |
| 2D grounding | **rendered-view OVS** | - | 更符合文献 |
| ScanNet direct point-query | **ScanNet OV point-cloud segmentation** | - | 如果要 SOTA，必须升格为标准 segmentation |

标题可以改成：

> **CTF-GS: Compact Teacher-Feature Gaussian Splatting for Open-Vocabulary 3D Scene Understanding**

或者更强调 memory：

> **CTF-GS: Compact Teacher-Feature Gaussian Scene Memories**

我更推荐第一个，审稿人一眼知道是 3DGS + teacher feature + open-vocabulary 3D。

---

# 6. 实验协议应该修成三条 canonical tracks

## Track 1：LERF-OVS Rendered-view OVS

**任务定义**：

> Given a text query and an evaluation view, render a feature/relevance map and predict a 2D mask.

**指标**：

- LocAcc：heatmap argmax 是否落在 GT mask 内；
- mIoU：thresholded predicted mask vs GT mask。

**需要修复的点**：

当前 threshold-0.60 作为主结果可以保留，但必须说明选择方式。你们现在说“GT-free threshold-0.60 mask readout”，但如果 0.60 是根据同一批 LERF masks 的 mIoU sweep 选出来的，那这个选择过程并不是 GT-free。建议改成以下二选一：

1. **fixed-prior protocol**：0.60 作为固定阈值，不通过 test masks 选择；引用其作为 OpenGaussian-style / common threshold 的合理值；
2. **leave-one-scene-out calibration**：每次用 3 个 scenes 选阈值，在剩下 1 个 scene 上评估，最后 macro average；
3. **validation split calibration**：如果有额外 validation scenes，阈值只在 validation 上选。

主表用 fixed 或 leave-one-scene-out，best threshold sweep 只放 appendix。

## Track 2：LERF-OVS Direct 3D Object Selection

**任务定义**：

> Given a text query, score/select 3D Gaussians directly. Render selected Gaussians only for metric computation.

这和 OpenGaussian-style query-select-render protocol 对齐。OpenGaussian repo 也明确提供了 LERF evaluation：先 render text-selected 3D Gaussians，再 compute metrics。([GitHub](https://github.com/yanmin-wu/OpenGaussian))

**指标**：

- mIoU；
- Acc@0.25。

**需要修复的点**：

SAM3-box row 必须和 pure direct-3D row 分开。建议主表变成两个小表：

**Table A: Pure text-to-Gaussian selection**

| Method | Query in 3D? | Boundary postprocess? | mIoU | Acc@0.25 |
|---|---:|---:|---:|---:|
| OpenGaussian | yes | no | ... | ... |
| Dr. Splat | yes | no | ... | ... |
| OccamLGS | yes/adapted | no | ... | ... |
| CTF-GS VPR | yes | no | ... | ... |

**Table B: Selection + common frozen boundary readout**

| Method | 3D selector | Same SAM3 readout applied? | mIoU | Acc@0.25 |
|---|---|---:|---:|---:|
| OpenGaussian + SAM3 box | OpenGaussian | yes | ... | ... |
| Dr. Splat + SAM3 box | Dr. Splat | yes | ... | ... |
| CTF-GS + SAM3 box | CTF-GS | yes | ... | ... |

如果只给 CTF-GS 用 SAM3 box，而不给 baseline 用同样后处理，就不能把 SAM3-box row 当成 strict SOTA comparison。

## Track 3：ScanNet-v2 Open-Vocabulary 3D Semantic Segmentation

**任务定义**：

> Given a ScanNet point cloud and category names, assign an open-vocabulary semantic label to every GT point / vertex.

OpenGaussian repo 提供 ScanNet evaluation for 19/15/10 categories。([GitHub](https://github.com/yanmin-wu/OpenGaussian)) OpenGaFF 也明确说在 ScanNet-v2 上用 ground truth semantic point cloud 做 3D open-vocabulary segmentation，报告 mIoU 和 Acc。([arXiv](https://arxiv.org/html/2605.06088v1))

**需要修复的点**：

你们当前 row 叫 “direct point-query transfer”，这在论文里已经保守说明不是 standard leaderboard。([GitHub](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex)) 如果想 SOTA，必须升级为 standard evaluator：

1. 使用 ScanNet-v2 GT semantic point cloud / vertices；
2. 使用同一批 scenes；
3. 使用同一套 19/15/10 class splits；
4. 使用同一 label mapping 和 ignore labels；
5. 每个 point 输出 class prediction；
6. 计算 per-class IoU、macro mIoU、mAcc；
7. 所有 baselines 用同一个 evaluator。

你们可以保留当前 gaussian-index row 作为 internal probe，但 SOTA 表必须使用标准 point-cloud segmentation evaluator。

---

# 7. 建议复现的开源方法清单

下面是我建议的优先级。目标不是“复现所有相关论文”，而是复现足以支撑 SOTA claim 的强开源 baseline。

## P0：必须复现

| 方法 | 覆盖任务 | 为什么必须复现 | 复现计划 | 工作量 | 风险 |
|---|---|---|---|---|---|
| **OpenGaussian** | T2 direct 3D；T3 ScanNet | 这是 direct point-level 3DGS open-vocabulary 的核心 baseline；repo 明确提供 ScanNet 19/15/10 eval 和 LERF object-selection eval。([GitHub](https://github.com/yanmin-wu/OpenGaussian)) | 用官方数据结构跑 LERF-OVS 和 ScanNet；导出 selected Gaussian masks / ScanNet point predictions；统一 evaluator 重新算 mIoU、Acc@0.25、mAcc | 中 | assets、text_features、scene path、官方脚本是否和论文版本一致 |
| **LangSplatV2** | T1 rendered-view OVS；可尝试 T2 | 近期强 2D baseline；官方 repo 提供 datasets/model weights 和 `eval_lerf.sh`。([GitHub](https://github.com/ZhaoYujie2002/LangSplatV2)) | 使用官方 pretrained 或 retrain；导出 LERF heatmaps/masks；用统一 evaluator 计算 LocAcc/mIoU | 中-高 | README 写实验使用 A100，环境和 CUDA 可能有成本。([GitHub](https://github.com/ZhaoYujie2002/LangSplatV2)) |
| **OccamLGS** | T1；T2 可适配 | 近期 published numbers 在 LERF 2D/3D 都很强；官方 repo 有 `run_lerf.sh`、`full_eval.py`、feature renderer 等。([GitHub](https://github.com/insait-institute/OccamLGS)) | 用同一 3DGS geometry 和 LERF language features；导出 feature maps 和 query masks；如果能得到 per-Gaussian scores，再纳入 T2 | 中 | feature format 依赖 LangSplat-style 512D features；T2 adapter 需要额外实现 |
| **GAGS** | T1 rendered-view OVS | OpenGaFF published context 中 GAGS 是强 2D baseline；官方 repo 提供 eval scripts、LERF-OVS 设置和 evaluation dataset。([GitHub](https://github.com/WHU-USI3DV/GAGS)) | 用官方 LERF-OVS setting 训练/评估；导出 relevance maps；统一 evaluator 计算 | 中-高 | 训练流程含 SAM / granularity factor；需要确认 query list 和 masks |
| **Dr. Splat** | T2 direct 3D；T3 ScanNet | CVPR 2025 direct language registration baseline，和你们 VPR claim 最接近；官方 repo 已开源。([GitHub](https://github.com/kaist-ami/Dr-Splat)) | 准备 COLMAP poses + pretrained 3DGS；提取 SAM masks + CLIP embeddings；导出 per-Gaussian language embeddings；统一 T2/T3 evaluator | 高 | feature extraction 和 SAM mask pipeline 成本高；不同 CLIP head 影响对比 |

## P1：强烈建议复现

| 方法 | 覆盖任务 | 复现价值 | 复现计划 | 工作量 | 风险 |
|---|---|---|---|---|---|
| **LangSplat** | T1 rendered-view OVS | 经典 3D language Gaussian baseline；官方 repo 提供 preprocessed datasets 和 pretrained models。([GitHub](https://github.com/minghanqin/LangSplat)) | 优先用 pretrained models 导出 LERF predictions；若结果偏差大再 retrain | 中 | 老环境；有些 issue 讨论 dataset/protocol 差异，必须统一 evaluator |
| **LEGaussians** | T1 rendered-view OVS；storage comparison | 经典 compressed language Gaussian baseline；官方 repo 含 LERF dataset setup 和 CLIP+DINO feature quantization。([GitHub](https://github.com/buaavrcg/LEGaussians)) | 跑官方 LERF scenes，导出 masks/heatmaps；统一 evaluator | 中 | Torch/CUDA 老版本；scene list 包含 bouquet，需和四场景 LERF-OVS 对齐 |
| **CAGS** | T2；T3 可尝试 | 直接基于 OpenGaussian data structure，适合比较 context-aware Gaussian semantic modeling。([GitHub](https://github.com/Wistzz/CAGS)) | 先跑 LERF-OVS；ScanNet 部分 README 仍标 TODO，所以优先 T2，T3 作为风险项 | 中-高 | ScanNet eval 脚本不完整；repo 规模小，可能需要 debug |
| **LaGa** | T1/T2/T3 可尝试 | 处理 view-dependent semantics，和你们 multi-view/VPR 叙事相关；repo 提供 LERF-OVS 和 ScanNet 数据入口。([GitHub](https://github.com/SJTU-DeepVisionLab/LaGa)) | 先跑 LERF；如果能导出 object descriptors / masks，再纳入 T2；ScanNet 作为扩展 | 高 | 推理流程偏 object/descriptor，可能不是直接 heatmap evaluator |
| **Semantic Gaussians** | T3 ScanNet | ScanNet open-vocabulary semantic segmentation baseline；repo 明确主打 ScanNet-20。([GitHub](https://github.com/sharinka0715/semantic-gaussians)) | 把 ScanNet-20 输出适配到你们 19/15/10 splits；统一 evaluator | 中-高 | class split / label mapping 不完全一致 |

## P2：有价值但不建议作为第一批

| 方法 | 原因 |
|---|---|
| **GOI-Hyperplane** | repo 有 evaluation code，但 README 说需要 GUI query objects and save segmentation masks，这不适合大规模自动化同协议复现。([GitHub](https://github.com/Quyans/GOI-Hyperplane)) |
| **OpenSplat3D** | 官方代码是 open-vocabulary 3D instance segmentation，更偏 ScanNet++ / instance evaluation；可以作为相关工作或 appendix，不适合作为 LERF-OVS / ScanNet-v2 三任务主 baseline。([GitHub](https://github.com/VisualComputingInstitute/opensplat3d)) |
| **SuperGSeg** | 论文和 published results 很相关，但 project page 仍显示 Code coming soon，当前不能严格复现。([supergseg.github.io](https://supergseg.github.io/)) |
| **OpenGaFF** | 近期 published numbers 很强，而且协议与你们目标高度一致；但当前 arXiv 页面说明 code will be publicly released upon acceptance。它现在只能做 published context，不能做 reproduced baseline。([arXiv](https://arxiv.org/html/2605.06088v1?utm_source=chatgpt.com)) |

---

# 8. 推荐的 reproduced leaderboard 设计

如果你们要把论文推进成“同协议 SOTA”，主结果建议是下面三张表。

## Table 1：LERF-OVS Rendered-view OVS

| Method | Native feature | Reproduced? | LocAcc | mIoU |
|---|---|---:|---:|---:|
| LERF | CLIP | yes/no |  |  |
| LangSplat | CLIP/SAM | yes |  |  |
| LEGaussians | CLIP+DINO | yes |  |  |
| GAGS | CLIP/SAM | yes |  |  |
| OccamLGS | CLIP | yes |  |  |
| LangSplatV2 | CLIP/codebook | yes |  |  |
| CTF-GS | RADIO/SigLIP2 | yes |  |  |

这里要注意：**native feature 不同不是 protocol 不公平，而是 method capability 的一部分**。公平性来自相同 dataset、query、masks、threshold selection rule 和 evaluator，而不是强行把所有方法都换成 SigLIP2。

可以额外加一张 controlled internal table：

| Method | Same teacher? | Same geometry? | LocAcc | mIoU |
|---|---:|---:|---:|---:|
| frame-wise RADIO | yes | no |  |  |
| nearest-view RADIO cache | yes | no |  |  |
| per-Gaussian 1280D memory | yes | yes |  |  |
| CTF-GS | yes | yes |  |  |

这张表服务于“方法必要性”，不是 external SOTA。

## Table 2：LERF-OVS Direct 3D Object Selection

分成两个 leaderboard。

**Pure 3D selection：**

| Method | Query target | Boundary postprocess | mIoU | Acc@0.25 |
|---|---|---:|---:|---:|
| OpenGaussian | Gaussians / clusters | no |  |  |
| Dr. Splat | Gaussians | no |  |  |
| CAGS | Gaussians | no |  |  |
| OccamLGS | Gaussians/adapted | no |  |  |
| CTF-GS VPR | Gaussians | no |  |  |

**Common SAM3 boundary readout：**

| Method | 3D selector | Same frozen SAM3 readout? | mIoU | Acc@0.25 |
|---|---|---:|---:|---:|
| OpenGaussian + SAM3 | OpenGaussian | yes |  |  |
| Dr. Splat + SAM3 | Dr. Splat | yes |  |  |
| CTF-GS + SAM3 | VPR / direct field | yes |  |  |

如果这个表里 CTF-GS + SAM3 赢，就可以 claim：

> CTF-GS provides stronger 3D object localization evidence for a common frozen boundary readout.

而不是说：

> CTF-GS pure 3D segmentation SOTA.

## Table 3：ScanNet-v2 Open-Vocabulary 3D Semantic Segmentation

| Method | Reproduced? | 19 mIoU | 19 mAcc | 15 mIoU | 15 mAcc | 10 mIoU | 10 mAcc |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenGaussian | yes |  |  |  |  |  |  |
| Dr. Splat | yes |  |  |  |  |  |  |
| OccamLGS | yes |  |  |  |  |  |  |
| Semantic Gaussians | yes/adapted |  |  |  |  |  |  |
| CAGS | yes/adapted |  |  |  |  |  |  |
| CTF-GS | yes |  |  |  |  |  |  |

这里 CTF-GS 当前需要提高，尤其是 15/10 split，否则很难打 SOTA。

---

# 9. 当前协议里我建议立刻修的几个问题

## 9.1 “novel-view”措辞要核查

如果 LERF annotated evaluation frames 被用于 3DGS geometry training 或 teacher-feature supervision，那么主文不要反复说 novel-view。可以改成：

> held-out annotated-view evaluation

或者：

> evaluation on annotated LERF-OVS views; masks are never used for training or calibration.

如果 annotated RGB frames 确实完全 excluded from feature supervision，则可以说 novel-view。但这个要在 protocol card 里写清楚。

## 9.2 threshold-0.60 不能像 test-set tuning

你们当前 threshold-0.60 结果很好，但主文需要说明它不是在 test masks 上挑出来的。最稳做法是：

- 主表：fixed 0.60 或 leave-one-scene-out；
- appendix：0.50 / 0.55 / 0.60 / 0.65 sweep；
- 不要用“best threshold”进入 main result。

## 9.3 ScanNet 必须从 probe 升级为 standard segmentation

当前 ScanNet row 可以保留为 compatibility probe，但如果它在主文三任务里叫 segmentation，就要变成：

```text
for each GT point p:
  aggregate nearby visible Gaussian / rendered feature evidence
  compute class logits against prompt ensemble
  assign argmax label
  compare against GT semantic label
```

不要依赖 row-aligned `gaussian_index` 作为唯一主结果。它适合内部审计，不适合和 OpenGaussian / Dr. Splat / OccamLGS 的 ScanNet semantic segmentation 对比。

## 9.4 SAM3-box row 需要给 baseline 同等机会

如果用 SAM3 box readout 提升边界，应该给所有 direct 3D baseline 的 selected mask 都套同一个 SAM3 readout。否则这张表只能叫：

> CTF-GS boundary-readout diagnostic

不能叫：

> direct 3D SOTA。

## 9.5 统一 text prompt protocol

建议固定：

```yaml
prompt_templates:
  - "{}"
  - "a photo of {}"
  - "a photo of the {}"
  - "the {}"
```

然后做两种 track：

1. **native-prompt track**：每个方法用官方 prompt/template；
2. **unified-prompt track**：所有方法尽可能使用统一 prompt bank。

主文优先 native-prompt reproduced，因为这是各方法最公平的性能；appendix 放 unified-prompt robustness。

---

# 10. 如何把主文变得简洁

我建议主文只保留下面结构。

## Introduction

1. 2D foundation features are image-bound。
2. 3DGS 是好 scene representation，但 RGB-oriented。
3. 问题：能否把 dense RADIO features 压进 compact 3D Gaussian scene memory？
4. 方法：CTF-GS = compact feature field + teacher decoder + VPR。
5. 三个任务验证：LERF rendered OVS、LERF direct 3D selection、ScanNet OV segmentation。

## Method

### 3.1 Compact teacher-feature scene memory

讲 compact latent field + teacher decoder。

### 3.2 Training with frozen teacher compatibility

讲 RADIO feature reconstruction + frozen-head consistency，一段即可。

### 3.3 Three inference readouts

只讲三个 readout：

1. rendered-view readout；
2. VPR text-to-Gaussian readout；
3. point-cloud semantic readout。

## Experiments

### 4.1 Unified protocols

一张表讲三任务、数据、输出、指标。

### 4.2 Main results

三张主表：T1 / T2 / T3。

### 4.3 Controlled evidence

teacher、cache、1280-D memory、ablation、storage。

### 4.4 Failure and diagnostics

只放最强 failure figure；细节进 appendix。

---

# 11. 现在最重要的判断

你的直觉是对的：**要想投顶刊并打 SOTA，最好把对比方法重新复现一遍。**

但更关键的是：

> **复现之前先冻结统一 evaluator，否则复现出来的数字仍然没有说服力。**

我建议行动顺序是：

1. 先定义三条 canonical protocols：T1 / T2 / T3；
2. 写统一 evaluator 和 prediction export format；
3. 先复现 OpenGaussian、OccamLGS、LangSplatV2、GAGS、Dr. Splat；
4. 再补 LangSplat、LEGaussians、CAGS、Semantic Gaussians；
5. OpenGaFF / SuperGSeg 作为 published context，等代码 release 后再进入 reproduced leaderboard；
6. 重新评估 CTF-GS，并针对弱项优化：T1 mIoU、T2 Acc@0.25、T3 15/10 split。

现在这篇论文最干净的主线应该是：

> **CTF-GS learns a compact RADIO-compatible Gaussian scene memory. It supports three standard open-vocabulary 3D evaluation interfaces: rendered-view OVS, text-to-Gaussian object selection, and ScanNet point-cloud semantic segmentation. Under a unified reproduced benchmark, it is compared against open-source language-Gaussian baselines.**

如果复现后 CTF-GS 确实三任务都领先，就可以打强 SOTA。  
如果只在 LocAcc、compactness、teacher-memory controlled baselines 或 SAM3-assisted boundary readout 上领先，那就把 claim 收敛成对应维度的 SOTA / best reproduced result，不要强行写全局 SOTA。

---

**Sources:**

- [OpenGaFF: Open-Vocabulary Gaussian Feature Field with Codebook Attention](https://arxiv.org/html/2605.06088v1)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/Arthurshen926/RADIO-GS/main/paper/radio_gs_draft.tex)
- [GitHub - yanmin-wu/OpenGaussian: [NeurIPS 2024] OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding · GitHub](https://github.com/yanmin-wu/OpenGaussian)
- [SuperGSeg: Open-Vocabulary 3D Segmentation with Structured Super-Gaussians](https://supergseg.github.io/)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)