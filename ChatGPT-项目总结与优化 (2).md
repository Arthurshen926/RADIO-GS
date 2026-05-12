# 项目总结与优化

**User:** Anonymous  
**Created:** 5/8/2026 13:39:01  
**Updated:** 5/12/2026 3:07:38  
**Exported:** 5/12/2026 3:13:07  
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



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)