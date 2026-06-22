# GaussFM 中文导师讨论版 PPT 设计

日期：2026-06-19

## 目标

生成一份面向导师讨论的中文 PPT。它不追求对外 10 分钟短讲的极简包装，而是要把故事讲完整、证据放充分、问题边界说清楚。

## 输出

- `paper/gaussfm_advisor_discussion_cn.pptx`
- `paper/gaussfm_advisor_discussion_cn_outline.md`
- `radio_gs/scripts/build_advisor_discussion_cn_presentation.py`

## 内容原则

- 中文为主，保留必要英文术语如 GaussFM、RADIO、LERF、ScanNet、VALA-aligned。
- 不出现 `OpenGaFF`、`CTF-GS`、`CTFGS`。
- 用 `frame-wise RADIO`、`RADIO 参考特征` 表达原始 RADIO 对比，不使用 teacher/student 作为主叙事。
- 实验页尽量放全：主结果、原始 RADIO 对比、frozen-head downstream、architecture ablation、Direct3D readout ablation、ScanNet diagnostic、storage/efficiency。
- 每页仍保持凝练：标题直接写结论，正文以表格和 2-4 个 bullet 为主。

## 页序

1. 题目与一句话主张
2. 为什么需要 3D foundation-feature memory
3. 核心问题与论文定位
4. 方法总览
5. 核心组件：compact field
6. 核心组件：重建与多视角支持
7. 主要贡献/创新点
8. 实验矩阵：claim 到 evidence
9. 主结果：LERF rendered-view OVS
10. 主结果：LERF direct 3D object selection
11. 主结果：VALA-aligned ScanNet point query
12. 原始 RADIO 对比：controlled evidence
13. 原始 RADIO 对比：SAM3/DINO frozen-head tasks
14. Architecture ablation
15. Direct3D readout/support ablation
16. ScanNet readout diagnostic
17. Storage / efficiency
18. 定性证据与可视化使用方式
19. 当前边界和风险
20. 导师讨论点

## 验证

- 运行生成脚本。
- PPTX zip/XML 结构有效，slide 数为 20。
- PPTX XML 中包含 `GaussFM`、`frame-wise RADIO`、`VALA-aligned`。
- PPTX XML 中不包含 `OpenGaFF`、`CTF-GS`、`CTFGS`。
