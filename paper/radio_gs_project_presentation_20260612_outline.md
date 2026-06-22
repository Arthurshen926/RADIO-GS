# GaussFM / RADIO-GS 项目汇报 PPT 大纲

- 输出文件：`/root/RADIO-GS/paper/radio_gs_project_presentation_20260612.pptx`
- 口径：导师/项目中期汇报 + 顶刊投稿准备

## 1. 标题与一句话主张
- 开场先定义本文不是做一个新的 3DGS RGB 重建，而是在冻结几何上学习 compact RADIO feature memory。
- 用一句话讲清楚：小存储、可重建、可查询，并支撑 2D/3D 开放词汇理解。

## 2. 汇报主线：从问题到投稿闭环
- 这一页用于给导师建立总路线，后面所有章节都围绕这个闭环展开。

## 3. 背景：2D foundation features 与 3D 场景记忆之间有断层
- 这里避免说别人完全没有 2D/3D 双接口，重点放在 compact + reconstructed feature quality + support stability。

## 4. 核心假设：压缩不是损失信息，而是多视角去噪与重组
- 这一页用于强调 compact 不是退化，是多视角信息瓶颈，有新意。

## 5. 整体流程：先 RGB 3DGS 几何，再学习 compact RADIO feature memory
- 这是纠正框架图误解的关键页。

## 6. 方法一：Hybrid Compact RADIO Feature Field
- 这里回答用户关心的：compact memory learning 不只是一个 decoder，而是 fine latent + spatial hash + fusion + HCD codec。

## 7. 方法二：Multiview Primitive Registration 压回 compact field
- 这页把 VPR cache 风险改写成训练机制，而不是方法依赖。

## 8. 训练目标：主干 RADIO 重建 + 任务相关一致性
- 这是答辩/汇报中最容易被问到的概念边界。

## 9. 推理/评估接口：同一 compact memory 支撑三类开放词汇任务
- 这页也是后续 qualitative 排版的逻辑依据。

## 10. 实验设计：四条定量证据链
- 四条证据链是论文实验章节的骨架。

## 11. 主结果一：LERF rendered-view open-vocabulary grounding
- 这里先放 LERF rendered 主表，说明 2D 查询能力。

## 12. 主结果二：LERF direct 3D object selection
- 这里要准确讲：compact row 是主线，但 strict no-RGB one-map ablation 另有数值；当前最佳使用轻量 RGB support guard。

## 13. 主结果三：VALA-aligned ScanNet-8 direct point-query
- 这页按用户要求使用与公开 VALA 协议一致的 direct point-query 设置，不在论文中暴露未发表方法名。

## 14. 重建 Scene Features vs. 原始 frame-wise RADIO
- 这里直接支撑用户关心的：rendered field 在 selected frozen-head tasks 上强于 frame-wise RADIO。

## 15. 定性一：LERF 2D + 3D Open-Vocabulary Query
- 主文定性图应选我们强、baseline 弱但合理的样本；2D baseline 建议 LangSplatV2，3D baseline 建议 Dr. Splat。

## 16. 定性二：ScanNet Open-Vocabulary 3D Query
- ScanNet 定性不宜太花，重点展示某一开放词汇类别是否被查到。

## 17. 定性三：消融可视化要服务核心贡献
- 这页给导师看图表取舍：不是每个模块都放 qualitative。

## 18. 定量消融：贡献大小排序
- 这页解决“模块太多”的问题：按贡献排序，不平均讲。

## 19. Storage / Efficiency：小存储与单次 query latency
- 这一页按用户要求突出 single-query latency，而不是总评估耗时。

## 20. 投稿叙事：建议面向 TPAMI 的组织方式
- 顶刊不是只看指标，还看叙事是否清楚、协议是否无歧义。

## 21. 当前风险与边界：提前防 reviewer 质疑
- 这页适合导师问答，口径要比论文正文更直白。

## 22. 下一步：把投稿包从“可投”打磨到“强稿”
- 收尾时给出明确可执行清单，而不是泛泛说继续优化。
