# GaussFM 中文导师讨论版 PPT 大纲

- PPTX: `paper/gaussfm_advisor_discussion_cn.pptx`
- 口径：标准会议论文 presentation 流程 / 中文导师讨论版 / 投稿故事线
- 结构：背景 -> 动机 -> 相关工作 -> 问题缺口 -> 方法 -> 实验设置 -> 主结果 -> 定性 -> 控制对比 -> 消融 -> 边界

## 1. GaussFM：开放词汇三维场景的紧凑基础特征记忆
- 首页给出论文 thesis 和报告路线。

## 2. 任务背景：foundation features 正在成为 3D 场景理解的通用接口
- 标准会议报告先解释任务和研究对象。

## 3. 动机：逐帧 2D 特征强，但不是可部署的 3D 记忆
- 动机页明确为什么不能停留在 frame-wise feature extraction。

## 4. 相关方法脉络：已有工作各解决了一部分，但接口仍然割裂
- 补上标准报告中的 related work overview。

## 5. 问题分析：顶会论文必须证明三个层面的“不是”
- 这页解释为什么后续实验设计是必要的。

## 6. 本文贡献：一个记忆，三个接口，两类闭环证据
- 贡献页把方法贡献和证据贡献绑定。

## 7. 方法总览：从 posed RGB views 到 compact Gaussian feature memory
- 使用新总框架图。

## 8. 方法核心：Compact Gaussian Feature Memory
- 使用新 compact Gaussian feature memory 图。

## 9. 查询接口：同一个 memory 服务三种评测协议
- 替代不严谨的 query interface 表述。

## 10. 实验设置：主结果、控制对比和消融各回答一个问题
- 实验设置页让后续结果有导航。

## 11. 主结果一：LERF rendered-view 2D OVS
- 表格字号提高到 10。

## 12. 主结果二：LERF direct 3D object selection
- Direct 3D 表补全对比方法。

## 13. 主结果三：VALA-aligned ScanNet point query
- ScanNet 表字号提升到 9，并保留完整方法集合。

## 14. 定性一：同一 memory 同时支持 2D 与 3D 开放词汇查询
- 补足主结果对应定性图。

## 15. 定性二：GaussFM 与 frame-wise RADIO 的 visual comparison
- 加入原始 RADIO 定性对比。

## 16. 定性三：冻结头 probes 中的 scene-feature usability
- 加入 SAM/DINO 原始 RADIO 对比定性图。

## 17. 控制对比：不是最近帧 cache，也不是显式 1280-D Gaussian memory
- 控制对比表字号从 7 提升到 9。

## 18. 核心消融：只保留支撑主要贡献的三项强对照
- 消融表字号提升，内容收束为核心贡献。

## 19. Storage / efficiency：compact memory 的实际收益
- 效率表删掉一列以提升字号。

## 20. 边界：哪些 claim 必须主动收束
- 讨论页避免过度声明。

## 21. Takeaway：从 frame-wise features 到 queryable 3D scene memory
- 会议报告式收尾。
