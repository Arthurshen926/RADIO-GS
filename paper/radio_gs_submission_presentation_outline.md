# RADIO-GS Submission Presentation Outline

## 1. RADIO-GS
- 开场用一句话定义：不是训练一个语言分类器，而是把 RADIO foundation features 重建成一个可渲染的 3D feature memory。

## 2. 一句话主张
- 这页给听众建立主线，后面所有方法和实验都围绕 feature memory 展开。

## 3. 动机：2D foundation features 与 3D deployment 的断层
- 强调问题不是传统 3D 语义分割，也不是 2D feature extraction，而是 3D feature reconstruction。

## 4. 待解决的问题
- 这页为后面的 HCD、hybrid、refiner、FDH 和 adaptor supervision 做铺垫。

## 5. 方法总览
- 这页可以作为论文 Figure 2 的口头解释版本。先讲训练流，再讲 inference 流。

## 6. 方法组件 1：Hybrid Gaussian Feature Field
- 这页回答用户之前关心的 LocAcc 下降原因：区域覆盖和 peak localization 是两种不同指标。

## 7. 方法组件 2：HCD Codec 与 Screen-Space Refiner
- 强调 HCD 消融是最强证据，不是调参带来的小差别。

## 8. 方法组件 3：FDH Warm-Start 与冻结头监督

## 9. 方法组件 4：RADIO Adaptors

## 10. 实验协议与证据边界
- 这页避免 reviewer 认为混用协议。强调 main table、ablation、diagnostic 的边界。

## 11. 主结果：LERF-OVS Rendered-Feature Grounding
- 注意主表是 current-best freeze，不和 seed-7 ablation 混为一谈。

## 12. Rendered Features vs. Original RADIO RGB
- 这页是最强 story 页之一：为什么重建的 feature 可能比单帧 teacher 更适合 novel-view grounding。

## 13. 定性结果：LERF grounding overlays
- 用这页展示四个场景的 rendered feature heatmap 和视觉质量，强调来自 frozen shortlist。

## 14. 核心组件消融：闭合证据链

## 15. 为什么 mIoU 提升但 LocAcc 下降？

## 16. Adaptor ablations：DINOv3 / SAM3

## 17. SAM/DINO 下游探针

## 18. ScanNet v67：跨域 direct point-query

## 19. ProFuse-inspired DINO cross-view diagnostics

## 20. 效率与成本：分开报告不同 measurement type

## 21. Baseline provenance：投稿前必须保守

## 22. 项目完成度与投稿主线

## 23. 论文结构建议

## 24. 投稿前剩余工作

## 25. Takeaways
- 最后把问题收回投稿目标：方法和证据链已经足够，下一步是 freeze-safe presentation and paper polish。

## 26. Backup：详细 LERF component table

## 27. Backup：artifact map
