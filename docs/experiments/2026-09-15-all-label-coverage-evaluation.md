# 全标签覆盖评估修复（2026-09-15）

支持区域内 IoU 会过滤没有地图支持的已知正样本，因此不能单独代表端到端召回。新增 image_all_labels / image_refined_all_labels：只使用已有非冲突标签，支持不足（贡献质量不超过 0.5）的预测置零，正标签计为漏检。原条件指标保留，训练监督不变。

同一冻结模型、相同 query 和标签重评；以下都是 source pseudo-label 的开发证据，不是 benchmark GT。ramen 使用新增六视角，text 7 / image 38 / prompt 38；figurines 使用原验证集。

|场景/模型|query|条件 IoU|全标签 IoU|正标签支持召回|
|---|---|---:|---:|---:|
|ramen/dense_view_coverage660|text|0.6424|0.6415|0.9990|
|ramen/dense_view_coverage660|image|0.5412|0.5213|0.9165|
|ramen/dense_view_coverage660|prompt|0.5797|0.5390|0.9165|
|ramen/dense_view_without_radio660|text|0.6383|0.6373|0.9990|
|ramen/dense_view_without_radio660|image|0.5411|0.5195|0.9165|
|ramen/dense_view_without_radio660|prompt|0.5940|0.5528|0.9165|
|figurines/dense_view_coverage180|text|0.4370|0.4370|1.0000|
|figurines/dense_view_coverage180|image|0.3427|0.3427|1.0000|
|figurines/dense_view_coverage180|prompt|0.3008|0.3008|1.0000|
|figurines/dense_view_without_radio180|text|0.4354|0.4354|1.0000|
|figurines/dense_view_without_radio180|image|0.3053|0.3053|1.0000|
|figurines/dense_view_without_radio180|prompt|0.2912|0.2912|1.0000|

结论：覆盖缺口与覆盖内匹配错误必须分别处理。新增指标不改变模型排序的合法性边界，也不允许使用验证标签补训练覆盖。后续训练视角选择继续仅依赖训练侧几何；特征优化同时检查条件 IoU 与全标签 IoU，防止靠缩小有效支持改善表面指标。

提示最小修正仍是可选实验，没有因本次报告变更默认策略。全标签指标不受 reference-run 的支持交集裁剪。3D 仍保留条件指标：不能把缺少可靠 3D 标签的元素制造为真值。

验证：相关 10 个测试文件 56 passed；包含未覆盖正样本计漏检、空地图不能获得完美正样本召回、未知标签不计负样本。两个场景四个冻结模型完成实际评估，原条件分数复现。
