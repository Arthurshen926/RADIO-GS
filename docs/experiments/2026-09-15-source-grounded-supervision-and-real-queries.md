# 源侧语义监督修复与真实跨视角召回（2026-09-15）

本轮完成19次训练及两场景真实标注评估。结论：旧碎片裁剪的词表 top1 分类污染了文本监督，修复源侧定位后，文本、图像、直接点提示的真实开发集召回均改善。查询时点提示优化仍有退化，因此不替换正式历史性能基线。后续开发使用 corrected v2 监督，不使用有缺陷的 v1。

## 实现与排除的错误方向

- 保持冻结 MoGe 几何和64维元素特征场；文本/图像共享原生 SigLIP 查询适配，提示使用原生 DINO 与注册世界坐标。四教师特征、关系、边界辅助监督保留。此次没有宣称提升几何。
- 新源侧 SAM3 文本定位只读取训练 RGB 和原有79词词表，不读取评估类别或目标标注。SAM3 使用 FP32 权重及 BF16 autocast，修复直接 BF16 权重的矩阵类型错误；官方 masks_logits 已是概率，避免重复 sigmoid。
- 几何贡献把像素定位映射到元素；正负标签分别要求至少两个训练视角证据。符号冲突元素弃权，不强行决定标签。像素监督要求对应符号元素贡献及几何支持均超过0.5。
- 关键 v2 修复：某查询在一帧未检出时必须弃权。只有该查询在同一帧存在明确正定位，其他明确对象内部的低概率区域才可贡献负例。未知背景不作负例。
- 训练器只替换训练文本任务，图像/提示任务与验证资产保持；实际2D/3D membership 使用新证据。报告新增真实优化目标拟合度，旧文本标签监控另行注明，避免把旧监控冒充训练质量。监督哈希写入最终 checkpoint。
- 上下文裁剪实验发现背景泄漏：扩大裁剪后周围碗可把桌面碎片判为碗。加入前景擦除控制后，两个场景只有6/3个合格区域，没有合法双端文本任务；明确拒绝训练，没有降低门槛。
- generic-negative readout 未获得跨场景一致收益，保持关闭。v1错误负例模型仅保留为诊断，不作为交付候选。

## 真实文本 GT：固定0.5阈值，完整正样本分母

figurines 为4帧21类56任务；ramen为7帧14类71任务。下表均为观察任务 mIoU/AP，数值范围0–1。figurines使用同366步旧监督控制，避免把增加训练预算误报成修复收益。

|场景|配置|mIoU|AP|AUROC|
|---|---|---:|---:|---:|
|figurines|grounded_v2366|0.1275|0.1767|0.7685|
|figurines|grounded_v2_seed43366|0.1213|0.1644|0.7823|
|figurines|grounded_v2_no_radio366|0.1273|0.1739|0.7674|
|figurines|旧监督 matched366_seed42|0.0588|0.1101|0.7191|
|figurines|旧监督 matched366_seed43|0.0589|0.1339|0.7155|
|ramen|grounded_v2660|0.1664|0.2581|0.8206|
|ramen|grounded_v2_seed43660|0.1635|0.2574|0.8160|
|ramen|grounded_v2_no_radio660|0.1641|0.2565|0.8190|
|ramen|旧监督660步|0.0482|0.1229|0.6489|

原180步 figurines 基线为0.0479；同366步控制为0.0588，新监督为0.1275，第二种子0.1213。ramen同660步从0.0482到0.1664，第二种子0.1635。改善并非仅靠训练更久或单个随机种子。

## 同一跨视角任务集合的三种查询

每类选择首个标注帧作参考，后续标注帧作目标。参考多边形作为图像裁剪/提示的显式用户输入，目标多边形仅用于指标。文本直接编码类别。目标图像不用于其自身查询。提示额外报告既有64步源种子条件优化，优化过程不更新地图。

这是由真实2D类别标注派生的开发协议，不是正式实例跟踪或官方三维评测；同类多个对象按类别并集，不能当作已核验的实例身份。

|场景/模型|任务数|文本 IoU|图像 IoU|直接提示 IoU|优化提示 IoU|
|---|---:|---:|---:|---:|---:|
|figurines/matched366_seed42|35|0.0702|0.0392|0.0237|0.1501|
|figurines/grounded_v2366|35|0.1335|0.0713|0.0485|0.1440|
|figurines/grounded_v2_seed43366|35|0.1243|0.0694|0.0565|0.1500|
|figurines/grounded_v2_no_radio366|35|0.1329|0.0699|0.0591|0.1563|
|ramen/dense_view_coverage660|57|0.0541|0.0596|0.0618|0.1944|
|ramen/grounded_v2660|57|0.1539|0.1044|0.0926|0.2021|
|ramen/grounded_v2_seed43660|57|0.1495|0.0903|0.0912|0.1908|
|ramen/grounded_v2_no_radio660|57|0.1517|0.1030|0.0914|0.1953|

优化提示未通过全面提升验收：figurines seed42 IoU从0.1501降至0.1440，AP从0.3169降至0.2780；ramen IoU从0.1944到0.2021，但AP从0.4127降至0.3779。ramen图像/直接提示AUROC也有小幅退化。不能只挑IoU声称所有指标改善。所有逐任务 AP/AUROC 保留在原始 JSON。

## 训练、几何与部署检查

- corrected v2 文本任务：figurines63个/14类，ramen99个/14类；冲突弃权的查询-元素对分别2107、12293。
- seed42实际训练文本 IoU分别0.9142、0.9429；图像0.9931、0.9125；提示0.9953、0.9008。它们是源侧已知支持伪标签拟合度，不是最终召回指标。
- 可见且语义训练元素分别92249、255984；无可见却未更新特征的元素被标成有效。所有最终模型保持原几何绑定。
- 四教师模型使用 radio/dino/sam/siglip；无RADIO模型只打开其余三教师。移除RADIO真实文本IoU为0.1273/0.1641。注意此次文本标签生成仍依赖SAM3，不能把去辅助SAM教师等价为整个流程无需SAM。
- 73项相关测试通过，覆盖未知负例弃权、双视角冲突、上下文监督、特征有效性、排序指标、带符号查询条件优化及现有召回；git diff --check 通过。
- 独立进程三类原生 query 冷加载结果见各场景 grounded_v2_no_radio_cold JSON；禁止读取训练教师、episodes、queries及新源侧监督缓存。

## 可复现开发入口

源定位：`scripts/audits/build_source_grounded_text.py --root ROOT --output ROOT/source_grounded_text_fp32_v2.pt`，使用 sam3 Python 环境。
编译：`scripts/audits/compile_grounded_text_supervision.py --root ROOT --grounding ROOT/source_grounded_text_fp32_v2.pt --output ROOT/grounded_text_supervision_v2.pt`。
训练：`radio_gs/scripts/train_three_query_field.py --root ROOT --geometry GEOMETRY --name NEW_NAME --steps 366或660 --share-siglip-adapter --grounded-text-supervision ROOT/grounded_text_supervision_v2.pt`。seed43增加 `--seed 43`；无RADIO增加 `--teachers dino,sam,siglip`。保留默认64维、密集教师覆盖、常数学习率。输出不可覆盖，复跑必须新名称。
评估入口：`evaluate_frozen_field_lerf_labels.py`、`evaluate_real_reference_queries.py --refine-prompts`、`cold_three_query_native.py`。完整输入绑定和输出哈希见配套 JSON。

## 尚未解决与下一阶段门槛

1. 有效文本覆盖仍只14个源概念，专名、颜色属性、细粒度组合和同类实例区分不足；figurines多类IoU仍为0。下一步需要来源可靠的细粒度区域描述/属性证据及跨视角身份监督，而不是加入更多未核验硬标签。
2. 提示训练与真实用户参考查询有差距；改进必须同时通过原始提示与源种子优化后的IoU/AP，而不能依赖查询时优化掩盖场缺陷。
3. 几何采样与足迹的细节瓶颈仍沿用上一报告结论。本轮几何冻结，不能声称已优于历史几何或完成独立3D质量验证。
4. 两场景均属开发数据，尚无正式同合约历史主线回归、独立3D真值和更多场景盲测。因此推荐v2为后续开发监督方案，保留历史性能基线；不把通用CLI默认偷偷切换成已验收正式主线。

可视化：`paper/artifacts/ramen_source_text_label_contact_20260915.png` 为原裁剪标签诊断；`paper/artifacts/ramen_source_grounded_text_contact_20260915.png` 为新定位；真实响应图在各场景 grounded_v2_previews_20260915 目录。
