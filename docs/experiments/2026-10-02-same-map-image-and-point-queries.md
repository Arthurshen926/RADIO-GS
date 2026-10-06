# 同地图图像与点提示：接口闭环完成，范围语义仍未解决

2026-10-02。已完成两个源侧探针 × 四种输入条件，共八次完整推理。不是新benchmark成绩，不用于挑选LERF默认参数。

## 实际完成的系统改动

沿用ramen文本主线的冻结几何、原生SigLIP特征与合法源图，增加显式image、prompt_2d、prompt_3d Query。所有参考实例路径经过同一SAM3源视图跟踪、连续响应重建、实际Gaussian集合求解和双读出。没有重新训练地图或使用旧查询预测。统一入口支持 `--scene-input` / `--query-input`；没有适用的冻结评估器时，不把自定义查询交给LERF文本评分。

图像query使用原生图像编码，在原生场中输出余弦排序；不套用文本概率标定。实际范围路径在声明的源区域缓存中检索最高相似度种子，再跟踪122个合法源视图。源区域缓存和照片仍是查询依赖，不能称为仅靠紧凑场完成召回。

3D点绑定地图坐标，寻找最近Gaussian，并在合法视图中找到该Gaussian贡献最大的可见像素，以其归一化坐标启动SAM3点提示。这是可见性约束的载体锚点，不是世界点的精确几何投影，也不是已经证明了最优视角选择。2D点可直接进入同一参考实例路径。

RGB-only与RGB+前景掩码是两种明确不同的输入条件。后者把掩码外RGB设为灰色再编码；不能用其效果替代前者成绩。

## 固定探针与实验边界

在授权源帧的中间帧68，从已有原生区域缓存选出最大的两个互相IoU<0.1的区域。每个区域取纯度>0.9的可见Gaussian作3D锚点；图像输入为区域矩形裁剪，未去背景。图像检索排除原裁剪帧，但原帧仍参加地图构建和之后跟踪。因此不是独立留出评估。

两个区域来自SAM伪标签，没有独立真值。第二个区域视觉上是碗外的桌面范围，RGB矩形裁剪仍包含非常显眼的碗：单纯RGB不能唯一表达要查询外围区域。辅助掩码来自同一源伪区域，属于更强输入。两个2D点由3D解析结果直接产生，是坐标转换等价对照，不是独立2D提示标注。

## 结果

以下是选中Gaussian再投影到源帧，与源伪区域的IoU，**不是检索准确率或官方IoU**。

|探针|RGB图像|RGB+前景掩码|3D点|等价2D点|
|---|---:|---:|---:|---:|
|0：碗及内部的源区域|96.85%|96.21%|43.72%|43.72%|
|1：碗外源区域|0.0044%|53.54%|56.71%|56.71%|

等价2D/3D输入的元素集合IoU分别为100%和99.99365%；第二例仅两个元素不同，连续响应最大差约2e-5。原生图像响应中，探针1的构造锚点排序从前24.07%变为前0.269%，但这是自构造锚点诊断，不能当分类准确率。

可视化显示：RGB图像0找回完整碗区域；点0主要选择碗的边缘/部分表面，不能自动表达“碗和内部全部内容”。图像1仍选碗；明确前景掩码后转向外围，但出现跨边界扩张。点1同样存在外围扩张。范围错误并没有因接口打通而解决。

输出：`output/same_map_queries/summary.json`，`source_comparison.jpg`，每个 `probe_<i>_<mode>/runtime/typed_readout.pt`。八次完整作业已结束，未添加到112份benchmark结果登记中。

## 本阶段结论与后续单一问题

完成的是共享地图与共享集合求解的多模态运行闭环，**不是完整四query能力验收**。文本任务是text_union，当前图像/点任务是reference_instance；不能仅因四种入口可运行就声称任务语义一致。NVOS正参考实例86.49%、LERF文本集合投影66.66%为先前结果，本轮没有重跑或提高这些分数。

下一阶段集中解决查询到范围的歧义：先明确图像输入到底提供RGB、前景掩码还是参考区域，点输入要求单表面、部件还是整体。保持这些条件分别报告。随后在固定独立标注下比较种子范围、跨视角传播范围与最终元素集合，判断错误先发生在哪一层。不得用本轮两张源伪区域调参后称为泛化收益；也不能单纯扩大集合来解决单点范围不足。ScanNet统一运行、独立三维评价、新场景与紧凑化仍未完成；UQIS保持暂停。

## 复现

仓库根目录，运行环境 `/root/miniconda3/envs/cybersim_agent/bin/python`，SAM3子任务使用sam3环境。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/prepare_same_map_probes.py
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_same_map_queries.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/summarize_same_map_queries.py
```

本轮最初四个及等价2D作业由批处理运行，两个掩码控制由独立并发调用运行；批处理现已包含相同掩码控制以便整体复现，该重新编排未重复运行八次模型。

统一入口命令（本轮确认了计划生成；完整模型运行通过其调用的run_reference_query完成）：

```bash
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_current_research.py \
  --scene-input output/same_map_queries/probes/scene.json \
  --query-input output/same_map_queries/probes/probe_0_image.json \
  --gpu 0 --output output/custom_reference --plan
```
