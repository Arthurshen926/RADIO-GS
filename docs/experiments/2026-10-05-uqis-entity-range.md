# UQIS：范围证据的类型、补全与误检

2026-10-05。本轮完成4组完整UQIS对照：36个场景预测／评估作业、1072次独立query执行。
全部9景、67目标、268条query均保留。**没有候选取得足够的整体净收益，唯一默认仍为v8。**
新增的范围接口、两类离线教师及复现入口已实现，均保留为关闭的研究选项。
本轮不是五个benchmark同时涨分，也不是新的SOTA或独立测试结论。

## 这轮真正排查了什么

v8局部匹配已改善图像身份，但认对身份的相同目标仍常只覆盖约一半。
原范围头要求种子与对应区域的共可见IoU和双侧可见比例均不低于0.5，容易维持原部件粒度。
我们固定几何、原生语义、局部图像匹配及文本／图像身份种子，分别改变对应依据和范围证据类型。
每组的原生场、锚点、种子控制各268份排名与v8最大差值均为0；134条文本／图像身份种子均未改变。
因而本轮退化不能归因于又一次压缩损坏query空间。

共同输入为原1107张合法采样源图，不额外裁掉区域或查询；这是声明的原建图预算，不是所有原始扫描帧。
UQIS查询阶段只读同一地图与当前query：自身图像裁剪、2D像素／相机或3D坐标；没有源RGB、深度、SAM或跨模态配对。
新教师只在建图运行。所有自动预测完成后才评分，标签辅助候选诊断与自动成绩完全分开。

## 四个固定条件

**第三视角对应。** 借鉴[MaskClustering](https://arxiv.org/html/2401.07745v2)的多视角包含证据。
该工作假设输入近似对象级panoptic分割；我们的SAM整体／部件候选重叠，不能直接套用其对象假设。
本实验每视角最多一票，排除候选自身的两个生成视角；可见比例0.3、包含比例0.8、共识比例0.9，至少两个其他视角。
见证区域自身须在至少两个其他可见视角中被一致包含，保留率0.8。
保留每个种子的邻域，不做传递连通分量合并，不平均或改写身份特征。
点提示仅在包含锚点的原区域中，按视角平衡的支持选种子。
同一图分别以可见多数和正证据并集读出，构成前两个条件；没有阈值扫描。

**通用概念范围。** 同一SAM3检查点切换到概念分割路径，所有原源图统一使用`object`、置信度0.5。
不读取目标名称，也不针对失败目标改写提示。1107图得到833区域，其中967图为空；828区域有mesh支撑。
它实际只作用于0条文本、2条图像、各1条点提示，说明“给一个通用词”不能代替对象级分割。

**整图分割范围。** 使用本地已有[FC-CLIP／Mask-Adapter](https://github.com/hustvl/MaskAdapter)完整预训练权重，
保持上游COCO推理词表及配置，不使用UQIS类别清单。1107图得到10763区域，10431有mesh支撑，没有空图。
只将mask归属写入地图，不将教师类别名改写为query语义。模型原有250个学习查询属于模型结构，没有额外输出截断。
这是新增教师信息与建图计算，不能称为相同证据预算下的新算法收益，也不是我们训练出的分割模型。

后两者使用相同范围接口：原生种子与包含它的实体区域通过几何绑定；实体沿原物理对应规则聚合，
至少两个支持视角才使用该范围；证据缺失时保留本次重新计算的v8范围。点提示用实体区域共归属。
没有读取保存过的最终答案，没有按query成绩决定回退。整图先验实际作用于45/48/54/56条文本／图像／2D点／3D点。
它的类别体系和stuff合并仍可能与官方实例范围不一致，不能把这些mask称为已验证的完整实例。

## 完整自动结果

均为场景宏平均百分数，每格为 **AP / 实际集合IoU**；v8是已完成的冻结对照。

|条件|text|image|2D正点|3D正点|
|---|---:|---:|---:|---:|
|v8默认|26.596 /16.387|44.034 /34.876|52.745 /42.090|52.359 /41.524|
|第三视角＋多数|24.945 /15.545|40.891 /31.624|42.775 /39.490|42.071 /38.669|
|第三视角＋并集|24.945 /15.602|40.891 /32.204|42.775 /39.398|42.071 /38.602|
|通用概念先验|26.596 /16.387|43.912 /34.610|52.301 /41.814|51.925 /41.246|
|整图分割先验|26.584 /16.578|42.597 /33.305|41.938 /38.707|41.950 /38.579|

整图先验文本IoU仅+0.191个百分点，场景配对95%区间[-2.779,+2.781]，不作为有效提升。
四类同时IoU>=0.5从7/67到8/67，场景宏平均10.384%→11.772%；其他分项退化，不能只摘出这项宣称成功。

![相同查询队列的范围结果](../../output/witnessed_scope_study/comparison.png)

**补全增加了，但错误归属增加得更多。** 整图先验的2D点覆盖48.520%→51.442%，纯度80.057%→64.859%。
3D点覆盖47.733%→51.035%，纯度80.352%→64.917%。图像覆盖42.893%→43.982%，纯度59.999%→53.053%。
图像身份种子完全不变，最终所选集合以目标为唯一主导实例的数量却45→43；两类点分别57→52、58→52。
这直接说明：保护原生特征和身份入口，**并不能保护后续扩大范围时的对象归属**。

在前后均以目标为主导的相同49条2D点查询上，覆盖52.100%→58.623%，纯度91.539%→81.462%，
IoU48.479%→49.374%。这是条件子集的query均值，不替代全量场景指标；真实补全收益被其他目标上的误扩张抵消。

## 标签辅助诊断：信息缺口与选择缺口都有

以下每个目标允许用GT从整个**有限候选家族**选最优范围，只在全部自动预测之后计算。
它们不是自动方法成绩，不是查询时可用选择器，也不是任意未来模型的数学上限。

|标签选优候选家族|IoU|目标覆盖|
|---|---:|---:|
|原SAM单区域，既有对照|45.333%|49.334%|
|原一跳物理范围，既有对照|52.995%|59.072%|
|第三视角并集|52.485%|60.621%|
|通用概念＋原物理范围|0.992%|1.196%|
|整图分割单区域|39.868%|44.336%|
|整图分割＋原物理范围|45.758%|53.073%|
|原一跳与整图范围两家族联合选优|56.173%|62.381%|

整图先验在24/67目标上提供了优于原一跳家族的候选，但整体替换不成立。两家族联合仍仅56.173%：
不能把全部问题归给最后一个选择器，也不能期待再换一个合并阈值就获得高性能。
现在更明确的限制是：**对象级范围证据仍不充分；存在新范围时，机制也缺少判断其正确归属的独立依据。**

## 共同架构与收敛决定

唯一默认保持`configs/pipeline/system_v8.json`，不把四个失败候选拼成按query选优的新主线。
本轮没有修改LERF／NVOS／ScanNet的有效数值配方，也未重复运行这些未受影响的路径；它们沿用已完成的v8共同结果，
不记作本轮新增收益。当前所谓LERF选集成绩仍是实际Gaussian集合的2D投影，不冒称独立3D真值IoU。

可保留的实现是统一范围证据接口：原生多教师身份 → 独立范围先验 → 几何关联 → 实际集合。
`entity_observation_scope.py`接受显式归属／可见性；教师选择发生在建图，四类query不增加源图权限。
这为后续独立训练的归属机制提供接入点，但本轮尚未得到应升级的范围模型。

下一阶段只围绕两件事推进，不继续扫描并集、图深或通用提示词：

1. **验证真正的实例范围监督。** 用不重叠的训练场景或独立预训练实例模型，区分同实例、整体／部件、不同实例与未知。
   不能把COCO语义区域、任意SAM部件或多视角一致性直接当作完整实例标签。若使用额外3D标注训练，单列监督条件。
2. **直接学习或验证实际补全部分的归属。** 身份坐标固定，训练与评价同时检查新增真阳性、范围外误检和被删掉的原目标，
   在相同教师／源图预算下与简单读出比较。先验证UQIS四类输入，再对类别集合与分区任务检查可迁移部分，
   不把实例规则直接套到LERF同类集合或ScanNet类别分区。

当前九景全部是开发集；独立最终场景、紧凑学生、部件／正负交互与公开SOTA等价评价仍未完成。

## 复现与成本

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_witnessed_scope_cohort.py --mode majority --gpus 0,1
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_witnessed_scope_cohort.py --mode union --gpus 4,5
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_entity_scope_cohort.py --gpus 0,1,4,5
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_entity_scope_cohort.py --provider panoptic --output output/panoptic_scope_study --gpus 0,1,4,5
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/summarize_witnessed_scope.py --entity-system output/entity_scope_study/system --panoptic-system output/panoptic_scope_study/system
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/diagnose_witnessed_scope_capacity.py
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/diagnose_entity_scope_capacity.py --root output/panoptic_scope_study --compare-capacity output/observation_hypotheses_study/label_capacity.json
```

完整比较`output/witnessed_scope_study/comparison.json`，三个研究根目录各自保留共同配方、全部预测及评分。
固定条件为`configs/recipes/uqis_witnessed_scope.json`、`uqis_entity_scope.json`、`uqis_panoptic_scope.json`。
新增图约0.485MiB，编译记录合计58.35秒；通用概念源处理412.66秒，整图分割源处理1033.96秒。
整图先验增加地图5.780MiB，总2858.590MiB，原共享query编码器不变；额外离线教师权重另计。
各完整四模态队列查询记录合计约406—411秒，包含模型加载且共享GPU，不是独占设备延迟基准。
没有新增单元测试、训练GT回流、人工逐query路线或未结束后台任务。
