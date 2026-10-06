# 自然指代学习、实际选集收益与训练观测域的缺口

本阶段完成两组自然语言指代学习、40个新场景作业（31项共同系统执行＋9景局部对照），
共536条UQIS四模态查询。所有参数和epoch只由独立训练／校准房间确定。
公共任务使用同一主架构和原有输入条件重新执行，未拼接历史答案。

**文本实际IoU有约2pp开发收益，但独立渲染房间退化、场景区间跨零，不能升级主线，
更不能宣称指代身份瓶颈已解决。** 唯一默认仍是`configs/pipeline/system_v8.json`。

## 这次真正新增了什么

过去是“整句话的向量与单个区域向量比较”。现在新增可选学习头：完整文本token、
原生区域特征和区域几何共同产生一个身份排序修正，再交给原来的范围解码器。

- 原生SigLIP特征、query编码器、几何、成员关系与范围解码不训练；原语义场逐值保留。
- `local`只看当前候选及完整文本；`scene`另让所有受支持候选交换内容与几何信息。
  不截断候选top-k，不向模型提供类别名或评估目标ID。
- 两组均560897个声明参数。局部组使用相同value/output投影，但场景注意力Q/K不参与，
  因而是同训练规则／输入预算对照，不声称有效参数或计算量完全一致。
- 修正头零初始化，原生logit项保持FP32；混合精度只用于学习修正。输出是排序logit，
  不冒充原生余弦或前景概率，不覆盖显式元素集合的评价。
- 监督是实际解码范围IoU的四次方归一化分布。它同时涉及身份、纯度和范围质量，
  **不等价于一个已经分离且完备的身份识别监督**；本轮结果也必须按这个限制解释。

这不是紧凑多教师学生，也不是额外attention模块已经产生新论文贡献。图像／两类点仍按
各自合法输入使用同一地图；单实例头没有被错误套到LERF类别并集或ScanNet类别分区。

## 实例对应和训练条件

下载32个既有独立房间的官方原始mesh、segments与aggregation，逐顶点核对训练载体。
坐标与顶点次序全部精确相同；旧实例编号与官方`objectId+1`在32景也全部一致。
因此这次排除了编号错位，**没有发现并宣称一个不存在的标签编号bug**。

既有24／4／4房间中，22／3／4有NR3D自然描述，分别1728／207／175条记录。
拟合只用1592条listener-correct记录；其中56条在现有范围家族中完全没有目标支持，
无法形成正分布，因此不进入梯度，共1536条实际训练记录。校准和开发评价保留全部记录。
房间与benchmark及其rescans分离；四个开发房间并非从未打开过的最终测试集。

两组各12个epoch，AdamW学习率0.0003、相同seed和批次规则。只按3个校准房间的
实际IoU选择epoch：local为5，scene为4。不按UQIS成绩选epoch或调参数。
额外自然语言／3D实例监督与此前纯VFM方案必须分开报告。

|场景宏平均实际IoU，%|原生排序|局部学习|场景学习|
|---|---:|---:|---:|
|22个拟合房间，全描述诊断|5.009|9.162|8.650|
|3个校准房间|2.568|3.601|3.320|
|4个训练／校准外开发房间|5.033|3.575|2.996|

训练损失下降且训练房间变好，不足以证明指代能力能迁移。不能因UQIS开发均值上升，
忽略这一独立房间的负面证据。也不能把该小样本结果解释为所有场景学习都无效。

## 五任务重新执行结果

场景组执行完整31项；局部组执行9景UQIS。以下为相同冻结评价方式，百分数。
v8对照取自上一阶段完成的9景重跑，仅供结果比较，不进入新预测过程；本轮40项均为新执行。

|UQIS文本条件|AP|实际IoU|目标占多数种子|四query同时IoU≥0.5|
|---|---:|---:|---:|---:|
|v8原生对照|26.596|16.387|20/67|7/67|
|完整自然语言＋局部候选学习|27.951|18.384|20/67|8/67|
|再加入场景候选注意力|28.172|18.330|22/67|7/67|

文本IoU变化分别+1.9977pp、+1.9431pp；按场景重采样的95%区间分别
[-1.5231,5.6506]与[-1.4438,5.1771]pp。没有证明稳定净增益或额外场景注意力优势。
种子主导身份正确数为23→21／24；局部组范围IoU上升但主导身份没有变好，
说明不能把“选集得分提高”直接翻译成“指定身份绑定已学会”。

两组原生文本语义输出逐值保持。图像／2D点／3D点全部实际集合与排序数组也逐值保持，
其实际IoU分别34.8762／42.0902／41.5238%，AP为44.0336／52.7449／52.3589%。

|本轮公共任务重新执行|结果，%|解释|
|---|---:|---|
|LERF语义2D IoU|70.4007|保持原规则|
|LERF实际Gaussian选集的2D投影IoU|66.9649|保持，微小重跑波动；不是独立3DGT|
|NVOS IoU|86.5134|保留正参考提示协议|
|ScanNet OVS 19／15／10类mIoU|61.0115／62.5276／72.4861|保持原规则|

公共结果是本轮重新执行得到的保持结果，不归因于新身份头；不是官方SOTA等价认证。

## 更上游的实际问题：训练时看到的对象已经残缺

旧独立训练资产是每房间12张合成点渲染图，不是部署时的真实RGB-D观测。
抽查能看到点渲染空洞、灰色区域和外观失真；定量范围检查为：

|集合|目标数|有目标占多数候选|目标平均源支持覆盖|有限范围家族标签最优IoU|
|---|---:|---:|---:|---:|
|拟合22景|246|175|65.04%|34.81%|
|校准3景|29|17|65.03%|23.76%|
|开发4景|25|19|80.19%|39.29%|

覆盖和最优IoU先按每个不同目标计算，再场景平均；与按描述重复次数加权的前表不同。
最优范围使用真值，仅诊断有限候选承载能力，不是自动成绩，也不是未来方法数学上限。
71个拟合目标连目标占多数区域都没有；范围质量监督必须在这些混杂或残缺候选上分配。
它提供了明确的数据承载限制，但尚不能把所有泛化失败归因于这一点。

本轮继续下载预先固定的首个拟合房间scene0271_00、首个开发房间scene0540_00的真实
ScanNet RGB-D源。直接复用部署用`registered_observation_map`，固定原mesh和教师，
分别处理全部stride20合法帧96／51张，无额外总数上限。另从同一真实地图均匀取12帧，
与旧12帧输入形成数量对照。12帧对照没有匹配相机位置，真实外观和相机分布共同改变，
不能把全部差值单独归因于纹理改善。该诊断不训练新模型、不改变benchmark。

|独立房间／输入|视图数|自动原生文本选集IoU|目标占多数候选覆盖|有限范围家族标签最优IoU|
|---|---:|---:|---:|---:|
|0271旧点渲染|12|0.876%|9/13目标|37.828%|
|0271真实RGB-D|12|7.541%|10/13目标|37.640%|
|0271真实RGB-D，完整既定采样|96|10.987%|12/13目标|62.215%|
|0540旧点渲染|12|2.042%|7/11目标|32.454%|
|0540真实RGB-D|12|9.508%|11/11目标|45.674%|
|0540真实RGB-D，完整既定采样|51|8.948%|11/11目标|55.277%|

两景共有168条独立训练／开发描述；这不是新增UQIS成绩。固定12帧时真实观测的
自动文本结果都明显改善；视图数量不是这部分差值的唯一解释，因为其数量已经匹配。
同时，相机分布仍不同，不能仅靠这个对照把外观质量与相机覆盖的贡献彻底分开。
增加真实帧明显提高有限候选承载，但0540自动IoU反而9.508→8.948%，所以“更多信息”
和“正确利用信息”仍是两步。不得把55%／62%的标签最优范围当成自动方法成绩。

这是一个已经落实的训练管线修正方向：独立房间也能复用部署的真实RGB-D观测编译器，
原始源文件可访问，并非缺少数据权限而必须使用12张合成图。两景构建完成，
**其余训练房间真实观测重建、在其上重训和新的独立确认尚未完成**。

## 决策与复现

本轮学习头保留为有完整结果的研究候选，不替换v8，不继续扫描attention宽度、学习率、
余弦融合权重或UQIS阈值。后续先把训练观测与实际建图方式对齐，再区分身份与范围质量
监督。完整成员恢复仍需独立解决，不能用约2pp文本增益代替其验收。

声明：`configs/recipes/referent_grounding.json`。结果：
`output/referent_grounding_20261005/comparison.json`；真实观测诊断在该根目录下
`real_observation_control/comparison.json`。没有新增单元测试或测试集参数搜索。

```bash
PY=/root/miniconda3/envs/cybersim_agent/bin/python
PYTHONPATH=. $PY scripts/research/prepare_referent_grounding.py --stage labels
PYTHONPATH=. $PY scripts/research/prepare_referent_grounding.py --stage features
PYTHONPATH=. $PY scripts/research/prepare_referent_grounding.py --stage targets
PYTHONPATH=. $PY scripts/research/prepare_referent_grounding.py --stage language
PYTHONPATH=. $PY scripts/research/fit_referent_grounder.py --condition local
PYTHONPATH=. $PY scripts/research/fit_referent_grounder.py --condition scene
PYTHONPATH=. $PY scripts/research/prepare_referent_grounding.py --stage recipes
PYTHONPATH=. $PY -m radio_gs.pipeline.cohort --recipe output/referent_grounding_20261005/scene_recipe.json --output output/referent_grounding_20261005/scene/system --gpus 0,1,2,4
PYTHONPATH=. $PY -m radio_gs.pipeline.cohort --recipe output/referent_grounding_20261005/local_recipe.json --output output/referent_grounding_20261005/local/system --benchmarks uqis --gpus 5
```

训练诊断读取的是所选checkpoint；原训练空间未改写。地图没有增加查询标签、类别名或
源图查询权限。两组各增加一个2.146MiB共享小头；拟合阶段分别24.17／36.66秒，
不包含特征提取、地图构建和评价，不能当完整训练成本。原始高容量地图与教师成本仍需计入。

真实观测对照复现：

```bash
PYTHONPATH=. $PY scripts/research/prepare_real_grounding_control.py
SAM_PY=/root/miniconda3/envs/sam3/bin/python
PYTHONPATH=. $SAM_PY -m radio_gs.pipeline.registered_observation_map --config output/referent_grounding_20261005/real_observation_control/config.json --output output/referent_grounding_20261005/real_observation_control/maps --scenes scene0271_00 scene0540_00 --stage masks
PYTHONPATH=. $PY -m radio_gs.pipeline.registered_observation_map --config output/referent_grounding_20261005/real_observation_control/config.json --output output/referent_grounding_20261005/real_observation_control/maps --scenes scene0271_00 scene0540_00 --stage compile
PYTHONPATH=. $PY scripts/research/measure_real_grounding_control.py
PYTHONPATH=. $PY scripts/research/summarize_referent_grounding.py
```
