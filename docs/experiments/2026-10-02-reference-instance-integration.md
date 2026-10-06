# 参考实例路径接入：保留NVOS精度，区分响应与集合

已将正参考提示路径接入当前统一入口，并完成NVOS八个查询的源证据重放、重新跟踪与重建、集合求解器对照和双读出评价。**新入口平均IoU86.4891%，与原高分结果逐查询一致。** 本阶段是系统整合和输出含义修正，没有新的NVOS精度增益；LERF默认identity66.66%没有改动或重跑。

## 实际完成的流程

显式Scene/Query → 参考掩码标识一个实例 → 仅合法源RGB上SAM3重新跟踪 → 冻结几何上的源证据 → 连续响应重建＋实际集合推断 → 双读出。

Scene文件声明几何、相机、跟踪/拟合帧和合法RGB；Query文件声明参考帧、参考图与参考掩码。评价帧与GT路径放在独立evaluation文件，推理程序不读取它。初次配置导出使用历史地图元数据定位资产，但新运行不读取旧field中的查询值、旧trained.pt或旧目标预测。

本轮复用的是**已冻结的正提示SAM参考掩码**。它是查询输入派生物，不是目标GT；未重新执行原始scribble→SAM步骤，也没有用负提示。因此只能称正参考掩码条件下的重新跟踪/重建，不能称正式正负scribble协议、纯图像查询或任意新点提示已经完整通过。

连续响应调用已有共享数值核心`radio_gs/querying/typed_evidence.py::solve_rendered_evidence`；集合调用文本主线的`decode_expanded_extent_next72.py`显式输入分支。几何与证据相同，两个目标分别适配其输出算子。参考实例已有身份锚点，未套用文本同名候选的GMM身份否决。此路径没有使用原生图像语义先验，不能用其成绩证明统一多教师检索特征已经成功。

## 冻结NVOS结果

|查询|原next74|新入口重新跟踪/重建|
|---|---:|---:|
|fern|82.18|82.18|
|flower|86.59|86.59|
|fortress|97.21|97.21|
|horns_center|95.73|95.73|
|horns_left|92.66|92.66|
|leaves|94.08|94.08|
|orchids|67.96|67.96|
|trex|75.49|75.49|
|宏平均|86.49|86.49|

冻结读取：全场景、观测支持归一化的连续响应；按原图栅格双线性缩放后固定`>=0.5`。无目标RGB、无负提示。AP约98.0139%；重放和重跑AP有约1e-6量级差异，不能声称所有数值逐位一致。IoU八查询一致。

主要产物：`output/reference_integration/replay`、`fresh`、`fresh_tracks`、`typed`。每组八份评价。`run_current_research.py --benchmark nvos --scene fern`也另行执行了完整入口，包含重新跟踪与双读出生成，验证顶层入口不是只打印计划。配置已冻结引用已有地图，未重建几何。

## 方法级发现：集合参数不能冒充连续响应

同一批新跟踪证据交给LERF实际集合优化后：

- 将其`selection_scores`错放进NVOS连续响应算子：平均IoU **6.8182%**。
- 真正取出`selection_scores>0.5`的元素，渲染其自身轮廓：诊断IoU **85.0794%**。
- 保留独立响应重建并按原NVOS算子评价：**86.4891%**。

集合解码可以用少量合适Gaussian解释对象轮廓。将这些成员值放回完整场景、按所有可见Gaussian归一化时，响应会被其他Gaussian稀释；小的成员连续值也不等同于语义概率。因此错用响应会崩溃，并不表示对应的元素集合完全错误。

85.08%使用子集alpha经缩放后`>10.5/255`的诊断，不是冻结NVOS指标，更不是独立三维GT IoU。不能通过切换到这个阈值把6.82%“修成官方成绩”。两种输出在方法层明确分开：

- `scores`、`semantic_geometry`、`semantic_observed`：连续响应。
- `selection_scores`、`selection_geometry`、`selection_observed`：实际集合。

`compose_reference_readout.py`按预先规定的含义组装，不读取评价分数，不按query或GT选择胜出分支。八个typed输出重新评分仍为86.49%。采用两种拟合目标增加了计算成本；这不是声称一个标量或一个目标已经完成两件事。

## 使用与复算

```bash
# 完整参考实例运行，默认重新跟踪，生成两类读出并按NVOS响应协议评价：
python scripts/research/run_current_research.py --benchmark nvos --scene all --gpu 0 --evaluate --output output/current_nvos_reference

# LERF文本默认不变：
python scripts/research/run_current_research.py --scene all --evaluate --output output/current_lerf_identity

# 原源证据重放、重新跟踪、错误读出对照依次运行：
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_reference_integration.py --mode replay --output output/reference_integration_replay
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_reference_integration.py --mode fresh --output output/reference_integration_replay
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_reference_integration.py --mode subset --output output/reference_integration_replay
```

任意显式Scene/Query参考输入由`run_reference_query.py`执行；可用`--tracks`声明证据重放，否则重新调用跟踪器。原始参考SAM掩码始终作为声明输入，不能把它藏进“完全不读查询缓存”的说法。

结果汇总：`output/reference_integration/summary.json`。配方：[reference_integration.json](../../configs/recipes/reference_integration.json)。共同入口配置：[current_research.json](../../configs/recipes/current_research.json)。本轮32份完整队列评价登记为重放、fresh、错误算子对照、typed四组，不能把重复复算当作32个独立查询或场景。

## 系统状态与下一步

当前已有一个统一入口，覆盖LERF文本和NVOS正参考实例；共享冻结几何、源证据表达、响应/集合分离和集合求解器。不同benchmark仍使用各自地图，不是四种query在同一场景地图上的验证。horns_center/left复用同一几何支持不同参考实例，也仍只是同一类提示能力。

尚未完成：纯图像query、3D点query、从原始正负提示重新生成查询、同地图四query、ScanNet共同入口、新场景与公开协议验证、紧凑多教师学生；UQIS继续暂停。没有新的SOTA或CVPR就绪结论。

下一步优先在一张已有地图上明确“参考图身份”和“提示给定范围”的输入区别，接入真实图像编码与3D锚点，验证它们是否召回同一目标，并保持双读出含义不变。不能把参考掩码经过视频传播的成功，直接作为图像编码器/多模态特征场成功的证据。
