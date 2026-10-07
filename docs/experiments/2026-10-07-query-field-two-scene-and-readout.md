# 新 Gaussian 场：双场景验证与身份到成员的损失定位

本轮完成三个新训练、四组真实 LERF 预测与评分、身份/尺度假设诊断、原生坐标交换，
以及两个场景的读出对照。代码继续位于 `codex/query-preserving-gaussian-field`，
没有引入 v10 的 source-assisted 推理、MaskAdapter 或额外 3D 标签成员树。

**主要结论：高容量场仍有可用的原生选集能力，但当前“估计单锚点→成员扩展→候选并集”
会丢失大量能力。扩充源视图和挑选尺度都不足以解决这个问题。**
当前共享解码器没有通过质量要求，不能进入以其为成功教师的压缩阶段，也不能替换 v10。

承接 [首次实现记录](2026-10-07-query-preserving-gaussian-field-bootstrap.md)。
完整数值、配置和内容散列见 [本轮证据索引](2026-10-07-query-field-two-scene-evidence.json)。
附件是研究设计参考；本轮执行依据用户后续“开始全面推进”“继续推进”的授权。

## 固定条件与执行范围

保留同一共享模型：32D group、固定原生 SigLIP/DINO、尺度 gate、2,000 次更新、
相同随机种子和学习率、query-selected soft-alpha 训练目标、成员阈值 0.5。
所有源掩码来自 query-free SAM。没有在建图或训练中使用 benchmark 文本或答案。

| 条件 | Gaussian 总数 | 有特征观测 | 可训练区域 | 2,000 步实际访问的区域 |
|---|---:|---:|---:|---:|
| figurines，6 视图，沿用上轮 | 554,467 | 242,262 | 489 | 485 |
| figurines，18 视图，新训练 | 554,467 | 380,715 | 1,455 | 1,093 |
| ramen，6 视图，新训练 | 592,709 | 319,623 | 405 | 400 |
| ramen，18 视图，新训练 | 592,709 | 452,765 | 1,320 | 1,032 |

视图清单在新结果产生前固定，采用源帧清单上的均匀位置，figurines 保留上轮训练帧；
ramen 的 6 帧属于 18 帧子集。具体索引在 `configs/query_field/*views.json`。
开发视图固定为 figurines 69/232、ramen 34/99，全部区域分别为 169/149。
所有训练特征帧都避开真实 LERF 目标视图。

这是**同训练步数下扩大输入预算**的实验，同时改变原生融合、观测覆盖、训练区域数量
和训练区域中位物理尺度；不是纯算法收益，也不是每区域优化次数匹配的对照。
几何仍来自较大的已有映射集合，不能把特征帧隔离描述成几何隔离或独立最终测试。

三个新训练分别约 79.63/86.35/99.09 秒，包含训练函数内部诊断和导出，不含编译教师缓存。
模型文件 figurines 为 2,657,563,621 bytes，ramen 为 2,840,857,509 bytes；
语义增量张量分别为 2,626,510,315 和 2,807,662,669 bytes。仍是高容量对照。

## 教师区域诊断：点提示改善不代表真实任务改善

全部开发 SAM 区域纳入，失败保留为零；使用默认尺度、成员阈值 0.5、重渲染 alpha > 0.5。

| 条件 | 2D 点实际子集 IoU | 注册 3D 点实际子集 IoU | 区域图像实际子集 IoU |
|---|---:|---:|---:|
| figurines 6 | 26.46% | 25.24% | 19.56% |
| figurines 18 | 31.63% | 27.46% | 17.05% |
| ramen 6 | 30.31% | 27.15% | 22.64% |
| ramen 18 | 31.72% | 29.57% | 24.14% |

未支持的 3D 点数量依次为 14/1/8/0，没有从分母移除。
图像输入仍是教师区域的 masked-crop SigLIP 加最多 16 个区域 DINO 像素；
这是受控诊断，**不是外部参考图像 benchmark**。注册点的身份来自主贡献 Gaussian，
也不是人工完整 3D 实例真值。归一化成员图另存，不能用其改善替代实际子集改善。

figurines 的图像退化随后进行了配对检查：固定 18 视图 group、尺度和两者共同的
242,262 个观测 Gaussian，只交换 6/18 视图原生语义与局部坐标。平均实际 IoU 为
17.62%/17.55%，差 -0.07pp；单例绝对差均值却为 12.76pp，69 个改善、57 个下降、43 个相同。
所以“均值几乎一样”不代表身份决策没变，也不能证明融合无损；只是不支持把原始总退化
全部归因于这次原生坐标替换。共同支持限制和 group 改动仍需区分。

## 身份与尺度假设：有限候选仍有缺口

在原 figurines 6 视图模型上，固定八个空间分离身份候选和三个尺度因子 1/0.5/2。
`hypotheses.py` 接受导出场与当前 QueryPacket，不接收源观测或目标掩码。
显式输入尺度不被覆盖；实际点支持沿用原路径；推断身份不会被硬钳为正样本。

无拟合的原生重排分数为：所选集合的语义均值占 1/2，双向局部 patch 匹配各占 1/4。
反向匹配要求每个 query patch 在选集中找到对应，分块执行并遵守 affine decoder。
它是一个未校准的固定研究对照，没有升级为默认方法。

| 169 个区域的自动或特权条件 | 实际子集 IoU |
|---|---:|
| 原自动最高分身份、默认尺度 | 19.56% |
| 固定原生双向重排，自动 | 21.60% |
| 特权选择八候选中的最佳身份，默认尺度 | 27.33% |
| 特权选择原身份的最佳尺度 | 20.60% |
| 特权选择 24 个身份/尺度组合 | 28.74% |
| 注册点、默认尺度 | 25.24% |
| 注册点、特权选择三个尺度 | 26.52% |

先保存所有候选预测，再计算目标 IoU；“最佳”列只存在于评价脚本。
这些数字是有限候选集合的诊断，不是通用理论上界。结果不支持继续堆尺度或扩大
候选数量来掩盖成员本身的问题。自动重排的 +2.04pp 只有一个场景的教师区域证据。

## 真实 LERF 接口：预测与评分分离

新增 `public_lerf.py`。预测只读取导出场、当前文本和共享 SigLIP 查询编码器；
不读取评测 scene、相机、源 RGB 或源 SAM。四组预测全部完成并保存散列后才启动评分。
评分逐元素核验几何，直接调用未修改的 `evaluation/lerf_readouts.py`。

两个语义读出均保留：原生响应使用固定四个通用负词 object/things/stuff/texture，
`sigmoid(10 * (query cosine - max negative cosine))`；成员响应使用共享成员函数。
两者配对**同一个成员选集**。没有继承旧系统按源 query 学习的 response calibration，
没有把两个读出的最好列拼成新的模型成绩。

| 场景/源视图 | 原生语义 IoU | 原生 AP | 成员语义 IoU | 共享成员实际子集 IoU |
|---|---:|---:|---:|---:|
| figurines 6 | 42.53% | 69.57% | 28.06% | 6.67% |
| figurines 18 | 47.29% | 78.85% | 30.31% | 11.60% |
| ramen 6 | 25.63% | 55.83% | 25.62% | 10.52% |
| ramen 18 | 35.14% | 68.72% | 18.64% | 6.74% |

figurines 共 21 个文本、56 个 query-view，ramen 共 15 个文本、71 个 query-view。
这里“实际子集 IoU”使用原历史协议的 **8-bit alpha > 10**，不是上述 SAM 诊断的 alpha > 0.5。
本地所谓 LERF3D 仍是选中 Gaussian 子集的 **2D 投影评测**，不等于独立 3D 真值。

v10 保存的同场景语义/子集指标为 figurines 76.73%/71.11%、ramen 59.41%/58.31%。
这两行没有重跑；它们是有源支持及不同预算的能力参照，不是同资源消融。
当前无源查询候选距其很远，不能宣称“已接近本地实现”。

## 读出干预：合并与成员扩展都损失能力

看到第一轮真实 LERF 的范围外溢后，固定了一个开发诊断，不包装成独立测试。
对两个 18 视图场复用完全相同的已保存 query 向量，所有新预测再次先于本轮评分保存：

1. 原共享成员并集：逐位重现上一轮的成员分数和选集。
2. 单最高分锚点：保留同一尺度和成员模型，仅取消多个假设的并集。
3. 原生直接选集：同一原生响应 > 0.5；绕过成员核心，仅用于定位能力损失。

| 固定导出场上的选集读出 | figurines 实际 IoU | ramen 实际 IoU |
|---|---:|---:|
| 共享成员，多候选并集 | 11.60% | 6.74% |
| 共享成员，单最高分锚点 | 20.39% | 9.81% |
| 原生语义直接选集，对照 | 46.45% | 25.42% |

并集的像素精度/召回分别为 figurines 11.76%/90.61%、ramen 6.86%/85.96%；
单锚点为 25.28%/57.14%、12.43%/54.93%；原生选集为 52.43%/86.82%、30.02%/63.59%。
都是 query-view 算术均值，不由平均精度/召回反算 IoU。

并集会放大外溢，但只取消并集仍远低于原生选集。单锚点也可能遗漏合法的多实例 union，
因此它不是语义等价的替代协议。原生直接选集展示了当前坐标中仍可读取的能力，
不构成统一成员架构的成功。该对照没有验证原生场已无融合损失，更没有排除成员监督、
估计锚点、尺度和软/硬选择差异的共同影响。

补充误差脚本最初把 GT 阈值写成 128；正式 benchmark 评分始终使用正确阈值 10。
已保留旧文件但不纳入证据主表，新增 `subset_errors.py` 直接使用历史 mask loader，
逐条断言与原评分 IoU 一致。当前数据上两者数值恰好相同；仍以修正后的 v2 为准。

## 工程验证与复现

19 项定向测试通过，包含真实 gsplat CUDA、未知/嵌套区域、原生写保护、
仿射双向 patch 匹配、候选重放、评分前完整预测门禁、不同控制选集绑定和 GT 阈值回归。
旧 `native_decoder_readout` 未修改，其另外五项测试沿用上轮记录，没有重复报成本轮执行。

ramen 18 视图新增真实文本/图像编码与两种点输入的独立进程重放：复制部署文件、
禁止打开源数据/原 output，四类分数和选集摘要逐位一致，文件打开违规为零。
功能 fixture 的固定文本为 “a figurine”，图像为当前完整查询图，不用于任何准确率主张。
这仍是 Python 文件打开审计，不是 OS 沙箱。

从 `/root/RADIO-GS-query-field` 使用现有 cybersim_agent Python，`PYTHONPATH=.`：

```sh
python -m radio_gs.query_field compile --config configs/query_field/ramen_18views.json --output NEW_RUN
python -m radio_gs.query_field fit --config configs/query_field/ramen_18views.json --output NEW_RUN
python -m radio_gs.query_field evaluate --config configs/query_field/ramen_18views.json --output NEW_RUN
python -m radio_gs.query_field smoke --config configs/query_field/ramen_18views.json --output NEW_RUN
python -m radio_gs.query_field.hypothesis_study --config configs/query_field/figurines_hypotheses.json --output NEW_HYPOTHESIS_RUN
python -m radio_gs.query_field.native_transfer --config configs/query_field/figurines_native_transfer.json --output NEW_TRANSFER_RUN
python -m radio_gs.query_field.public_lerf predict --config configs/query_field/lerf_two_scene_probe.json --job figurines_18 --output NEW_PROBE
```

真实 probe 必须先对配置内全部四个 job 执行 predict，随后逐个执行 score。
`readout_study` 同样先完成两个 job 的 predict 再 score。
示例配置指向已完成的只读导出；重新训练需另存配置并更新模型引用。
所有执行输出目录拒绝覆盖；本轮原始数据在 `output/query_field_20261007/`。
`scripts/research/summarize_query_field_two_scene.py` 在 CPU 上复核预测散列、完整 job 清单、
训练/目标帧关系和并集评分重放，生成本轮证据索引。

## 下一步架构约束

下一次主要改动必须面向**身份条件下的成员学习与读出**，保持单一 Gaussian 场和共享核心。
目前训练只见可靠的区域内部 Gaussian anchor，推理却把原生身份压缩成少量估计锚点；
上述干预表明这种转换未能保留现有响应。训练/推理支持分布失配是待验证假设，尚非唯一病因。

优先构造仅来自合法训练视图的 query-free SAM 区域伪查询，显式保留 native 语义与局部
证据、估计支持的不确定性和尺度；用独立开发视图同时检查身份和真实集合输出。
以同一源预算比较可靠注册支持与原生推断支持，逐项检查跨视图同组/可靠排斥/unknown，
再决定是否改变成员参数化。不能直接把 benchmark 文本或 GT 加入场景拟合。

进入压缩前至少要建立：共享成员读出对同一场上原生能力的稳定保留；注册点能力不被
身份改进破坏；真实参考图像局部对应可用；源数据禁读下完整包能重放。
这是一条开发方向和后续验证要求，不是本轮已证明的效果或已获支持的新颖性主张。

ScanNet OVS、NVOS、UQIS 的新候选完整评测、关系语言、类别 affinity、真正的查询保持
蒸馏及压缩仍未完成。本轮没有把五任务名称当作已统一成功的证据，也没有新建隐藏回退。
所有本轮作业已完成。GitHub Issues 尚未同步，因为环境没有 `gh`；未创建远程 issue 或 PR。
