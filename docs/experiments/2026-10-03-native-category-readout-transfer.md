# 完整原生地图与有效类别读出的整合

本轮完成8个匹配ScanNet场景、4条完整方法对照，并从统一公开入口重新计算全部8景。保留组合为**完整原生地图 + 显式源训练校准参数 + 原有源范围求解器**。逐点19/15/10类mIoU为 **55.3029/55.8230/69.1902%**，较旧源辅助方案提升 **1.3653/1.3204/1.5101个百分点**。这是开发集系统整合收益，不是独立测试、公开SOTA或已证实的新机制贡献。

## 核心问题与实际修复

新地图存储原生1536维语义方向，查询输出余弦。旧范围求解器接收的是经查询阈值、尺度、类别竞争校准后的logit。删除旧读出，或直接对余弦使用sigmoid，都不能公平检验新地图的能力。本轮将小型读出参数作为独立地图资产导出，不再要求运行时加载旧紧凑特征场、字典或旧预测。源辅助接口显式要求这个校准头；原始纯余弦地图仍保留自己的map-only配方。

校准头包含查询线性阈值及其上下界、原有有界阈值修正、响应尺度、原生查询坐标中的类别竞争方向。冻结新地图和query encoder，不训练新的匹配空间。最终源范围仍用原来的60步求解，源伪标签、置信度、边界和一致性规则保持不变。

8景使用新地图的全部886张建图源图，类别范围证据固定为此前176张合法源帧的教师缓存。同一份176帧证据用于所有本轮对照；它不是额外的训练图数量上限，也不是1158张源图的全量类别检测已经完成。另两景仍缺类别证据，未与这8景拼成十景源辅助成绩。

## 完整匹配结果

同几何、同类别、同点映射、同场景平均规则。下表为标签网格顶点等权mIoU，非体积加权Gaussian伪标签指标。

| 组合 | 19类 | 15类 | 10类 |
|---|---:|---:|---:|
| 旧原生/紧凑混合地图读取 | 50.1230 | 51.3217 | 64.4320 |
| 旧地图 + 原源辅助 | 53.9376 | 54.5026 | 67.6800 |
| 新完整原生地图，原始余弦 | 46.9743 | 48.2340 | 59.0662 |
| 新地图 + 移植校准，未加源求解 | 45.5634 | 46.8344 | 58.1297 |
| **新地图 + 移植校准 + 原源求解** | **55.3029** | **55.8230** | **69.1902** |
| 新地图 + 源重新拟合校准，未加源求解 | 49.3109 | 50.7184 | 61.8200 |
| 新地图 + 源重新拟合校准 + 原源求解 | 51.4771 | 54.4806 | 67.6567 |
| 新地图 + 移植校准 + 旧范围结构 + 原源求解 | 55.2497 | 55.7629 | 69.1462 |
| 新地图 + 移植校准 + 联合类别正证据求解 | 53.9496 | 54.9311 | 66.9100 |

统一`radio_gs.pipeline`入口重新编码查询、读取新地图和小型校准资产、重新求解源范围，并对全部8景复算。其场景平均指标与保留实验分支差值为0。公开运行不读取本轮实验的预测结果。所有场景使用同一组合，不按逐场景分数选择分支。

| 场景 | 旧源辅助19类 | 新整合19类 |
|---|---:|---:|
| scene0000_00 | 45.5560 | 47.2305 |
| scene0062_00 | 68.9409 | 71.3533 |
| scene0070_00 | 38.8913 | 40.2185 |
| scene0097_00 | 53.1062 | 50.5477 |
| scene0140_00 | 57.8441 | 59.6898 |
| scene0347_00 | 56.6764 | 55.7621 |
| scene0400_00 | 65.7238 | 69.2434 |
| scene0590_00 | 44.7620 | 48.3779 |

6景改善，2景退化。不能称为全场景稳定提升，也未测独立新场景和重跑波动。

## 两条核心改动为何不采用

### 源标签拟合改善不等于最终召回改善

校准重拟合固定200步Adam，学习每类有界阈值偏移和公共尺度；只使用现有源掩码内部及原规则允许的负区，未知元素无损失。正负、各类等权，并向移植参数正则。只针对声明的19类，不称为开放词汇query映射。结果为map-only提高3.7475个百分点，最终source反而下降3.8258个百分点。

这直接说明原生排名、类别校准、源逆渲染的交互不能用源拟合损失或map-only单项分数代替评价。它不证明一切校准学习无效，但该固定候选没有达到最终目标，不纳入配方，也不继续针对评测类别调阈值。

### 独立二值改成联合类别并不足以解决证据错误

联合候选只消费检测掩码腐蚀后的正区，重叠类别形成软分布，范围外未知；在互斥类别单纯形内求解60步可见性逆投影并保留原生先验。它仅适用于明确互斥的ScanNet类别，不适用于整体、部件和任意文本集合。结果53.9496%，不足以替代55.3029%的保留组合。该对照同时改变负证据使用和联合约束，不能把损失独归于其中之一。

迁入旧非负重叠范围组合结构的辅助对照只有55.2497%，也不加入配方。保留源码和完整结果，但不将这些负结果变成默认模块。

## 统一实现与复现

- 参数资产：`radio_gs/querying/native_category_head.py`。
- 共享源证据构造：`radio_gs/querying/scannet_runtime.py::declared_source_views`。旧provider仍调用相同规则。
- 原生类别查询：`scripts/research/run_scannet_query.py`。有显式头的native provider可进行source-assisted；原始无校准余弦不能冒充概率。
- 入口：`radio_gs/pipeline/runtime.py`。
- 场景：`configs/pipeline/scannet_native_calibrated/cohort.json`及8个Scene文件。
- 汇总：`output/native_category_transfer/summary.json`。每场景保存四种候选及公开入口预测和评价，不覆盖旧结果。

```bash
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_native_category_transfer_cohort.py --mode heads
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_native_category_transfer_cohort.py --mode scope
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_native_category_transfer_cohort.py --mode joint
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/export_native_category_transfer.py
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_native_category_transfer_cohort.py --mode public
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/summarize_native_category_transfer.py
```

单场景从地图资产出发：

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m radio_gs.pipeline \
 --scene configs/pipeline/scannet_native_calibrated/scene0062_00.scene.json \
 --queries configs/consolidated/scannet/scene0062_00.queries.json \
 --access source-assisted --output output/native_category_example --gpu 0
```

## 系统结论和边界

原生坐标保护与有效源范围推断可以兼容。此前新图46.97%的结果不能据此认定其全部特征质量不如旧图；接回匹配读出后55.30%超过旧完整组合53.94%。但本轮没有把差值证明为新原生写入机制的独立贡献：新图的建图图像数量、特征内容等也与旧图不同。相对裸图的8.33个百分点更不能全部归为新创新。

这次完善了实际统一入口中的ScanNet新原生地图路径；LERF/NVOS保留此前已验证组合，不声称本轮给它们涨分。仍未完成全部十景源辅助、独立3D集合真值、未调参场景、统一四query标注准确率、紧凑多教师学生和公平公开SOTA比较。UQIS继续按用户要求暂停。

下一步有意义的研究问题是：在保持原生语义能力和有效源读出的前提下，如何减少写入时的整体/部件混合与错误身份证据。继续只拟合有限源伪标签或堆范围平滑没有获得正证据。
