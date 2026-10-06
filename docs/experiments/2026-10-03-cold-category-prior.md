# 不继承旧场景参数的类别读出

本轮完成三条候选在ScanNet匹配8景的完整评价，并从统一入口重新编码查询、复算冷启动路径的全部8景。无需旧校准参数的原生类别先验达到19/15/10类逐点mIoU **55.7051/56.5188/68.4376%**。保留配方为55.3029/55.8230/69.1902%，因此19/15类略升，10类退步。**冷启动实现可用，但不全局替换system_v1，也不是大幅性能或SOTA突破。**

## 消除的实际依赖

上一阶段已经不需要在运行时加载旧语义场，但仍需要从旧场景训练结果导出查询阈值、尺度和类别竞争参数。这些参数可以作为部署资产，却没有解决一个新场景从原生地图独立建立读出的问题。

新路径仅使用当前原生场、原生query encoder、合法源证据及几何贡献。其类别先验为：

`z_c = 100 * cosine(feature, query_c)`

`prior_log_odds_c = z_c - logsumexp(z_j for j != c)`

温度100为本轮预先固定的常数，没有用评价标签调节。然后接原有60步源范围求解器。没有新VFM、没有学习查询空间、没有读取旧校准头或类别竞争向量。源图教师置信度仍按原共享规则由当前证据构造。

这是**明确互斥的完整类别菜单**下的先验，不是经过实测校准的正确率。它保持同一个元素内不同类别的原生排序，不保证某一query在不同元素之间的排序不变，也不适用于随意重叠的整体/部件文本集合。LERF text-union和参考实例路径没有套用这一转换。

实现直接使用前后缀logsumexp计算one-vs-rest log-odds。先裁剪softmax概率再取logit会把低分候选压成相同值，尤其在19类结果转到15/10类子集时制造排序错误，因此该近似没有进入最终结果。纯地图19/15/10类结果与原始余弦原生地图完全一致，符合本转换的类别排序性质。

## 完整同条件结果

所有候选使用同8景新原生地图、相同几何、同176帧源教师缓存及冻结的逐点评价。GT只用于最后评价；源教师mask不称为真实标签。所有场景运行同一配方，未逐景选赢家。

| 路径 | 地图19类 | 源辅助19类 | 15类 | 10类 |
|---|---:|---:|---:|---:|
| 保留旧参数头 | 45.5634 | 55.3029 | 55.8230 | **69.1902** |
| **冷启动互斥类别先验** | 46.9743 | **55.7051** | **56.5188** | 68.4376 |
| 从无冲突源类别核心学习校准 | **48.8559** | 51.0940 | 53.6811 | 65.2598 |
| 冷启动先验 + 排他类别核心范围证据 | 46.9743 | 53.9071 | 54.3013 | 65.2232 |

冷启动相对保留源辅助的19/15/10差值为 **+0.4023/+0.6958/−0.7526个百分点**。没有额外重复试验来建立显著性区间，不将这些小幅正差值包装成稳定泛化优势。

统一`radio_gs.pipeline`入口对冷启动路径重新编码、读取新场、构造源证据并求解全部8景；场景平均指标与研究路径最大差为0。公开路径没有复用本轮保存的最终预测。来源参数头训练路径和范围负证据路径均不进入默认系统。

## 两条源监督候选与负结果

源学习校准只对当前源图中唯一被腐蚀正区支持的类别建立正证据，用其他类别核心提供排除证据，不以全部mask外区域作为训练负例。只学习当前声明类别的阈值和公共尺度，共200步；完全无旧参数初始化。有正负证据的类别参与拟合，没有正证据的类别保留原生类别先验。它不是开放词汇query适配器。地图评价提高，但最后源辅助下降，说明这一拟合目标仍不足以保证最终使用效果。

范围候选直接在原求解器中使用唯一类别正核心；原始mask互相重叠的像素及无类别核心覆盖的像素保持未知，以其他类别核心排除当前类别。它同时改变了证据覆盖、目标与置信度分配，因而不能仅将退步归因于某一个变化，也不能反向断言原mask外所有负例都是正确的。

这两条负结果说明：保守标签构造本身没有形成有效的完整对象监督。后续需要额外可靠证据来兼顾身份、完整范围与排除能力，而非只靠削减负例或改拟合损失。当前有限证据还不能把责任精确分配给教师、几何对应和求解器。

## 实现和运行

核心：`radio_gs/querying/cold_category.py`。统一类别入口读取Scene显式的`native_category_prior`，不再要求每个场景都有导出的`native_category_head`。

- 冷启动场景：`configs/pipeline/scannet_cold_category/cohort.json`。
- 可执行替代系统：`configs/pipeline/system_cold_category.json`。
- 默认仍为：`configs/pipeline/system_v1.json`。
- 完整逐景汇总：`output/cold_category/summary.json`。

```bash
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_cold_category_cohort.py
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_exclusive_scope_cohort.py
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/export_cold_category_recipe.py
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_cold_category_public.py
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/summarize_cold_category.py
```

单场景独立运行：

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m radio_gs.pipeline \
 --scene configs/pipeline/scannet_cold_category/scene0062_00.scene.json \
 --queries configs/consolidated/scannet/scene0062_00.queries.json \
 --access source-assisted --output output/cold_category_example --gpu 0
```

## 研究边界与下一步

本轮消除了历史**校准参数**依赖，没有声称取消预训练教师、几何、合法源证据缓存或全部场景资产。验证仍是原8个开发场景，不能称为未调参新场景实验。没有改动LERF/NVOS，也没有宣称它们涨分。

两个尚缺类别源证据的ScanNet场景仍未完成源辅助评价；新路径让它们不再必须先具备旧校准模型，但源证据本身仍须生成并验证。176帧在本轮是同条件对照预算，不是新的永久源图上限。

统一紧凑多教师训练、完整多模态标注评价、独立3D集合真值和公平公开SOTA比较仍未完成。UQIS按用户要求保持暂停。该简化提供了可冷启动基线，不构成已确立的CVPR新颖性或原始核心claim证明。
