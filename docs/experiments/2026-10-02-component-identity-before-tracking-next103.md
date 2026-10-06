# 跟踪前的连通区域身份判断：next103

2026-10-02。两组四场景共8次评估完成；有限收益，不替换 next98。

## 改动

针对 next102 发现的起始掩码混入无关纸片，将身份判断前移至多连通区域种子。逐区域先做不含查询的图像描述，再做文本关系判断；A为相容对象/部件，B为明确无关类别，C为不确定。只移除 B，保留 A/C，避免按面积删除真实的不连通部件或同类实例。单连通种子不改，原有已接纳第二实例保持不变。若整张种子都被拒绝，规则回退原种子；实际两组均未触发。

两条配方均作用于全部13个主假设：

- query：新描述提示强调不清楚时弃权，再与原始查询比较，允许兼容部件、同类实例。
- reference：复用 next98 的盲描述提示，与此前缓存的种子参考描述比较宽类别，允许部件和材料猜测差异。区域描述和判断都重新生成，不沿用旧连通编号。

两组共82次区域描述和82次分类，使用本地 Qwen2.5-VL-3B。query 改变3个种子，reference改变2个；只对这5个改变的种子重跑全源图跟踪，未改变的种子复用原完整轨迹。使用新轨迹重建范围监督、初始化和代价，不使用 next98 的旧轨迹准入权限。因此匹配基准是未作轨迹类别筛查的 next97，不是已保留的 next98。

这不是物理实例身份认证；参考描述本身可能包含污染，类别相容也不能证明整体/部件范围完全一致。两组同时改变描述提示和文本比较对象，不能将差异完全归因于其中某一个因素。

## 源侧发现

| 配方 | 判断数 | A | B | C | 改变种子数 |
|---|---:|---:|---:|---:|---:|
| query | 41 | 4 | 3 | 34 | 3 |
| reference | 41 | 8 | 12 | 21 | 2 |

query 对杯子主区域给出 unclear，纸片多被保留为 C，说明新提示并未可靠完成清理。reference 保留杯子主区域13036像素，移除1440像素的 B，但另有1295像素的 C 仍进入跟踪提示。

reference 将 ketchup 两个区域均判 A 并保留，不再像最大连通区域规则那样删掉第二区域；pikachu、handle 等种子也保持原状。仍有错误或含糊描述，例如小区域被描述成图标、汽车等，不应把分类字母解释为校准后的真值概率。

## 完整结果

所选Gaussian子集2D投影 IoU，单位%。不是独立3D真值指标。

| 场景 | 匹配 next97 | query | reference |
|---|---:|---:|---:|
| figurines | 67.159 | 67.103 | 67.159 |
| ramen | 54.740 | 54.757 | 55.536 |
| teatime | 63.809 | 63.809 | 63.809 |
| waldo_kitchen | 61.572 | 61.572 | 61.572 |
| 四场景均值 | 61.820 | 61.810 | 62.019 |

reference 的主要有效变化是 sake cup **56.94→66.36%**：precision67.1→78.2%，recall69.4→70.3%。其他查询分数保持匹配基准水平，onion与ladle仍为0。它保住了 next102 面积规则回退的 ketchup、handle 和 pikachu，但杯子的收益弱于纯主连通区域对照77.03%。不能拼接这两组的逐类赢家。

总体仍低于 next98 62.430%，所以不替换保留方案。query 没有净收益。所有8输出原生语义分数及未重拟合54类选择完全保持父输出一致；没有新的2D语义能力提升。

## 方法结论与后续

将明确无关区域在跟踪之前排除有局部收益，也避免了最大面积筛选的一些破坏。但“保留不确定区域”目前仍意味着把它作为目标一起初始化跟踪：不确定性没有被表达在后续传播中。这个设计仍可能把残留纸片扩散为可靠正监督，尚不能用本轮结果证明每个残留区域就是误差来源。

后续应将已确认目标核心与待确认区域分开表示：核心用于身份锚定，不确定区域保留为独立候选，不能直接混成同一个目标提示；再以新的跨视角证据判断其是否属于目标部件或同类实例。不能用统一面积阈值或“全部弃权即负例”代替，也不应继续只调描述提示。

本轮按源图问题顺序开展两组实验，不是预注册；没有目标 GT 参与种子选择。没有新 NVOS、ScanNet OVS、UQIS、多query学生或SOTA结果。

## 复现

- `scripts/research/run_component_tracks_next103.py --scene <scene> --gpu <gpu>` 与 `run_component_scope_next103.py`：query。
- `run_reference_component_tracks_next103.py` 与 `run_reference_component_scope_next103.py`：reference。
- `scripts/research/summarize_components_next103.py`、`measure_components_next103.py`、`measure_component_errors_next103.py`。
- `output/next103/{summary,source_counts,retained_outputs,error_decomposition}.json`。
- 区域描述及判断：`output/{component_seeds_next103,reference_component_seeds_next103}/<key>/`。
- 配方：`configs/recipes/next103_research.json`。
