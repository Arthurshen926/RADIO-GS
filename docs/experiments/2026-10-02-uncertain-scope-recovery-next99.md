# 不确定范围恢复：next99

日期：2026-10-02。状态：两组、四场景共 8 次评估完成，均不替换 next98。next80 稳定参考和主线默认配置不变。

## 问题与方法

next98 的类别核验将源侧连通区域分为 A（接受）、B（冲突）、C（不确定）。查看 handle 的 12 个源图样本，C 中既有碎片，也有清楚可见的把手。因此需要检验能否恢复其中有用的正证据，而不是把 C 当作错误标签。

固定几何、原生语义、源图全集、教师缓存和评估，运行两个统一规则：

- all：接纳 A 和全部 C；B 不成为正例。
- geometry：保留 A，仅恢复有至少两个其他 A 源视角支持的 C。先用 A 的腐蚀正核心计算各 Gaussian 的可见投影纯度，纯度大于 0.5 算一个视角见证；排除被判断区域所在视角。将拥有至少两个其他见证的 Gaussian 投影回 C，区域内平均支持响应大于 0.5 才恢复。恢复区域不能递归为其他区域作证。

未接纳区域保持正监督未知，不因 B/C 标签新增负例。原有负监督规则保持不变。恢复使用 192×256 全源图投影，拟合仍沿用高分辨率交错采样、balanced BCE、0.1 footprint prior、corroborated negatives 和 source initialization。未调用新教师，增加的是投影计算。只重拟合既有 13 类范围，其他 54 类沿用父输出；无逐类结果拼接。

## 完整结果

指标为所选 Gaussian 子集的 **2D 投影 IoU**，不是独立 3D 真值指标。宏平均沿用冻结评估方式。

| 场景 | next98 | 全部恢复 C | 几何筛选恢复 C |
|---|---:|---:|---:|
| figurines | 67.415 | 67.149 | 67.257 |
| ramen | 56.308 | 55.912 | 55.930 |
| teatime | 63.693 | 63.851 | 63.851 |
| waldo_kitchen | 62.302 | 61.632 | 61.952 |
| 四场景均值 | **62.430** | 62.136 | 62.248 |

几何规则恢复 655 个 C 中的 583 个，加原有 689 个 A 共接纳 1272 个区域；133 个 B 保持排除。恢复数量不代表正确率。

| 查询 | next98 IoU | 几何恢复 IoU |
|---|---:|---:|
| handle | 68.92 | 70.77 |
| waldo | 72.30 | 69.64 |
| kamaboko | 86.10 | 82.62 |
| sake cup | 69.92 | 69.51 |
| paper napkin | 79.95 | 81.63 |
| ketchup | 78.60 | 73.18 |
| spatula | 49.36 | 48.46 |

onion segments 和 plastic ladle 仍为 0。所有 8 个输出的原生语义分数、其他 54 类选择均与 next98 完全一致。原生语义 IoU/AP 仍为 66.930/84.146，没有新的语义空间提升。

## 机制解释

不能将本轮概括成“恢复正例只会产生更多误检”。后验像素分解显示：

- Waldo：precision 97.2→97.5%，recall 73.9→70.9%。选中元素从 202 减少到 127，移除 142、增加 67，主要是完整性回退。
- kamaboko：precision 96.4→98.9%，recall 89.0→83.5%，同样损失完整性。
- handle：precision 76.6→79.2%，recall 87.5→86.1%，收益主要来自误检减少，不能解释为恢复出了更多目标。
- spatula：全部恢复的 precision/recall 为 45.3/85.7%；几何恢复为 55.1/80.1%；父输出为 53.8/85.6%。几何筛选减少错误范围，但付出漏检代价。

接纳更多源区域会同时改变初始化、参与训练的视角、视角采样分布和 footprint 代价的相机权重，所以不等价于在同一个目标函数中只增加几项正标签。不能把全部差异归因于几何对应准确率，也不能从本轮认定几何表示存在不可突破的上限。

本轮拒绝的是这个具体恢复规则。A 本身只是宽泛类别相容，不是物理实例身份确认；来自多个视角的支持可能相关。几何一致性只能提供空间对应证据，无法单独证明同一实例或整体/部件范围一致。恢复 C 数量很大但均值下降，说明继续调见证数或扩大接纳量没有充分依据。

## 后续优先级

保留 next98，不按查询挑选 next99 收益。下一步应把“是否同一实例”“当前仅看到哪个部件”“未覆盖部分是否未知”作为不同的源证据关系。对关系建模的对照应固定相机集合、采样权重与几何代价归一化，避免证据接纳同时改变优化问题。先用源视角的独立对应和可见范围验证这些关系，再检查完整冻结评估；不使用目标 GT 生成关系或挑参数。

本轮没有新的 NVOS、ScanNet OVS、UQIS 或四类 query 统一结果，也没有得到可发布的 SOTA 结论。

## 复现与产物

- 构造：`scripts/research/recover_uncertain_scope_next99.py --scene <scene>`。
- 拟合及评估：`scripts/research/run_uncertain_scope_next99.py --scene <scene> --gpu <gpu> --arms all geometry`。
- 汇总：`scripts/research/summarize_uncertain_next99.py`。
- 完整结果：`output/next99/summary.json`。
- 源区域恢复：`output/next99/recovery_counts.json`。
- 后验误差分解：`output/next99/error_decomposition.json`。
- 元素变化：`output/next99/selected_sets.json`。
- 源图样本：`output/next99/handle_unknown.jpg`。
- 规则与轨迹：`output/uncertain_tracks_next99/{all,geometry}/<key>/`。
- 最终输出：`output/uncertain_scope_next99/{all,geometry}/<key>/`。
