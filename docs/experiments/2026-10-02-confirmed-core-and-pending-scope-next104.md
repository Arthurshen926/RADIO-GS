# 相容核心与待确认范围分开使用：next104

2026-10-02。12次四场景评估完成。保留一个有小幅净收益的源条件研究候选；全局默认与next80稳定参考不变，next98可回退。

## 实现机制

使用 next103 已有源区域判断，不新增VLM调用。对每个已有主假设，A为相容区域、B为明确无关、C为不确定。只有 A 非空且 A 像素数至少为 C 像素数，才允许 A 单独作为跟踪核心；否则保留原源处理路径。这个统一的多数条件是未经独立标定的启发式，用于避免把一个被误判为A的像素当作整件目标。A仍是类别相容判断，不是真实实例认证。

待确认 C 掩码另存，不直接混入新起始正提示。保持原有第二实例与 text-union 处理；不按最大面积只取一个区域。源规则在新评估之前产生4个实际变更：handle、sake cup、dall-e brand、plastic ladle。ketchup两区域都A，掩码未改变；pikachu只有1像素A而22734像素C，不满足核心条件；未被核验的单连通种子不改。

核心掩码与已有轨迹种子逐位相同时复用对应完整源轨迹，因此只新增1条dall-e全源轨迹，其余复用原始、next102或next103中输入一致的轨迹。没有用评估分数选择轨迹。

## 三组实验

1. **core**：13个已有范围查询全部重新拟合；使用核心轨迹，重建源监督与优化条件，未继续施加next98的旧轨迹类别准入。它与next103 reference比较。
2. **pending**：固定core的初始化、相机顺序、置信度、归一化范围代价。把旧A+C轨迹作为另一种可能范围，只撤销其覆盖处与core冲突的负例，不增加正例。它并不是独立跟踪C，而是一个保守的备选范围；旧轨迹仍可能污染。直接移除的负例数量为handle178、cup3163、dall-e/ladle0。
3. **core_update**：实际集成配方。只重拟合源规则确定的4个改变查询，其他63个查询继续采用next98类别筛查流程的输出。更新集合不由GT、IoU或逐类赢家决定。不是全13个查询统一替换后的结果，也不是独立的新模型从零训练结果。

前两组用于解释机制，第三组验证增量接入现有候选。它仍是同一开发集上的顺序研究，不是独立最终测试。

## 结果

所选Gaussian子集2D投影IoU；不是独立3D真值指标。

| 配方 | 四场景均值 |
|---|---:|
| next103 reference | 62.019% |
| core，13类重拟合 | 62.236% |
| pending，13类重拟合 | 62.263% |
| 保留父候选next98 | 62.430% |
| **core_update，源规则决定4类更新** | **62.599%** |

core_update相对父候选提高0.169个百分点，3场景提高、1场景持平，不称为大幅突破。

| 场景 | next98 | core_update |
|---|---:|---:|
| figurines | 67.415 | 67.451 |
| ramen | 56.308 | 56.909 |
| teatime | 63.693 | 63.733 |
| waldo_kitchen | 62.302 | 62.302 |

实际变更查询：

- sake cup：69.92→77.03%，precision80.8→94.2%，recall71.9→80.4%。
- handle：68.92→69.92%。
- dall-e brand：49.96→51.16%。
- plastic ladle：0→0%。

其余63类输出严格保留，所有原生语义分数不变。onion和ladle仍为0；没有解决缺失身份/范围证据的普遍问题。

pending对照相对core只高0.027个百分点，cup77.03→78.48但handle69.92→69.46；没有重复运行方差估计，也没有完整集成评估，因此不把pending并入保留配方。它说明一种可能的范围补全作用，但不足以证明已经建立了可靠的部件对应。

## 方法结论

不确定源区域不必直接当作跟踪正提示。对具有足够相容核心的种子，可以先从核心传播，保留其他查询现有可靠路径，获得局部净收益。不能据此把C判成背景；core本身仍利用新轨迹生成补集负例，待确认范围的排除可靠性仍未完全解决。

主线集成不能顺带撤销未发生证据变化的查询已有类别筛查；全13类冷重拟合低于父候选，增量来源必须明确。当前结果支持这条源条件更新配方，而非任意逐类混合或性能路由。

下一阶段需要独立验证相容核心资格与未确认部件关系，尤其是无足够A核心、同类多实例、细长物体。仍不能把“多数A面积”当作已证明正确的身份标准。当前只覆盖既有13个种子假设，不宣称所有开放词汇查询的普适改进。

没有新的NVOS、ScanNet OVS、UQIS、多query学生或SOTA结果。保留为可回退的LERF研究候选，全局默认不变。

## 复现与产物

- `scripts/research/run_core_tracks_next104.py --scene <scene> --gpu <gpu>`。
- `scripts/research/run_core_scope_next104.py`：core/pending。
- `scripts/research/run_core_update_next104.py`：保留的增量配方，需next98父输出。
- `scripts/research/summarize_core_scope_next104.py`、`measure_core_outputs_next104.py`、`measure_core_errors_next104.py`。
- `output/core_scope_next104/{summary,retained_outputs,error_decomposition}.json`。
- 源决策：`output/core_seeds_next104/<key>/report.json`。
- 配方：`configs/recipes/next104_research.json`。
