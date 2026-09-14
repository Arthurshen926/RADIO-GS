# v4：重叠约束修复与候选融合反证（2026-09-14）

## 结论

确认并修复了同视角“disjoint”判断错误，但四场景复评没有带来整体精度提升。
另一个保留重叠候选的 max-product 方案四场景全部下降，不进入主线默认路径。
不能把单元测试通过或约束正确性修复，写成 benchmark 精度改善。

## 参考的历史实现

- `docs/experiments/2026-08-17-lerf-identity-extent-dual-posterior-closure.md`：身份/范围分开承担职责；历史开发集结果约 39%，不是当前 v4 结果。
- `docs/method/universal-field-typed-readout-v1.md`：same/different/unknown 区分；候选竞争发生在查询之后，包含 null。
- `radio_gs/querying/latent_proposal_posterior.py`：检查了关系约束与查询条件下的候选选择逻辑；没有将历史代码导入 v4。

## 已修复的确定错误

旧 `disjoint_same_view` 与 `exact_group_disjoint_assignment` 的 assignment 分支使用
`geometry < 0.5`，其中 geometry 是 `(IoU + containment)/2`。
支持集合 `{a,b}` 与 `{b,c}` 的 geometry 为 5/12；明明相交，却被错误地禁止合并。
旧实现还把空支持当成负证据。

现在使用共享 `_disjoint_same_view_cannot_link`：两侧支持非空、同视角、非自身，且 geometry 等于零。
训练 partition 验证与部署构建都调用这个函数；审计加入 `nonempty_zero_overlap_v2` 标识。
未修改训练权重、合并阈值、文本编码或像素阈值。旧模型附带的历史 partition 指标不能冒充新规则的验证结果。

| 场景 | 旧禁止关系数 | 修复后 | 移除的错误关系 |
|---|---:|---:|---:|
| figurines | 5215 | 4626 | 589 |
| ramen | 3242 | 2683 | 559 |
| teatime | 6614 | 5554 | 1060 |
| waldo_kitchen | 2311 | 1873 | 438 |

总计移除 2646 对错误关系。这里的“不相交”仍然只是**观测支持约束**，不是不同物理对象的证明；同一物体的两个部件仍可能不相交。这项方法假设尚未解决。

## 同协议四场景结果

使用修复后的原生分辨率 carrier、已重新验证的规范化 SigLIP2 文本缓存、相同 frozen full-v2 affinity checkpoint。
所有分支先生成共享三维 element field，再渲染原生分辨率 PNG；22 帧、208 个 object-frame 观测。
像素阈值保持 0.2，未做 GT 阈值搜索。每场景 32 个源视角投影完全一致。

| 场景 | 已修复输入的基线 mIoU | 仅修 cannot-link | 仅换 max-product |
|---|---:|---:|---:|
| figurines | 6.183% | 6.066% | 4.612% |
| ramen | 3.996% | 4.074% | 3.275% |
| teatime | 7.495% | 7.495% | 5.171% |
| waldo_kitchen | 12.617% | 12.104% | 11.371% |
| 四场景等权平均 | 7.573% | 7.435% | 6.107% |

这是 OpenGaussian 发布脚本的 mask 指标口径复现；`upstream_protocol_validated=false`。
不是已认证的完整 LERF2D/3D 双 benchmark 成绩，也不是与历史/SOTA严格同设置的比较。

候选替代方案证明了一个局部不变性：增加与查询无关的候选，不应压低原目标响应。
但最大值融合只是有界证据，不是校准概率；同一 0.2 阈值的直接替换在真实任务失败。
它仅保留为显式诊断选项，不能据此宣布原混合概率公式数学错误。

## 误检/漏检分解

下表是逐 object-frame precision/recall 的算术平均，区别于像素池化统计。

| 场景 | 基线 precision / recall | max-product precision / recall |
|---|---:|---:|
| figurines | 8.61% / 32.77% | 4.88% / 56.82% |
| ramen | 4.15% / 59.30% | 3.30% / 89.48% |
| teatime | 8.64% / 59.53% | 5.25% / 89.46% |
| waldo_kitchen | 24.83% / 56.66% | 19.98% / 55.55% |

前三场景扩大候选支持明显提高召回，却增加误检、降低 IoU。
因此现在不能再把“扩大覆盖率”当成主要修复手段。
这些结果尚不能单独区分“文本选错对象”和“选对候选但范围已混杂”；下一步要分别验证它们。

## 产物与测试

结果根目录：`/mnt/pool/sqy/results/RADIO-GS/output/`

- `v4_candidate_composition_20260914/{scene}/report.json`：基线和候选替代；包含 208 观测的各分支分数、PNG 哈希、输入缓存哈希。
- `v4_disjoint_constraint_repair_20260914/{scene}.pt`：修复约束后独立重建的对象记忆。
- `v4_disjoint_constraint_repair_20260914/{scene}/report.json`：对应原生分辨率评估及新记忆哈希。
- `v4_repaired_query_inputs_20260914/manifest.json`：本轮使用的正确文本输入权威清单。

旧产物未覆盖，新建对象记忆未替换之前的 sealed 权威输入。
实验分配 GPU 2/3 并行场景任务；主体 carrier 渲染及对象聚合仍是 CPU 工作，不能将低显卡利用率称为满载 GPU 训练。

验证：v4 与相关 summary/cache/公开 mask 指标测试，279 passed、2 skipped；增加了部分重叠、空支持、跨视角及训练 partition 的回归测试。

## 下一步优先级

1. 用正确 summary/text 编码复现源图 crop 检索基线，检查身份选取在进入 3D 组织之前是否已经失败。
2. 冻结同一批候选，分开测文本选取和 oracle 范围：区分错对象与混合范围，GT 只作诊断、不进入部署。
3. 借鉴历史三态关系：将缺少同对象证据视为 unknown，部件关系不应直接变成硬 different；相关学习输入改动必须重新训练验证，不能静默改 pretrained 特征语义。
4. 对照改进要继续保持同输入、同阈值、四场景全量复评；不再仅凭局部测试就推进新结构。
