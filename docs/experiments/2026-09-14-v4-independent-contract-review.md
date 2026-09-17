# v4 独立方法与实现审查

审查提交：`0ad5dbbf5336f983d6a891c4162e0944da8f1463`；日期：2026-09-14。

结论：附件对主因的判断基本成立。当前候选路线不具备替代历史性能主线的证据。问题同时存在于几何坐标、证据语义、对象组织、查询决策与实验比较；不能用增加训练/补全或更换聚合公式解决整条链路。建议恢复历史可复现链路作为控制组与性能基线，v4 保留研究身份。此次提交仅增加审计材料，不修改算法默认值、不覆盖缓存、不执行破坏性回退。

## 证据范围与限制

实际阅读 geometry/carrier、fragment association/membership、canonical composition、cold evaluator、completion adapter、public scorer、ADR 0005、历史方法和最新实验记录。使用当前生产函数运行 CPU 反例，而非只重写公式。相关现有测试 28 passed、1 skipped；这些测试通过仍能复现关键合同失败，说明测试覆盖的是已实现行为，不足以证明物理/统计正确性。

没有重新训练基础模型、跑五 benchmark 或重新生成全量历史预测。真实场景 v4 数字来自仓库 20260914 同输入实验记录，不是本次新测。历史结果原文件做了独立 SHA 盘点。GitHub CLI 在当前环境缺失，因此未获取在线 issue/spec；未发布 issue。本文不是穷尽所有代码路径无遗漏的保证。

可执行证据：`scripts/audits/v4_contract_review_20260914.py`；输出：`paper/artifacts/v4_contract_review_mechanisms_20260914.json`。运行：

```bash
PYTHONPATH=. /root/miniconda3/envs/sam3/bin/python scripts/audits/v4_contract_review_20260914.py
```

## 1. 致命的方法错位：候选证据被当成实例后验

`fragment_to_object.py::_compose_observed_membership` 逐假设融合源碎片。这些列可以重叠、重复、部件/整体嵌套。`lerf_text_cold_evaluator.py:101` 却无条件送入 `SparseObjectAssignments.from_dense(..., top_k=2)`；后者按整行总质量归一化，删除 top-2 之外的质量并并入 unknown。

生产算子反例：同一个表面有 1/2/4/8/16 个完全相同的目标候选，且全部被正确选择，最终响应为 1/1/0.5/0.25/0.125。复制表示即可改变查询答案，16 个时低于 0.2 阈值。候选集合中无关但重叠的列也能稀释原目标。

这里不是 categorical mixture sum 的数学公式本身错误，而是它的前提未成立。若已有互斥真实实例，sum 合理；若保留重叠假设，必须保留候选证据语义，查询后消歧、去重、选范围。不能通过简单换 max 声称修复概率：仓库真实对照 max-product 7.57%→6.11%，误检扩大。

修复：明确 candidate-support 与 instance-membership 两种类型，禁止隐式互转；研究阶段保留未经 top-2 压缩的原始候选，分别记录未观测、关联歧义与压缩丢失质量。验证重复候选不变性及独立负例后，再决定存储压缩。当前 top-2 是方法约束，不是无损实现优化。

## 2. 证据关系：缺失不等于不同对象，重复不等于更多信息

`fragment_to_object.py:150` 的 visibility conflict 没有显式 negative 输入。生产反例：两个互不相交的部件，两视角都可见，conflict=1；双向缺失仍不能证明不同物理实例。已修复 geometry<0.5 的误判，但 geometry=0 仅证明支持不相交，也不能证明物体不同。

默认 probabilistic_union 对同视角重复 mask 按 noisy-OR 融合：同一错误位置每次 0.05，5 次为 0.226219，20 次为 0.641514。代码虽有 same_view_max_then_union 诊断选项，却不是默认，且跨相近视角仍相关。对象内重复使错误更强，对象间重复又被 top-2 使正确更弱。

修复：same/different/unknown 关系必须携带粒度和来源；不能把所有 SAM disjoint 当 cannot-link。按独立观测来源控制重复贡献，保留可撤销的原始支持。改关系特征语义需重训已有 affinity，不能直接复用旧权重并声称匹配。

## 3. 几何存在明确 bug，但修复不保证恢复历史分数

| 位置 | 实际证据 | 结论与处理 |
|---|---|---|
| tsdf_fusion.py:69 与 sparse_voxel.py:197 | 反投影 x+0.5，投影 round；生产投影返回 0,2,2,4,4,6,6,越界丢弃 | 同相机 8 像素有 4 个不回原像素。统一边界/中心坐标、resize/stride/grid_sample/所有 carrier，不能只换一个 round；重建依赖缓存 |
| tsdf_fusion.py 默认 dispersion | 同体素 RMS≤sqrt(3)/2*s，门槛 1.5*s | 默认无法拒绝厚单元；用跨视角深度残差、法线厚度、自由空间冲突，而不是盒内 RMS 假保证 |
| fuse_lerf_moge3.py:238 | confidence=validity.float；融合取权重平均 | 有效点平均 confidence=1，未建模可靠程度。有效与可靠必须分开 |
| sparse_voxel.py:230 | 深度 band 内加权均值，无表面交点 | 前景背景间隔 1cm、4cm voxel，遮挡目标输出 0.437824；对 full-scene visible segmentation 可产生穿透 |
| SAM source raster | native 写入选项已存在；默认/旧资产仍可 reference raster | 输出放大不能恢复已丢边界。分别管理几何、特征、mask 分辨率及版本 |

文件名含 TSDF，但实现是 voxel-binned weighted point fusion，没有 TSDF 积分或自由空间约束；这不是命名本身造成低分，而是不能据文件名推断已满足表面一致性。几何 confidence=1 本身也不是未校准情况下非法数值，而是不能把它当作已验证可靠性。

stride 主点公式修复与 native footprint 修复已经在当前提交中，不应重复报为未修。单元往返正确只是必要条件，仍须检查跨视角遮挡、边界泄漏和小物体保持。

## 4. 身份出口退化与训练部署不一致

历史有效系统保留 dense field identity unary、身份峰保护、查询条件范围以及 primitive fallback。当前完整文本评估直接从 crop prototype 选择对象假设，经 canonical composition 产生输出。仓库存在 local-semantic API，不代表当前入口已使用它；compose_object_queries 固定 QueryPacket('multi_instance')，不自动承担部件/材质/stuff 查询。

本任务包含部件、整体、集合、材质等不同范围。要求所有文本先通过完整对象库，实际给 benchmark 加了一个更难且未验证的前置任务。身份和范围必须分开验证：源教师能定位→写入 3D 后仍能定位→候选辅助范围。对象先验不能成为唯一身份出口，也不能让错误范围覆盖可靠局部身份。

仓库 affinity domain 审计显示 projected RADIO 训练外观与 genuine SigLIP2 crop 部署域不一致。两者同为 cosine 不意味着数值/实例语义等价。geometry-only 实验未救回性能，因此这是风险与验证缺口，不能认定是唯一主因。应使用一致 descriptor 和真实 SAM 碎片错误分布验证关联；edge precision 不足以反映一次错合并破坏整个对象，需要对象等权 merge/fragmentation/extent precision 指标。

summary checkpoint、crop/frame 对应、文本规范化已有修复证据，不能继续拿已修 bug 解释全部低分。

## 5. 查询概率与补全

generic-negative sigmoid 的 0.2 接受边界对应相似度差 -0.041589；相等输出 0.5。它是相关性打分，不自动是校准实例概率。生产函数还取 prototype max，更多 prototype 带来更多机会命中；重复/相关 token 下 multi-instance null=prod(1-p) 同样不应解释为已经校准的不存在概率。

应拆开测检索排序、候选数量、absent-target 拒绝、选中范围纯度。校准只能在冻结的 source-heldout/开发划分，不能对最终 benchmark 扫阈值。0.5 固定点的既有对照仅 7.04%，说明单独改阈值没有救回链路。

**修正附件的一项泛化说法**：`completion/lerf_adapter.py:104` 当前对多个超阈值 token 的行保留 unknown，不再对所有歧义行强制 argmax。它只对唯一超阈值支持行硬化。然而“唯一支持”不等于正确，错误唯一归属仍可被 clamp；structured_extent 要求 observed positives 精确固定。尚未证明对真实错合并与噪声支持鲁棒。completion 不能先于身份/范围修复，也不应作为可见表面分割的必需步骤。

## 6. 评估和任务定义确实有问题，但不能归结为 IoU 算错

本次对照上游公开源码：

- https://raw.githubusercontent.com/yanmin-wu/OpenGaussian/main/scripts/compute_lerf_iou.py
- https://raw.githubusercontent.com/yanmin-wu/OpenGaussian/main/render_lerf_by_text.py

本地 evaluate_opengaussian_lerf_masks.py 的 >10、缺文件计零、empty union=0、object-frame 平均、严格 Acc cutoff 与所查上游一致；2 个相关测试通过。未发现该适配器基本 IoU 计算造成巨大低分的证据。上游 URL 是移动 main，正式冻结需另绑定提交。

但以下协议必须分开：

1. 冷评估 `_evaluate` 返回 macro_category_miou；公共脚本返回 object-frame mean；micro_iou 是像素池化；四场等权又是另一层汇总。不能统称同一个 mIoU。
2. 历史 43.7304% 是 208 样本加权结果；该历史结果四场等权约 42.92%。当前 7.57% 是场景等权。不能直接相减，尽管差距远大于该口径差异。
3. 当前 full-scene continuous render 与历史 selected_only_alpha 输出不同。既有同 field selected-only surfel 约 7.64%，未恢复历史效果；协议差异不是全部解释，也不代表 surfel 适配等价 Gaussian alpha。
4. 冷评估跳过没有 GT 的 category-frame，公开列表也只覆盖已标注目标。可维持公开协议，但它们不能证明 absent-target 拒绝能力；需另建完整标注诊断，不能把未标注等同目标不存在。
5. LERF2D 的定位、LERF3D 三维选择再投影、ScanNet 点/类 IoU、NVOS RGB-assisted 等应独立声明；一张 PNG 评分不能同时认证五 benchmark。
6. source-only、query 授权、target RGB-assisted 要分表；已有 provenance 检查是必要设施，不是对所有外部资产自动证明无泄漏。本次没有发现可证实 target GT 进入正式预测的新增证据。
7. 19.83% 只是当前候选、阈值、跨视角固定单候选的 oracle 诊断，不是任意集合组合理论上限；GT 不能据此选部署候选。

## 7. 真实损失定位：不要再扫更多模块

来源：paper/artifacts/sugm_v4_primary_failure_localization_20260914.md。

| 对照 | 四场等权 mIoU | 合法结论 |
|---|---:|---|
| 完整 generic+canonical | 7.57% | 当前链路失败 |
| 同 cosine top-1，raw 范围 | 8.21% | 候选有部分信息 |
| 同选择，canonical | 5.33% | 组合进一步损失信息 |
| source crop top-1 | 10.87% | 对象聚合并未保留 source 能力 |
| 同 source 路径，native 写入 | 11.98% | 写入分辨率局部有效 |
| native 回到完整组合 | 7.46% | 局部收益未传到整体 |

这些比较不能精确分解每个 bug 的百分点贡献。既有细几何探针同时更改射线/voxel/footprint，不可用来证明坐标修复有害。固定阈值比较是诊断，不替代不同输出概率的独立校准。

## 8. 历史恢复方案与此次核验

**新的独立核验与附件不同**：沿 `lerf_latent_proposal_null_full4_result_20260820.json` 的 run_root 找到八个原始结果，8/8 SHA 与该清单一致，包括 figurines。见 `paper/artifacts/v4_review_historical_inventory_20260914.json`。附件的 7/8 不应无条件继承；可能核对了另一路径/另一版本，本次未确定原因，未改历史记录。匹配证明文件出处，不等于本次复跑预测。

恢复时分别锁定：LERF2D retained identity–extent 39.584%；LERF3D latent proposal null 43.7304%；不要拿该 latent 路线未提升的 LERF2D 37.0796% 替代 retained 版本。保留表跨任务数字也不自动成为单一 checkpoint 的 Joint Development Baseline。

执行顺序与验收：

A. 在隔离 checkout/显式 baseline adapter 恢复历史源码版本、checkpoint、teacher/mask/source list、query scores、renderer 和原执行 args；args 可从上述原始结果读取。先重放旧预测，再冷生成同预测，两步分别记录 hash/数值容差。当前分支仍保留，不 reset。

B. 同一批历史预测通过现有 public scorer 复评，同时保留原聚合与统一样本级/场景级汇总。确认渲染阈值、GT、尺寸与 query 字符串一致，再允许引用历史性能对照。

C. 固定 source teacher/mask/query/视角，只换几何与注册，保留直接局部身份出口，不做全场对象合并和 completion。先通过坐标/可见性反例，再检查跨视角边界和小目标；所有受影响几何和 lift 重新构建并绑定实现版本。

D. 在 C 上仅加入可撤销对象组织，先不 top-2。必须保留身份与候选召回，不用局部 edge score 替代完整范围验证。对同对象部件、重复候选、错合并、缺观测做独立诊断。

E. D 有证据后单独校准查询拒绝与范围融合；最后才是 completion、压缩。每轮一项改动，固定开发协议、完整报告逐场结果和 precision/recall，不靠挑最好场景提升。

A/B 不需要等待 v4 所有理论问题解决。若 A/B 可复现、C/D 无增益，性能主线继续历史系统；这是可测的决策，不是“分析不出来才回滚”。本次没有宣称 A/B 完成。

## 9. ADR 与停止规则修正提案

ADR 0005 仍为 Proposed，却在第 3 步要求 top-2，第 6 步又把压缩放在 exact 方法闭合后，存在顺序冲突。它还让后续 benchmark 受严格前置 gate 阻塞，而此次已证明局部 gate 不能保证端到端。需以新 ADR 明确修订，历史 ADR 与资产保留。

建议硬 gate 仅阻止坐标/类型合同错误、非法信息、训练部署不匹配、无有效评分定义和结果冒充；允许固定开发端到端诊断与局部 gate 并行。最终确认集继续保护。以直接局部身份为基础、对象范围为辅助；不要求先完成整场互斥实例划分或不可见区域补全。

优先级：P1 坐标与表面合同；P1 candidate/instance 类型和 top-2 信息损失；P1 false-negative 关系；P1 历史控制组与评估绑定；P2 真实域训练与查询校准。没有证据支持立即增加模型宽度、更多 slots、mass/completion 网络或新架构代号。
