# v4 合同修复与历史性能控制恢复

实施依据：用户要求全面推进完整修复；ADR 0006 修订 ADR 0005 的实验顺序。基准提交为 `0ad5dbbf5336f983d6a891c4162e0944da8f1463`。

## 结论与当前使用方式

历史 identity–extent 性能控制已实际恢复；v4 继续作为研究候选。已修复一批基础合同，但不能宣称完整 v4 已达到历史性能，更不能宣称五 benchmark 联合达标。没有破坏性 reset、覆盖旧资产或将历史输出冒充新 v4 结果。

| 控制 | 本次结果 | 验证范围 |
|---|---:|---|
| LERF2D retained identity–extent | 0.3958417485 mIoU，0.8798076923 LocAcc | 四场景 checkpoint/历史字段与分数冷加载，重新渲染评分 |
| LERF3D retained latent proposal null | 公共 scorer 0.4373935865，四场等权 0.4293161368 | 四场景、208 样本，重新渲染导出 PNG，再经独立公共 scorer |
| LERF3D 查询重建 | 四场景所有 query_scores、identity_query_scores、valid、xyz 逐位相等 | 从哈希核对后的 persistent field cache、SAM membership、crop teacher 与文本重新计算；不是直接复制旧 query scores |

LERF3D 原评分约 0.4373038248，公共 scorer 略有差异；本次没有将不同 GT 栅格/聚合/渲染约定称为完全同一协议。`upstream_protocol_validated=false` 继续保留。没有重新训练 RADIO/SAM，也没有重新运行所有源 RGB 的 foundation-model 编码。

恢复命令使用实际历史结果中保存的参数，经当前 parser 检查后重放。脚本在 v4 方法模块之外：

```bash
# 一个场景的 LERF3D；RESULT 是原始结果 JSON，OUT 必须是新目录。
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/audits/replay_retained_lerf3d.py "$RESULT" "$OUT"

# LERF2D 指定其实际 evaluator，不能拿 latent 3D 的 2D 退化版本替代 retained。
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/audits/replay_retained_lerf3d.py "$RESULT" "$OUT" radio_gs.scripts.eval_lerf_grounding

# 四场景从字段和 teacher 重建查询，校验每项输入 SHA，并比较实际张量。
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/audits/rebuild_retained_lerf_queries.py --source-root "$OLD_SCORES" --output-root "$NEW_SCORES"
```

实际 replay_receipt 位于 `/mnt/pool/sqy/results/RADIO-GS/output/v4_repair_control_20260914/{scene}/` 和 `v4_repair_control_2d_20260914/{scene}/`。查询重建校验位于 `v4_rebuilt_control_queries_v2_20260914/verification.json`。

## 代码修复

1. **像素坐标与版本绑定。** 新投影 floor 对应 `x+.5` 的反投影中心；Camera 文档明确 resize 与 stride 的不同内参变换。保留 `legacy_round_band_v1`，新建使用版本字段，历史缺字段配置仍走旧投影。独立 `_build_carrier`、bundle builder、selected-only 子 carrier 均传递版本，receipt 与外层配置一致性检查包含该字段。
2. **可见性。** `pixel_centres_nearest_v2` 只保留最近采样深度层，零 confidence 不遮挡可信元素；`pixel_centres_band_v2` 仅修坐标，用于单独对照。最近采样深度不等于法线 ray–surface intersection；本次没有冒充 TSDF/ray-casting 闭合。两种输出都不能自动认证 selected-only Gaussian alpha 协议。
3. **融合可靠性。** 新融合对源深度重投影，累计 agreement 与 free-space conflict。表面后方被遮挡的样本视为 unknown；表面前方违反观测自由空间才是 conflict。以有限观测统计更新权重并筛除支持不足/冲突过大的单元，保存两种计数。体素内 RMS 仅是诊断，不再宣传默认 1.5 倍门槛可检验物理厚度。新增原始 MoGe prediction 和 source authority 哈希验证、融合实现 SHA 与合同版本。
4. **候选与实例分开。** categorical composition 验证行质量并直接使用完整 dense membership，拒绝明显重叠的输入，不再自动 top-2。候选使用独立接口；explicit boolean selection 的范围并集不伪装校准概率。旧三种文本 evaluator 的候选归一化须显式 `--allow-legacy-candidate-normalization`；历史 transfer runner 明确请求旧语义并标记诊断结果。
5. **压缩损失单列。** SparseObjectAssignments 记录 compression_discarded_weight，与压缩前 unknown 分开报告。为了旧查询重放兼容，原 unknown_weight 仍包含二者；不把压缩损失称为“从未观察到”。默认 fragment assignment 新增 full_uncompressed，不再先删 top-2 外的软关联。
6. **正负证据。** 新 visibility conflict 必须来自显式同粒度实例负证据；没有负证据时输出未知关系，而非 mask complement。默认不将同图不同 masks 当硬 cannot-link，同图层级重叠本身也不作为直接合并证据。默认先对同视角 masks 取最大，再跨视角合并，防止同一照片重复证据累积。近重复跨视角相关性仍未被完全建模。
7. **旧模型隔离。** 旧 visibility-absence 函数保留为命名明确的 replay 运算；历史 affinity 训练入口显式调用旧特征语义，新关系合同拒绝无匹配 metadata 的 learned model。当前权重的真实 crop 外观域差异没有通过这次代码修改自动解决。
8. **写入与评分。** 新 CLI 默认 native mask 写入；source-mask、feature raster 分离。冷评分同时报告 object-frame mean、category macro、pixel micro；无有效观测报错，不输出 NaN 伪结果；无 GT 的组合不算 absent-target 测试。

## 真实单变量对照

共同固定 source crop descriptors、文本、SAM proposal 身份、source views、0.2 阈值；source_masked_top1 绕过对象组织，是单源检索诊断而非完整多实例方法。各行是四场等权平均，不能直接与历史 208 样本汇总相减。

| 对照 | Figurines | Ramen | Teatime | Waldo | 四场均值 |
|---|---:|---:|---:|---:|---:|
| 像素中心修复，旧几何、原 depth band，重新 native 写入 | 10.549% | 6.517% | 20.246% | 10.349% | 11.915% |
| 同一几何，只再换最近采样深度层，重新写入 | 9.486% | 6.241% | 16.725% | 9.947% | 10.599% |

最近层虽然修复最小遮挡反例，却在噪声几何上四场全部下降。因此不能称它是已验证提分方案。历史 source native 路径约 11.98%，坐标修复也没有独自恢复性能。同一 4cm 尺度/同一 reference footprint 的新融合几何复评已完成：Figurines **20.701%**、Ramen **10.401%**、Teatime **29.065%**、Waldo **10.223%**，四场均值 **17.598%**，相对于同一最近表面路径的 **10.599%** 四场全部正向。该比较包含重建几何中的射线与一致性改动，不能归因于单独 confidence 公式。它仍是 source-only 单候选诊断，不是完整 v4 多实例成绩。

四场新几何已全部重建：33178 / 15440 / 91697 / 37571 个元素；均值 confidence 约 0.749 / 0.754 / 0.868 / 0.950。这些是观测统计权重，不是校准正确率。旧资产继续保留。

## 验证与边界

新增回归覆盖：多 raster/stride 的逐像素往返、近前后层遮挡、zero-confidence 遮挡、版本冷加载、16 候选未压缩质量、重复候选不变性、明确正负证据矛盾、同视角重复弱 mask、自由空间与遮挡 unknown、压缩损失、完整 fragment assignment、不同 IoU 汇总口径。

测试命令：

```bash
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python -m pytest \
  tests/v4 tests/test_opengaussian_lerf_masks.py tests/test_latent_proposal_posterior.py -q
```

最终验证：**310 passed**，`git diff --check` 通过。最近表面与新融合几何两组共八个 source 诊断，在独立进程重新读取保存的 native evidence 后重建 masked/dual 查询 field，均逐位一致；同时核对 evidence/field SHA 与 projection recipe（JSON tuple/list 规范化后）。见 `paper/artifacts/v4_repair_cold_verification_20260914.json`。这验证诊断 field 保存/重读，不冒充完整部署 bundle。

本环境 sam3 的 Python 缺少 gsplat，不能收集完整测试；完整测试使用仓库可运行的 cybersim_agent 环境。早期重放脚本的 query-names 输入格式错误已修复，失败目录保留，不冒充成功记录。

**未完成且不应掩盖的事项：** v4 完整多实例查询与真实 noisy-fragment affinity 还未达到历史成绩；新几何不是严格物理表面模型；generic-negative 的拒绝校准与跨视角相关证据没有自动解决；ScanNet/NVOS/SPIn-NeRF 本轮没有全量重跑，因此不是新的 Joint Development Baseline。当前应使用已复现历史路线做性能控制，并继续单独验证 carrier 信息保存与对象组织。

另已查明前次 SHA 表述差异：附件的 7/8 来自 20260817 v9 结果集，figurines 文件确与该清单不同；本次前一轮 8/8 核对的是 20260820 latent-proposal 结果集，两者不是同一批文件。原始历史清单均未改写。

## 产物

仓库摘要：

- `paper/artifacts/v4_repair_historical_control_20260914.json`
- `paper/artifacts/v4_repair_historical_control_2d_20260914.json`
- `paper/artifacts/v4_repair_rebuilt_query_verification_20260914.json`
- `paper/artifacts/v4_repair_projection_ablation_20260914.json`
- `paper/artifacts/v4_repair_fusion_receipts_20260914.json`
- `paper/artifacts/v4_repair_fused_source_ablation_20260914.json`

完整 source 重写/导出/评分：`/mnt/pool/sqy/results/RADIO-GS/output/` 下 `v4_pixel_contract_repair_20260914`、`v4_nearest_contract_repair_20260914`、`v4_fusion_contract_repair_20260914`、`v4_fused_source_repair_20260914`。所有新 source 诊断明确不是 sealed deployment bundle，禁止拿旧 scene-state hash 冒充新 lift 的部署身份。
