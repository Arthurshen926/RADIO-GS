# 监督覆盖与已注册正负提示：进一步实现和验收

本轮保留原 `all_four_shared` / `without_radio_shared` 场 checkpoint。扩充监督的六个训练候选没有同时改善三类 query，因此不替换原场。有效的改动是让已注册正负提示在推理时真正约束临时 query 向量，并增加能区分未知区域与误召的评估。

## 监督覆盖：实现了扩展，但拒绝把退化当进步

原跨视角训练只涉及 23 个目标区域，6 个训练视角实际上有 92 个 SAM 源区域。新增确定性的视角轮转采样，让全部 92 个区域参与教师辅助监督，并显式检查 query 视角和 target 视角均不跨越训练/开发边界。

新增 SigLIP 软 query–区域关系损失：79 个固定开发文本与同视角区域的原生相似度作为软分布，约束实际 query 投影与元素平均方向的相似度；不将 top-1 当成确定类别，不生成未知背景的负标签。此项仍是教师关系蒸馏，不是新增真实语义标注。

分别执行了六个 180 步实验，均使用同一冻结几何、源帧、随机种子和既有开发 episode：

| 候选 | 文本伪 IoU | 图像伪 IoU | 原始提示伪 IoU |
|---|---:|---:|---:|
| 原四教师共享投影 | 0.4291 | 0.3428 | 0.3372 |
| 扩充教师 + 单视角正例 | 0.5085 | 0.3935 | 0.2523 |
| 上述方案 + 软文本关系 | 0.4729 | 0.3186 | 0.2435 |
| 上述方案去 RADIO | 0.4655 | 0.2670 | 0.2433 |
| 仅扩充教师覆盖 | 0.4731 | 0.3500 | 0.2942 |
| 仅扩充教师 + 软文本关系 | 0.4099 | 0.2472 | 0.2690 |
| 上述方案去 RADIO | 0.3899 | 0.2339 | 0.2644 |

特别是单视角正例扩充令提示的已知负例误召率从 0.3051 增至 0.5914。实验说明“区域更多”不等于“检索监督更可靠”，也没有证明软文本关系有收益。没有把这些候选设为默认，也没有根据每个任务分别挑选 checkpoint 组成虚构的主线。

开关：`--source-coverage` 独立扩充教师区域采样；`--soft-query-relations` 需要启用 SigLIP；`--source-membership` 是额外的单视角正例实验，默认关闭。三个新开关均默认关闭。旧 `coverage_*_v1` 在拆分开关前运行，包含单视角 membership 扩充；`auxiliary_*_v2` 不包含该项。对应报告、权重和实现哈希保留在原产物根目录。

## 正负提示问题与修复

旧推理只使用 `Linear(concat(正点DINO, 负点DINO, 世界坐标))` 得到 query，再与地图匹配。它在训练中见过提示点损失，但在一个新的 query 上，并不检查或保证当前用户证据得到满足。

四教师场的 8 个开发提示任务中，只有 2 个同时满足实际召回决策：正点响应 > 0.5、负点响应 ≤ 0.5。若要求更强的 0.9/0.1 响应，原先是 0/8；两个标准不能混用。

新增 `radio_gs/models/signed_query_refinement.py`：

1. 从绑定冻结几何的原始源像素贡献取得正负点的可见元素混合；未知元素不参与，保留后遮挡特征支持归一化。
2. 检查原 query 是否已经正确响应给定正负点。已经正确则原样保留，不为追求置信度而改变身份。
3. 对违反证据的 query，仅优化一个临时 64 维 query 向量，64 步 Adam，学习率 0.05。损失为源点 BCE + 0.01 的原 query 方向约束。
4. 场特征和 query adapter 均 detach；不优化地图参数、几何或 opacity。不读取目标视角标签。
5. 输出仍是原公式 `sigmoid(10*cos(元素特征, query)-5)`，同一个元素后验用于 3D 和 2D。未满足的约束显式返回，不覆盖分数或硬填充目标。

这属于已注册 signed prompt 的查询过程，不是场重建质量提升。它使用已有协议提供的正负点、源视角与冻结几何贡献，没有增加目标 mask、目标 RGB 编码或物理实例码本。普通文本和图像 query 不执行这一步；需要额外每查询优化时间。任意未注册点击及更多 benchmark 协议仍需单独接入、验收。

## 固定场上的配对结果

以下仍是 **源视角开发伪标签、已知支持区域** 的指标，不是 benchmark 成绩。2D 与 3D 是同一元素后验的两种输出。

| 固定场 / 提示执行 | 2D 伪 IoU | 3D 元素伪 IoU | 已知负例误召率 |
|---|---:|---:|---:|
| 四教师，原提示 | 0.3372 | 0.2831 | 0.3051 |
| 四教师，条件校准提示 | **0.8416** | **0.5996** | **0.0261** |
| 去 RADIO，原提示 | 0.3135 | 0.2753 | 0.2823 |
| 去 RADIO，条件校准提示 | **0.7327** | **0.4926** | **0.0623** |

四教师场 6 个 query 执行校准、2 个保持原样；去 RADIO 场 5 个执行、3 个保持。两组最终均 8/8 满足 0.5 决策约束。仍有源点没有达到 0.9/0.1 的强置信度目标，不能声称任意几何载体都能精确满足所有点约束。

初版无条件校准的四教师提示均值为 0.8075，但前两个原本已满足正负决策的 query 退化。因此最终使用统一的源证据判定规则，避免无必要的更新；没有按目标 IoU 为单个 query 选择是否执行。初版与最终版均保留独立报告。这些开发视角已经用于方法选择，未来正式验证必须使用独立封存的样本。

## 评估与冷启动

新增 `partial_retrieval_metrics` 和 `evaluate_three_query_coverage.py`，报告已知区域 IoU、precision、正例 recall、负例误召率，以及未知区域响应占比。无标签时的指标返回 undefined/null，不当作零分或满分。未知区域响应可以诊断，不能算成可靠的 false positive。

当前可评估像素仍只有约 16%–17%；原场约 78%–83% 的响应质量位于未标注区域，这既反映覆盖不足，也意味着无法据此证明全图精准。图中第一例虽然源正负点已满足，目标仍有明显宽泛响应，保留原 query 不代表已经解决该例的实例范围问题。

冷启动在去 RADIO 原场上验证了一个实际违反负点约束的 query：

- 原生 DINO 从源 RGB 重新提取正负点特征，原生 SigLIP 文本/图像也重新编码。
- 禁止打开四教师训练特征、episodes、queries 与 RADIO 特征目录。
- 几何贡献表完整绑定，源 frame 88、像素栅格和最大贡献元素与已封存的 query 注册信息一致。
- 正点响应 0.7768 → 0.9744；负点 0.8538 → 0.0111。
- 全部 274,805 行生成同一个后验，有效性单独标记；没有场参数梯度或几何变更。

冷启动不是质量评估，目标标签未传入校准函数。它只证明这条查询链路可脱离训练缓存执行。

回归测试 **479 passed，4 warnings**，覆盖源/目标两侧 split 泄漏、未知指标、软关系教师 detach、正负点不可满足情况、场参数不变、已正确 query 不动、冷几何绑定与原有测试。未增加新的主线推广声明。

## 可复现入口与证据

产物根目录仍为：
`/mnt/pool/sqy/results/RADIO-GS/output/four_teacher_three_query_v2_20260915/figurines`。

配对评估：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
/root/miniconda3/envs/cybersim_agent/bin/python \
scripts/audits/evaluate_three_query_coverage.py \
--root /mnt/pool/sqy/results/RADIO-GS/output/four_teacher_three_query_v2_20260915/figurines \
--geometry /mnt/pool/sqy/results/RADIO-GS/output/geometry_only_repair_final_20260914/figurines/independent_scale.pt \
--runs all_four_shared without_radio_shared --refine-prompts \
--output /absolute/path/to/new_report.json
```

原生冷启动沿用 `cold_three_query_native.py`，添加 `--authority`、`--authority-sha256` 和 `--prompt-frame`，输入 `cold_signed_query_fixture.pt`。只有冻结几何贡献可以在冷启动打开，训练标签和教师特征仍禁止访问。

仓库结果：

- `paper/artifacts/three_query_coverage_eval_v1_20260915.json`：扩充正例训练。
- `paper/artifacts/three_query_coverage_eval_v2_20260915.json`：教师覆盖消融及无条件提示校准。
- `paper/artifacts/three_query_signed_guarded_20260915.json`：最终条件提示校准的逐 query 结果。
- `paper/artifacts/cold_three_query_signed_guarded_20260915.json`：去 RADIO 原生冷启动。
- `paper/artifacts/signed_query_guarded_20260915.png`：已正确、被修正、仍低置信度三种案例。

下一阶段仍应补可靠跨视角身份和文本标注，扩大独立场景与 query 协议的验证。当前证据支持提示查询实现的修复，尚不支持扩充教师损失后整体特征场更优，也不支持完整五 benchmark 指标已恢复。
