# v4 主因定位：同输入、逐段对照（2026-09-14）

## 结论先行

这轮**没有把完整 v4 修复到历史 40% 左右的水平**，不能宣称问题解决。
但已把“到处找小 bug”缩小为两个明确失败接口：

1. **对象范围没有组织成完整、稳定的目标。** 在当前候选和 0.2 阈值下，即使 GT 选择正确对象假设，稳定单候选的四场景等权 mIoU 也只有约 19.83%。这不是任意多候选组合、任意阈值或整个 v4 方向的理论上限。
2. **身份到范围的组合进一步损失效果。** 同一个 cosine top-1 选取，直接读取范围为 8.21%，经 canonical top-2 混合后降为 5.33%。完整 generic-negative 路径则每条查询平均激活约 26%–54% 的对象候选，造成大量误检。

前几轮仅凭源视角往返、单模块测试和覆盖率，就过早认为几何/对象组织已经通过；这些检查不等于跨视角可用的对象范围，更不能代替完整查询评估。

## 为什么历史主线正常，当前却低

实际阅读并对照：

- `docs/experiments/2026-08-17-lerf-identity-extent-dual-posterior-closure.md`
- `docs/method/universal-field-typed-readout-v1.md`
- `radio_gs/scripts/build_lerf_sam_siglip_object_posterior_scores.py`
- `radio_gs/querying/latent_proposal_posterior.py`

历史有效链路不是当前 v4 的等价重构：它有重建 Gaussian 场上的 dense identity unary，范围候选受 immutable field peak、源视角一致性约束，失败时还有 primitive fallback；候选在查询条件下与 null 竞争。
当前分支用 SAM crop 原型承担身份，先完成场景级碎片合并与 top-2 表示，再用独立 generic-negative sigmoid 激活候选；没有上述已验证的 dense identity anchor/fallback。
此外，当前 full-v2 affinity 报告明确写着训练外观来自 `mean_local_projected_radio_F71_slice_4_68`，`deployment_siglip2_crop_summary_domain_validated=false`，真实 SAM 噪声未建模。实际对象构建却使用 genuine SigLIP2 region-crop 原型。这是尚未解决的训练/部署域差异，不能用代理验证集的高 edge precision 证明对象组织已可用；它是风险证据，不是本轮已经单独量化的唯一根因。
几何也从重建 Gaussian 场换成 MoGe 独立深度融合、4 cm 体素，SAM 写入降到约 46×62（实际尺寸逐场景读取）。不能把“用了同一个 RADIO/SAM”视为仍然保留历史方法的有效链路。

历史 teacher 与当前 teacher 的 checkpoint、genuine crop-summary 类型相同，但 proposal 集合不同：figurines 历史 277 个，当前 579 个。
本轮没有把历史代码导入 v4，没有拿旧模型输出冒充新模型预测。

### 评估协议也不能混同

直接核对历史结果 JSON：历史 LERF3D 使用 `selected_only_alpha`，先阈值选择三维 Gaussian，再移除未选 Gaussian、渲染选中部分；当前 v4 默认是 full-scene 连续概率渲染后像素阈值。遮挡和概率含义不同。
新增 selected-only surfel support 对照，保持原三维查询 field 不变，四场景为 6.47%/3.44%/5.11%/15.53%，平均 **7.64%**；相对 full-scene 7.57% 没有恢复历史水平。因此协议差异真实存在，但这个适配对照没有解释全部性能差距。

上游 [OpenGaussian 查询渲染脚本](https://github.com/yanmin-wu/OpenGaussian/blob/main/render_lerf_by_text.py) 在导出前对 cluster silhouette 使用 0.7；[mask 评分脚本](https://github.com/yanmin-wu/OpenGaussian/blob/main/scripts/compute_lerf_iou.py) 再对 PNG 使用 >10。两者不是同一个阈值。
历史本地 VALA/自定义轨迹中的 `10/255` alpha 配置不应自动被称为完整 OpenGaussian 等价复现；没有据此静默改写历史 VALA 预设。当前 surfel 适配不含 Gaussian alpha，也不因此成为官方协议认证结果。

历史八个结果文件的 SHA 核对为 7/8 一致；figurines LERF3D 文件实际 SHA 为 `a802bc4af07bc4d3d6c9abfce3692927ab00a26ff8b00fae81e29d666fd97971`，与历史汇总清单所填 SHA 不同，但原始文件的 0.5164960622787476 mIoU 与汇总数值一致。这里只记录出处完整性问题，不据此宣布旧分数虚假，也不改写历史权威清单。

## 四场景逐段结果

所有实际预测都先产生同一个三维 element field，再渲染 native mask；GT 不参与预测。
表内为每场景 object-frame 平均 IoU，汇总为四场景等权平均；像素阈值始终 0.2。

| 路径 | figurines | ramen | teatime | waldo_kitchen | 等权平均 |
|---|---:|---:|---:|---:|---:|
| 原完整 generic + canonical 路径 | 6.18% | 4.00% | 7.50% | 12.62% | 7.57% |
| 同对象 cosine top-1，raw 范围 | 11.17% | 1.10% | 11.76% | 8.79% | 8.21% |
| 同对象 cosine top-1，canonical 混合 | 5.01% | 0.25% | 7.57% | 8.49% | 5.33% |
| 绕过对象聚合，源 masked-crop top-1 | 10.01% | 4.54% | 18.87% | 10.08% | 10.87% |
| 同上，仅恢复 native SAM 写入 | 10.53% | 6.65% | 20.39% | 10.36% | 11.98% |
| native 写入 + 单身份锚定范围扩展 | 10.92% | 6.14% | 19.25% | 8.79% | 11.28% |

直接将对象 cosine softmax 与 raw 范围做凸组合仅为 0.37%，不能把数学归一化正确等同于适配当前概率阈值。
补充固定 sigmoid 中性决策点 0.5（非 GT 搜索选阈值）后，完整原路径四场景为 3.84%/5.72%/6.02%/12.57%，平均 7.04%；没有恢复正常精度。它不排除进一步做独立校准，但排除了“直接改成 0.5 即解决”的解释。
top-1 和单锚定都是诊断/候选方法，不支持完整多实例语义，不能替换正式 multi-instance 接口。
11.98% 不是完整方法成绩；“7.57% → 11.98%”包含绕过对象组织的变化，不能全部归因于 native 写入。
native 写入的独立效果是同路径 **10.87% → 11.98%，四场景全部正向**。
把 native 证据重新送入原来的 frozen association/generic/canonical 完整组合，则约为 5.50%/4.36%/6.22%/13.77%，平均 **7.46%**，没有超过原完整组合 7.57%。因此 native 写入的局部修复并未解决整体方法退化。

## GT 只作诊断：范围上限

遍历现有全部候选，native raster，当前固定阈值 0.2；没有把 GT 选出的 ID 写入预测记忆。

| 场景 | 对象假设：跨视角固定一个候选 | 对象假设：逐视角任意最优候选 | 原始碎片：固定一个候选 |
|---|---:|---:|---:|
| figurines | 18.52% | 28.36% | 19.12% |
| ramen | 10.32% | 15.46% | 12.10% |
| teatime | 25.31% | 33.02% | 26.69% |
| waldo_kitchen | 25.17% | 25.24% | 27.96% |

这组数据区分了两个问题：候选本身不完整/跨视角不稳定，以及文本与组合未能兑现已有候选的能力。
它不能证明增加多个合适候选后的上限仍小于 40%，也不能单独断言几何是唯一原因。
GPU sparse renderer 仅用于穷举诊断，每帧与 CPU 首候选参考比较并记录误差；部署预测表仍由原 CPU renderer 生成。

## 实际修复

### 深度切片相机射线偏移

`fuse_lerf_moge3._scaled_camera` 的 depth 输入是 `[::stride]`，并非 resize；反投影使用 `j + 0.5`。
原主点 `(c + .5)/s - .5` 应为 `(c - .5)/s + .5`，才能与原样本 `s*j + .5` 对应。
stride=4 时旧公式对应原深度栅格 3 像素的射线偏移。已修复；新增 stride 1/2/4/7 的射线一致性测试。
现有旧几何缓存不会因代码变化自动修复，本轮没有把旧缓存结果标成这一修复的收益。

### 分离语义特征栅格和源掩码栅格

正式 `lerf_fragment_surface_memory` 构建器新增 `--source-mask-raster native`：

- 原始 SAM 尺寸决定掩码采样相机，不再继承 46×62 特征栅格；
- 要求 scene state 已绑定对应 carrier reference raster，避免高分辨率重新退回 radius-one 小点；
- 每帧记录真实 `mask_raster_shape`，全局 `raster_shape` 明确是 carrier reference；
- 每帧用完立即释放投影缓存，避免 native 投影累积占满内存；
- 旧 reference 模式保留，以便历史回放，不覆盖旧产物。

### 未接受的替代方案

- `anchored_fragments.py` 实现身份锚定后、按视角选择语义与几何相容的范围，使用 max 防止重复证据累加。但平均 11.28% 低于 native 单源 11.98%，**不提升为默认**。
- 使用修正射线的新 1 cm 几何，并在 native 输出像素量化物理 footprint：figurines masked top-1 为 6.23%，ramen 为 4.29%，低于原 4 cm/native 写入的 10.53%/6.65%。这是射线/分辨率/footprint 的组合探针，不是单变量证明；**不推广到另外两场景，不覆盖旧几何**。
- 单纯增加点数或换归一化公式均没有救回主线，不继续沿这些失败替代扩大训练。

## 产物索引

根目录：`/mnt/pool/sqy/results/RADIO-GS/output/`

- `v4_chain_causal_20260914/{scene}/report.json`：身份选取/范围混合拆分。
- `v4_probability_half_causal_20260914/{scene}/report.json`：预先指定 0.5 概率决策点的全量对照。
- `v4_selected_only_causal_20260914/{scene}/report.json`：同三维 field、仅改变 selected-only 渲染语义的对照。
- `v4_source_causal_20260914/{scene}/report.json`：原始碎片检索。
- `v4_native_ceiling_causal_20260914/{scene}.json`：GT-only 穷举范围上限。
- `v4_native_relift_causal_20260914/{scene}/report.json` 与 `native_relift.pt`：native 写入对照及源证据。
- `v4_anchored_extent_causal_20260914/{scene}/report.json`：未接受的锚定扩展。
- `v4_geometry_repair_causal_20260914/{figurines,ramen}_1cm.{pt,json}`：修正切片射线后的新几何探针。
- `v4_fine_geometry_causal_20260914/{figurines,ramen}/report.json`：新几何预测对照。
- `v4_native_fragment_repair_20260914/{scene}.pt`：正式构建器的 native 源碎片记忆；绑定已有 corrected sealed scene state。
- `v4_formal_native_evaluation_v2_20260914/{scene}/report.json`：正式产物重新冷加载、渲染后的 source-only 与完整 canonical 组合评估。

正式重建曾出现相对早先 GPU 原子累加结果最大 0.000244 的 FP16 差异，没有将其谎称 bit-exact；因此重新评估正式产物，而不是直接沿用早先实验分数。
重新评估后，`v4_native_fragment_repair_20260914/verification.json` 已确认四个正式记忆的两种源查询 field 与各自实际评估 field **逐位一致**（独立进程冷验证）。

所有实验使用 GPU 2/3；投影构建仍有 CPU 开销。没有后台进行未报告的长期训练。
最终完整相关回归测试：**287 passed、2 skipped**；`git diff --check` 通过。selected-only 新增了“未选前景不得遮挡选中背景”的回归测试，且不会修改持久场景。
这不是完整 LERF2D/3D 已达标结论；公开 mask 评分口径不自动认证完整上游协议。

## 科研决策

当前 `learned fragment association → scene top-2 → generic sigmoid` 分支不应继续被视为已经验证的完整框架。
保留有效的 summary/text/射线/native 写入修复，暂停把该分支当成性能基线继续加训练。
如继续研究，应以历史可复现 identity–extent 接口为对照，先单独恢复可靠身份定位，再用 source-heldout 对象范围验证跨视角组织；不能继续先压缩候选、再指望查询头补回丢失的身份与范围。
