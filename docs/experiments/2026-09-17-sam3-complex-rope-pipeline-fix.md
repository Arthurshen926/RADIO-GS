# SAM3 复数 RoPE 被精度转换破坏：系统级实现错误

## 确认的根因

`build_sam3_foundation_cache.py` 中 `model.to(dtype=torch.bfloat16)`，以及 point-only 路径中对子模块执行 `to(dtype=target_dtype, device=...)`，会将注册的复数 buffer 转为实数。SAM3 visual ViT 的 `freqs_cis` 是 complex64；当前官方 image builder 的 `use_rope_real=False`，实际 attention 消费该复数 buffer。虚部消失后不再执行正确的位置旋转。point-only 路径即使请求 float32 也受影响；非 point-only 的 `model.float()` 路径保留复数，不应一概判坏所有 SAM3 结果。

修复使用 `Module.float()` / `Module.bfloat16()` 仅转换浮点参数与 buffer，再单独移动 device。没有改 checkpoint、模型结构或官方 attention 代码。自动 cache 合约升为 v3，记录复数保护及 loader SHA，防止 `skip_existing` 混用旧结果；NVOS 查询裁剪也增加对应身份。

## 可复现证据

同一官方 checkpoint、同一 fern 源图、16个固定源点，各3个候选，保留实数权重不变，分别使用完整复数相位与旧转换产生的实数相位：

- 对应候选 mask 平均 IoU：**40.08%**；
- mask 像素改变比例：**5.01%**（完整图像分母）；
- 质量预测平均绝对差：**0.2253**；
- 恢复复数后再次推理：mask **完全一致**。

这是输入/实现正确性的对照，不是语义 GT 质量评分。不能由这组数值直接声称最终召回提高了多少。实测及三个输出保存于 `output/sam3_complex_cast_audit_20260917/`。

历史 LERF multiscale cache 的 producer binding 中，旧 loader SHA 为 `b2e849a3a83c0fffb297742c55074ae98da7b6c30d6999b89bafc5def95f9313`。本轮由修改前内容复原的文件 SHA 与之完全相同，确认旧 source32/grid16 自动 proposal 也经过这条转换。保留 LERF inventory 同时含有这些旧 proposal 与另行生成的 grounded regions，不能把整个 inventory 统称为全正确或全错误。以 FP32 非 point-only 生成的文本 grounding/caption masks 不受该转换错误影响。

## 对此前结论的影响

原生查询适配器崩塌、几何投影误差、signed 上限不可行等独立诊断仍成立。但在受影响 SAM inventory 上进行的 hash、MoGe 结构辅助、范围选择和教师替换实验，只能解释为该输入条件下的结果，不能作为干净教师条件下的架构优劣验收。已有数值保留，禁止把重新生成的 mask 套在旧 lifting、监督或 field 上。

两个尚在生成错误源 mask 的 NVOS 任务已中止，旧部分输出连同哈希移入 `interrupted_legacy_rope_source_sam`，不删除、不复用。重建顺序为：正确 SAM → 原生 crop 编码 → 对应/监督 → 场 → 范围与校准 → 实际查询。几何算法与输入不因 SAM 修复而改变；fern/orchids 重建几何的 scale/opacity 完全相同，中心最大差为 7.63e-6，法线最大差为 2.39e-7，不能声称整个几何文件逐位一致。见 `paper/artifacts/sam_ropefix_geometry_retention_20260917.json`。

## 运行中的闭环

- 修复后的8项 NVOS参考查询已生成，目标 RGB/GT 在构造阶段禁止读取。
- “旧地图＋新查询”的分离控制已执行 fern/orchids；最终源地图重建尚需单列。只换查询并不能补救旧图中的教师污染。
- 七个物理 NVOS 场景排入全量重建，horns 对应两个官方查询。
- LERF 追加干净单网格 source inventory 的完整重建，与 NVOS采用相同 grid12 自动 proposal 入口。与旧混合 multiscale/grounded inventory 比较属于完整 recipe 对照，不冒充只改一行代码的召回因果实验。

当前尚未给这批干净重建颁发最终性能结论。测试已覆盖 float32/bfloat16 下复数相位和整数 buffer 保留；相关54项测试通过。后续真实评估和最终封存以执行收据为准。
