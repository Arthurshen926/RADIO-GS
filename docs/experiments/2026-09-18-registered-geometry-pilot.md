# 注册相机与 MoGe 几何迁移试跑

本轮接入 ScanNet OpenGaussian 提供的 RGB、相机和无语义标签重建输入，不读取 `.labels.ply`、语言特征、查询或评价 mask。十场各32张源RGB已封存，实际 MoGe/校准/融合仅完成 scene0000_00；其余场景未声称重建完成。公开 train/test 相机重叠，不能叫独立新视角测试。

## 实现和可见性修复

- 注册针孔相机直接进入 MoGeGeometryPrior，明确 OpenCV 轴方向、内参、逆变换和哈希；不伪造 COLMAP 轨迹。
- MoGe 官方缓存 FP32 点图/法向，在 sam3 环境完成32帧推理。第一次缺失依赖的失败执行材料保留。
- 第一版仅顶点 z-buffer 会穿过稀疏投影的空隙，把被挡的背景当作可见参考。该结果保留，不用于后续地图。
- 修复版用提供的 points3d.ply 三角面确定可见性，再用顶点深度作校准参考。显式消耗了提供的网格几何，不能声称仅RGB路线。合成测试验证上/下方向和前/后遮挡。
- 全局顶点编号模5划分：0仅审计、1候选选择、2/3/4拟合。身份变换也必须通过选择集质量门槛；有界仿射变换还要求选择集 median/p90/mean 均不退化。没有根据审计集选择参数。

## 结果与结论

27/32视角通过：20个有界仿射、7个保留原预测、5个拒绝。通过视角中，审计顶点的视角宏平均 median AbsRel 4.408→2.334%，p90 10.919→8.938%，mean 6.057→4.314%。参考网格/相机共享，点ID划分不提供独立传感器真值或空间独立性；拒绝的5个视角不能从覆盖分母消失。

但接着实际融合并在融合留出源视角渲染，独立尺度版本覆盖仅22.83%，median AbsRel 15.74%，free-space fraction 28.21%；joint候选覆盖20.94%、free-space fraction 34.39%，没有通过渲染质量门槛。这里只对齐留出MoGe深度，未提供独立几何GT。留出视角各自也曾用网格校准，因此仅从联合拟合及融合留出，不是从所有几何预处理留出。

**不批准把该几何用于声称已修复的语义主线。** 下一定位应固定全部32视角分母，检查相机覆盖、源对深度一致性与双视角保留率，分离源分布稀疏和错误跨视角融合。不能因为单视角校准数字较好就启动大规模语义训练。

## 可复核材料

- `output/scannet_registered_sources_20260918/`
- `output/scannet_moge_current_20260918/scene0000_00/calibration_report.json`：保留失败参考方式。
- 同目录 `calibration_mesh_report.json`：修复可见性与三分割选择。
- 同目录 `geometry_mesh_calibrated/report.json`：真实融合与留出渲染。
- `radio_gs/geometry/registered_pinhole.py`、`mesh_visibility.py`，`scripts/audits/calibrate_scannet_moge_geometry.py`。

P5 的相机输入和真实几何试跑向前推进，但完整 ScanNet/NVOS 语义任务没有完成，不能填入历史方法分数。

单项有界控制把双视角相对深度容忍从1%增为3%，保持同一校准、分割和其余融合条件。覆盖22.83→22.93%，median AbsRel 15.74→15.81%，几乎没有恢复覆盖，否决“单纯容忍阈值过紧是主因”的解释；未保存或晋升新几何。结果在 `depth_consistency_tolerance_control.json`。

进一步分离“同一体素有几个观测”与“独立深度是否支持”后，只放宽前者到1、仍要求至少2个源深度支持且支持数大于自由空间冲突数：覆盖升至35.22%，但free-space fraction恶化到36.21%，agreement coverage仅13.90→16.28%。也不能晋升。这说明同体素占据确实限制覆盖，但放宽占据会暴露不可靠表面；下一步须定位错配/遮挡和覆盖，而非继续放宽门槛。材料为 `voxel_occupancy_control.json`。
