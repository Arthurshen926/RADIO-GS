# Canonical query optimization — 2026-07-15

## 结论

本轮保留了两类经完整评估验证的改动：

1. LERF text query 恢复 completion-aware primitive score、旧版 multiscale readout 和保守的官方 SAM3 rendered-RGB refinement。四场景 LERF3D scene-macro mIoU 从 primary-only 的 0.417858 提高到 support completion 的 0.433461，再提高到最终 0.552567。
2. ScanNet 3D point query 修复了“可选 SAM3 分支反向扰动后续 core”的评估错误，并加入固定的官方 predicted-IoU 0.5 接受门槛。完整 63 个实例的 mIoU 从 0.303096 小幅提高到 0.303616，Acc@0.25 不变，Acc@0.5 从 0.174603 提高到 0.190476。

没有保留的尝试包括：把单视图 mask complement 当成 3D hard negative、强制使用 MPR footprint 像素作为 world-point prompt、把所有 point seed 限制在 Euclidean KNN，以及根据测试标签搜索 SAM/几何阈值。

这些结果仍是 development-stage controlled experiments。LERF 四场景、ScanNet `scene0000_00`、NVOS fern 和 SPIn-NeRF fern 已参与方法诊断；冻结方法后仍需在未参与开发的完整集合上一次性运行，才能作为无偏 benchmark 主结果。

## 当前方法主干

当前方法不是逐任务训练多个 3D head，而是一套共享 canonical primitive field：

- teacher：官方 C-RADIOv4 raw spatial feature；
- 重建：depth、alpha、raster responsibility 和 robust fusion 组成的 query-free MPR；
- 紧凑表示：每个 Gaussian 的 local code 与共享 affine basis 解码到 raw RADIO；
- capability：冻结的官方 DINOv3、SAM3 adaptor 用于 appearance/boundary capability loss 和 query evidence；
- semantic：官方 SigLIP2 text encoder；由于官方 spatial adaptor 的二维 oracle 不足，text 另用一个全局、跨场景冻结的 region-summary bridge，并与官方 adaptor 明确区分；
- query compiler：text、registered 2D prompt、world 3D point 都编译为同一 primitive-domain prototype/soft-seed evidence；
- inference：同一 query-independent support graph 与 solver；task-specific readout 只负责把 primitive support 投影到 benchmark 输出域。

MPR 一直存在于 canonical feature reconstruction 和多视图 support 中。本轮验证表明，MPR 的 camera/footprint provenance 不能无条件替代 world point 的可见性判断；因此它没有被强塞进 ScanNet point prompt routing。

## 本轮有效改动

### Primary/fallback contract

旧实现把 primary field 未覆盖区域和 fallback feature 混在一个表面上，导致 primitive direct readout 与 rendered readout 不一致。本轮明确为：

- primary 行是 canonical truth；
- fallback 只补 primary 缺失支持，不覆盖 primary score；
- support completion 使用 primary-anchored directed graph；
- text score、validity 和 confidence 同时按 completion contract 传播；
- 不生成新的 query-conditioned feature，也不读取 GT mask。

这使 LERF3D 四场景 core mIoU 从 0.417858 提高到 0.433461，说明修复的是 primitive coverage，而不只是二维边界后处理。

### Official SAM3 rendered-RGB refinement

LERF2D/3D 的最终 mask refinement 使用方法自己的 Gaussian RGB render、当前 query heatmap/box 和官方 SAM3；没有自训练 SAM head。接受规则固定检查初始支持、heatmap mass/mean 和 peak containment。它属于方法内部 readout，但论文必须披露，并同时报告 refinement 前结果。

ScanNet 3D point 使用官方 interactive point decoder。world click 先解析到最强 canonical primitive；只有该 primitive center 在固定训练相机中通过 rendered depth/alpha 可见性检查时才调用 SAM3，否则严格回退纯 3D 结果。官方 predicted-IoU 低于 0.5 的 mask 直接拒绝。单个二维视角的背景不会成为 3D hard negative。

### Core freezing

所有 ScanNet direct 3D core 在加载、调用第三方 SAM3 推理栈之前一次性计算并冻结。此前逐实例交替执行 `core -> SAM3 -> core` 会让后续 CUDA 数值路径漂移，使同一 core 从 0.303096 变成 0.302777。修复后 SAM on/off 的 `macro_core_iou` 精确一致，保证后处理消融可信。

## 当前结果

### LERF2D text query

| Scene | mIoU | Localization accuracy |
|---|---:|---:|
| Ramen | 0.433840 | 0.859155 |
| Figurines | 0.433591 | 0.857143 |
| Teatime | 0.510098 | 0.830508 |
| Waldo Kitchen | 0.302513 | 0.636364 |

协议：官方 annotation frames/query strings；固定五模板；absolute threshold 0.5；原图分辨率；bbox-smoothed peak localization；方法渲染 RGB 上的官方 SAM3 box refinement。没有 threshold sweep、per-scene query rewrite 或 target-mask calibration。

### LERF3D text query

| Scene | Primary only | + support completion | + official SAM3 | Boundary-F | trimap IoU |
|---|---:|---:|---:|---:|---:|
| Ramen | 0.557151 | 0.558108 | 0.592006 | 0.810090 | 0.355045 |
| Figurines | 0.388392 | 0.390972 | 0.598451 | 0.798958 | 0.317940 |
| Teatime | 0.487025 | 0.498721 | 0.649439 | 0.797328 | 0.428546 |
| Waldo Kitchen | 0.238865 | 0.286045 | 0.370373 | 0.454910 | 0.229720 |
| Scene macro | 0.417858 | 0.433461 | 0.552567 | 0.715321 | 0.332813 |
| Sample weighted | 0.458159 | 0.467488 | 0.586590 | 0.765906 | 0.352648 |

协议：与 LERF2D 相同的 text embedding/cache；固定 0.5 primitive selection；full-scene visibility-aware selected-membership composite；官方标注帧和 GT；方法渲染 RGB 上的官方 SAM3 refinement。`+ support completion` 是不含二维 refinement 的 core 对照。

### ScanNet world 3D point query (`scene0000_00`)

| Variant | mIoU | Acc@0.25 | Acc@0.5 | 判断 |
|---|---:|---:|---:|---|
| Pure canonical field | 0.303096 | 0.476190 | 0.174603 | 主对照 |
| Global Euclidean K16 | 0.301375 | 0.476190 | 0.206349 | 拒绝：主指标下降 |
| Adaptive K16 guard, ratio 2 | 0.302283 | 0.476190 | 0.190476 | 拒绝：主指标下降 |
| Official SAM3, no quality gate | 0.301729 | 0.460317 | 0.206349 | 拒绝 |
| Official SAM3, predicted-IoU >= 0.5 | **0.303616** | **0.476190** | **0.190476** | 保留为 `+SAM3` variant |

协议：每个不少于 100 个 mesh vertices 的实例按固定 seed 采一个 GT 内 world point；query compiler 只接收该坐标；63 个实例；canonical Gaussian domain 推理；固定 adaptive-sigma KNN 投影到官方 ScanNet mesh vertices；不使用目标 mask、类别或实例大小选参数。

提升很小，说明当前 3D point 瓶颈仍是“一点对应哪个实例”的 instance discrimination，而不是 graph iterations 或简单 point-to-Gaussian registration。`+SAM3` 应作为有额外计算成本的 variant 报告，不能掩盖 pure field 对照。

### 其余已跑通任务（本轮未重新优化）

| Task | Scope | Current result | 限制 |
|---|---|---:|---|
| ScanNet text | `scene0000_00`, 19 classes | mIoU 0.413309 | 单 development scene |
| NVOS registered scribble | 8 prompts/scenes in current score-lift run | mean foreground IoU 0.618940 | 尚需冻结后的正式全集复核 |
| SPIn-NeRF registered full mask | fern | foreground IoU 0.561235 | 不是 sparse-point 主协议，且仅单场景 |

## 被否定的具体假设

1. **单视图背景可作 3D negative**：一个 window query 从 0.595794 降到 0.176009。原因是单相机未观察到的 3D 区域被错误钳成背景。现实现只允许正支持增强。
2. **MPR footprint centroid 可作 world click**：前五实例从 0.286161 降到 0.285621；footprint 常落在低分辨率图像边缘，不等于用户点击。
3. **把原 world point 投入 MPR provenance camera 即可**：前五实例降到 0.284522；Gaussian 的最佳相机不保证邻近 mesh point 在该相机内可见。
4. **全局 Euclidean KNN 能修复 Mahalanobis 远锚点**：虽然 Acc@0.5 上升，完整 mIoU 下降。远锚点是局部错误，但不是总体瓶颈。
5. **SAM3 predicted-IoU 能普遍判断收益**：quality 与 IoU delta 的 Spearman 仅 0.112。固定 0.5 门槛只作为保守拒绝规则，不再搜索阈值。

## 评估与可比性边界

- RGB/SAM refinement、MPR、bridge、graph 和 deterministic projection 都可以是方法内部模块；是否可比较取决于 query、GT、target frames、输出域和指标是否一致，而不是所有方法内部处理必须相同。
- 必须在方法和 implementation details 中披露 SAM/GrabCut/RGB refinement，并给出无 refinement 消融；不能把它说成纯 feature-field 输出。
- 当前所有保留结果均为 `test_calibration=false`：target masks 只在最终 metric 代码中打开。
- LERF 结果可与使用同一官方 annotation/query/metric 的论文做通常意义上的 paper-level comparison，但 refinement 与 prompt templates 需要脚注说明。
- ScanNet text/point、NVOS 和 SPIn 当前范围不足以作为完整 benchmark 主表，只能作为 pilot/ablation。

## 产物

- LERF2D：`output/optimization_20260715/support_completion_evidence_confidence/*_fixed/lerf_ovs_results.json`
- LERF3D：`output/optimization_20260715/support_completion_evidence_confidence/lerf3d_*_primary_anchored_support_sam3_*/**/lerf_direct_3d_selection_results.json`
- ScanNet pure core：`output/optimization_20260715/scannet_point_local_registration/scene0000_current_core_full63.json`
- ScanNet retained `+SAM3`：`output/optimization_20260715/scannet_point_rendered_sam3/scene0000_quality05_final_full63.json`
- ScanNet rejected point variants：`output/optimization_20260715/scannet_point_local_registration/` 与 `output/optimization_20260715/scannet_point_rendered_sam3/`

## 验证

- canonical query、point-query、score-calibration、NVOS focused tests：37 passed；
- repository tests：848 passed / 3 failed；三个失败仍是未修改的 legacy nearest-view 文案、train script 行数 release audit、OpenCLIP explicit frame-id 排序；
- 本轮所有正式 JSON 保留 protocol、cache/checkpoint path 和 refinement provenance。
