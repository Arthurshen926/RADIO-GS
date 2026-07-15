# Canonical compact capability iteration — 2026-07-14

## 本轮最终决策

本轮没有为了单个 benchmark 指标替换 canonical field。继续保留的表示主干是：

- 官方 C-RADIOv4 backbone 的 raw spatial teacher；
- 基于 depth、alpha、raster responsibility 和 robust fusion 的 query-free MPR；
- 每个 Gaussian 一个 128 维 local code，经 primitive-local residual fusion 得到 256 维 affine basis coefficient，再解码为 1280 维 raw RADIO；
- 训练时只增加冻结的官方 DINOv3、SAM3 adaptor capability loss；
- render fine-tuning 只读取 query-free 训练帧，并排除 LERF 标注帧；
- text、registered 2D prompt 和 world 3D point 都从同一 primitive field 派生证据。

主 checkpoint 为：

`output/canonical_fields/ramen_canonical_radio_fusion_d256_l128_nohash_mpr40_caploss_renderft256_seed0.pth`

SHA256：`c7a930c3e8fc83a9a1484e198fc02928f8de97ca010b83b0f6a79edeb4610c4e`。

`Canonical-D384` 仍只作为高保真 oracle，不是最终紧凑设计。当前 compact checkpoint 约 196 MiB，d384 oracle 约 569 MiB。

本轮另有两个有条件的组件决策：

1. Text bridge：下一轮多场景冻结候选改为 image-disjoint 的 COCO-10000 crop-context bridge；旧 COCO-1000 bridge 只保留为 Ramen development baseline。不能按场景选择 bridge。
2. Query score：registered 2D 继续使用原始 cosine margin；`robust_tanh_zero` 只保留为 `WORLD_3D` 候选策略，绝不设为所有 query 的默认值。

Bridge candidate 为 `output/semantic_bridge/global_region_summary_coco10000_centered_imageholdout.pth`，SHA256 为 `20035e5fa8e5da14de9d2efa92ed92f8f4f54bd2e74006372d407a9a0e9b8e8b`；legacy COCO-1000 bridge SHA256 为 `2f2f09b815d9c5e0b849aa67fc628ae006cb55f077f1335be681d48b831c37ff`。

## 数据使用边界

这些边界决定下面的数值能否进入论文正式主表：

- Ramen 已反复参与结构诊断，是 development scene，不再算无偏测试场景。
- Figurines、Teatime、Waldo Kitchen 已用于一次性 teacher bridge 比较，也已经被消耗为 bridge audit/development scenes。
- ScanNet `scene0000_00` 用于 point-query 设计与 score-normalization 选择，是 development scene。
- NVOS fern 与 SPIn-NeRF fern 目前只是单场景协议诊断。
- 所有正式/诊断 JSON 都保持 `test_calibration=false`；field、bridge、graph、threshold 和 solver 不读取 benchmark mask 来训练或选择。

因此，本文件报告的是受控方法迭代，不是可直接复制到完整 benchmark 主表的最终数值。后续必须冻结全部设计，再在未参与开发的完整场景集合上一次性评估。

## 按附件诊断逐项闭环

| 附件诊断 | 对应检查 | 本轮判断 |
|---|---|---|
| primitive 直接读取与二维 render 读取可能不是同一事实 | held-out render 的 raw / DINO / SAM cosine 为 0.765277 / 0.850628 / 0.683892；Ramen primitive capability 相对 d384 oracle 的 DINO / SAM cosine 为 0.998479 / 0.994620 | primitive truth 已基本保真；二维 compositing，尤其 SAM boundary，仍是未完全解决的上限 |
| 过去的高精度主线是否丢失 MPR | 当前 checkpoint 明确由 `mpr40` cache 训练，MPR 继续使用多视图几何、alpha 和 raster responsibility | MPR 没有被移除，也不是改成独立逐视图 head |
| d384 是否被误当成最终架构 | compact 与 d384 在 NVOS、ScanNet point、LERF3D 上逐项对照 | d384 只作 oracle；compact 在 ScanNet/LERF3D 不低于原 oracle 单次诊断，保留 compact |
| prompt 是否真正进入 primitive-consistent support | SPIn fern full-reference-mask：unary 0.403194 → propagated 0.933798 → connected 0.942710 IoU | registered prompt、primitive unary 和共享 3D support 的断点已解决 |
| 一个 calibration 是否能服务所有 query | compact ScanNet 5-seed 提升；NVOS/SPIn 4-prototype 均轻微下降 | 只能按 query modality 显式配置，不能设成全局默认 |
| multichannel graph 是否应替换原 graph | geometry/appearance/boundary、mutual-kNN 和 typed graph 变体没有稳定提高 mIoU | 拒绝替换；保留固定 legacy support graph |
| query-free semantic capability loss 是否可直接提高文本 | generic semantic cosine 0.861327→0.866410，但 LERF2D mIoU 0.246027→0.246131、Loc 0.816901→0.788732 | proxy 提升没有保持 text margin，拒绝 semanticFT checkpoint |
| official SigLIP2 spatial adaptor 是否已经足够 | 原始二维 teacher 的 Level 1/2/3 分层 oracle | Level 1/2 明显不够；当前仍需要独立声明的 global semantic bridge |

零均值 view residual 只允许修正二维 rendered readout，不能写回 primitive canonical truth，也不能被用来支撑 2D/3D primitive 一致性的主张。

## 当前任务诊断

### Text query

所有 LERF2D 结果使用相同的严格 readout：annotation 原始 query string、模板 `{query}`、四个固定 generic negatives、logit scale 10、temperature 0.1、absolute threshold 0.5、30-pixel bbox-smoothed peak localization、原图分辨率，无 mask refinement、threshold sweep 或 label calibration。

| Ramen / bridge | 2D mIoU | Loc |
|---|---:|---:|
| 旧 COCO-1000 bridge（development baseline） | 0.246027 | 0.816901 |
| COCO-10000 image-disjoint bridge（下一轮冻结候选） | 0.233423 | 0.816901 |
| COCO-10000 full-context variant（拒绝） | 0.194381 | 0.845070 |

LERF3D 固定使用同一 query cache、primitive score compiler、12-step shared support solver、0.5 selection threshold、`selected_only_alpha` 投影和 `png_uint8_gt10` 二值化；没有 GrabCut、SAM refinement 或 sweep。

| Ramen / bridge | mIoU | Acc@.25 | Acc@.50 | Boundary-F | trimap IoU |
|---|---:|---:|---:|---:|---:|
| 旧 COCO-1000 bridge（development baseline） | 0.334770 | 0.591549 | 0.253521 | 0.460724 | 0.435930 |
| COCO-10000 image-disjoint bridge（下一轮冻结候选） | 0.317273 | 0.605634 | 0.154930 | 0.453474 | 0.431175 |

COCO-10000 不能被表述成“在 Ramen 全面提升”：它只提高了 LERF3D Acc@.25，其他 Ramen 下游指标下降。选择它的依据是更严格的 generic split 和跨场景 teacher audit，而不是逐场景挑最好结果。

### Registered 2D prompt

| 数据 / prompt | 协议 | IoU | Pixel accuracy |
|---|---|---:|---:|
| NVOS fern scribble | 官方正负 scribble；raster-responsibility registration；target view 不参与 support；4 prototypes；固定 0.5；score calibration `none` | 0.744900 | 0.913760 |
| SPIn-NeRF fern full reference mask | image000 为唯一注册 reference；其余 19 帧为 target；4 prototypes；固定 solver/threshold；score calibration `none` | 0.942710 | 0.991886 |

SPIn-NeRF 当前不是 SAGA 的 sparse-point 协议，也不是完整 10-scene 可比结果，不能进入正式主表。

### World 3D point

ScanNet `scene0000_00` 每个实例从 GT instance 内按固定随机种子采一个 world point，但 query compiler 只收到该坐标。点通过 anisotropic Gaussian Mahalanobis responsibility 提升到 primitive，appearance/boundary 使用官方 DINOv3/SAM3 capability bank，最后投影到官方 mesh vertices。所有设置和 63 个实例保持不变。

compact field 的五个固定 point seeds 为 0、1、7、42、123：

| Score policy | mIoU mean | mIoU std | Acc@.25 mean | Acc@.50 mean |
|---|---:|---:|---:|---:|
| `none` | 0.305540 | 0.011061 | 0.526984 | 0.171429 |
| `robust_tanh_zero` | 0.309061 | 0.012961 | 0.530159 | 0.184127 |
| 差值 | +0.003521 | — | +0.003175 | +0.012698 |

在 d384 oracle 上方向也一致：mIoU 0.304221→0.308348，Acc@.25 0.530159→0.536508，Acc@.50 0.174603→0.180952。

`robust_tanh_zero` 不读取标签，也不减去 scene median；它只用当前 query 在未标注 primitive 上的 MAD 估计固定 tanh scale，并保持原始零点。它应如实声明为 label-free query-time scene-score normalization，而不能隐藏成“没有后处理”。该公式是在 `scene0000_00` development scene 上选出的，正式评估时必须冻结并同时保留 `none` 对照。

## 为什么 calibration 必须按 query 类型区分

为排除 prototype 数量不一致，NVOS 和 SPIn 回归都显式使用与主协议相同的 4 prototypes：

| Query | `none` IoU | `robust_tanh_zero` IoU | 差值 |
|---|---:|---:|---:|
| NVOS fern scribble | 0.744900 | 0.744514 | -0.000385 |
| SPIn fern full mask | 0.942710 | 0.941991 | -0.000719 |

registered prompt 已有显式正/负 prototype，原始 cosine margin 的零点有直接语义；world point 则只有一个正点和 unlabeled scene-mean negative，score scale 更不稳定。两者不应被强行统一。代码现在支持显式 `score_calibration_by_modality`，默认仍为 `none`，并在 `QueryResult.score_calibration` 中记录实际策略。

当前候选策略是：

- `TEXT`：`none`；
- `IMAGE`：`none`，待正式 image-query 实验；
- `REGISTERED_2D`：`none`；
- `WORLD_3D`：`robust_tanh_zero` 候选，同时报告 `none`。

## Text oracle 与 Bridge v2 系列

### 官方 head 是否足够

在未进入三维场之前，先对原始二维 teacher 做完全相同的 LERF readout。跨 Figurines、Teatime、Waldo Kitchen 共 137 个样本：

| Teacher route | Sample mIoU | Scene-macro mIoU | Loc |
|---|---:|---:|---:|
| Level 1：官方 SigLIP2 spatial adaptor | 0.029272 | 0.036783 | 0.051095 |
| Level 2：官方多尺度 crop visual summary | 0.107753 | 0.128007 | 0.386861 |
| 旧 COCO-1000 global bridge | 0.274875 | 0.276126 | 0.759124 |
| COCO-10000 image-disjoint crop-context bridge | **0.292964** | **0.294732** | **0.781022** |
| COCO-10000 full-context bridge | 0.283907 | 0.286982 | 0.773723 |
| COCO-15000 local-scale full-context bridge | 0.209071 | 0.192840 | 0.729927 |

因此，官方 spatial/crop-summary 仍不足以支持当前 dense text query。global bridge 是独立的 custom semantic component，不能与官方 SigLIP2 adaptor 混为一谈。

### Generic proxy 与下游为何失配

固定旧 bridge 未见过的 COCO 图像，COCO-10000 crop-context 将 generic descriptor selection score 从 0.857926 提高到 0.917492。它同时修复了旧版按 crop 而非 source image 划分 validation 的泄漏风险。

但后续隔离实验表明 generic proxy 不是充分条件：

| Variant | Generic selection score / baseline | Ramen teacher mIoU | Ramen teacher Loc | Heldout3 sample mIoU | 判定 |
|---|---:|---:|---:|---:|---|
| COCO-10000 crop-context | 0.917492 / 0.857926 | 0.213873 | 0.774648 | **0.292964** | 下一轮冻结候选 |
| COCO-10000 full-context | 0.903591 / 0.897334 | 0.181849 | 0.859155 | 0.283907 | 拒绝 |
| COCO-15000 local-scale full-context | 0.817507 / 0.543328 | 0.102581 | 0.718310 | 0.209071 | 拒绝 |
| 3/7/8 token-density crop-context | 0.910245 / 0.906427 | 0.198711 | 0.873239 | 未再打开 | 拒绝 |

local-scale variant 在 generic proxy 上大涨、在所有 LERF 场景上大跌。检查 source token 后，极小 full-image region 实际只覆盖约 1–2 个独立 backbone token，再上采样为 8×8；bridge 学到的是 crop-summary prior，而不是可重建的局部视觉信息。故不构建更昂贵的 primitive cache。

3/7/8 density 实验保持 cache、target、bridge、loss、image-disjoint split 与旧图像 exclusion 不变，只改变 source token pooling。generic 平均只提高 0.003818，Ramen mIoU 相对旧 bridge 下降 0.052021，Loc 持平；按预注册规则停止，不再构建 15×15/225-token cache。固定 token 数量不是当前主要矛盾。

下一步 text 改进不应继续增加 bridge 局部复杂度，而应在全新、未消费的多场景上验证 COCO-10000 crop-context；若仍不稳定，需重新设计真正具有足够二维 token support 的 region observation，而不是在 LERF 上调 bridge。

## 评估协议是否允许这些内部组件

允许，但必须满足并披露以下条件：

- MPR、官方 adaptor、global bridge、graph、solver、query-time normalization 和 deterministic projection 都可以是方法内部；
- 它们不得读取目标 mask 来训练、选 checkpoint、选 threshold 或选 graph policy；
- query text、scribble、point、reference mask 本身是 benchmark 给定输入，可以进入 query compiler；
- 输出域、query 集合、target frames 和最终指标必须与对应 benchmark 协议一致；
- 每个阶段应分别记录 unary、propagated、connected/projected 结果；
- 任何使用 evaluation-scene 无标签统计量的 normalization 都必须明确披露，不能伪装成原始 cosine；
- 不允许逐场景选择 bridge、threshold 或 refinement。

## 关键产物

- compact field：`output/canonical_fields/ramen_canonical_radio_fusion_d256_l128_nohash_mpr40_caploss_renderft256_seed0.pth`
- retained bridge candidate：`output/semantic_bridge/global_region_summary_coco10000_centered_imageholdout.pth`
- legacy Ramen bridge：`output/semantic_bridge/global_region_summary_coco1000_centered.pth`
- heldout3 bridge oracles：`output/semantic_oracles/*heldout3_vala_paper_2d_exact.json`
- Ramen old/candidate LERF2D：`output/lerf2d_compact_d256_l128_caploss_renderft256_primitive_query*/ramen/lerf_ovs_results.json`
- Ramen candidate LERF3D：`output/lerf3d_compact_d256_l128_caploss_renderft256_coco10000_imageholdout_primitive_text_support_k16/ramen/lerf_direct_3d_selection_results.json`
- compact ScanNet 5-seed score audit：`output/audits/query_calibration_v1/scannet_compact_scene0000_seed*_score_*.json`
- registered-prompt calibration regressions：`output/audits/query_calibration_v1/*robust_tanh_zero_4proto/fern_evaluation.json`
- NVOS retained result：`output/audits/compact_query_v1/nvos_fern_compact_d256_l128_caploss_streaming_repeat/fern_evaluation.json`
- SPIn retained diagnostic：`output/audits/compact_query_v1/spin_fern_compact_d256_l128_caploss_full_reference_mask/fern_evaluation.json`

## 尚未完成的正式范围

- LERF2D/3D 尚未用同一个冻结 bridge 与 compact field 跑完全部未参与开发的场景。
- NVOS 与 SPIn-NeRF 尚未覆盖完整官方场景；SPIn 还缺正式 sparse-point 协议。
- ScanNet point 只有一个 development scene，必须在其他冻结场景验证五种 point seed 的总体趋势。
- Pose-free image exemplar query 尚未进入本轮完整 benchmark。
- 二维 SAM render cosine 仍明显低于 primitive fidelity；这是下一轮 representation/rendering 的首要剩余问题。

## 代码验证

- `query_engine.py` 与 bridge trainer 通过 Python 3.9 syntax compilation。
- canonical field、MPR、compositing、render ceiling、capability、bridge split、query compiler、score calibration、support solver、NVOS evaluator 等相关测试共 59/59 通过。
- 仓库完整测试为 810 passed / 3 failed；三个失败来自本轮未修改的 legacy nearest-view 文案断言、`train_feature_field.py` release 行数阈值和 OpenCLIP 显式 frame-id 排序，与 canonical/query 路径无关。
- `git diff --check` 通过。
