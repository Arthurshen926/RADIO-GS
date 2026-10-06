# 多粒度原生语义场：完整开发集结果（2026-09-27）

结论：恢复小区域证据、分开保存不同粒度、在原有语义方向上作有界补充，带来了完整 LERF 四场景的实际提升；ScanNet 八场景总体仍退化。该方法保留为研究候选，不替换跨 benchmark 主线。所有本轮启动的构建、400 步先导训练和完整评估均已结束。

## 实现改变

1. 恢复原先被面积阈值过滤的小型源侧 SAM 区域，并重新编码原生 SigLIP 特征。仅使用已有允许的源图。
2. 每个源像素分别保留覆盖它的最小、中位、最大面积区域。三层独立通过冻结几何的渲染责任权重传递到元素，避免整体与细节在写入前直接平均。面积排序不是经过验证的包含树，也不证明跨视角对象身份。
3. 源训练区域形成 512 个原生 1536 维球面码字，每元素每层保留 4 个码字及其权重。查询以原生编码匹配码字；源图、源掩码不参与查询阶段。
4. 保留原有 64 维语义方向和 32 维结构特征。新语义仅加入与原方向正交的分量，范数上限 0.25，然后归一化。查询与源侧码字的关系决定三层读出权重。保护的是原有状态和主方向，不是最终排序严格不变。
5. 继续使用现有范围头和输出方式；新响应的阈值修正仅由源侧伪掩码拟合。统一两轮局部搜索和 ridge=1，不按目标场景指标选择超参数。image/prompt 分支未接入该扩展。

代码：`radio_gs/models/native_scope_dictionary.py`；`scripts/research/{prepare_small_scope_evidence,build_scope_dictionary,train_scope_dictionary,calibrate_scope_dictionary,evaluate_scope_dictionary}.py`。

容量：原有场之外新增每元素 48 字节的序列化索引/权重，以及共享 512×1536 fp16 码本（约 1.5 MiB）等元数据。运行时转 float/long 的内存更大。原型仍显式依赖原有场文件，不是同容量、单向量部署替换；ADR 0009 已说明这个实验性偏离。

## 完整结果

下表单位为百分比，按场景等权平均。基线来自同次执行。几何、评估帧、GT、输入权限和评分器保持相同；新方法增加了语义存储和小区域编码预算。开发集已经反复用于研究选择，不能称为独立 test 或 SOTA 证据。

| 指标 | 同次基线 | 分层有界读出 | 加源侧标定 |
|---|---:|---:|---:|
| LERF2D IoU | 49.7408 | 50.9411 | 51.1248 |
| LERF2D AP | 68.4795 | 69.2259 | 69.2707 |
| LERF2D 定位准确率 | 81.2684 | 83.5781 | 83.5781 |
| LERF 元素选择投影 IoU | 33.5676 | 35.1964 | 35.5784 |
| ScanNet OVS 19 类 mIoU | 54.5182 | 54.3823 | 54.5110 |
| ScanNet OVS 15 类 mIoU | 55.6658 | 55.5092 | 55.5740 |
| ScanNet OVS 10 类 mIoU | 69.8154 | 68.8653 | 69.1482 |

LERF 为四场景 208 个查询-视角观测。元素选择投影使用既有 selected-only 渲染和 PNG 评分协议；历史称为 LERF3D，但此处没有独立物理 3D 真值。ScanNet 使用既有 provided-unlabelled-mesh 轨道与 VALA Gaussian 伪真值分配/体积加权协议。其结果不等于 RGB+MoGe 输入轨道，也不能直接混比其他协议。

| 场景 | 基线 | 分层+源标定 |
|---|---:|---:|
| lerf_ramen | 35.353 / 19.951 | 36.205 / 22.302 |
| lerf_teatime | 60.548 / 46.970 | 60.661 / 46.935 |
| lerf_waldo_kitchen | 53.470 / 37.692 | 56.044 / 42.144 |
| scannet_scene0062_00 | 70.901 / 70.834 / 78.235 | 71.846 / 71.864 / 80.541 |
| scannet_scene0070_00 | 37.876 / 41.297 / 62.683 | 36.963 / 40.246 / 60.219 |
| scannet_scene0097_00 | 63.440 / 60.327 / 68.896 | 62.268 / 58.497 / 65.192 |
| scannet_scene0140_00 | 48.614 / 55.257 / 72.495 | 48.163 / 55.063 / 72.456 |
| scannet_scene0347_00 | 58.626 / 56.003 / 74.105 | 63.240 / 60.760 / 73.519 |
| scannet_scene0400_00 | 67.355 / 67.475 / 68.887 | 66.298 / 65.923 / 67.859 |
| scannet_scene0590_00 | 41.905 / 49.273 / 71.686 | 40.069 / 47.872 / 71.609 |
| lerf_figurines | 49.592 / 29.658 | 51.589 / 30.932 |
| scannet_scene0000_00 | 47.428 / 44.860 / 61.538 | 47.242 / 44.367 / 61.791 |

LERF 行为 2D / 选择投影 IoU；ScanNet 行为 19 / 15 / 10 类 mIoU。全部场景使用同一候选配方，没有按场景拼接获胜分支。NVOS 的图像/交互路径未改变，本轮没有新 NVOS 提升或复测结论；UQIS 继续暂停。

## 失败分支和核心解释

- 直接以新码本替换原有语义，以及对多层响应取高值，没有保留原有有效语义；figurines 原生 AP 从 67.21% 降到 53.92%。保留原方向的有界补充更有效。
- 400 步源像素响应重建训练不能自动解决跨视角冲突：分层先导在 figurines 的原生 AP 从未训练的 68.73% 降到 66.79%；ScanNet 先导部分分类指标改善。完整候选统一使用未训练的分层融合，没有逐场景挑训练版。
- figurines 的 waldo、门把手等细粒度目标有改善，但带救生圈的鸭子退化，例如 frame_00195 的 IoU 63.97%→49.58%。粒度分离确实保住部分细节，尚未解决对象范围相容性。
- 源侧标定使 LERF 2D 与实际选择进一步改善，但没有恢复 ScanNet 的全局增益。因此阈值失配不是全部原因，继续搜阈值无法替代语义对应修复。
- 当前同一层仍会平均不同视角的证据；每像素的面积次序会因遮挡、裁切变化，同一层不等于同一对象或部件。查询的层权重主要依赖全场码字统计，也没有逐元素验证哪一层具有可靠对应。这是实现中可直接指出的剩余机制缺口；本轮尚未量化它们各自的责任比例。

## 下一步聚焦

优先让局部元素的粒度读出受跨视角相容证据约束：只聚合相互支持的区域，将包含关系与冲突关系分开，未覆盖保持未知；不要仅凭相同面积层或高文本相似度合并。先固定当前码本和几何，比较逐元素、逐视角的一致性加权与当前平均，使用未参与对应决策的源视角选定方案，再完整评分。源监督同时观察概率图和实际所选元素；停止把降低教师响应重建误差当作召回能力提高。

现阶段结论是 LERF 获得有限、可复现的候选收益，ScanNet 仍未解决；没有达到可投稿的跨任务 SOTA 状态。

## 复现入口与结果

先导两个场：`output/scope_hierarchy_next33/{lerf_figurines,scannet_scene0000_00}`；其余十场：`output/scope_hierarchy_full_next33/<key>`。每场 `anchored/pipeline/evaluation.json` 是未重标定读出，`anchored/pipeline_calibrated/evaluation.json` 是源侧重标定读出。汇总：`output/scope_hierarchy_full_next33/summary.json`。

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/calibrate_scope_dictionary.py --key lerf_ramen --base output/scope_hierarchy_full_next33
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/evaluate_scope_dictionary.py --key lerf_ramen --base output/scope_hierarchy_full_next33 --readout anchored --pipeline --calibrated
```
