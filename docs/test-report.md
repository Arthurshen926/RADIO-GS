整体定位
这个仓库现在可以理解成一个独立出来的 RADIO + 3D Gaussian Splatting 研究线：它不是做 RGB 重建，而是把冻结的 RADIO 1280d foundation features 蒸馏进 3DGS 场景表示里，再从新视角渲染出仍然可用于 depth / segmentation / grounding 的特征图。
先看哪些文档
文档的“可信度”建议这样排：
1. 代码本身
   radio_gs/scripts/train_feature_field.py
   radio_gs/rendering/feature_renderer.py
   radio_gs/models/hybrid_gaussian.py
   radio_gs/models/screen_refiner.py
   radio_gs/data/benchmark_paths.py
2. README.md
   仓库入口文档。最适合快速知道仓库做什么、主要脚本是什么、推荐工作流是什么。
3. docs/current_algorithm_framework.md
   这是当前最完整的方法说明文档，适合理解“主线算法 + 代码映射 + 研究目标”。
4. docs/feature_reconstruction_analysis.md
   更偏“特征蒸馏方法解释”，适合理解为什么要做 feature reconstruction。
5. docs/submission_status.md
   讲论文包装成熟度，不是实现文档。
6. docs/benchmarking_plan.md
   讲论文主表该和谁比。
7. docs/ablation_execution_plan.md
   讲当前实验队列和消融优先级。
读文档时要注意
有几处“文档叙述”和“当前代码现实”不完全一样：
- refiner 在代码里是作用在紧凑/latent feature 上，不是在最终 1280d decode 之后再修。
- grounding 的辅助训练不是所有数据集都全开，当前 query-level grounding aux 主要还是 Replica 路线。
- 一些较早文档把 bottleneck 说成 32-64d，但当前主线配置已经大量使用更大的 bottleneck，例如 192d。
所以最稳的原则还是：代码 > README > docs/current_algorithm_framework.md > 其他 docs。
算法主线
当前主线算法可以概括成一句话：
冻结 RADIO 教师特征 + 冻结 3DGS 几何骨架 + 可学习 hybrid feature field + 屏幕空间 refiner + HCD decoder + 多任务辅助监督
展开后是这几层：
1. 几何骨架
   用预训练好的 3DGS PLY 作为场景几何，不重新学几何拓扑，主要提供投影关系、几何深度、alpha/visibility。
2. 教师特征
   每帧 RGB 先离线送进冻结的 RADIO 编码器，得到 1280d 教师特征，存成 .pt 文件。
3. 特征场主干
   当前主线不是简单 explicit Gaussian feature，而是 hybrid。
   它由两部分组成：
   - per-Gaussian latent，负责 fine/local 细节
   - 3D spatial hash field，负责 coarse/global 结构先验
4. 屏幕空间渲染
   FeatureFieldRenderer 用 gsplat 把 Gaussian latent 渲染到当前视角，输出：
   - feature_map
   - depth_map
   - alpha_map
5. 局部修正
   渲染后的紧凑特征先过 FeatSharp3D，再可选过 ScreenSpaceRefiner，利用 RGB、depth、depth gradient、alpha、boundary 等 guide 修边界和局部结构。
6. Hybrid 解码
   HybridFeatureGaussian 会把：
   - 屏幕空间 latent fine 分支
   - 由深度反投影得到的 3D position map 送入 hash field 得到的 coarse 分支
   融合成最终紧凑特征。
7. HCD codec
   最后再通过 HCDCodec.decoder 把紧凑特征恢复到 RADIO 1280d 空间。
8. 多目标监督
   当前训练不是只做 feature reconstruction，还会联合：
   - L2 / cosine distillation
   - TV / gradient / depth-guided / geometric-edge / boundary-aware loss
   - depth aux
   - segmentation aux
   - frozen depth head supervision（FDH）
   - frozen segmentation head distillation
   - SigLIP 对齐
   - grounding aux
实现流程
如果按“程序是怎么跑起来的”来看，主流程是：
1. 准备几何
   Replica 用 train_rgb_gs.py，LERF 用 train_colmap_gs.py。
2. 抽取教师特征
   extract_radio_features.py 把每帧 RADIO 特征离线存盘。
3. 解析路径与 split
   benchmark_paths.py 负责兼容 Replica / ScanNet / LERF 的 pose、frame id、rgb/depth/semantics、feature 路径。
4. 组装训练样本
   train_feature_field.py 里的 SimpleRadioDataset 读取：
   - teacher features
   - pose
   - optional depth
   - optional semantics
   - optional RGB guide
5. 渲染当前视角特征
   FeatureFieldRenderer 用当前 pose 从 Gaussian 场渲染出 compact feature/depth/alpha。
6. 做 hybrid 与 refiner
   先 FeatSharp3D，再 ScreenSpaceRefiner，然后走 hybrid fine/coarse/fusion。
7. 恢复到 1280d
   通过 HCDCodec 解码回 RADIO feature space。
8. 计算所有 loss 并更新
   更新的通常是：
   - Gaussian latent / feature params
   - hash field
   - fine/coarse decoder
   - fusion head
   - codec
   - refiner
   - 训练期辅助 heads
   冻结的通常是：
   - RADIO encoder
   - 几何骨架
   - frozen heads
   - 部分 SigLIP 对齐模块
9. 评测与可视化
   - eval_rendered.py：综合 depth/seg/geom/fused/direct-head 评估
   - eval_grounding.py / eval_lerf_grounding.py：grounding
   - generate_visualizations_v2.py：定性图
代码阅读顺序
如果你要接手代码，建议顺着这个顺序读：
1. README.md
2. docs/current_algorithm_framework.md
3. radio_gs/scripts/train_feature_field.py
4. radio_gs/rendering/feature_renderer.py
5. radio_gs/models/hybrid_gaussian.py
6. radio_gs/models/screen_refiner.py
7. radio_gs/data/benchmark_paths.py
8. radio_gs/scripts/eval_rendered.py
整个项目现在是什么状态
分两层看。
第一层是“论文成熟度”。
按 docs/submission_status.md 里的自我评估，这个项目当前属于：
- 强研究原型
- 部分论文包
- 距离完整 top-conference submission 大约 55%-60%
这意味着：
- 方法已经成型
- 训练/评测链路已经比较完整
- 但跨域、seed、效率表、最终冻结叙事还没完全收口
第二层是“当前工程推进到哪了”。
现在不是在发散想新方法，而是在做一轮很关键的 paper-critical rerun：
1. 已完成的关键工作
   - 找到并修掉了 3DGS depth 输出与 trainer 侧 shape 不一致的主 blocker。
   - 训练报告链路已经补强，开始系统保存 manifest/report/history/failure/visualization。
   - eval_rendered.py 也补上了结构化 JSON 落盘，并修了一个 split/pose 兼容问题。
2. 当前正在跑的主线实验
   现在核心是 5 个 nofdh_240ep 基线在重跑：
   - room0_hybrid_v14_nofdh_240ep
   - lerf_figurines_v14_nofdh_240ep
   - lerf_ramen_v14_nofdh_240ep
   - lerf_teatime_v14_nofdh_240ep
   - lerf_waldo_kitchen_v14_nofdh_240ep
3. 这些 run 现在都还没 finished
   marker 目录里目前还是：
   - room0_nofdh.done.lock/
   - figurines_nofdh.done.lock/
   - ramen_nofdh.done.lock/
   - teatime_nofdh.done.lock/
   - waldo_kitchen_nofdh.done.lock/
   这表示 wrapper 正在持锁运行，还没落成 .done。
4. 我刚核到的实时进度
   - room0: epoch 22，best val cosine 约 0.8865
   - figurines: epoch 160，best val cosine 约 0.7427
   - ramen: epoch 236，best val cosine 约 0.7840
   - teatime: epoch 134，best val cosine 约 0.7089
   - waldo_kitchen: epoch 203，best val cosine 约 0.7053
5. 当前意味着什么
   - ramen 已经接近 240 epoch 终点
   - waldo、figurines、teatime 在中后段
   - room0 还相对早，所以整条队列真正的拖尾大概率还是 room0
项目进入到哪里了
如果按阶段看，现在大致在这里：
1. 方法设计阶段
   已完成。
2. 主干实现与关键 bug 修复阶段
   已完成，核心训练/渲染/评测链路都能工作。
3. paper-critical baseline 重跑阶段
   正在进行。
   现在就在这一阶段。
4. 下游自动解锁阶段
   还没真正进入。
   但等待脚本已经挂好了，等 nofdh.done 一出现，就会自动触发后续：
   - room0_pure_frozen
   - room0_pure_frozen_depth_only
   - 4x LERF pure_frozen_depth_only
   - 4x LERF fdh_ws240_240ep
   - auto-eval
5. 汇总表格 / submission artifacts 阶段
   还没到。
   这一步要等上面那些训练和 auto-eval 真的出完结果。
一句话总结当前局面
这个项目的方法层面已经不是“还在搭框架”，而是一个比较完整的 foundation feature rendering system；但工程推进层面，现在正处在“把 paper-critical 基线重新跑稳并产出可追溯结果”的阶段，还没进入最终的 auto-eval 汇总和 submission freeze。
如果你愿意，我下一步可以继续给你一份更“接手导向”的版本：
1. 只讲代码结构和模块依赖
2. 只讲实验管线和输出目录
3. 按“新成员 onboarding 文档”格式重写一版