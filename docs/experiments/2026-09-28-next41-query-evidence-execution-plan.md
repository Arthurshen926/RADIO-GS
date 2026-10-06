# next41：未知监督一致性与查询身份—范围证据优化

## 0. 状态与执行边界

状态更新（2026-09-28，主助手）：**委派接口已恢复。按用户指定的原始完整请求及同一幂等键，已取得唯一 Run ID `wc_agent_run_x78PcpTl_ASkZeBNB2g1woreoVuE2KeJGxcEsrcE7uU`。最终状态为 completed / completed，stop_reason=end_turn，error_code=null，completed_at=1790535609。代理只完成前置只读核查，未实现任务 A、未测试或评测；completed 仅表示这次代理回合结束，不表示方法任务完成。**

代理终止报告仍引用旧方案中的历史403/无Run ID，并要求主助手提供实际模型回执和原启动状态。主助手已独立取得下方真实回执，但更新未在代理结束前进入其上下文。这是交接未闭合，不是本次再次遭遇启动403/404。主助手没有另建Run，没有用shell或CLI替代coding_agent执行实现。

代理另报告默认沙箱两次只读命令退出1，完整错误为：
```text
bwrap: No permissions to create a new namespace, likely because the kernel does not allow non-privileged user namespaces. On e.g. debian this can be enabled with 'sysctl kernel.unprivileged_userns_clone=1'.
```
其后经代理已有审批机制获准的只读核查成功。主助手未执行上述错误文本中的系统修改建议，也未改变沙箱权限。该错误不能改写为本次coding_agent_start失败。

本次恢复的幂等键为 `radio-gs-next41-collaborative-resume-gpt-6-astra-low-20260927-v1`，不是下文历史 403 尝试的旧键。不要再启动另一代理来“恢复自身”。本次获用户授权的任务 A 代理回合现已结束；保留原键和精确Run身份。再次观察completed不等于恢复执行，不得伪称实现仍在后台推进。

主助手已独立核验真实运行配置（非请求值回显）：委派 app-server PID 913501 的主线程 `01a0e43b-810c-7263-810d-0f36f23f583e`，其 rollout 的 `turn_context`（UTC `2026-09-27T18:58:23.617Z`）记录 `model=gpt-6-astra`、`effort=low`、`cwd=/root/RADIO-GS`；`session_meta.originator=webcodex-runner`。主线程用户任务指令 SHA256 为 `fce9663e3a749227eae0d894b54a9e5741c0d3c8f06e79e1cc350d5317c7a2d3`，与上次保存的完整启动请求逐字一致。证据文件：`/root/.codex/sessions/2026/09/28/rollout-2026-09-28T02-58-20-01a0e43b-810c-7263-810d-0f36f23f583e.jsonl`；仅核验了元数据和任务指令哈希，未修改 provider、模型或权限。

以下 403/无回执记录保留为恢复前历史，不是本次运行的现状；任务 A 的原有范围、文件所有权、先导队列和验收条件完全不变。

本轮用户要求：主助手负责困难方案设计、拆解和独立验收；具体实现/实验由 Eden 本地 Codex coding_agent 执行，模型 GPT-6 Astra，推理 low。不得静默换模型或改成主助手自行执行实现。

- Project：`agent:eden-container:radio-gs`，目录 `/root/RADIO-GS`。
- Workflow Session：`wc_sess_KjK5rkiWA-XB3zzh`。
- 观测 HEAD：`dabcaa9b5fc6d89fa4474bcaea5db52ca7f74030`；分支 `codex/surface-object-memory-v4`；原工作区已有大量修改，必须保留。
- 最新源记录仍为 next40；next38 是本轮完整任务配方共同对照。
- 已实际尝试 `coding_agent_start`，provider=`codex`，请求 config 为 model=`gpt-6-astra`、reasoning_effort=`low`。这些是请求值，**没有收到 provider 接受或实际使用的确认**，不能当作已验证的模型 ID。
- 同一执行意图的 idempotency key：`radio-gs-next41-supervision-KjK5-01`。恢复时先查这个精确启动记录；不要在未知结果时创建替代任务。
- 委派启动返回 HTTP 403，support_id=`cmd_33ad5b6f_012f_4f7e_883e_581f711797bd`，没有 Run ID。另一次 durable-goal admission 也返回 HTTP 403；没有已创建 Goal 的证据。
- 聚焦 runtime 显示 Eden 在线、Codex provider 已注册；本会话 list_jobs 为空。GPU 观测为 6 张 RTX4090、每卡约 1 MiB 占用、0% utilization。硬件可用不等于接口授权成功。
- `coding_agent_start` 工具声明需要 `coding_agent:run` 与 `project:write`。403 原因未细分，需检查 WebCodex 服务侧授权/网关策略，不能断言只缺某一个 scope；不改权限、不借 shell 绕过委派权限限制。

## 1. 研究目标及不可变条件

目标不是再做一次全局平滑，而是分离并处理：

1. 缺失检测保持未知时，模型选择不能把未知区预测计成假阳性。
2. 查询身份证据要直接约束范围选择，不能仅让范围解释已有场分数。
3. 概率图、实际元素选择、类别竞争、其他查询能力必须一起验收。

固定：几何、原生编码器、原 next38 底座、合法 source 数据权限、benchmark scorer/阈值/分母。原 checkpoint、历史 report、seal 不覆盖；新结果进入独立 next41 目录。不向地图构建反灌 benchmark 查询/GT，不读取 NVOS target RGB 或新增负提示，不恢复 UQIS，不按场景或 benchmark 拼接赢家。底座已看过的 source views，只能叫新增头的留出视角，不能叫独立地图/最终测试。

## 2. 共同对照

来源：`output/taskwise_scope_composition_next38/{recipe,execution,summary}.json`。

| 指标（%） | next38 |
|---|---:|
| LERF4/208 text IoU / AP | 52.0258 / 69.2143 |
| LERF selected-only projection IoU | 36.0223 |
| ScanNet8 19/15/10 mIoU | 54.7502 / 55.7542 / 69.7506 |
| NVOS8 IoU / AP | 81.2693 / 91.5393 |
| LERF reference image IoU | 45.8996 |

同时报告 next40 修正 odds 的 52.1397/68.9263/36.0022 与 ScanNet 57.0272/56.7982/68.8205。next40 不是逐指标优于 next38，不能替换共同父对照后隐藏退化。其 reference-scopes NVOS 81.4939/93.6034 属于另一显式分支，不合并成虚构统一结果。

## 3. 任务 A：修复部分监督的模型选择约定

### 已定位的具体机制

`scripts/research/fit_source_kernel_extent.py`：81–100 行构建 known；149–151 行训练仅在 known 上计残差；但 129–141 行 measure 及 169–181 行 ridge 选择的 IoU 没有一致限定 known，未知预测进入 union。这是源模型选择问题，不是已证明的 benchmark scorer 错误，真实指标影响待测。

### 数据和指标定义

对每个 source query-view episode：

- V：可见且具有语义支持的像素；K：确认正或 qualified negative 的像素。
- P=V & K & target；N=V & K & ~target；U=V & ~K。
- known IoU 只在 P∪N 计 TP/FP/FN；positive recall 只在 P 计；qualified negative FPR 只在 N 计。
- 对 U 只报告预测质量、选择比例、相对父响应的变化，不称其 TP/FP。
- 没有有效监督的 episode 不伪造 0 或 1；报告不可用及覆盖分母。只有正或只有负支持时，仅计有支持一类并标注。

主要源选择标准预先固定为 class-balanced known Brier：对 P 和 N 各自求概率平方误差，再对存在的两类等权平均。以 query-view 等权聚合，同时报告类支持计数及 known IoU。固定 ridge 集合仍是 {0.1,1,10}。误差在 1e-6 内视作数值平局，优先更大 ridge；未知区变化仅作已声明的保守平局依据，不作未知真值。全可见域旧 IoU 保留为 legacy 诊断，不决定 partial-known 候选。

### 实现限制

保留显式 `legacy_visible_iou` 路径用于旧结果重放；新增 `partial_known_brier`，不得改未标记旧 payload 的解释。训练/选择均使用实际部署 odds 修正及相同 projection/support。不要只修日志而遗漏实际 ridge selection。

同一次 60-step pipeline/odds 拟合产生相同 residual，导出两种源选择规则的候选；不得用两个不同训练随机过程伪装选择规则消融。可复用已保存且绑定一致的 residual，不能凭路径相似复用。分开测量 partial-known 指标修复和选参规则改变，不把两者的联合效应称为单一损失函数收益。

### 文件所有权与执行

Agent A 仅修改 `scripts/research/fit_source_kernel_extent.py`，新增 `radio_gs/training/partial_scope_metrics.py`、`scripts/research/run_partial_scope_next41.py`、必要直接测试、`output/partial_scope_next41/`。不改根文档、其他 agent 模块、历史资产和 scorer。

初始固定三个先导：lerf_figurines、lerf_ramen、scannet_scene0000_00。使用 GPU0，运行前再次确认空闲，不杀外部进程。Python：`/root/miniconda3/envs/cybersim_agent/bin/python`。真实调用现有 evaluator，记录 text IoU/AP/selected 和 ScanNet 三类集结果、选择 ridge、source coverage、未知变化。即便指标不涨，正确性修复仍可保留，但不得因此提升模型或宣称解决召回。

必要测试仅覆盖未知不影响 known 统计、不可见目标排除、有依据负例仍计错、空支持不造分、zero-delta odds 恒等及旧路径重放。禁止花主要预算新增审计框架。

产物：每臂 training/evaluation、`report.json`、`notes.md`、精确命令、源码差异和未完成事项。

## 4. 任务 B：独立身份证据，不再只拟合旧响应

### 需要检验的假设

`ScopeComposition.refine` 通过旧 scores 间接依赖查询，主要优化 ||M a - u||，固定 u≤0.25/u≥0.75。它能整理模糊响应，却无法对被固定的错误身份自行提供反证。新模块必须使用旧 u 之外的合法 source-region 证据；去掉 fixed_cores 本身不是新方案。

### 设计：源区域前景—上下文对照证据 + 部分标签身份/范围联合学习

先检查现有 crop teacher 是否已经包含同等对照，防止重复实现。固定源帧、相机、几何、原生 encoder，保留每个源 mask 的区域身份和单视角来源，分别提取：原 crop、保留 mask 内内容的 crop、抑制 mask 内内容的 context crop。替换像素方案必须固定并记录；这种干预会带来分布偏移，是待验证证据而不是真实身份标签。

查询 q 的区域关系量包括原 native similarity、foreground-vs-context response difference、跨视角同一范围的一致性/分歧、可见贡献、独立 mask support。不将整个 crop embedding 作为每个元素标签，不把同名区域或单向包含视作同实例；面积只描述尺度。对无法确认整体/部件/同类实例的样本保留关系未知。

候选范围 M 来自已有连续逆渲染范围与未匹配局部证据；不把缺对应的局部证据删除。保持多个可能范围，类别并集允许多选，reference-instance 使用授权参考锚点，部件与整体只能在有关系证据时约束。未有类型证据时不得自动生成物理 part 标签。

用一个共享、低容量、具有显式 task_type 输入的 compatibility learner 对上述关系量给出 signed scope evidence；先从可解释线性/小型低秩关系模型开始，而不是再加自由 query encoder。参数在 source query-view episodes 上学，监督为 (P,N,U)，预测通过固定 R_v 与范围 M 回到 source 像素。scope coefficients 由 query-region evidence 决定，旧 u 只是先验，不是训练目标或真值。

元素响应可写为 p(q)=sigmoid(b_parent(q)+M a_theta(q,E,task))。这只是候选参数化：须让正、负修正都具备独立的 source 支持，对没有新证据的元素严格回到 parent；不将所有高置信 parent 永久固定，也不将未知当成反证。先保持 text-only 变化，image/reference prompt 路径精确回放；以后扩展不能直接复制 text 监督。

学习目标使用可见 P/N 的概率重建、困难兼容/不兼容查询的 ranking、来源独立的范围一致性和 parent trust-region。source relation 未确认时对应项关闭；不以 loss 下降替代身份正确。根据实际评价算子检查 visible selected projection，同时单独保留 historical selected-only score；不对合法隐藏表面施加二维背景负例。

### 有辨别力的最小对照

B0：相同 budget 的 parent/旧 response-composition。
B1：相同来源和范围，仅使用原 crop 语义关系。
B2：加入 foreground/context 对照，其他容量/训练条件相同。
B3：在 B2 之上加入实际可确认的范围关系监督；没有足够有效关系时报告阻断/低覆盖，不伪造该对照。

不开展大规模 loss/threshold sweeps。先检验新证据对困难源样本是否有信息增量、region/query margin 和可见范围是否一起改善。只有源准入后才把一个固定候选扩到完整队列；先导 benchmark 看过后的修改全部记为开发选择，不叫独立 test。

### 文件所有权

Agent B 使用独立 `radio_gs/querying/query_region_evidence.py`、`scripts/research/build_query_region_evidence_next41.py`、`fit_query_region_extent_next41.py`、`run_query_region_extent_next41.py` 和 `output/query_region_evidence_next41/`；不得同时编辑 A 文件。用 GPU1/2，源 cache 只读，新增证据单独目录。需要公共 evaluator 接入时先给主助手提交最小 diff 需求，不与其他 agent 同写。

## 5. 任务 C：参考身份约束的可解释纠错

这是 B 的不同任务验证，不与 text 分支拼分。现有 `reference_scope_completion.py` 只增加正证据，无法纠正已选假阳性。可研究在授权 reference 上共同选择相容多范围，再仅对独立已确认冲突域降低 parent；reference 外未覆盖部分维持未知，不能整体压为背景。

先报告多少目标具有可确认的排除证据。输入仍仅参考正提示/SAM；不得凭空新增负 scribble。没有可靠负证据时只做正向补全，不承诺解决原 FP。保留 next38 与 next40 reference-scopes 全 NVOS8 对照。此任务依赖 B 的证据接口稳定后执行，不先并行写其共享模块。

## 6. 独立验收，不由 coding_agent 自行宣布成功

主助手负责检查实际 diff、模型配置回执、输入权限、同父模型绑定、少量直接测试和原始预测文件；自行复算汇总，不只接受 agent 最后一句总结。

验收分层：

1. 正确性：源未知不当负；部署/训练算子一致；有效域分母完整；没有读取目标 GT 调参。失败则不得继续解释性能。
2. 机制：新证据对困难身份/范围样本有可复现信息增量；去掉证据后收益消失；不依赖只增加维数/预算。
3. 开发效果：完整 LERF4/208 概率 IoU、AP、定位、selected；ScanNet8 19/15/10；NVOS8；不删失败场景。image 保留路径的 exact replay 只证明未改动，不称已升级。
4. 统计：固定种子先完整跑，随机训练部分至少复核若干种子；按 scene 聚类报告差异/区间。LERF只有4场，不能靠208观测伪造208独立样本的显著性。
5. 采用：没有整体稳定收益就保留为 opt-in 研究候选。源修复可独立接受，checkpoint 不自动 promotion；任何小幅 tradeoff 必须逐项展示。不可要求事后调整阈值让新方法过门。
6. 成本：总 scene state 包括96D、codebook、稀疏范围、query-kernel residual、encoder依赖；报告新增字节、峰值显存、离线时间与单query延迟。不把 N×Q 的源query残差藏在96D名义下。

真实全系统语义身份仍可能受伪标签可辨识性限制。若同一证据支持多个互斥解释，需要新独立观测或明确 source-side 人工标注；assistant-reviewed mask 仍是诊断伪证据，不能改称人类GT。

## 7. 恢复后的顺序与当前交付

顺序：先确认服务侧允许 coding_agent_start 及 live-advertised GPT-6 Astra/low 配置 → 恢复同一精确启动意图 → A 实现并给真实对照 → 主助手审阅 A；B 可独立开发但不改 A 文件 → 选择一个源准入方案完整评测 → 主助手验收并更新执行报告。

原方案落盘时只新增本设计文件，模型、训练脚本、默认方法和历史结果未由当时那轮更改。现已完成上方注明的委派启动和实际配置核验；A/B/C 的实现、训练、GPU评测、收益和最终采用仍需分别用真实产物验收。本文件不能引用为已取得 next41 指标。
