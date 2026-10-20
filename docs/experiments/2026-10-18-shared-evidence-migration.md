# 公共证据核心：第一阶段实现与验收

日期：2026-10-18。分支：`codex/tmm-query-response`。依据：[ADR0042](../adr/0042-shared-evidence-core-and-five-output-migration.md)。

已完成第一版代码迁移、覆盖全场景的组件检查及受存储预算约束的重评入口。
**尚未完成五项新推理和性能验收，也没有证明精度已经不下降。**
当前没有 GPU 同时满足利用率不超过 10% 和至少 16 GiB 可用显存的启动条件。
没有启动后台排队任务，也没有改动其他 GPU 进程。

## 已落地的计算边界

`radio_gs/evidence_core/` 提供真实共用的运算，不接收 benchmark 名称：

- `compile_domains`：支持、排除、未知和不可见分开；声明负证据来源；拒绝同时断言正负。
  `unknown` 撤销两种断言，`exclusion_veto` 只撤销负证据，避免被拒绝候选抹掉独立成立的正证据。
- `signed_weights`：保留适配器原权重，不把身份、可见性和质量分数解释为可直接相乘的正确率。
- `project`：共用稀疏传输，保留原稀疏格式、计算次序和像素域。
- `grouped_membership_features`：保留同视图嵌套掩码、共同可见性及分歧，不把重复掩码当新增视图。
- `scope_response`：保留原区域范围策略及可见补集排除。
- `EvidenceStore`：只读接入现有 UQIS 地图，校验区域、视图、载体对应，不复制地图。

| 输出 | 共用运算 | 保留的能力 |
|---|---|---|
| LERF 2D | 正负域、权重、反投影 | 连续语义响应、原生身份及无证据处的原响应 |
| LERF 3D | 最终正负域、跨视图稀疏传输 | 真实选集渲染及原优化目标 |
| ScanNet OVS | 可见域、正负域、稀疏累计 | 区域内类别竞争、完整菜单、原双教师基线 |
| NVOS v10 | 正负域、权重、渲染响应重建 | 像素证据、原迭代次序和无支持处的任务策略 |
| UQIS | 只读地图、视图分组、范围响应及投影 | 15 维成员输入、学习式成员头、图像与点输入许可 |

候选身份判断和权重策略仍由现有适配器提供；尚未完成所有候选准入策略的迁移。
N14 的参考修复及目标图像条件后端尚未完成本轮接入，当前 NVOS v10 路径不等同于 N14。
默认仍是 `legacy`，迁移配置显式启用 `shared_v1`。
迁移前 15 个源文件及配方保存在 `docs/history/2026-10-18-evidence-core-baseline/`，带 SHA256。
历史分数没有替换为新结果；旧实验源码哈希不代表当前修改后的代码已获新性能验证。

## 已做的验证与限制

封存报告：[组件审计 v2](2026-10-18-shared-core-component-audit-v2.json)。
31 个场景／任务全部通过，226 项比较的数组形状、dtype、元素值完全一致。
报告绑定 220 个源码文件和 142 个实际读取的输入文件。

- LERF 4 景、NVOS 8 任务：每项抽两帧实际源掩码，验证正负域及权重；可见域使用测试样本。
  尚未重跑实际高斯渲染、最终选集和评分器。
- ScanNet 10 景：每景两帧实际区域与几何贡献、完整类别菜单，检查源证据累计、特征和双教师融合。
  原生先验使用固定测试值，教师编码器及评分器未重跑。
- UQIS 9 景：每景三个确定性锚点，检查抽取载体上的 15 维特征及原成员头输出；
  文本范围运算检查完整载体。图像／文本身份编码器未重跑。

公共核心开启后，注册研究套件 **209 passed**，两条已有依赖警告。
新增测试检查未知域、重复视图、稀疏后端、旧求解器逐元素一致、不完整队列拒绝、核心调用轨迹，
以及预算超限时仅释放本轮临时文件。另有协议／迁移组合测试 45 passed，与研究套件存在重叠。
这些结果证明所检查组件的迁移一致性，不是新的 benchmark 精度。

## 新重评入口与输出规则

配置：`configs/research/shared_evidence_migration.json`。
入口：`scripts/research/run_shared_evidence_migration.py`。
预检：`output/shared_evidence_20261018/shared_v1_preflight_v2/manifest.json`。

入口固定完整 31 项任务，拒绝缺景或重复任务；全部预测完成后才评分。
ScanNet 同一批新预测分别报告十景 mesh 回归和正文八景 Gaussian 协议，保留 19/15/10 类。
Gaussian 评分使用原 Mahalanobis 伪标签及 opacity-volume 权重；仍含提供的未标注 mesh 输入，
不声称与 VALA 的 RGB 建场输入完全等价。
每项记录实际核心调用和查询时间；完整建场成本与峰值显存计量待补齐。

- 既有地图、几何、教师缓存只读引用。
- 稠密输出放在独占 `/dev/shm/radiogs-core-*`；每 5 秒检查 40 GiB 阈值，保留 4 GiB 空间。
  这是监测阈值，不是内核硬配额。
- 成功、失败、超预算后均释放本轮临时目录，不清理共享输入和历史资产。
- 每轮持久化上限 32 MiB，单文件 8 MiB，整个迁移研究 256 MiB。
  仅保留配置／源码哈希、结果、日志、调用轨迹、最终预测哈希。
- 配置及已有资产声明已绑定，抽样读取文件有内容哈希；全量教师缓存的独立内容封存待完成。
  目前不宣称拥有完整封闭的原始字节复现包。

GPU 可用后，同配方分别执行两条完整回归，不同时更换教师：

```bash
cd /root/RADIO-GS-query-field
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_shared_evidence_migration.py --run legacy_full_v1 --arm legacy --execute
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_shared_evidence_migration.py --run shared_full_v1 --arm shared_v1 --execute
```

run 名称不可覆盖；源码变化使组件审计过期时拒绝执行。本次只运行预检，完整评估尚未启动。

## 后续收敛顺序

1. 完成五项新推理，逐场景／任务检查新旧差异、真实选集、AP、UQIS 四路一致性及成本。
2. 接入 N14 完整强配方，在相同正负涂画、图像权限、传播与后端下隔离 DINO 参考修复贡献。
   v10 回归不能替代 N14 能力迁移。
3. UQIS 图像在合法查询裁剪上增加 SAM 净化，对比现有 SigLIP2+DINO；不读取隐藏原帧或位姿。
4. 分开测 ScanNet 单 SigLIP2 替换，以及 LERF 跨视图对应改善。
5. 五项统一回归通过后再改默认和论文主配置；不逐景拼最高分。
