# 实验评估协议冻结与历史清理决策单（2026-08-01）

本轮没有删除、移动或覆盖任何历史资产。唯一的正式选择器是
`evaluation_protocol_freeze_20260801.yaml`；详细行仍保存在两个 registry，
本清单只供决策，机器版为
`evaluation_protocol_cleanup_inventory_20260801.yaml`。

## 已冻结的七条路线

| 任务 | Canonical 方法/范围 | 固定结论 |
| --- | --- | --- |
| LERF-2D | OccamLGS，四场景、22 帧、208 queries | `63.6200 / 82.8487` mIoU/LocAcc；旧 LangSplatV2 不再选择协议 |
| LERF direct 3D | VALA 三层 semantic/evaluator + compatible Occam RGB geometry | `54.1249 / 79.3526 / 56.6114` mIoU/Acc@.25/Acc@.5；旧 Dr. Splat L1/L3 仅消融 |
| ScanNet OVS | VALA paper8、19/15/10 text-query splits | `34.5269/51.5906`、`37.9606/56.7696`、`47.3642/67.4650`；scene0645 仅 code9 sensitivity |
| AGILE3D | Easy3D official checkpoint，312 scenes / 10,357 objects | `agile3d_release` 交互、max-clicks 10、BF16、batch 4；released-code forward 仅 sensitivity |
| NVOS | LUDVIG-SAM，8 tasks × 3 seeds | `91.2577%`；严格复现 released online-multiview protocol，但不是 strict-unseen |
| SPIn-NeRF | LUDVIG-SAM，本地 9 scenes × 3 seeds | `93.7200%`；Fork 缺失，禁止冒充论文 10-scene row |
| PFPR | 固定 v2 privacy/metric contract + exact-LUDVIG 一场景自定义 adapter | 当前 `scene0050_02` 只作解释性诊断；不补 oracle，不晋升正式 20-scene 结果，不与论文比较 |

协议变更今后必须创建新 ID，写清 before/after sensitivity，绑定结果 hash，
并通过：

```bash
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.scripts.validate_evaluation_protocol_freeze
```

ScanNet 的两个活跃默认值已修正：Gaussian evaluator 默认是 `paper8`，
`code9/custom` 必须显式选择；feature wrapper 默认 `extract-only`，旧
mesh-kNN 只有显式 `--legacy-mesh-eval` 才会执行。

## 建议决策波次

### A. 低风险直接清理，约 2.1 GB

建议批准。主要是 Occam iteration-7000、Ramen 错误 suffix 泄漏
checkpoint、VALA pilot/failed shards/smokes、无 manifest 的 Pinecone
undistortion v1、PFPR 并发 smoke 和运行缓存。它们均不在 freeze 的
`authoritative_artifacts` 中。

需要注意：实际执行时仍会先逐项解析成明确路径，不会对仓库根目录、
`output/` 根或通配后的未知目标做递归删除。

### B. 先冷归档再移除，约 115–120 GB

建议分项目批准：

- Concept 约 31 GB：LangSplatV2、Dr. Splat、Occam resume checkpoint、
  旧 VALA res2/local-site。保留小 summary、配置、manifest 和 SHA。
- LUDVIG 约 9.2 GiB：每次 evaluation 的 `gaussians.ply`、
  `features.npy`、SAM masks/removal 可再生；不得删除 training PLY、
  accepted manifests、protocol result 和 checkpoint。
- PFPR 约 78 GB：v1、旧 v2 field、三组 600k geometry ablation 和重复
  no-gate/support diagnostics。先保存 release/field contracts、结果 JSON、
  support reports 和 hashes。

### C. 处理状态（2026-08-16 更新）

- `output/agile3d_scannet40` 约 553 GiB 已按独立退休清单清除。它是项目自身方法的
  历史实验树，并非 Easy3D baseline；结果收据哈希和保留边界见
  `paper/artifacts/agile3d_retirement_20260816.json`。轻量 Easy3D/AGILE3D
  外部协议冻结仍保留，但不属于 UQIS 或五 benchmark 主线。
- PFPR 当前 `reconstruction_v1` 至少 410 GB，重建成本高。若 1–2 个月内
  恢复正式 PFPR，应保留；若长期暂停，可先保留每场景 final checkpoint、
  geometry/render/source contracts、manifest 和摘要，再优先归档约 300 GB
  的逐帧 `radio_features`。

## 小文件建议保留

以下几类几乎不占空间，删除反而损害可解释性：修正后的 LangSplatV2
exact-camera receipt、Dr. Splat L1/L3 summary、VALA proxy/P0/code9、Easy3D
pilot policy、LUDVIG intermediate summaries/dry-runs/recovery receipts、PFPR
v1 协议变更说明和 oracle/no-gate 诊断代码。最容易误用的三个旧 Markdown
已加 `SUPERSEDED / DIAGNOSTIC ONLY` banner；项目 strict-unseen prompt 文档和
Easy3D 通用 20-click 文档也加了 scope note。

## 清理前仍需迁移的活跃引用

这些是删除大型目录前的阻塞项，建议先修引用再执行 B 波次：

1. `audit_external_baselines.py` / `external_baseline_audit.json` 仍把旧
   Occam pre-rendered 结果当当前完成项。
2. `sync_external_reproduction_summaries.py`、
   `validate_external_reproduction_summaries.py` 仍把旧 Dr. Splat summary
   当活跃 reproduction source。
3. `final_rows.yaml` 和部分 submission/status 日志仍记录旧
   LangSplatV2/Occam/Dr. Splat queue 状态；日志可保留原文，但入口应指向 freeze。
4. `paper/radio_gs_tpami.tex` 及 LERF/ScanNet qualitative manifests 仍说明
   2D prior=LangSplatV2、3D prior=Dr. Splat、ScanNet=旧 VALA compatibility。
   在删除其大型资产前，要么用 Occam/VALA/exact-paper8 重生成，要么明确降级
   为 historical diagnostic。
5. VALA LERF-3D 当前结果和 split receipt 已哈希冻结，但尚缺一个仓库内
   端到端 wrapper/checkpoint-hash manifest；在补齐前不要删除其 staging、
   12 个 semantic checkpoints 或最终 masks。
6. PFPR launcher 名称混合 benchmark v2 与 field MPR-v3 版本，旧
   `full_sens` 脚本还默认较早路径。后续应增加一个 canonical launcher，
   旧脚本只作兼容转发。
7. `paper/artifacts/checksums.txt` 覆盖整个 paper snapshot；当前工作树还有
   其他会话改动，不能在本轮盲目全量重写。待这些修改合并后统一刷新。

## 请用户决策

- **A：是否批准约 2.1 GB 的 safe-remove 波次？**
- **B1/B2/B3：是否分别批准 Concept、LUDVIG、PFPR 的冷归档波次？**
- **C-PFPR：当前 410+ GB materialized field 是近期保留还是长期暂停后裁剪？**
- **C-AGILE3D：553 GiB 项目方法历史树交给方法改进会话逐目录判定，还是另开清理审计？**

在收到选择前，所有候选都保持原位。
