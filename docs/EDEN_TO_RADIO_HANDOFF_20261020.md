# eden → radio 开发交接（2026-10-20）

本交接对应本地迁移提交；父提交为 `9363333eba24bb9d3f707293928edeb40c52f511`，分支为 `codex/tmm-query-response`。新机器用 `git log -1` 确认交接提交，不要只恢复父提交。本次不推送、不操作 radio、不运行 GPU、不修改共享资产。

## 当前状态与入口

先读 [CONTEXT](../CONTEXT.md)、[来源与建场报告](experiments/2026-10-19-region-provenance-and-build-core.md)、[N14 接入报告](experiments/2026-10-19-complete-reference-provider-integration.md)。

- 公共证据核心已被五类输出消费；Gaussian 建场代数已抽取，注册地图保留原视图均衡。**共享建场尚未全部完成**，统一读取不等于全部建图流程统一。
- 20 份旧 ScanNet 场已有显式来源附加层，仍缺独立几何可见性；新构建可选保留可见性，未完成 GPU 重建验收。
- N14 已有完整参考补全、源端整响应选择、两帧传播和原生射线拟合的显式入口。使用新生成的源端初始化，不用历史答案；目标 RGB、负涂画和实际读出随 profile 披露。
- feasible 是历史主策略，quality 是固定次策略；八任务同策略，不拼逐任务赢家。独立选集仍继承 source-only 初始化，不等于 N14 新连续响应的选集。
- **默认仍是 legacy/source-only；GPU 端到端和冻结五项完整重评未完成。** 单 SigLIP2、同权限 DINO/SAM、UQIS crop-only SAM 仍为独立待评实验。

| 用途 | 配置或入口 |
|---|---|
| 历史 v10 | `configs/pipeline/system_v10.json` |
| 原 source-only 的 legacy/shared 回归 | `configs/research/shared_evidence_migration.json` |
| N14 quality / feasible 扩展 | `configs/research/shared_evidence_n14_quality.json` / `shared_evidence_n14_feasible.json` |
| 执行入口 | `scripts/research/run_shared_evidence_migration.py` |

基础迁移配置中 “N14 pending integration” 保留了 source-only 路径的历史语境；10 月 19 日已有单独接入配置，但 GPU 验证仍 pending。此次未修改实验行为、配方或默认。README 的论文里程碑不代表新架构已全部验收。

## 验证时间边界

- **历史记录**：`docs/experiments/2026-10-19-shared-core-component-audit-v7.json` 为 31/31 任务、349 项组件比较，不是新推理分数。
- **历史记录**：`docs/experiments/2026-10-19-complete-reference-provider-integration.md` 记载两种 profile 各 253 passed、1 skipped；跳过 CUDA/NVCC。不是两套独立科学证据。
- **本次执行**：7 个固定 CPU 测试文件，legacy 与 shared_v1 各 **75 passed、1 warning**；CUDA 禁用、离线、仅写本轮专用临时夹具。CSR beta 警告不影响通过。
- 本次检查候选 Python/JSON 解析、凭据特征及暂存内容；见 [检查记录](migrations/2026-10-20-eden-radio-checks.json) 与 [筛选清单](migrations/2026-10-20-eden-radio-selection.json)。
- 未重跑教师、真实渲染、31 任务审计、253 项全套测试、benchmark、历史评分或环境重建。旧审计哈希是当时记录，不能解释为本次新验收。

## Git 筛选

初始为 53 个已跟踪修改、3,141 个未跟踪文件，暂存区为空。纳入全部 53 个修改和 740 个候选文本文件，共 793 项，再加本次交接和检查清单；包含源码、配置、测试、论文文本与必要小型报告。

不提交而原地保留：2,123 个嵌套历史快照/操作回执、265 个原临时文件、9 个生成二进制/日志格式文件、4 个超过 1 MiB 的生成报告。本次 `tmp/eden-radio-migration-20261020/` 也不提交。不删除用户文件。

这不是所有历史实验的封闭字节复现包。大型历史反馈和旧存储审计的部分夹具仍在 eden，不属首轮开发验收范围。例外见筛选清单。

## radio 接续（本次未执行）

1. eden 审阅本地提交后再推送；当前分支原先无 upstream。radio 正常 clone/checkout 交接提交，不复制当前 worktree 的 `.git` 指针。
2. 按 [外部依赖清单](EDEN_RADIO_EXTERNAL_ASSETS_20261020.md) 检查共享盘、模型字节绑定、外部源码及环境。保持旧绝对路径或显式登记路径重绑定；不静默改写有哈希的共享配置。
3. 在 checkout 根目录先跑 CPU 夹具。将下面的 `<python>` 换成新机目标解释器，basetemp 使用新的专用目录：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RADIO_GS_EVIDENCE_CORE=shared_v1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
<python> -B -m pytest -q -p no:cacheprovider --basetemp /tmp/radiogs-radio-cpu-UNIQUE \
tests/test_shared_evidence_core.py tests/test_shared_evidence_cpu_extension.py \
tests/test_shared_evidence_migration.py tests/test_region_provenance.py \
tests/test_reference_completion_pipeline.py tests/test_response_reconstruction.py \
tests/test_response_permissions.py
```

4. 再做固定代表任务 ramen、fern、horns_center、scene0000_00、scene0030_00 的端到端对照，之后单模块小对照，最后冻结五项重评。预检/推理会写新 run，须用唯一名称，不复用 eden 产物目录；完整推理需要显式 `--execute`。
5. 检查两台机器无同名 run 并发写入。此次快照未见项目写进程或目标路径打开的写文件，不是持续锁定保证。

此前部分 output 读取被 WebCodex 以 `read_file failed: invalid_path` 拒绝；本次未重试或绕过。radio 的路径、权限、资产完整性仍待确认；清单里的模型哈希抄录保留配置，未重读大权重。
