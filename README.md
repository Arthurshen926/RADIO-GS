# RADIO-GS — TMM query-response study

迁移接续（2026-10-20）：先读 [eden → radio 交接](docs/EDEN_TO_RADIO_HANDOFF_20261020.md) 和 [外部资产/环境清单](docs/EDEN_RADIO_EXTERNAL_ASSETS_20261020.md)。10 月 19 日已有公共建场运算、来源附加层及 N14 显式接入；默认仍是 legacy/source-only，GPU 端到端与五项新重评尚未完成。下文保留论文历史里程碑，不能当作新架构验收。

当前主线：**查询条件下的证据融合与 3D 响应形成**。v10 是我们的方法实现，
其身份判断、源证据融合和完整成员恢复，以及相应高分与受控实验，属于本文主体。
特征场重建、维度和压缩不作为本篇研究变量。

工作分支为 `codex/tmm-query-response`，工作树为 `/root/RADIO-GS-query-field`。
完整实现使用 [v10 配方](configs/pipeline/system_v10.json)，覆盖五项输出、31 个场景任务。
[ADR0024](docs/adr/0024-v10-method-ownership-and-nvos-evidence.md) 纠正了此前只围绕
后续替换器写论文的定位。L11 完整响应替换作为独立扩展保留，不能替代或冒领 v10 的贡献。

NVOS 保留的均衡三维方案 N14 为 **93.1730 IoU / 98.9451 AP**；当前最高实测
三维 IoU 来自 N16 局部重解码，**93.2805 IoU / 98.4663 AP**，存在 AP 代价，
尚不能宣称同协议 SOTA。主线采用完整参考响应、源端选择、跨视角传播和原生三维拟合。
全部八任务整体比较，不拼接逐任务最高成绩；其他四 benchmark 保留原结果，
未切换全局默认。等权多视角融合和目标重解码已达到各自停止条件。
[方法与能力迁移](docs/experiments/2026-10-18-nvos-reference-completion.md)；
[最新性能取舍与停止证据](docs/experiments/2026-10-18-nvos-target-redecode.md)。
最新 [A24 数值审计](docs/experiments/2026-10-18-nvos-score-precision.md) 已完成八任务：
高精度评分对平均 AP 的影响小于 0.002 点，24 张硬掩码均不变；重解码退化
不能由该数值因素解释，维持当前细化路线的停止决定。

## 从哪里开始

- [当前上下文与工作边界](CONTEXT.md)：接续工作先读这里。
- [方法、实现、实验与复现说明](paper/tmm_response/README.md)：本课题的主入口。
- [实验登记表](configs/research/tmm_response.json)：唯一维护的实验编号、状态及运行入口。
- [指标表 CSV](paper/tmm_response/artifacts/metrics.csv) 与 [证据目录 JSON](paper/tmm_response/artifacts/study_catalog.json)：由原始结果生成。
- [论文方法](paper/tmm_response/sections/method.tex)、[正文](paper/tmm_response/main.tex)、[诊断附录](paper/tmm_response/diagnostics.tex)。
- [ADR0022](docs/adr/0022-query-response-paper-on-frozen-providers.md)：范围和主要研究决策。

## 统一入口

在工作树根目录运行。`status` / `verify` / `results` 只依赖 Python 标准库。
实验、作图和测试使用现有 `cybersim_agent` 环境；构建 PDF 还需 `pdflatex`。

```bash
conda activate cybersim_agent
python -m radio_gs.response_study status
python -m radio_gs.response_study verify
python -m radio_gs.response_study results
python -m radio_gs.response_study refresh
python -m radio_gs.response_study test
python -m radio_gs.response_study paper
```

`verify` 核验保留报告、封存哈希、配方与聚合指标；不等于重新推理或读取 GT 重评分。
`results` 导出统一指标；`refresh` 重建证据、表格及方法清单；`paper` 构建正文、诊断和新机制定义 PDF。
这些命令不训练、不启动 benchmark 推理，不复制大权重。

```bash
# 仅展示完整 31-job 重跑命令；--execute 才启动推理。
python -m radio_gs.response_study reproduce E04 --run e04_replay_new
```

重跑使用唯一目录、冻结队列和已有存储限额。E01/E02 的原磁盘写入器不开放自动重跑；
E01 还存在旧源码版本缺失。E03/E04 的临时预测已释放，重新评分需要重算。

## 历史资料

历史数值实现保留原路径，避免破坏证据哈希和调用关系。旧训练路线及系统文稿不再作为
当前入口；完整旧 README、3349 行 CONTEXT 和旧论文说明已逐字保存于
[历史快照清单](docs/history/2026-10-16-normalization-snapshots.json)。
[其他历史论文](paper/README.md) 与 `docs/experiments/` 继续保留，不能把历史指令视为当前任务。
