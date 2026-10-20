# RADIO-GS — TMM query-response study

当前主线：**学习完整响应的保留与替换**（Learned complete-response replacement）。
研究如何根据查询与观察证据判断替换一个完整答案是否有益。教师特征、场表示和几何保持固定；
本篇不再以紧凑特征场重建或系统压缩为研究变量。

工作分支 `codex/tmm-query-response`，工作树 `/root/RADIO-GS-query-field`。
当前性能参考仍为 [v10 配方](configs/pipeline/system_v10.json)。四轮完整机制实验已经完成，
尚无新候选在所有任务上形成共同增益，因此没有替换默认方法或拼接各任务最优结果。

当前具体推进方向是**学习完整响应的替换收益**：保留原答案及冻结解码器，训练一套
共同的保留／替换决策。独立房间训练与文字／图像迁移已执行，当前结果有收益也有退化。
架构见 [ADR0023](docs/adr/0023-learned-complete-response-replacement.md)，
结果见 [L01/L02 报告](docs/experiments/2026-10-16-tmm-learned-response-replacement.md)。

## 从哪里开始

- [当前上下文与工作边界](CONTEXT.md)：接续工作先读这里。
- [方法、实现、实验与复现说明](paper/tmm_response/README.md)：本课题的主入口。
- [实验登记表](configs/research/tmm_response.json)：唯一维护的实验编号、状态及运行入口。
- [指标表 CSV](paper/tmm_response/artifacts/metrics.csv) 与 [证据目录 JSON](paper/tmm_response/artifacts/study_catalog.json)：由原始结果生成。
- [论文方法](paper/tmm_response/sections/replacement_method.tex)、[正文](paper/tmm_response/main.tex)、[诊断附录](paper/tmm_response/diagnostics.tex)。
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
