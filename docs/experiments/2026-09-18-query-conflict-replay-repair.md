# 实际query冲突监督：保留查询区别的有界修复

本轮没有新增范围选择器。源侧自动检出“相同被接受范围头、同一视角、不同mask且IoU<0.1”的query对；只把新修正训练视角中的配对用于训练。figurines共2对，其中训练侧1对（plastic bag/container），另一对在新修正验证侧；ramen共1对（wooden chair/table）。范围是自动教师证据，物理身份未知，没有把助手审阅标签加入训练。

既有 sigmoid 查询条件修正模块继续冻结全部父模型张量。固定600步：一半步骤重放自动冲突对，另一半保持原来的广泛源范围采样；原有源验证完整像素分母和步骤0候选不变。两场都由源验证选择100步。该验证仅从新修正拟合留出，父模型见过源数据，仍不是P4独立家族验证。

|方式|figurines全文本IoU/AP|ramen全文本IoU/AP|
|---|---|---|
|V2父模型|43.812 /60.083|33.474 /50.874|
|冲突对重放|44.503 /60.442|33.681 /51.139|

两场图像/提示任务逐观测完全等于V2；全部父张量精确不变。源袋子三视角完整IoU从0.029%到42.119%，但仍低于原生文本70.495%。这是已知训练侧冲突驱动的局部修复证据，**不能把42.119%当成独立泛化收益**；完整标注开发成绩仅小幅上升。

原两次查询条件控制没有修好袋子；本次差别是明确同时监督互斥query，而非稀疏地遍历源范围。它证实现有实际query条件分支可表达部分修正，也暴露平均源IoU选择与罕见身份错误并不完全一致。不按袋子结果选择更晚检查点。

该候选尚未优于历史保留控制的所有模态，不作全局晋升。与“统一禁用概念接管”对照不同，它没有造成ramen的大幅回退。下一验收仍需可靠多视角身份/粒度监督、真正家族隔离和完整跨benchmark；不能宣称核心问题全部解决。

可复核：

- `scripts/audits/build_query_scope_conflicts.py`
- `scripts/audits/train_query_scope_correction.py --mode sigmoid_v1 --steps 600 --conflict-replay ...`
- `output/expert_plan_20260918/{figurines,ramen}_query_scope_conflicts.json`
- `output/v2_scope_rebuild_20260918/{scene}/field/query_scope_conflict_replay/`
- `output/query_scope_conflict_replay_20260918/{scene}/{text,reference}.json`
- `paper/artifacts/query_scope_conflict_replay_evaluations_20260918.json`
