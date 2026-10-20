# 跨任务候选能力、决策证据与独立监督审计

完成 A08：22 个源辅助任务的完整答案决策审计；完成 A09：29 个独立房间的完整类别
菜单与类别并集目标准备。二者是诊断／准备，不是新的五任务性能结果。默认仍为 v10。

## 决策单位及约束

- LERF：同一查询在所有评估帧共享一个完整语义响应，共 67 个被评分查询。
- NVOS：8 个请求各选一个完整参考条件响应。
- ScanNet：10 个场景各选一套完整 19 类输出，19/15/10 类指标共享这个选择。
- LERF 独立选集没有新候选，保留投影 IoU 66.9633%，本轮未重新渲染评分。
- UQIS 引用 A07、L06 已核验的 134 个候选池，本轮未新增推理／评分。

A08 重算所有源任务的合法源证据，在读评分记录前封存选择。两候选为 v10 与 E02
原有完整响应，均核验大数组／原响应哈希。共同判据是逐视图、按授权与置信度归一化的
Brier 拟合损失下降；未知位置不作背景。类别菜单先整体汇总再决定。对照在视图可见
位置共同置乱目标、正负权限和置信度，保持类别边际与共同关系。没有拟合门限。

源证据已参与候选构造，不能称为独立验证证据。决策 IoU/AP 由已评分的完整候选记录
精确选择，没有重新渲染目标图像。更大的历史池含 E01–E04 共 9 个条件，只对评分记录
做诊断，未声称重新生成已释放的稠密预测。

## 源拟合分数不是可靠替换效用

| 指标（%） | v10 | 源证据决策 | 置乱证据决策 |
|---|---:|---:|---:|
| LERF semantic IoU | 70.4007 | 70.9319 | 70.9319 |
| LERF AP | 87.6193 | 87.3874 | 87.3874 |
| NVOS IoU | 86.5134 | 86.4656 | 86.4896 |
| NVOS AP | 98.0222 | 97.9957 | 97.9926 |
| ScanNet mIoU19 | 61.0115 | 60.5911 | 61.0505 |
| ScanNet mIoU15 | 62.5276 | 61.8786 | 62.5089 |
| ScanNet mIoU10 | 72.4861 | 72.0285 | 72.2158 |

LERF 替换 58 个被评分查询：43 改善、13 损伤、2 不变。IoU 收益全部被置乱对照保留，
不能归因于正确的证据对应。NVOS 替换 6 个：4 改善、2 损伤。ScanNet 选择全部 10 个
替代菜单，实际仅 3 个改善、7 个损伤。

当前结果不支持把源拟合残差直接当作替换效用。它不证明所有源证据无用，也不能断言
训练分布、候选相关性或特征形式中哪一项是唯一原因。

## 各任务的候选空间缺口不同

| 指标（%） | v10 | 历史池主指标 oracle | 同时不伤害所报告指标的 oracle |
|---|---:|---:|---:|
| LERF semantic IoU | 70.4007 | 72.1550 | 71.7441 |
| NVOS IoU | 86.5134 | 86.8972 | 86.5615 |
| ScanNet mIoU19 | 61.0115 | 61.5137 | 61.2168 |
| ScanNet mIoU15 | 62.5276 | 62.5336 | 62.7342 |
| ScanNet mIoU10 | 72.4861 | 71.3781 | 72.5128 |

所有 oracle 都使用标签，只是有限池诊断，绝不是可部署效果或模型理论上限。第一种
按查询平均 IoU／完整菜单 mIoU19 选同一候选，其他指标随该候选报告。最后一列额外
要求每个决策单位的所有报告指标均不下降，比只保护场景宏平均更严格。

UQIS 候选池相对 v10 仍有 text 15.3947、image 9.4923 点 IoU 空间，主要需要学会选择。
源任务旧池的多指标余量仅 LERF 1.3434、NVOS 0.0481、ScanNet mIoU19 0.2053 点。
它们多为同一响应的数值修改，需要功能上有区别的完整候选，不能仅继续优化选择器。
LERF 独立选集完全没有新增候选。

ScanNet 还有目标不完整的问题：即使标签 oracle 只按 mIoU19 选，mIoU10 仍会下降
1.1080 点。整体菜单避免类别拼接，但没有自动解决多指标冲突。

## 独立目标已准备好，效用训练数据尚未齐备

A09 保留 22 fit／3 calibration／4 development 房间，检查房间前缀与 benchmark
不重叠。通过官方 segIndices、aggregation 和 raw-category→NYU40 表重建类别目标，
逐顶点确认地图与官方 mesh 坐标和顺序完全相同，没有用单实例代替类别并集。

- 29 个完整 19 类菜单目标，缺席类别也保留。
- 551 个类别并集目标定义，包含空目标，并非 551 个正例。
- 87 个连续时间三段划分的跨视图菜单定义，仍只有 29 个独立房间。
- 保存哈希、计数、视图索引与覆盖率，不保存新稠密标签／地图／特征。
- 视图块不重叠，但边界近邻位姿未剔除，几何来自整幅扫描，不宣称统计独立。

fit 覆盖 19 类，但 shower curtain、toilet、bathtub 各只出现于 1 个房间。
calibration 缺少 bookshelf、counter、shower curtain、toilet、sink、bathtub 六类。
原三房间校准不能支持完整类别覆盖主张。

现有实例数据含 SigLIP、SAM 区域、RGB-D、几何与标签，UQIS 图像使用 DINO；ScanNet
还要求 Mask-Adapter 和 literal-category SAM 源证据。它们不能由 DINO 代替，在这些
29 房间上也未齐备。清单明确为 `utility_training_ready=false`。

## 下一阶段的具体收敛方向

保持一个研究对象：完整候选答案之间的实际相对效用与保留决策。

1. 源任务复用有效原生／源条件完整解码器，产生功能有区别的完整候选。独立选集必须
   显式进入候选契约，不能仅改变语义数值后宣称五输出被优化。
2. 补齐独立房间上的原配置教师、完整类别源证据与校准类别覆盖；分别监督实例、类别
   并集和整体菜单的真实输出效用，菜单内部不选择逐类赢家。
3. 用跨视图划分检验收益／损伤证据。若 incumbent 已使用某视图，就不能称该视图为
   它的独立验证数据；交叉构造候选与最终完整答案的关系必须明确。
4. 在校准数据上共同保护实际选集、排序与菜单各子集指标，再冻结同一决策机制做完整
   评估。不能把本次标签 oracle 直接部署。

方法贡献尚未闭环；本轮将候选能力、效用证据、任务目标和监督覆盖分成可检验的缺口。

## 复现

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/research/audit_tmm_source_decisions.py --run source_decision_audit_new
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/research/prepare_tmm_cross_task_targets.py --run cross_task_targets_new
python -m radio_gs.response_study refresh
python -m radio_gs.response_study verify
```

配置：`configs/research/source_decision_audit.json`。生成反馈／表格：
`paper/tmm_response/cross_task_feedback.json`、`artifacts/cross_task_metrics.csv`。
运行目录为 `source_decision_audit_v1`、`cross_task_targets_v1`，沿用 2 GiB 研究预算。

验证完成：相关测试 90 项通过，`refresh`、`verify`、`paper` 与 `git diff --check`
通过。当前正文 6 页，编译无 LaTeX 警告／未解析引用；未另做 PDF 图像渲染检查。
两个新运行分别占 880,770 与 746,936 字节，共约 1.63 MB，无新增稠密预测。
研究目录预算余量 52,903,484 字节。E01 仍有 8 条历史源码版本记录缺少对应文本，
该历史复现限制没有被本轮核验消除。
