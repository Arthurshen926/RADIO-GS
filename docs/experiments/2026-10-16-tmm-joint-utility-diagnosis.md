# 联合效用与相对差异对照：结果和失败归因

由封存结果自动生成。主线保留完整答案、原教师和提供器；模型只决定是否替换。
同一个模型和统一门限服务所有已接通任务。开发与 benchmark 早已打开，默认仍为 v10。

## 独立房间开发：两种输入结构

joint 使用候选签名、集合均值和相对差异；relative 只使用相对差异。
其余教师、候选、数据划分、样本权重、模型容量和 15 分组校准保持一致。
relative 的设计和代码在 joint 结果产生前封存；设计时读取过无标签目标签名。

| 模型 | 条件 | 输出 | 原答案 % | 输出 % | Δ pp | 改善/损害 |
|---|---|---|---:|---:|---:|---:|
| joint | full | text | 6.3250 | 6.3250 | +0.0000 | 0/0 |
| joint | full | image | 20.5197 | 29.2991 | +8.7794 | 3/2 |
| joint | full | category | 65.1393 | 65.1393 | +0.0000 | 0/0 |
| joint | full | menu19 | 61.8142 | 61.8142 | +0.0000 | 0/0 |
| joint | full | menu15 | 62.1668 | 62.1668 | +0.0000 | 0/0 |
| joint | full | menu10 | 71.4368 | 71.4368 | +0.0000 | 0/0 |
| joint | query_blind | text | 6.3250 | 6.3250 | +0.0000 | 0/0 |
| joint | query_blind | image | 20.5197 | 30.2499 | +9.7302 | 3/2 |
| joint | query_blind | category | 65.1393 | 65.1393 | +0.0000 | 0/0 |
| joint | query_blind | menu19 | 61.8142 | 61.8142 | +0.0000 | 0/0 |
| joint | query_blind | menu15 | 62.1668 | 62.1668 | +0.0000 | 0/0 |
| joint | query_blind | menu10 | 71.4368 | 71.4368 | +0.0000 | 0/0 |
| joint | permuted | text | 6.3250 | 6.3051 | -0.0199 | 0/1 |
| joint | permuted | image | 20.5197 | 25.3691 | +4.8495 | 2/1 |
| joint | permuted | category | 65.1393 | 65.1393 | +0.0000 | 0/0 |
| joint | permuted | menu19 | 61.8142 | 61.8142 | +0.0000 | 0/0 |
| joint | permuted | menu15 | 62.1668 | 62.1668 | +0.0000 | 0/0 |
| joint | permuted | menu10 | 71.4368 | 71.4368 | +0.0000 | 0/0 |
| joint | unrestricted | text | 6.3250 | 10.5606 | +4.2356 | 61/42 |
| joint | unrestricted | image | 20.5197 | 30.8834 | +10.3637 | 7/9 |
| joint | unrestricted | category | 65.1393 | 65.1393 | +0.0000 | 0/0 |
| joint | unrestricted | menu19 | 61.8142 | 61.8142 | +0.0000 | 0/0 |
| joint | unrestricted | menu15 | 62.1668 | 62.1668 | +0.0000 | 0/0 |
| joint | unrestricted | menu10 | 71.4368 | 71.4368 | +0.0000 | 0/0 |

joint：统一门限 0.09309921251417219；开发门槛通过 False；分任务检查 {'text': False, 'image': False, 'category': False}。

| relative | full | text | 6.3250 | 6.3250 | +0.0000 | 0/0 |
| relative | full | image | 20.5197 | 20.5197 | +0.0000 | 0/0 |
| relative | full | category | 65.1393 | 65.1393 | +0.0000 | 0/0 |
| relative | full | menu19 | 61.8142 | 61.8142 | +0.0000 | 0/0 |
| relative | full | menu15 | 62.1668 | 62.1668 | +0.0000 | 0/0 |
| relative | full | menu10 | 71.4368 | 71.4368 | +0.0000 | 0/0 |
| relative | query_blind | text | 6.3250 | 6.3250 | +0.0000 | 0/0 |
| relative | query_blind | image | 20.5197 | 20.5197 | +0.0000 | 0/0 |
| relative | query_blind | category | 65.1393 | 65.1393 | +0.0000 | 0/0 |
| relative | query_blind | menu19 | 61.8142 | 61.8142 | +0.0000 | 0/0 |
| relative | query_blind | menu15 | 62.1668 | 62.1668 | +0.0000 | 0/0 |
| relative | query_blind | menu10 | 71.4368 | 71.4368 | +0.0000 | 0/0 |
| relative | permuted | text | 6.3250 | 6.3250 | +0.0000 | 0/0 |
| relative | permuted | image | 20.5197 | 20.5197 | +0.0000 | 0/0 |
| relative | permuted | category | 65.1393 | 65.1393 | +0.0000 | 0/0 |
| relative | permuted | menu19 | 61.8142 | 61.8142 | +0.0000 | 0/0 |
| relative | permuted | menu15 | 62.1668 | 62.1668 | +0.0000 | 0/0 |
| relative | permuted | menu10 | 71.4368 | 71.4368 | +0.0000 | 0/0 |
| relative | unrestricted | text | 6.3250 | 10.0493 | +3.7243 | 59/43 |
| relative | unrestricted | image | 20.5197 | 27.0511 | +6.5315 | 9/8 |
| relative | unrestricted | category | 65.1393 | 65.1393 | +0.0000 | 0/0 |
| relative | unrestricted | menu19 | 61.8142 | 61.8142 | +0.0000 | 0/0 |
| relative | unrestricted | menu15 | 62.1668 | 62.1668 | +0.0000 | 0/0 |
| relative | unrestricted | menu10 | 71.4368 | 71.4368 | +0.0000 | 0/0 |

relative：统一门限 None；开发门槛通过 False；分任务检查 {'text': False, 'image': False, 'category': False}。

## 固定模型的误差与输入域

以下范围由独立训练房间定义；目标仅使用签名，不读取 benchmark GT。
越界与树重要性只能描述分布差异和模型使用情况，不能单独证明错误原因。

| 签名 | 训练范围 | Gaussian 范围 | 超出训练范围的候选/30 |
|---|---|---|---:|
| selected_fraction | 0.0365–0.0484 | 0.0501–0.0525 | 30 |
| local_rank_outside_response_mean | 0.5058–0.5213 | 0.5028–0.5073 | 26 |
| local_rank_outside_response_q25 | 0.5015–0.5136 | 0.4983–0.5034 | 18 |
| local_rank_outside_response_std | 0.0062–0.0137 | 0.0048–0.0098 | 10 |
| mean_best_region_recall | 0.1284–0.4440 | 0.2074–0.6056 | 8 |
| selected_visibility | 0.0240–0.1169 | 0.0766–0.2291 | 7 |
| query_weighted_recall_std | 0.0475–0.1161 | 0.0512–0.1367 | 7 |
| mean_selected_probability | 0.2248–0.5457 | 0.2028–0.6460 | 6 |

| 数据 | 条件 | 任务 | 候选可用增益 pp | 决策损失 pp | 实际增益 pp |
|---|---|---|---:|---:|---:|
| calibration | full | text | 9.6672 | 8.1871 | +1.4801 |
| calibration | full | image | 29.0891 | 29.0891 | +0.0000 |
| calibration | full | category | 0.0000 | 0.0000 | +0.0000 |
| calibration | query_blind | text | 9.6672 | 9.3356 | +0.3316 |
| calibration | query_blind | image | 29.0891 | 8.4173 | +20.6718 |
| calibration | query_blind | category | 0.0000 | 0.0000 | +0.0000 |
| calibration | permuted | text | 9.6672 | 9.6672 | +0.0000 |
| calibration | permuted | image | 29.0891 | 29.0891 | +0.0000 |
| calibration | permuted | category | 0.0000 | 0.0000 | +0.0000 |
| calibration | unrestricted | text | 9.6672 | 5.2166 | +4.4506 |
| calibration | unrestricted | image | 29.0891 | 24.7449 | +4.3442 |
| calibration | unrestricted | category | 0.0000 | 0.6319 | -0.6319 |
| validation | full | text | 10.8819 | 10.8819 | +0.0000 |
| validation | full | image | 16.1040 | 7.3246 | +8.7794 |
| validation | full | category | 0.0000 | 0.0000 | +0.0000 |
| validation | query_blind | text | 10.8819 | 10.8819 | +0.0000 |
| validation | query_blind | image | 16.1040 | 6.3737 | +9.7302 |
| validation | query_blind | category | 0.0000 | 0.0000 | +0.0000 |
| validation | permuted | text | 10.8819 | 10.9019 | -0.0199 |
| validation | permuted | image | 16.1040 | 11.2545 | +4.8495 |
| validation | permuted | category | 0.0000 | 0.0000 | +0.0000 |
| validation | unrestricted | text | 10.8819 | 6.6463 | +4.2356 |
| validation | unrestricted | image | 16.1040 | 5.7402 | +10.3637 |
| validation | unrestricted | category | 0.0000 | 0.0000 | +0.0000 |

分解恒等式为：实际增益 = 完整候选池可用增益 − 决策损失。
有界 oracle 只用于归因，不是本方法结果；所有数值先按房间聚合。

## 同一 joint 模型和门限的实际 benchmark 迁移

| 条件 | ScanNet19 % | ScanNet15 % | ScanNet10 % | 文本 IoU % | 图像 IoU % | 文本 AP % | 图像 AP % |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 61.0115 | 62.5276 | 72.4861 | 16.3865 | 39.8980 | 26.5959 | 59.3088 |
| full | 61.0115 | 62.5276 | 72.4861 | 16.3865 | 40.8668 | 26.5959 | 60.4594 |
| query_blind | 61.0115 | 62.5276 | 72.4861 | 16.3865 | 40.2613 | 26.5959 | 59.5043 |
| permuted | 61.0115 | 62.5276 | 72.4861 | 16.3865 | 40.5070 | 26.5959 | 60.2724 |
| unrestricted | 60.1179 | 61.9458 | 71.4094 | 21.6410 | 38.6856 | 34.2296 | 58.1350 |

全 10 个完整菜单与 134 个文本/图像答案先封存，再评分全部 268 个 UQIS 查询。
点查询保留原答案；各类别子菜单使用同一个候选。relative 尚无本轮 benchmark 推理。

图像 IoU full − query_blind：+0.6056 pp，描述性场景 bootstrap 95% 区间 [-0.35730474895341685, 2.1739736724676493]。

图像 IoU full − permuted：+0.3598 pp，描述性场景 bootstrap 95% 区间 [-0.35730474895341685, 1.436635047746158]。

全队列实际选集改善/损害/不变：{'improved': 1, 'harmed': 1, 'unchanged': 266}。

此前仅文本/图像联合模型 L06 的图像 IoU 为 40.9859%；本轮没有刷新这一最好结果，且未将历史较优分项拼入统一策略。

## Benchmark 候选能力与决策损失

| 输出 | 可用增益 pp | 决策损失 pp | 实际增益 pp |
|---|---:|---:|---:|
| 完整菜单三输出均值 | 0.9131 | 0.9131 | +0.0000 |
| text | 15.3947 | 15.3947 | +0.0000 |
| image | 9.4923 | 8.5235 | +0.9689 |

10 个类别决策中，与 query-blind 相同 10，与查询置乱相同 10。

## 尚未闭环的范围

LERF semantic 2D、独立选集的二维投影和 NVOS 尚无此模型的有效迁移结果。
NVOS 现有每视角单个目标区域不满足区域签名的比较条件，需补齐适用的观察接口。
训练 mesh 与 benchmark Gaussian 的可见域不同；六个类别未被校准房间覆盖；
AP 和独立 LERF 选集没有直接效用监督。不能将保留原输出、兼容性或开发检查等同于机制成功。

relative 的门限为 reject-all，因此正式策略按定义始终返回原答案；不再浪费一次稠密重跑。
这是一条决策恒等性结论，不记录成新的 benchmark 实测或精度提升。

