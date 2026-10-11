# 正范围、排除许可与原子拒识：固定四场协议

2026-10-12。已知上一轮结果，本轮无新的GT读取或调参后冻结。四场68文字查询，
沿用18观察/场、同一场包、原生先验、完整观察提升/足迹/集合光学和4遍增删。
目的：把原生35.0014%与特权匹配55.0409%的差距分解成可检验的范围职责，
并检验一个有限的三态证据假设。不会用特权条件的成绩提升标准默认。

三项特权功能干预，均使用原生条件完全相同的逐view confidence：

1. matched_positive_native_negative：匹配组的二值target和P，原生N。
2. native_positive_reference_negative：原生target/P，原始参考N。
3. matched_positive_reference_negative：匹配target/P，原始参考N。

加上既有native_complete_scope，形成P来源×N来源的2×2操作性对照。每次
组合后，与所用target>.5冲突的N改未知；这是维持标签一致的既定规则，不是
部署时正证据应优先的结论。报告全部冲突数。P变化可能改变active views、
提升和成本，因此是完整范围函数替换效应，不冒称固定观测域上的纯像素效应。
第三项与既有matched_positive_scope仅confidence来源不同，其活动域变化也
属于该替换。原参考N与匹配组已清冲突的N不混用。

唯一新标准候选native_atomic_abstention：每个atom取所有覆盖候选的原生
SigLIP probability最小值与最大值。仅lower>.5记正、upper<.5记负；有分歧、
等于.5或无覆盖均未知。这里是响应集合的区间，不是统计校准的置信区间。
重复相同候选不增加证据。对正atom和负atom分别要求完整3×3邻域一致，再
应用有效观察与既有原生正响应保护；不自动对正并集的补集授权负证据。
confidence仍为既有原生max(2p−1)，view仍须存在正证据才激活。
所有未知射线不进入BCE/overlap；全局足迹cost仍可能影响未知成员，不能因此
声称未知等于硬保持先验。候选不存新的场状态，不调用新teacher或训练。

这是“重叠范围的语义分歧应导致拒识”的固定候选，不宣称所有尺度的候选都
同等可靠，不宣称已解决part/instance/category类型差异。可能过度拒识，需
同时报告已知覆盖、fallback、误选和漏选。没有温度/阈值/步数/场景参数搜索。

四项新条件全部重新执行成员决策。原生prior与scope从冷field+当前合法文字
向量实际重算；旧prior误差<=2e-6、阈值bit差0，旧native scope逐位相同。
原生和原子范围及preparation在builtins.open/io.open/torch.load禁读环境执行；
这不是OS沙箱。特权数据只传入命名的hybrid条件，不传入原子函数。
四个旧控制成员精确继承。全部四场68查询×4新条件预测封存后，才评分。

独立审计需核验全部原子scope的NumPy定义、混合条件/冲突/active域、控制成员、
真实mask IoU/P/R；每场每新条件至少一个实际query的初末渲染目标独立复核。
源伪范围质量仅用于归因，不当评测GT或性能上界。标准精度恢复、五benchmark、
真正LERF3D GT、公共多模态链与论文机制创新仍分别待验收。
