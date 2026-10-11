# 完整范围组合：固定四场诊断协议

预先固定，2026-10-11。继承 extent_preparation 四场18训练观察/场、完整观察提升、
完整成员光学、成本归一、4遍双向离散成员政策。根目录 scope_topology_v1。
本轮验证范围信息及身份准入两个职责，不改变标准默认，不声称五任务完成。

构建仅使用原始query-free SAM层级候选与其原生SigLIP描述子。恢复全部5756个
候选，另记录原来保留的5137个坐标。两种候选域均在容量诊断中报告，不混淆
新增候选和新的组合表示。只使用既有18训练帧，不使用development/benchmark GT。
原始native masks直接bilinear到384×512、转half再>.5；新范围目标为二值。
这与条件检测器的soft target不完全相同，不能宣称数值复现检测器。

将全候选成员模式相同的位置合并为atom，仅存atom-proposal incidence、18组
192×256观察射线的3×3邻接atom索引、描述子和retained标记。没有源栅格坐标、
RGB/相机/mask文件或查询答案。绑定已有Gaussian观察提升/光学域，随同一field.pt
冷加载。此表示保留query-free分割信息，属于未压缩功能参考；不可通过重命名
宣称分割信息消失或证明存储新颖。所有新增字节计入场开销。

运行时先联合候选，再取邻域all为正/邻域none为负；未知候选内部不授权负证据。
空接受不产生补集负证据。边界忽略域外邻居，匹配canonical max_pool2d约定。
构建每帧验证10个query-free条件（空/全/8随机）的全部射线与完整raster独立实现
逐位相同。新状态及全部父状态冷序列化逐tensor相同。

四个实际集合条件：

- seed_addition：既有46.6535%标准参照，预测固定。
- extent_discrete：既有56.9282%特权完整范围参照，预测固定。
- matched_positive_scope：只为诊断，用已有源伪范围的完整采样射线选择候选并集；
  固定16步贪心，每次严格增加IoU，序号确定tie，不看GT。报告单候选、retained
  并集、全部候选并集的匹配率。用全部候选并集重建target/eroded-positive，
  保留参考negative/confidence，但新正范围内部冲突的negative改未知。
  当前条件仍依赖特权准入、negative和confidence；不是标准成绩，也不是全局
  容量上界。它回答已有候选的正范围是否足以支撑真实集合，而非检测器已迁移。
- native_complete_scope：对所有query-free候选复用既有原生文字相对四个通用负词
  的sigmoid(10 margin)>0.5准入，置信度max(2p-1)，完整并集后生成P/N。
  N仅在有效观察且原生Gaussian投影<=.5处获准；空接受未知。此项是固定身份
  规则的诊断控制，不把再次试阈值或完整补集当新身份机制。没有源伪范围、
  detector confidence或GT输入。运行时原生先验由field+合法文字向量实际重算，
  对固定prior最大误差<=2e-6/阈值差0验收；不拿旧预测当计算输入。

两项新条件使用同一初始化/成本与4遍成员函数，无连续查询拟合。无新增教师调用、
训练或参数搜索。所有四场68查询全部预测封存后，才允许独立score读取评测GT。
既有两控制的实际集合/语义分数必须精确保持。语义response继承控制，不以此声称
新范围语义AP改善。报告分场实际集合IoU、P/R、源伪范围匹配、活动view及fallback。
原生范围的质量决定下一项身份迁移，不以源拟合率或更低能量替代benchmark。

尚未完成instance/part/attribute/partition公共类型适配、真正LERF3D GT及完整五项。
未通过完整能力验收前不更改默认，不宣称v10恢复或多个创新已闭环。
