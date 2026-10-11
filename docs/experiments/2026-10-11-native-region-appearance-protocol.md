# 原生区域外观汇聚：匹配候选的功能迁移实验

这是已知此前四场开发结果后的适应性实验，不是独立确认。当前 seed_addition
46.6535%、frozen_visibility 45.1953%、条件扩展44.2876%。本轮不改阈值、森林、
候选、可靠可见性、原生语义、种子、几何或投票规则，只替换区域外观描述子。

主干预 native_448：沿用原local教师448宽度和LANCZOS整图缩放；同一原生SAM
proposal按现有source_records绑定（先native_proposal_indices，再retained region
IDs），mask用nearest-exact缩放到图像尺寸。调用既有NativeRegionAppearance.encode，
按区域面积汇聚未额外做逐token L2归一化的DINO x_norm_patchtokens，最后归一化。
这同时恢复区域原生支撑和汇聚次序，不能把收益只归因于某一次归一化。

诊断干预 native_full：相同函数、原始图像和mask尺寸，额外分离输入分辨率因素。
两组均使用同72个训练源视图；不读开发视图或目标标签，不增候选，不重新训练。
既有质量森林有24个独立ScanNet训练房间的3D监督，继续明示该条件。

构建逐帧核对RGB/SAM/区域缓存/observation哈希、原始mask到训练raster的精确
绑定，以及field中的view、scale和semantic descriptor。描述子编译进入各自完整
field.pt；冷加载要求其他所有张量逐值相同、总字节数不变。查询只读部署场。

完整四场68文本：保留此前三组强对照和旧conditional_addition；新增两种外观各自
的conditional_addition、conditional_replace，共八组。每个新场重新执行公共函数，
验证前三组逐Gaussian选集精确保持。旧conditional仅使用已冻结输出作参考。
主组native_448_addition，主要比较旧conditional_addition检验汇聚功能；同时必须
比较更强seed_addition，不能以超过较弱旧条件组就声称超越当前最佳。
native_full两组和replace两组只作诊断，不按结果重命名主组。

全部预测及哈希完成后才统一评分实际Gaussian选集的重渲染IoU，语义响应固定。
报告所有场景和全部组，并审计新增/删除像素、P/R及特征分布。小型重复开发集的
置信区间不能代替泛化证明。预测暖机后禁止Python open；这不是OS隔离证明。
不自动更改默认、不声明独立3D IoU、五任务恢复或论文闭环。失败后按证据停止
该方向的无依据阈值搜索，进一步定位测度、候选范围或条件质量函数的剩余差异。
