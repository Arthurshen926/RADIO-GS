# Pipeline正确性优先：修复、验收与空间对齐对照（2026-09-15）

后续状态：独立单位区域对齐被进一步证明在重叠目标上可能不可同时满足，现已要求显式历史控制开关。本文保留当时的实验记录；当前修复与推荐开发入口见[源区域与三类召回修复报告](2026-09-15-component-pipeline-repair.md)。

本轮完成18次训练（包含2次早期正确性回归）、两场景真实文本及三类参考查询评估；另6次标定训练在发现兼容性测试失败后主动中止，不纳入质量结论。正确性保护成为默认，空间对齐和标定仍是显式实验，未通过全面质量门槛。

## 必须保留的流程修复

1. **输入契约不再静默兼容旧错误监督。** grounded监督升级为schema v2，显式声明同视角正定位才可产生负证据，记录源定位文件与编译实现哈希。训练核对帧/元素/像素/query轴、bool标签、正负互斥、任务索引及训练视角归属；源定位文件也做哈希复核。已有正确v2负例策略重新编译为contract_v3文件，任务及标签不变。旧文件名中的v2不等于schema v2。
2. **默认排除同query、同元素的相反训练伪标签。** 仅统计实际使用的任务，避免把已被替换的旧文本标签算入新监督冲突。figurines实际剩余此类冲突为0，ramen为1122个query-element对。
3. **默认保护注册提示输入。** ramen发现2条任务（原episode索引197、221）的目标伪标签与输入正负点冲突，目标membership弃权，输入点监督保留；figurines没有这类任务。
4. **区域教师采样独立于匹配任务。** 所有训练源区域轮转，预算不足或覆盖不全即拒绝；figurines92/92、ramen174/174。密集教师仍独立遍历训练视角及像素。
5. **明确拒绝隐式旧词表top1硬文本监督。** 新训练必须指定新版定位证据、上下文验证证据或显式软监督。历史控制需 --allow-legacy-text-pseudo-labels，不能无提示回退。旧schema及隐式旧标签均已实际测试为创建输出前失败。
6. **几何贡献与归属验证。** 检查贡献索引、权重有限非负、帧顺序、region/query轴和提示像素范围；加载器继续验证几何、教师文件与query编码器绑定。训练实现关键文件哈希在开始时记录，运行期间变化则拒绝发布模型。
7. **监控不混淆弃权与拟合。** 当前代码分别报告有有效像素的任务数和完全弃权任务数；旧文本监控保留历史标签说明，不冒充新目标优化质量。

输入一致性和监督自洽检查不是“所有伪标签都语义正确”的证明。SAM3定位、区域对应、跨视角尺度仍是有误差的证据；不能把契约通过改称几何/语义真值已验证。

## 空间一致性改进：已实现但未推广

新增 --native-space-alignment：要求原生训练输入SVD基、冻结共享文本/图像适配器、SigLIP教师及完整源区域覆盖。实际平均单位元素方向直接对齐投影后的原生区域描述，不再允许额外教师头独立吸收这项误差。教师和投影不接收该损失梯度。其他教师辅助头仍保留。
新增 --calibrated-readout：每模态仅学习正尺度和偏置，共6个标量；默认初始化仍10/-5，仅使用训练监督优化，不对目标GT搜阈值。它不改变单次查询内的余弦排序，但训练时会改变梯度，不能声称最终AP必然不变。
普通召回和瞬态提示优化共用 cosine_logits；查询优化显式detach标定参数，场不更新。旧模型保持原固定读出、旧权重格式可加载。首次新增接口在仅图像模态旧模型上触发一个回归失败，训练被中止；修复为显式模态或单模态推断后，完整测试通过。

## 真实文本开发集：完整任务分母

同前轮56/71任务、原生类别文本、固定最终概率0.5阈值。下表为mIoU/AP，所有对照启用相同正确性保护。

|场景|方案|mIoU|AP|
|---|---|---:|---:|
|figurines|pipeline_checked366|0.1272|0.1766|
|figurines|pipeline_fixed_basis366|0.0534|0.1182|
|figurines|pipeline_native_aligned366|0.0559|0.1234|
|figurines|pipeline_calibrated_fixed366|0.1103|0.1689|
|figurines|pipeline_basis_calibrated_fixed366|0.0893|0.1518|
|figurines|pipeline_aligned_calibrated_fixed366|0.0982|0.1686|
|figurines|pipeline_checked_no_radio366|0.1268|0.1746|
|figurines|pipeline_aligned_calibrated_no_radio366|0.1073|0.1698|
|ramen|pipeline_checked660|0.1448|0.2512|
|ramen|pipeline_fixed_basis660|0.0304|0.1476|
|ramen|pipeline_native_aligned660|0.0304|0.1414|
|ramen|pipeline_calibrated_fixed660|0.1586|0.2440|
|ramen|pipeline_basis_calibrated_fixed660|0.1294|0.2107|
|ramen|pipeline_aligned_calibrated_fixed660|0.1131|0.1862|
|ramen|pipeline_checked_no_radio660|0.1386|0.2468|
|ramen|pipeline_aligned_calibrated_no_radio660|0.1156|0.1860|

与前轮相比，正确性基线figurines基本不变（0.1275→0.1272），ramen下降（0.1664→0.1448）。本轮同时修正冲突保护与区域教师覆盖，不能把下降纯归因于某一项；更不能因分数下降重新启用矛盾标签。正确性修复是否保留不取决于最终指标。
源文本拟合：固定基为figurines0.501、ramen0.655；加标定后分别0.939、0.992。加直接对齐且标定后分别0.938、0.981。分数尺度错配被缓解，但高拟合并未转成足够的开放词汇泛化。

## 同一跨视角参考任务集合的三类查询

仍是由真实2D类别多边形派生的开发协议，参考GT仅作为显式查询裁剪/点输入，目标GT只计分；不是正式3D实例基准。

|场景/模型|文本IoU|图像IoU|直接提示IoU|优化提示IoU|
|---|---:|---:|---:|---:|
|figurines/pipeline_checked366|0.1341|0.0720|0.0487|0.1468|
|figurines/pipeline_aligned_calibrated_fixed366|0.1020|0.0536|0.0309|0.0818|
|figurines/pipeline_checked_no_radio366|0.1326|0.0695|0.0538|0.1588|
|figurines/pipeline_aligned_calibrated_no_radio366|0.1105|0.0497|0.0329|0.0770|
|ramen/pipeline_checked660|0.1345|0.0807|0.0829|0.1947|
|ramen/pipeline_aligned_calibrated_fixed660|0.1074|0.0680|0.0624|0.1543|
|ramen/pipeline_checked_no_radio660|0.1285|0.0835|0.0797|0.1860|
|ramen/pipeline_aligned_calibrated_no_radio660|0.1108|0.0657|0.0612|0.1521|

空间对齐+标定的三类查询总体未胜过正确性基线，尤其提示优化仍退化。因此不推广，不用局部AUROC改善掩盖任务IoU/AP下降。无RADIO模型完整训练成功，新增标定读出也完成独立原生三query冷启动，禁止训练特征和监督缓存访问。

## 复现与默认行为

推荐正确性开发入口（ROOT分别为four_teacher_three_query_v2_20260915/figurines和three_query_coverage_expansion_20260915/ramen的原绝对目录）：

```bash
python radio_gs/scripts/train_three_query_field.py \
  --root ROOT --geometry GEOMETRY --name NEW_RUN \
  --steps 366 --share-siglip-adapter \
  --grounded-text-supervision ROOT/grounded_text_supervision_contract_v3.pt
```

ramen使用660步。默认启用冲突弃权、提示保护、独立源区域覆盖；不默认启用native-space-alignment/calibrated-readout。历史行为仅通过 --allow-conflicting-pseudo-labels、--allow-conflicting-signed-targets、--matched-source-regions 等明确选项恢复。评估旧模型不要求重写旧资产；新版训练拒绝旧监督schema。
当前代码在最后加入了隐式旧文本拒绝与弃权任务计数；它们不改变显式grounded训练的优化方程。各已完成模型报告记录实际执行时的源代码哈希，不篡改旧报告以冒充新代码训练。

## 验收与剩余问题

94项相关测试通过，含新增16项输入契约测试、3项直接空间对齐测试、2项标定/提示路径测试；新代码语法编译及git diff --check通过。两个实际拒绝路径均未创建训练输出；18个完成模型的字段与几何哈希、教师区域覆盖检查通过。
当前已验收的是这些明确契约与执行路径，不能宣称整个研究pipeline绝无剩余bug。现存主要方法风险仍是区域教师描述与细粒度目标不一致、完整对象范围/跨视角身份监督不足，以及多教师在同一紧凑空间的目标冲突。
下一步应在正确性基线上检查源区域教师与元素匹配目标的一致性和可满足性，再引入更可靠的细粒度监督；不继续盲目加大对齐权重、放宽未知负例规则或搜索目标阈值。几何细节、独立3D真值和正式历史同合约回归仍未完成，不在本轮冒充解决。
