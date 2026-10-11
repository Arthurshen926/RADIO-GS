# 公共文字入口：原始查询、范围证据与实际集合

2026-10-12。新增 `QueryFieldRuntime.text_extent`，将已验证的原子三态范围、
观察提升、足迹成本与四遍集合决策接入公共运行时。它只接收单场状态和当前
合法原生文字向量；不依赖研究脚本、已存查询范围或实验预测。全部68查询
已实际复算，选集逐位一致；连续成员值的更强严格检查在一个回退查询失败。

## 功能合同

场包需要同时包含 `semantic_measure`、`scope_topology`、
`extent_preparation`、`extent_factors`。场在CPU加载，原生查询与范围计算
建立指定CUDA设备上的临时镜像，集合引擎在CPU执行。镜像是计算副本，
不是另一任务场。初始化用一个与真实查询无关的合成射线预热数值内核。

入口先重算原生先验与每个观察范围；`scope_rule='atomic'` 使用正/负/未知
原子规则，再生成初值和成本，运行固定四遍成员增删。没有活动范围时保留
原生先验。`membership_scores > .5` 得到唯一实际集合；不能把成员值当作
校准概率，或把原生语义热图称为集合投影。2D集合评估须重渲染这一选集。

`other_view_positive` 是显式实验选项；其有效性见本轮范围实验。原子规则
仅是这个新入口的默认项，未替换已有全局 `query` 路径或其他任务配置。
此接口尚不负责类别菜单、笔画、图像或点提示，不能强行把它们转成文字。

## 使用示例

```python
import torch
from radio_gs.query_field.adapters import NativeQueryEncoder
from radio_gs.query_field.runtime import QueryFieldRuntime

# field_dir 必须是上述完整场包；siglip_path 对应其原生语义坐标。
runtime = QueryFieldRuntime.load(field_dir, device='cpu')
runtime.prepare_text_extent(scope_device='cuda:0')
encoder = NativeQueryEncoder(siglip_path, device='cuda:0')
negative = torch.stack([
    encoder.text(t).semantic.float()
    for t in ('object', 'things', 'stuff', 'texture')
])
q = encoder.text(expression).semantic.float()
result = runtime.text_extent(q, negative, scope_rule='atomic')
selected_xyz = runtime.field.xyz[result.selected]
```

编码器首次加载允许读取共享checkpoint；这与查询时读取场景源图像不同。
实际集合入口的文件禁读验证在场和数值内核初始化后进行。例中编码器/
场包路径由调用者提供，不需要读取研究实验预测或benchmark答案。

## 验证记录

- 四场各一个固定首查询冷启动：成员值及选中位全部精确一致，见
  [四探针记录](2026-10-12-text-extent-public-audit.json)。每个worker仅接收
  场和当前向量；旧预测只由外层控制器在实际执行后读取比较。
- 原始68个合法查询字符串和固定四个通用负词，重新经过原有SigLIP2
  编码器，输出与公共入口完整复算所用向量全部逐值一致，最大差为0，见
  [编码记录](2026-10-12-text-extent-encoder-audit.json)。没有源场景输入、
  目标标签或新的准确率评估。严格保持checkpoint的1152→1536输出头。
- 全部68查询完成实际公共入口复算，并先封存四场结果再比较。**所有68个
  最终选集逐位一致**，67个查询的成员值也精确相等。Ramen无活动范围的
  `a small piece of food resting on a plate`有12,797个连续先验值不相等，
  最大差5.960464477539063e-08，选中位差为0。

完整验证器预设“成员值与选中位均精确相等”，因此它在Ramen报错退出；
**该严格检查没有通过**。之后的[全部四场差异诊断](2026-10-12-text-extent-public-full-diagnostic.json)
单列 `frozen_strict_gate_passed=false` 和 `all_actual_sets_exact=true`，
没有更改预测、阈值、容差或原门槛。完整失败日志另存于本实验ROOT的
`strict_failure.log`。差异定位在连续原生先验回退，具体浮点运算原因尚未
确定；不以猜测的CUDA归约原因宣布修复。

固定同一场几何和评估函数，全部实际选集精确保持意味着原子链50.8519%的
实际投影能力得到保留；这不是新的准确率测量，也不证明连续后验逐值保持。
重新编码逐值一致与公共实际选集逐位一致，组合给出原始文字到实际集合的
功能证据，但不是一次重新运行的一体化原始字符串benchmark。该全量公共
复算选择原子规则；补全52.0773%仍属于另外的研究运行器成绩。

禁读覆盖实际查询期间的 `builtins.open`、`io.open` 和 `torch.load`，
不是OS沙箱。完整复算已先封存全部四场结果再比较旧成员值/选中位，
未根据参考结果修补查询。代码收据冻结为 `5abe900`。

## 与五任务闭环的关系

[共同包清单](2026-10-12-common-extent-readiness.md)核实只有四份LERF包
具备上述完整算子。ScanNet、NVOS、UQIS仍需补齐构建状态及各自输入合同。
公共文字功能保持消除了一个实际接入缺口，但不等于共同五任务、真正3D-GT、
独立泛化或v10完整性能已经恢复。四场现有完整tensor共43.762GiB，仍是
未压缩功能参考，本轮没有紧凑性或时延改进主张。
