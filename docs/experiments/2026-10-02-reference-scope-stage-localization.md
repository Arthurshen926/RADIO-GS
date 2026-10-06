# 范围损失逐级定位：八查询独立目标评价与参考帧对照

本阶段完成八查询全源侧投影诊断、八次仅参考帧响应重建及其独立目标评价。固定现有NVOS地图、正参考掩码、源跟踪缓存与原评分方式；没有新跟踪、目标RGB推理、参数搜索或默认方法变更。

## 核心结果

仅参考帧证据的目标IoU **77.9961%**；全部合法跟踪源帧 **86.4891%**，八查询均改善，宏平均差 **8.4930pp**。这是已有跨视角机制的消融证据，不是本轮相对当前主线的新涨分。使用更多视角本身改变证据/计算预算，不能把它归因于新的融合方法贡献。

|查询|仅参考帧目标IoU|全源目标IoU|全源目标精确率|全源目标召回率|源响应/跟踪IoU|源集合/跟踪IoU|
|---|---:|---:|---:|---:|---:|---:|
|fern|76.29|82.18|89.47|90.98|93.11|92.19|
|flower|81.32|86.59|98.80|87.51|95.57|94.63|
|fortress|91.31|97.21|99.92|97.29|98.43|96.95|
|horns_center|86.29|95.73|98.52|97.13|96.74|95.76|
|horns_left|77.13|92.66|95.88|96.50|96.64|95.97|
|leaves|83.53|94.08|99.58|94.46|96.05|95.88|
|orchids|63.50|67.96|99.91|68.00|92.73|91.08|
|trex|64.59|75.49|83.54|88.68|84.87|82.13|

单位%。源响应平均拟合94.27%，实际集合平均拟合93.07%。二者比较的是源伪证据，不是源侧独立准确率。源分辨率192×256；目标沿用各自冻结全分辨率及连续响应阈值0.5。源、目标数字不能简单相减并分配责任百分比。实际集合仍单独按alpha>10.5/255做诊断，不替代NVOS响应指标。

## 排除了什么，定位到什么

1. **没有证据支持删除跨视角传播。** 固定参考实例身份时，完整传播比只用参考帧普遍好。该结果也不能证明每个传播掩码都正确，或推及LERF文本union任务。
2. **剩余问题不是统一的范围过大。** orchids几乎没有目标外误检，主要漏检；trex则精确率与召回率均有缺口。对所有任务扩大范围会损害后者的精确率。
3. **trex在源侧已经有较大拟合残差。** 冻结几何、低分辨率与冲突的源证据都可能造成这类残差；尚未分别识别，不能直接宣布是几何bug。后端重建仍值得做固定证据控制。
4. **orchids有明显的跨视角目标完整性缺口。** 高精确率低召回和较高源拟合说明，单纯继续压源拟合损失未必恢复漏掉部分。仍需测目标真值上的几何可见支持与源证据覆盖，并检查提示和跟踪的范围粒度。

当前无法严格区分“原始种子遗漏”与“传播后来遗漏”：没有逐源帧独立标注。脚本记录参考帧跟踪对输入掩码的一致性，但输入掩码本身也是SAM结果。没有运行目标图像SAM来填这个空缺，也没有把助手审阅冒充人工真值。

## 实现与复现

- `diagnose_reference_scope_stages.py`：只读取冻结输出，分别渲染连续响应与实际集合；所有源帧与源track比较，独立目标预测保存后才读取GT。输出逐帧IoU、精确率、召回率。
- `fit_reference_instance.py --reference-only`：明确的证据消融，只参考帧提供监督；仍构建全源几何可见支持，步数保持120。未被参考证据约束的元素按既有策略清零，因此这个对照比较完整现有流程，不是纯粹隔离跟踪准确率的因果试验。
- `run_reference_scope_ablation.py`：六GPU执行完整八查询消融。
- `summarize_reference_scope_stages.py`：从原始诊断与评分文件汇总，不手写指标。

```bash
# 每个查询先运行；scene取配置中的八个查询名
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/diagnose_reference_scope_stages.py --scene fern --output output/reference_scope_stages/fern
/root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_reference_scope_ablation.py
/root/miniconda3/envs/cybersim_agent/bin/python scripts/research/summarize_reference_scope_stages.py
```

结果：`output/reference_scope_stages/summary.json`；每查询`diagnosis.json`及`reference_only/evaluation/evaluation.json`。全源目标IoU与上一阶段结果一致至浮点精度。此前112份历史登记未改写；本阶段结果独立登记在`configs/recipes/reference_scope_stages.json`。

## 下一步的限定问题

先做**独立目标上的几何支持与源证据覆盖分解**，判断漏掉的像素是否有可用载体、载体是否获得过正证据；分析只在预测冻结后读取目标标签，不把目标标签加入推理。之后仅针对已证实的源拟合残差比较固定证据下的渲染分辨率/求解，不按每查询目标分数挑方法。源侧范围标注缺口保持公开，UQIS暂停。当前没有新增benchmark精度、四query独立验收或SOTA结论。
