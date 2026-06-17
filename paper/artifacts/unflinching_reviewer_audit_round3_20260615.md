# 顶刊审稿式穿透审阅 Round 3：合格投稿包检查

日期：2026-06-15  
对象：TPAMI 主文、补充材料、投稿清单、投稿模式指南、封面信草稿与 active source root  
目标：不再只检查论文叙事，而是按“能否作为合格投稿包上传”的标准检查污染源、协议表述、source package 风险和最终可验证性。

---

## 第一轮：核心缺陷诊断

1. **投稿包污染源：旧 CVPR 草稿仍在 `paper/` 根目录。**  
   这是合格投稿前必须处理的问题。旧草稿中残留 VPR/readout/teacher/旧数值/旧 CVPR 模板口径。如果 source package 或人工检查误扫到它，会直接制造“作者自己都没统一版本”的印象。

2. **ScanNet 表述仍有回避强对比方法的语气。**  
   “OpenGaFF method row omitted by design” 这句话过于生硬，容易被审稿人理解成故意回避。正确写法应是：OpenGaFF/VALA 是 protocol source，不作为 reproduced baseline row。

3. **Efficiency 段落仍像未完成计划。**  
   “A future deployment study should...” 放在主文会削弱合格投稿状态，像作者承认关键效率证据未完成。应改为当前 submission 对 latency 的解释边界：protocol-level conservative evidence，不和 storage claim 混淆。

4. **补充表格术语仍有少量 readout/teacher 残留。**  
   这些词不全是错误，但在当前论文主线中会造成不一致。Appendix 里的 caption 也必须遵守主文定义。

5. **投稿指南缺少 source-package exclusion rule。**  
   即使 README 写了 legacy draft，若 ScholarOne 允许上传源文件，作者仍可能把 archive 一起打包。必须在 submission mode guide 明确排除 `paper/archive/`。

---

## 第二轮：修正与重建

### 1. 旧 CVPR 草稿移出 active root

**修正：**  
将 `paper/radio_gs_draft.tex` 移动到：

```text
paper/archive/radio_gs_draft_legacy_cvpr.tex
```

并在 README / submission checklist 中明确：该文件只作 traceability，不是 active submission source，不应上传。

**验证：**  
`test ! -f paper/radio_gs_draft.tex && test -f paper/archive/radio_gs_draft_legacy_cvpr.tex`

### 2. OpenGaFF/VALA 表述修正

**修正：**  
主文和 provenance table 中将 “OpenGaFF method row omitted by design” 改为：

> OpenGaFF/VALA split is used as the protocol source rather than as a reproduced baseline row.

这仍然符合当前比较策略，但语气不再像回避。

### 3. Efficiency 从 future work 改成当前边界

**修正：**  
将 “future deployment study should report...” 改为当前论文对 latency table 的解释：它是 conservative protocol-level evidence，直接 3D latency 包含 selected-primitive rendering，primitive scoring 才是 compact-memory operation。

### 4. Appendix caption 清理

**修正：**

- `Feature reconstruction error ... rendered-grounding readout` -> `rendered-grounding query protocol`
- `raw center readout` -> `raw center scoring`
- `boundary readout` 已改为 `boundary diagnostics`

### 5. Source-package 上传规则

**修正：**  
在 `paper/tpami_submission_mode_guide.md` 中明确：source package 上传时不要包含 `paper/archive/`，因为它只保存 legacy working drafts。

---

## Round 3 后的合格投稿状态判断

当前 active submission entry points 已经清楚：

```text
paper/radio_gs_tpami.tex
paper/radio_gs_tpami_supplement.tex
paper/radio_gs_tpami.pdf
paper/radio_gs_tpami_supplement.pdf
```

主动排除的非投稿文件：

```text
paper/archive/radio_gs_draft_legacy_cvpr.tex
```

论文主线已收束为：

> compact reconstructive RADIO Gaussian memory, trained by dense rendered RADIO reconstruction and sparse MPR semantic anchoring, evaluated through fixed 2D/3D/ScanNet/frozen-head/storage/latency protocols.

剩余 human-only 上传项仍存在，但它们不是论文科学内容缺陷：

- 最终作者、单位、邮箱、ORCID；
- funding / acknowledgements；
- AI/tool disclosure；
- dataset/code/model license forms；
- ScholarOne double-anonymous vs single-blind 路线确认；
- source package 是否需要上传，以及是否排除 archive。

## 结论

第三轮修正后，当前项目已经从“论文内容基本闭环”推进到“投稿包边界更干净”的状态。最重要的变化不是指标，而是去除了会让审稿人或编辑在预审阶段产生不信任的版本污染、回避式措辞和未完成式效率表述。
