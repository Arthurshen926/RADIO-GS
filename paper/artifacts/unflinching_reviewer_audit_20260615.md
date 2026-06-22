# 顶刊审稿式穿透审阅与修正记录

日期：2026-06-15  
对象：`paper/radio_gs_tpami.tex` 及主文引用表格  
审阅口径：以 TPAMI / 顶级视觉期刊投稿为标准，优先检查贡献边界、协议可信度、术语一致性、实验解释和过度声称风险。

---

## 第一轮：核心缺陷诊断

1. **致命风险：方法叙事一度像“模块堆叠”，而不是一个凝练的科学问题。**  
   主文已有 strong results，但如果 Related Work 只罗列 LERF、LangSplat、OpenGaussian、Dr. Splat 等工作，而不正面回答“这些方法是否已经支持 2D/3D query”以及“本文到底新在哪里”，审稿人会把 GaussFM 解读为又一个 Gaussian language feature variant。

2. **概念风险：`teacher` 和 `readout` 残留会削弱“compact memory”主线。**  
   `teacher` 容易让人误解为本文只是模仿 frame-wise RADIO；`readout` 容易让人觉得 2D/3D/SAM/DINO 是为 benchmark 外挂的接口。顶刊审稿人会追问：统一表示在哪里？为什么不是任务特化？

3. **实验呈现风险：LERF 2D 单独小表与 LERF 2D+3D 主表重复。**  
   同一组数值在主文连续出现两次，会让论文显得像实验日志拼接。更严重的是，重复表会抢占版面，却没有增加论证强度。

4. **协议风险：受控 PCC-only feature comparison 与最终 2D OVS row 没有明确分界。**  
   `0.5707` 与主表 `64.98` 同时出现，如果不明确前者是 controlled PCC-only comparison，审稿人会认为数字口径混乱，甚至怀疑结果 cherry-pick。

5. **ScanNet 解释硬伤：旧诊断句与最终主表数值冲突。**  
   主文写“calibration scale 0.75 raises 10-class mIoU to 0.4612”，但最终主表 10-class mIoU 是 `57.85%`。这类句子会直接暴露实验版本残留，是顶刊审稿中非常低级、但杀伤力很大的问题。

6. **Direct3D claim 边界必须严谨。**  
   当前主结果不使用 MPR cache 或 official RGB SAM decoder，但使用 GT-free color-edge / RGB component support calibration。若写成完全 image-free selector，会被质疑；若写成“轻量 support calibration，不调用额外 learned feature model”，则边界合理。

7. **结论过弱。**  
   原结论只说“turns a scene into memory”和列举结果，缺少对核心贡献、证据闭环和限制边界的最终收束，不符合顶刊论文的 closing standard。

---

## 第二轮：深度解剖与已落地修正

### 1. 相关工作和贡献边界

**问题定性：**  
如果不明确指出现有 3DGS open-vocabulary 方法多为 language-aligned semantic embedding storage / grouping / registration，本文的 compact reconstructive RADIO memory 就会被低估成“又一个语义特征高斯场”。

**修正：**  
在 Related Work 中新增 gap paragraph：现有方法证明 Gaussian scene 可以携带 language-aligned evidence，但并未直接回答是否能以低维 Gaussian codes 重建高维 foundation feature，并同时保持 rendered-view、primitive-level、point-level 和 frozen-head downstream usability。

**修改位置：**  
`paper/radio_gs_tpami.tex` Related Work。

### 2. 术语统一

**问题定性：**  
`teacher` 和 `readout` 会诱导错误理解：前者像简单蒸馏，后者像任务外挂。

**修正：**  
主文可见文字中将多数 `teacher` 改为 `frame-wise RADIO`、`RADIO target`、`RADIO-space feature`、`training target`；将 “Support-Calibrated Primitive Readout” 改为 “support-calibrated primitive selection”。保留数学上必要的 frozen target 概念，但不再作为主叙事。

**修改位置：**  
`paper/radio_gs_tpami.tex` Introduction / Method / Experiments / Discussion；  
`paper/storage_footprint_table.tex`、`paper/efficiency_cost_table.tex`、`paper/compression_downstream_correlation_table.tex`、`paper/lerf_per_gaussian_1280d_baseline_table.tex`、`paper/quantitative_ablation_summary_table.tex`、`paper/lerf_direct3d_confidence_coverage_table.tex`。

### 3. 方法图和方法段落的核心变量

**问题定性：**  
原文把输入 compact map 和融合输出都写作 `Z_v`，容易让 Method 看起来不够严谨。

**修正：**  
将 fused compact feature map 记为 `C_v`，明确 fine/coarse paths 经 `H_phi` 生成 `C_v, q_v, u_v`，再由 `D_theta(C_v)` 重建 RADIO feature。

**修改位置：**  
`paper/radio_gs_tpami.tex` Contextual Gaussian Feature Field。

### 4. 主表压缩和口径分界

**问题定性：**  
重复 LERF 2D 小表不是贡献，只会制造版面噪声。

**修正：**  
删除独立 LERF 2D 小表，只保留 LERF 2D+3D 主表；在文字中补充 2D row 的协议解释。

**修改位置：**  
`paper/radio_gs_tpami.tex` LERF-OVS Main Result。

### 5. Controlled PCC-only vs final 2D OVS

**问题定性：**  
`0.5707` 与 `64.98` 两个数字必须清楚区分，否则同一任务似乎出现两个 Ours。

**修正：**  
明确 `0.5707` 是 controlled PCC-only mask conversion，不是 final 2D benchmark row；final row 额外使用 feature-only SAM3-adaptor boundary head，达到 `64.98` mIoU。

**修改位置：**  
`paper/radio_gs_tpami.tex` LERF-OVS Main Result 与 Rendered Features vs. Original RADIO Features caption。

### 6. ScanNet 版本残留

**问题定性：**  
旧 `0.4612` 诊断句与最终主表冲突，属于必须清除的低级硬伤。

**修正：**  
删除该数值句，改为更稳健的结论：diagnostic proposal-memory variant 改善部分 19-class fine-grained 类别，但损害最终 15/10-class balance，因此留在补充材料。

**修改位置：**  
`paper/radio_gs_tpami.tex` ScanNet Direct Point-Query Transfer。

### 7. Direct3D support calibration 边界

**问题定性：**  
Direct3D 主结果应声明“不使用 MPR cache / official RGB SAM decoder”，但不能把 GT-free RGB/color-edge guard 写成不存在。

**修正：**  
保留并强化限制表述：deployed compact direct-3D result uses GT-free color-edge and score-component support calibration; it is not an official RGB SAM decoder and not a learned RGB segmentation network, but it is a lightweight support-calibration step.

**修改位置：**  
`paper/radio_gs_tpami.tex` Direct3D / Limitations。

### 8. 结论重写

**问题定性：**  
顶刊结论不能只是结果摘要，必须回到科学问题并明确边界。

**修正：**  
重写 Conclusion，强调：低维 contextual Gaussian field 可重建 RADIO-compatible scene features、吸收 sparse MPR evidence，并支持 rendered-view、direct-primitive、point-level open-vocabulary queries；同时明确小物体和边界敏感查询仍依赖 support calibration。

**修改位置：**  
`paper/radio_gs_tpami.tex` Conclusion。

---

## 修正后仍需在投稿前警惕的质询

1. **RGB/color-edge support calibration 是否会被认为削弱 pure one-map claim？**  
   当前写法已经避免过度声称，但答辩时必须坚持：它不是 learned feature model，也不是 official SAM decoder；它是 GT-free support calibration。若审稿人坚持 image-free，使用 strict no-RGB one-map row 作为防御。

2. **DINO/SAM/SigLIP 是否真的属于同一 foundation-feature memory？**  
   必须继续坚持：存储的是 RADIO-compatible compact memory；DINO/SAM/SigLIP 是 adaptor/probe spaces，不是并列 raw feature memories。

3. **ScanNet 是否等同 full semantic segmentation benchmark？**  
   不能。必须保持 “VALA-aligned direct point-query feature probe” 的边界。

4. **Storage 是否在超大场景上更有优势？**  
   目前主表已有 LERF 场景证据，论证 fixed decoder overhead amortization 合理；但如果投顶刊长周期版本，建议补一个大规模室外/室内场景 memory scaling appendix。

5. **是否存在单场景调参嫌疑？**  
   主文需要持续强调 fixed protocol、same reproduced protocols、GT-free calibration、controlled ablation 和 checksums/provenance manifest。

---

## 当前判断

修正前，论文最大问题不是结果，而是叙事和协议边界会被顶刊审稿人抓住。修正后，主线更清楚：

> GaussFM 是一个 compact reconstructive RADIO Gaussian feature memory。它通过 dense rendered RADIO reconstruction 和 sparse MPR semantic anchoring 学习三维基础特征记忆，并通过 support-calibrated primitive selection 将 primitive scores 转成稳定对象支持。

当前版本更接近顶刊投稿状态，但仍应在最终提交前进行一次全 PDF 人工排版审查，尤其检查 figure readability、caption 长度、表格拥挤度和补充材料中的旧术语残留。
