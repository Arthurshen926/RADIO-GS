# 顶刊审稿式穿透审阅 Round 2

日期：2026-06-15  
对象：`paper/radio_gs_tpami.tex`、`paper/radio_gs_tpami_supplement.tex`、主文/补充引用表格  
目的：在上一轮修正后继续查找更深层的叙事、协议和术语风险，并将可修复问题直接落到论文。

---

## 第一轮：新的核心缺陷诊断

1. **方法结构仍有误读风险：MPR 被放在 Query Modes 下面，像推理接口而不是训练监督。**  
   这会直接削弱 “no MPR cache at inference” 的可信度。顶刊审稿人会问：既然 MPR 在 query section 里，它到底是不是部署时必须读取的第二套特征系统？

2. **SigLIP sparse supervision 的理论边界不够硬。**  
   如果只写 3D primitive supervision 用 SigLIP2 summary，审稿人会怀疑方法为了 direct 3D OVS 特化到 text-aligned space，而不是学习通用 RADIO feature memory。

3. **RGB/几何重建边界没有在 Method 开头压实。**  
   图和局限性中已经说不改 RGB reconstruction，但 Method Problem Setup 没有明确“3DGS geometry/RGB scene is pretrained and fixed”。这会造成贡献边界含混：到底本文有没有优化 RGB？是否利用 RGB loss 训练特征？

4. **补充材料残留旧术语会污染主文叙事。**  
   主文已将 teacher/readout 口径改为 frame-wise RADIO / query / support calibration，但补充表格 caption 里仍有 `cached-teacher`、`teacher vector`、`boundary readout`、`cached registered summary features` 等旧表达。审稿人经常查 appendix，这类不一致会被视为作者没有统一方法定义。

5. **legacy draft 是目录级风险。**  
   `paper/radio_gs_draft.tex` 仍残留 CVPR 旧口径、旧数值和旧术语。虽然 README 已声明它不是 active entry point，但它仍然是投稿前人工检查中的污染源。当前不强行同步它，以免改动无关草稿，但最终打包时必须排除。

---

## 第二轮：法医式解剖与修正

### 1. MPR 训练桥位置错误

**破坏性影响：**  
MPR 若被看作 query-time readout/cache，论文最重要的部署主张会变弱：one compact memory no multiview-registration cache。

**修正：**  
新增 `Sparse Primitive Anchoring by Multiview Registration` 方法小节，把 MPR 从 `Query Modes and Support Calibration` 中抽出。新小节明确：

- MPR 是 training and analysis bridge；
- 它渲染 decoded RADIO-space features，再经过 frozen SigLIP2 summary head；
- 注册目标只作为 sparse primitive semantic target；
- inference 不读取 registration targets；
- Dr. Splat-inspired rasterizer-level variants 是 negative controls。

**修改位置：**  
`paper/radio_gs_tpami.tex` Method。

### 2. “直接把 latent 对齐到 SigLIP” 的误读

**破坏性影响：**  
如果 compact latent 被理解为直接 SigLIP embedding，本文就会从 “compact reconstructive RADIO memory” 降级成 “text-head-specific Gaussian semantic field”。

**修正：**  
在 Contextual Gaussian Feature Field 和 MPR 小节中明确：

- all downstream adaptor scores are computed after RADIO-space reconstruction；
- compact latent is not a stored SigLIP/DINO/SAM embedding；
- MPR objective uses `A_SigLIP2(D_theta(c_i))`，即先 decode 到 RADIO space，再过 SigLIP2 summary head；
- compact latent is never directly stored or optimized as standalone SigLIP embedding。

**修改位置：**  
`paper/radio_gs_tpami.tex` Contextual Gaussian Feature Field / Sparse Primitive Anchoring。

### 3. Sparse SigLIP vs dense SAM/DINO 的理论解释不足

**破坏性影响：**  
审稿人会问：为什么 3D sparse supervision 不同样用 SAM/DINO/SigLIP 全部空间？如果回答“因为 Direct3D 用文本”，那就是任务特化。

**修正：**  
新增理论解释：DINO/SAM adaptor signals encode dense neighborhood topology, region continuity, and boundary structure；如果只在 isolated primitive centers 上施加，会丢失其空间关系。SigLIP2 summary features 是 global/text-aligned semantic descriptors，因此更适合作为 sparse semantic anchors。这个解释不依赖 Direct3D OVS 任务本身，而来自特征空间的结构差异。

**修改位置：**  
`paper/radio_gs_tpami.tex` Dense Reconstruction and Adaptor-Space Regularization。

### 4. RGB/geometry reconstruction 边界

**破坏性影响：**  
如果不写清楚，读者可能以为 CTF-GS 同时优化 RGB/geometry/feature，导致贡献和 storage accounting 全部变模糊。

**修正：**  
Problem Setup 增加说明：RGB/geometry 3DGS scene is reconstructed before feature-memory learning and kept fixed；CTF-GS does not introduce RGB reconstruction loss or claim improved radiance-field appearance。

**修改位置：**  
`paper/radio_gs_tpami.tex` Problem Setup。

### 5. Appendix 术语一致性

**破坏性影响：**  
补充材料术语不一致会让主文统一叙事失效，尤其是 teacher/readout/cache 这些词会被审稿人用来质疑是否存在 hidden inference dependencies。

**修正：**

- `teacher vector` -> `RADIO vector`
- `Nearest-view cached-teacher baseline` -> `Nearest-view frame-wise RADIO cache baseline`
- `boundary-error readout` -> `boundary-error diagnostics`
- `local boundary readout` -> `local boundary diagnostic`
- `cached registered summary features` -> `registered summary targets`
- `Registered MPR readout` -> `Registered MPR control`

**修改位置：**  
`paper/radio_gs_tpami_supplement.tex`、`paper/lerf_nearest_view_cache_baseline_table.tex`、`paper/boundary_error_readout_table.tex`、`paper/lerf_direct_3d_context_table.tex`、`paper/lerf_vpr_field_consistency_table.tex`。

---

## Round 2 后仍需警惕

1. **`radio_gs_draft.tex` 必须在最终投稿包中排除。**  
   README 已说明 active entry points 是 `radio_gs_tpami.tex` 和 supplement，但 legacy draft 中仍有旧数值和旧口径。最终压缩包不要包含它，或将其移入 archive。

2. **MPR 的命名要持续保持“registration/anchoring”，不要回到 VPR cache/readout。**  
   论文现在的主线是 sparse primitive anchoring，不是部署时的 registered feature bank。

3. **SigLIP sparse anchor 的解释要在答辩中坚持“feature-space structure”，不要说成“Direct3D task needs text”。**  
   正确逻辑是：dense topology signals belong to rendered feature maps; sparse semantic summary belongs to primitive anchoring.

4. **RGB/color-edge support calibration 仍是可被质疑的轻量图像辅助步骤。**  
   当前边界写法可接受，但必须保留 strict no-RGB controlled row 作为防御。

---

## 当前判断

第二轮修正后，方法章节更像一个统一训练框架，而不是一组为 benchmark 拼起来的接口：

> pretrained 3DGS geometry/RGB backbone -> compact contextual RADIO feature memory -> dense rendered RADIO reconstruction + dense structural adaptor regularization -> sparse MPR semantic anchoring after RADIO decoding -> support-calibrated open-vocabulary querying.

这条链条比上一版更能承受顶刊审稿人的关键追问：是否任务特化、是否依赖 hidden cache、是否真的学习 RADIO memory、是否改了 RGB reconstruction、是否 appendix 和 main claim 一致。
