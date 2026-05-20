# External Baseline Audit

Created: `2026-05-20`
Baselines root: `/root/baselines`
LERF root: `/mnt/pool/sqy/3d_understanding/lerf_ovs`

## Repositories

| Method | Commit | Exists | Dirty | Missing submodule | Blocker |
|---|---:|---:|---:|---:|---|
| OpenGaussian | 1f99db1 | True | True | False | ScanNet is reproduced locally; four-scene LERF language_features are available, but a strict official-policy LERF training/evaluation rerun is pending. |
| LangSplatV2 | 1667303 | True | True | False | Local LERF compatibility reruns completed all four scenes through all three feature levels plus eval_lerf.py --quick_render; current summary is LocAcc 0.6176 / mIoU 0.4601 scene-mean and LocAcc 0.6010 / mIoU 0.4487 object-weighted over 208 queries. This remains a compatibility rerun, not a strict released-checkpoint macro. |
| OccamLGS | eb98bcb | True | True | False | all four LERF compatibility scenes completed RGB training, language-feature extraction, test feature-map rendering, and tracked normalized pre-rendered readout: LocAcc 0.8221 / mIoU 0.4515 over 208 objects. This remains a compatibility readout, not a strict released-checkpoint macro. |
| GAGS | 4ce2721 | True | True | False | Full local GAGS LERF compatibility training/eval completed on all four scenes from local feature extraction/training. Shared-summary metrics: scene-mean LocAcc 0.7273 / mIoU 0.4893; object-weighted LocAcc 0.7308 / mIoU 0.4935 over 208 queries. This remains a local compatibility rerun because pretrained/preprocessed GAGS models are not released. |
| Dr. Splat | 764f608 | True | True | False | Dr. Splat local LERF compatibility mask export/evaluation completed on all four scenes with the shared nested-mask evaluator: mIoU 0.1762 / Acc@0.25 0.2561 / Acc@0.5 0.1137 over 208 objects, with 0 missing masks counted. This is a same-evaluator local compatibility row; upstream released-checkpoint/protocol caveats remain separate from the metric export. Official evaluation remains TBA upstream, so the local wrapper path is recorded explicitly. |
| LangSplat | d70edb8 | True | True | False | The local simple_knn/langsplat-rasterization/segment-anything-langsplat local site is built with NumPy pinned to 1.26.4; train.py/render.py/evaluate_iou_loc.py reach CLI help. All four scenes completed local compatibility training/render/eval after fp32 dim-3 feature conversion, chunked decoder eval, and split-aware train/test feature path fixes. Current summary: scene-mean LocAcc 0.7335 / mIoU 0.4433 and object-weighted LocAcc 0.7356 / mIoU 0.4613 over 208 queries. This remains a compatibility rerun, not a strict released-checkpoint macro. |
| LEGaussians | c2230af | True | True | False | LEGaussians official quantize_features.py, train.py, and render_mask.py compatibility pipeline completed on all four LERF scenes. Shared evaluator metrics: scene-mean mIoU 0.2694 / Acc@0.25 0.3974 / Acc@0.5 0.2312; object-weighted mIoU 0.2694 over 208 objects with 0 missing masks counted. This is a local compatibility rerun, not a released-checkpoint macro. |
| CAGS | 9136592 | True | True | False | OpenGaussian-compatible LERF training/render/eval scripts are available; the rasterizer ABI blocker and PyG source builds are cleared locally via output/baselines/cags/local_site, and train.py/render_lerf_by_text.py reach CLI help. The vectorized clustering fix, PyTorch checkpoint-loading wrapper, CPU-only FAISS fallback, and train/test render wrapper fix are in place. All four scenes completed local compatibility training/render/eval from OpenGaussian 30k starts: scene-mean mIoU 0.2627 / Acc@0.25 0.3997 and object-weighted mIoU 0.2394 / Acc@0.25 0.3558 over 208 objects, with 34 missing rendered masks counted. This is a diagnostic reproduced row, not a SOTA claim. ScanNet evaluation is marked TODO upstream. |
| Semantic Gaussians | ae53137 | True | True | False | Semantic Gaussians ScanNet compatibility distill/eval completed on the four tracked ScanNet scenes using the local label-PLY evaluator. Mean IoU is 0.0280 (scene0000_00 0.0209; scene0062_00 0.3634; scene0070_00 0.0213; scene0097_00 0.1315). This is a ScanNet-20 compatibility reproduction row; class-split leaderboard claims remain governed by the dedicated ScanNet protocol. |
| LaGa | 9df7586 | True | True | False | LaGa scene point cloud export, affinity feature training, descriptor building, mask export, and shared evaluator pass completed. LaGa local LERF compatibility mask export/evaluation completed on all four scenes with the shared nested-mask evaluator: mIoU 0.2337 / Acc@0.25 0.3660 / Acc@0.5 0.1535 over 208 objects, with 0 missing masks counted. This is a same-evaluator local compatibility row; upstream released-checkpoint/protocol caveats remain separate from the metric export. The row is reported as a compatibility adaptation of the inference notebook rather than an upstream paper-table macro. |
| OpenGaFF | - | False | None | None | Repository is not cloned locally. The arXiv paper reports state-of-the-art LERF-OVS and ScanNet context numbers, but its source states that code will be publicly released upon acceptance; no public implementation was found in the arXiv metadata/source or web search on 2026-05-18. Keep as published context, not a reproducible local baseline. |

## LERF Language Features

| Scene | Images | Labels | direct *_s/*_f | LangSplat *_s/*_f | Direct ready | LangSplat complete |
|---|---:|---:|---:|---:|---:|---:|
| figurines | 300 | 4 | 299/299 | 299/299 | False | False |
| ramen | 132 | 7 | 131/131 | 131/131 | False | False |
| teatime | 178 | 6 | 177/177 | 177/177 | False | False |
| waldo_kitchen | 188 | 5 | 187/187 | 187/187 | False | False |
