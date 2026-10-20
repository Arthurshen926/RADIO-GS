# eden → radio 外部资产与环境清单（2026-10-20）

配合 [交接](EDEN_TO_RADIO_HANDOFF_20261020.md) 使用。本轮未搬模型、修改共享盘、访问 radio 或导出凭据。

## 共享 /mnt/pool

| 路径/入口 | 用途及绑定 | radio 要检查 |
|---|---|---|
| `/mnt/pool/sqy/results/RADIO-GS/output/` | 几何、原生场、区域/SAM 缓存、轨迹、UQIS 地图/成员模型和历史结果；`output` 是已跟踪软链接 | 同路径与读权限；新运行独立写目录。Git 带链接，不带内容 |
| `output/shared_evidence_20261018/region_provenance_20261019_v1/` | 20 场来源附加层和 recipe；报告 3,728,810 字节；基础迁移配置直接引用 | 小型附加目录与原缓存齐全，不重写大场 |
| `output/query_conditioned_membership_20261006/image_convergence/` | v10 历史绑定、UQIS 地图/成员等引用 | 静态地图/几何/成员依赖可读；不以历史答案替代新推理 |
| `output/compiled_native_observations/`、`output/adapter_observation_control/` | ScanNet 两语义教师编译场 | CSR、描述子、载体和来源附加层保持；旧场仍无独立可见性 |
| `output/nvos_carrier_next72/`、`output/typed_evidence_next44/full_source_tracks/` | NVOS 几何、拟合记录、轨迹 | 原几何/源帧绑定；不能把目标 RGB 混入几何训练 |
| `/mnt/pool/sqy/3d_understanding/segmentation_benchmarks/NVOS/` | RGB、正负涂画；16 份 N14 profile 声明逐文件 SHA256 | 对照 `configs/pipeline/reference_completion_20261019/{quality,feasible}/` |
| 其余 LERF/ScanNet/UQIS 声明 | 路径在 configs、共享 scene/map/query 内 | 可挂载不等于闭包完整；权限内逐项预检，不改原数据 |

以上来自源码、配置及保留报告。本轮不读取被拒绝的 output 目标，不全量遍历共享资产。

## eden 本地：Git 与共享盘不会自动提供

| 路径 | 来源/绑定及用途 | radio 要检查 |
|---|---|---|
| `/root/.cache/huggingface/hub/models--google--siglip2-giant-opt-patch16-384/snapshots/a713301b217d38485fb2204c808367d10bc3cc40` | SigLIP2 giant，1536 维原生语义；reconstruction、ScanNet/UQIS 配置 | 同 snapshot、processor/tokenizer、本地文件，不能自动换最新版 |
| `/root/RADIO-GS/checkpoints/sam3_modelscope/sam3.pt` | N14 SAM3；profile SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` | 旧主仓绝对路径或明确的新绑定；新 checkout 的 checkpoints 不自动含它 |
| `/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth` | DINOv2 ViT-B/14；profile SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` | 字节一致；哈希是配置记录，本轮未重读 |
| `/root/.cache/torch/hub/facebookresearch_dinov2_main` | 外部 DINO 源码，N14 绑定 models/layers/hub 文件 | 实际源码和本地修改；目录名 main 不是精确修订号 |
| `/root/external/sam3/` | model_builder、image processor、SAM1/tracking predictor 等源码 | 同文件版本/修改；同包名不足以证明一致 |
| `/root/.cache/huggingface/hub/models--owl10--Mask-Adapter/snapshots/99662f6a64430bdb458f56f8285bd13e2bd1ac06/fcclip_convnext_large_maskadapter.pth` | 历史双教师及替换对照 | 重跑教师需同 snapshot 和 FC-CLIP 依赖；只读编译场与重跑教师需求不同 |
| `/root/miniconda3/envs/cybersim_agent/bin/python` | eden 解释器，多处计划使用绝对路径 | 新机解释器路径及启动器绑定；不复制 eden 环境或凭据 |

## 本轮读取的 eden 包元数据

| 包 | 版本 |
|---|---|
| torch / torchvision | 2.0.1+cu118 / 0.15.2+cu118 |
| gsplat | 1.2.0 |
| transformers / timm | 4.49.0 / 1.0.25 |
| numpy / scipy | 1.26.4 / 1.13.1 |
| scikit-learn / joblib | 1.2.2 / 1.5.3 |
| pytest | 8.4.2 |

这不是完整 lock 或 radio 验证。`requirements.txt` 声明 `gsplat>=1.4.0`，与 eden 实际 1.2.0 不同；先核对工作环境，不能为满足宽泛声明直接升级。SAM3 无同名 distribution 元数据，不表示外部源码不可用。

真实 renderer/footprint 还需 CUDA、驱动、NVCC、编译扩展；不能假设 eden CUDA 缓存跨机可用。radio 需独立验证编译和数值。PDF 构建另需 pdflatex，不影响首轮 CPU 检查。

## 尚未确认

外部资产全量内容封存、radio 权限、SAM/DINO 源码精确修订、GPU 编译和端到端结果仍待确认。大型生成报告与历史夹具保留在 eden，完整历史再生成可能需要它们。本地旧检查点分支不因本次新迁移提交自动推送；是否发布旧分支另行决定。现有文档记录不替代这些验证。
