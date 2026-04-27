# RADIO-GS 环境配置验证与云服务器部署指南

> 验证日期：2026-04-25
> 目标：确认在云服务器上快速复现当前环境并直接运行实验管线

---

## 1. 当前环境诊断（Source Machine）

### ⚠️ 关键问题：当前 Conda 环境使用 PyPy 而非 CPython

```bash
# 当前默认环境 (iclpose)
Python 3.9.16 [PyPy 7.3.11]    # ← PyPy，不是 CPython！
PyTorch 2.0.1+cu118              # 已安装但无法加载 CUDA 扩展
```

**PyTorch CUDA 扩展在 PyPy 下无法工作**。当前环境的 GPU 训练实际无法运行——此前的结果可能来自其他环境或用 CPython 跑的。

### 已验证的依赖清单

| 依赖 | 当前版本 | 兼容性 |
|------|---------|--------|
| Python | **3.9.16 (PyPy)** | ❌ 必须换为 CPython 3.9+ |
| CUDA Driver | 580.105.08 | ✅ |
| GPU | RTX 4090 ×6 | ✅ |
| torch | 2.0.1+cu118 | ✅ CPython 下正常 |
| torchvision | 0.15.2+cu118 | ✅ |
| gsplat | 1.4.0 | ✅ 需对应 torch 版本 |
| numpy | 1.26.4 | ✅ |
| timm | 1.0.25 | ✅ |
| tinycudann | 2.0 | ⚠️ 需编译，可选依赖 |
| open_clip_torch | 3.3.0 | ✅ |
| transformers | 4.46.3 | ✅ |
| huggingface_hub | 0.36.2 | ✅ |

### 数据存储布局（Source Machine）

```
RADIO-GS repo:          /root/RADIO-GS/                          (206M, 不含 output)
Checkpoints (in-repo):  /root/RADIO-GS/checkpoints/              (119M, git-tracked)
Data mount:             /mnt/pool/sqy/  (NFS, 44T, 82% used)
├── dataset/            (47G)     — Replica / ScanNet 数据集
├── lerf_ovs/           (1.0G)    — LERF-OVS 原始数据 (COLMAP)
└── results/RADIO-GS/output/
    ├── 3dgs_models/             (679M)  — 预训练几何 PLY
    ├── radio_features_1280d_reextract_20260407/   (11G)  — Replica 特征
    ├── radio_features_lerf/     (5.5G)  — LERF 特征
    ├── radio_gs/                (56G)   — 训练输出 + 检查点 + eval 结果
    └── sp_gs/                   — 其他

Torch hub cache:        /root/.cache/torch/hub/NVlabs_RADIO_main/  (26M)
```

---

## 2. 云服务器部署步骤

### 2.1 基础环境

```bash
# 1. 创建 CPython 环境（关键：不要用 PyPy！）
conda create -n radio-gs python=3.9 -y
conda activate radio-gs

# 2. 安装 PyTorch（根据云服务器 CUDA 版本选择）
#    检查 CUDA 版本: nvidia-smi | grep "CUDA Version"
#    CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
#    CUDA 12.1:
# pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 gsplat (必须与 torch 版本匹配)
pip install gsplat==1.4.0

# 4. 安装其他依赖
pip install numpy scipy PyYAML tqdm pillow matplotlib tensorboard plyfile timm
pip install opencv-python
pip install open_clip_torch transformers huggingface_hub

# 5. 安装本项目
# 先传 repo 到云服务器，然后:
cd /path/to/RADIO-GS
pip install -e .
```

### 2.2 数据迁移

需要从 Source Machine 传输到云服务器的数据：

| 数据 | 大小 | 必要性 | 位置（目标） |
|------|------|--------|-------------|
| Git repo | ~206M | 🔴 必需 | git clone / scp |
| Dataset | ~47G | 🔴 必需 | 任意路径，config 中配置 |
| LERF-OVS | ~1G | 🔴 必需（做 grounding 实验） | 任意路径 |
| RADIO features (Replica) | ~11G | 🟡 可选（可重新抽取） | 任意路径 |
| RADIO features (LERF) | ~5.5G | 🟡 可选（可重新抽取） | 任意路径 |
| 3DGS models | ~679M | 🔴 必需 | 任意路径，config 中配置 ply_path |
| RADIO cache | ~26M | 🟡 可选（torch.hub 自动下载） | ~/.cache/torch/hub/ |

**推荐传输方式：**

```bash
# 方式 A：rsync（推荐，支持断点续传）
rsync -avz --progress -e ssh /mnt/pool/sqy/dataset/ user@cloud-server:/data/dataset/
rsync -avz --progress -e ssh /mnt/pool/sqy/lerf_ovs/ user@cloud-server:/data/lerf_ovs/
rsync -avz --progress -e ssh /mnt/pool/sqy/results/RADIO-GS/output/3dgs_models/ user@cloud-server:/data/3dgs_models/
rsync -avz --progress -e ssh /mnt/pool/sqy/results/RADIO-GS/output/radio_features_1280d_reextract_20260407/ user@cloud-server:/data/radio_features_1280d/
rsync -avz --progress -e ssh /mnt/pool/sqy/results/RADIO-GS/output/radio_features_lerf/ user@cloud-server:/data/radio_features_lerf/

# 方式 B：先打包再 scp
tar -czf radio_gs_data.tar.gz \
  /mnt/pool/sqy/dataset/ \
  /mnt/pool/sqy/lerf_ovs/ \
  /mnt/pool/sqy/results/RADIO-GS/output/3dgs_models/
scp radio_gs_data.tar.gz user@cloud-server:/data/
```

### 2.3 路径配置调整

云服务器上的路径与 Source Machine 不同，需在 configs 中修改以下字段：

每个 YAML config 需要检查和修改：

```yaml
# 核心路径修改（必改）
ply_path: /data/3dgs_models/{scene}/.../point_cloud.ply
feature_dir: /data/radio_features_1280d/{scene}/{split}
val_feature_dir: /data/radio_features_1280d/{scene}/{split}

# 如果 LERF 场景路径变了
scene_root: /data/lerf_ovs/{scene}

# 如果数据集路径变了
rgb_dir: /data/dataset/{scene}/{split}/rgb
depth_dir: /data/dataset/{scene}/{split}/depth

# 输出路径
output_dir: /data/output/radio_gs
```

### 2.4 RADIO 仓库设置

```bash
# 方案 A：通过 git clone（推荐，与论文代码对应）
git clone https://github.com/NVlabs/RADIO.git /path/to/RADIO
export RADIO_REPO=/path/to/RADIO

# 方案 B：通过 torch.hub 自动下载（首次运行时下载一次）
# torch.hub.load('NVlabs/RADIO', 'radio_model', ...)
# 会自动缓存到 ~/.cache/torch/hub/NVlabs_RADIO_main/
```

---

## 3. 环境验证脚本

```bash
#!/bin/bash
# save as: env_check.sh
# 验证云服务器环境是否就绪

echo "=== Python ==="
python --version 2>&1 | grep -q PyPy && echo "❌ WARNING: Using PyPy! Must use CPython."
python --version

echo "=== PyTorch ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    x = torch.randn(2,3).cuda()
    print(f'Tensor test: {x.sum().item():.4f} ✅')
"

echo "=== gsplat ==="
python -c "
import gsplat
print(f'gsplat {gsplat.__version__} ✅')
"

echo "=== Import check ==="
python -c "
from radio_gs.config import load_config; print('config ✅')
from radio_gs.models.hcd_codec import HCDCodec; print('codec ✅')
from radio_gs.models.hybrid_gaussian import HybridFeatureGaussian; print('hybrid ✅')
from radio_gs.models.screen_refiner import ScreenSpaceRefiner; print('refiner ✅')
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer; print('renderer ✅')
from radio_gs.heads.depth_head import DepthHead; print('depth_head ✅')
from radio_gs.heads.segmentation_head import SegmentationHead; print('seg_head ✅')
from radio_gs.heads.grounding_head import GroundingHead; print('grounding_head ✅')
from radio_gs.losses.distillation_loss import DistillationLoss; print('losses ✅')
from radio_gs.models.siglip_projection import SigLIP2FeatureProjection; print('siglip ✅')
print('All imports OK ✅')
"

echo "=== Data paths ==="
ls /data/dataset/ 2>/dev/null && echo 'Dataset ✅' || echo 'Dataset ❌'
ls /data/lerf_ovs/ 2>/dev/null && echo 'LERF ✅' || echo 'LERF ❌'
ls /data/3dgs_models/ 2>/dev/null && echo '3DGS models ✅' || echo '3DGS models ❌'
echo "\$RADIO_REPO = $RADIO_REPO"

echo "=== Smoke test ==="
python radio_gs/scripts/smoke_test_radio_gs.py 2>&1 | tail -5
```

---

## 4. 快速启动：一键训练测试

```bash
# 在云服务器上验证完整管线

# Step 1: 确认环境
bash env_check.sh

# Step 2: 配置路径
export RADIO_REPO=/path/to/RADIO

# Step 3: 运行训练（验证 GPU 训练可用）
CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/train_feature_field.py \
  --config radio_gs/configs/replica_hybrid_v14_room_0_nofdh_240ep.yaml \
  --epochs 5

# Step 4: 运行评估
CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/eval_rendered.py \
  --config radio_gs/configs/replica_hybrid_v14_room_0_nofdh_240ep.yaml \
  --checkpoint output/radio_gs/.../checkpoints/best.pth
```

---

## 5. 常见问题

### Q: PyTorch 版本与 gsplat 兼容性？
gsplat 1.4.0 兼容 PyTorch 2.0-2.1。如果云服务器 CUDA 版本更高（如 12.4），建议使用 PyTorch 2.1+。

### Q: 能否不传 RADIO features 直接在云服务器重新抽取？
可以，但需要：
1. 安装 RADIO 仓库（git clone）
2. 有 RGB 图像作为输入
3. 运行 `extract_radio_features.py`
抽取 LERF-OVS 4 个场景的 1280d 特征需要约 2-4 小时（取决于 GPU）。

### Q: 训练结果（checkpoints）是否需要传输？
不强制。云服务器可以从头开始训练。如果需要继续已有的训练，把 checkpoint 传过去即可。

### Q: 如果云服务器没有足够磁盘？
最小可运行数据只需：repo + 3DGS models + 至少一个场景的 RADIO features + dataset。约 **20G**。
```bash
# 最小依赖（单场景 LERF 训练）
du -sh RADIO-GS/                    # 206M  repo
du -sh 3dgs_models/figurines/        # ~100M PLY
du -sh radio_features_lerf/figurines # ~1.4G features
du -sh lerf_ovs/figurines/           # ~250M images + COLMAP
# 总计: ~2G
```
