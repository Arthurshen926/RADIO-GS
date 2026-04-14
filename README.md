# RADIO-GS

`RADIO-GS` is a standalone split of the scene-understanding / foundation-feature rendering line from the original `ICLPose` workspace.

This repository keeps only the `RADIO + Gaussian Splatting` pipeline for novel-view foundation feature rendering and downstream scene understanding. The old localization / pose-estimation line is intentionally left out.

## What this repo does

The goal is to distill `RADIO C-RADIOv4-H` spatial features (`1280d`) into a Gaussian-splat scene representation so that a rendered novel view can be decoded back into RADIO-like features and consumed by downstream adaptors / heads.

The current pipeline supports:

- feature reconstruction / cosine evaluation
- novel-view depth estimation
- novel-view semantic segmentation
- text grounding with SigLIP2-style projections
- geometry-depth fusion for stronger depth prediction
- qualitative visualization for `RGB / PCA feature / depth / segmentation / grounding`

## Algorithm overview

```text
RGB images
  -> RADIO encoder (frozen)
  -> 1280d teacher features
                  |
                  v
3DGS / 2DGS geometry + learnable feature field
  -> render compact latent feature maps
  -> HCD codec / decoder
  -> screen-space refiner
  -> reconstructed 1280d novel-view features
  -> task heads / adaptors
       - depth
       - semantic segmentation
       - text grounding
```

### Main components

1. `radio_gs/scripts/train_rgb_gs.py`
   - trains the RGB geometry model used as the scene backbone
   - supports Replica-style RGB/depth/pose inputs

2. `radio_gs/scripts/extract_radio_features.py`
   - extracts and stores RADIO teacher features for all frames
   - now saves a `frame_manifest.json` and uses numeric frame ordering

3. `radio_gs/scripts/train_feature_field.py`
   - the main feature-distillation training entry point
   - supports explicit and hybrid Gaussian feature architectures
   - includes HCD compression, screen-space refinement, geometry-aware losses, and optional grounding auxiliary loss

4. `radio_gs/scripts/eval_rendered.py`
   - evaluates reconstructed novel-view features with downstream probes / heads

5. `radio_gs/scripts/eval_grounding.py`
   - standalone SigLIP2 text-grounding evaluation

6. `radio_gs/scripts/generate_visualizations_v2.py`
   - renders qualitative grids and per-frame figures for debugging and paper figures

## Repository layout

```text
RADIO-GS/
├── README.md
├── checkpoints/
├── requirements.txt
├── setup.py
└── radio_gs/
    ├── config.py
    ├── configs/
    ├── data/
    ├── heads/
    ├── losses/
    ├── models/
    ├── rendering/
    ├── replica_constants.py
    ├── scannet_constants.py
    └── scripts/
```

## Environment setup

This repo assumes:

- Python `3.9+`
- a working PyTorch + CUDA environment
- `gsplat >= 1.4`
- a local clone of the official `RADIO` repository

The repo already ships with the small-but-required grounding resources under `checkpoints/`:

- `checkpoints/siglip2_feat_projection.pth`
- `checkpoints/siglip2_text_embeddings_v2.pt`
- `checkpoints/siglip2_text_embeddings_v4.pt`

So a fresh clone does **not** need to redownload those files.

Example setup:

```bash
conda create -n radio-gs python=3.9 -y
conda activate radio-gs

# install PyTorch first using the command that matches your CUDA version
pip install -r requirements.txt

# optional but recommended for package-style imports
pip install -e .
```

Set the RADIO repo path explicitly:

```bash
export RADIO_REPO=/path/to/RADIO
```

`radio_gs/config.py` will pick up `RADIO_REPO` automatically. You can also override `radio_repo` inside YAML configs or per-command flags.

No code change is needed to load a local RADIO checkout or local checkpoint files:

- `radio_repo` already accepts a local directory path
- `siglip_projection_weights` and `grounding_text_embeddings` already accept local file paths
- current defaults now point to the bundled `checkpoints/` directory, with fallback to the old `output/radio_gs/...` paths for backward compatibility

## Expected data layout

The current codebase assumes Replica-style data by default:

```text
dataset/
└── room_2/
    ├── Sequence_1/
    │   ├── rgb/
    │   ├── depth/
    │   ├── semantic_class/
    │   └── traj_w_c.txt
    └── Sequence_2/
        ├── rgb/
        ├── depth/
        ├── semantic_class/
        └── traj_w_c.txt
```

`radio_gs/data/benchmark_paths.py` also supports mixed split logic and ScanNet / LERF path resolution.

Raw LERF downloads are also supported directly now. If your scene root looks like:

```text
/mnt/pool/sqy/lerf_ovs/
  ├── figurines/
  │   ├── images/
  │   └── sparse/0/{cameras.bin,images.bin,...}
  └── label/figurines/frame_XXXXX.json
```

then `LERFDataset` can read the COLMAP sparse model and polygon JSON annotations without a manual conversion step. You still need to extract RADIO features into a separate feature directory.

## Recommended workflow

### 1. Train or prepare geometry

Train a clean RGB Gaussian scene model first:

```bash
CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/train_rgb_gs.py \
  --scene room_2 \
  --sequence Sequence_1 \
  --iters 30000
```

This writes the geometry backbone that later configs reference through `ply_path`.

### 2. Extract RADIO teacher features

Training split:

```bash
CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/extract_radio_features.py \
  --scene room_2 \
  --image_dir dataset/room_2/Sequence_1/rgb \
  --output_dir output/radio_features_1280d/room_2/Sequence_1 \
  --radio_repo "$RADIO_REPO" \
  --radio_version c-radio_v4-h \
  --batch_size 4
```

Validation split:

```bash
CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/extract_radio_features.py \
  --scene room_2 \
  --image_dir dataset/room_2/Sequence_2/rgb \
  --output_dir output/radio_features_1280d/room_2/Sequence_2 \
  --radio_repo "$RADIO_REPO" \
  --radio_version c-radio_v4-h \
  --batch_size 4
```

If you need to repair legacy feature folders created before the numeric frame-order fix:

```bash
python radio_gs/scripts/repair_legacy_radio_feature_indices.py \
  --image_dir dataset/room_2/Sequence_1/rgb \
  --feature_dir output/radio_features_1280d/room_2/Sequence_1 \
  --output_dir output/radio_features_1280d_corrected/room_2/Sequence_1
```

### 3. Train the feature field

Example with the current hybrid config:

```bash
CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/train_feature_field.py \
  --config radio_gs/configs/replica_hybrid_v14_room_2_reextract.yaml
```

Key config fields to check before training:

- `ply_path`
- `feature_dir`
- `val_feature_dir`
- `train_split` / `val_split`
- `output_dir`
- `grounding_text_embeddings`
- `siglip_projection_weights`

On a shared GPU machine, you can queue training and let it auto-start once a card is free enough:

```bash
MIN_FREE_MIB=4096 MAX_UTIL=70 GPU_LIST=0,1,2,3,4 \
  bash radio_gs/scripts/wait_and_train.sh \
  radio_gs/configs/replica_hybrid_v14_room_2_clean_retrain.yaml \
  output/radio_gs/train_room2_v14_clean_retrain_waiter.log
```

### 4. Evaluate reconstructed novel-view features

Rendered multi-task evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/eval_rendered.py \
  --config radio_gs/configs/replica_hybrid_v14_room_2_reextract.yaml \
  --checkpoint output/radio_gs/room2_hybrid_v14/checkpoints/best.pth
```

Standalone text grounding:

```bash
CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/eval_grounding.py \
  --config radio_gs/configs/replica_hybrid_v14_room_2_reextract.yaml \
  --checkpoint output/radio_gs/room2_hybrid_v14/checkpoints/best.pth \
  --gt_features output/radio_features_1280d_reextract_20260407/room_2/Sequence_2 \
  --semantic_dir dataset/room_2/Sequence_2/semantic_class \
  --vis_dir output/radio_gs/vis_grounding_v14_room2_reextract
```

### 5. Generate qualitative visualizations

```bash
CUDA_VISIBLE_DEVICES=0 python radio_gs/scripts/generate_visualizations_v2.py \
  --config radio_gs/configs/replica_hybrid_v14_room_2_reextract.yaml \
  --checkpoint output/radio_gs/room2_hybrid_v14/checkpoints/best.pth \
  --output_dir output/radio_gs/room2_hybrid_v14_reextract/visualizations_v14 \
  --num_views 10 \
  --grounding_device cpu \
  --probe_device cpu
```

This writes:

- `feature_pca/`
- `depth/`
- `segmentation/`
- `grounding/`
- `grounding_seg/`
- `composite/`

## Current implementation notes

### Architectures

The code currently supports two feature-field variants:

- `explicit`: per-Gaussian learnable feature embeddings
- `hybrid`: compact per-Gaussian latent features + hash-grid semantic branch

The current research line mainly uses the hybrid family.

### Important fixes already included in this standalone copy

- numeric frame-order RADIO feature extraction
- `frame_manifest.json` export during extraction
- legacy feature repair utility
- mixed-split path resolution utilities in `radio_gs/data/benchmark_paths.py`
- corrected visualization path resolution for fresh re-extracted features
- grounding evaluation fixes for split pose resolution and GT feature lookup

### Important scientific caveat

Fresh teacher features can fix evaluation / visualization correctness, but old checkpoints trained against contaminated teacher supervision are still only diagnostic.

For final claims, retrain the feature field with clean teacher features first, then re-run:

- `train_feature_field.py`
- `eval_rendered.py`
- `eval_grounding.py`
- `generate_visualizations_v2.py`

## What is intentionally not included

This standalone repo does **not** carry over:

- pose-estimation / localization models
- localization training entry points
- unrelated paper notes and legacy docs
- datasets
- checkpoints
- output directories

Those should live outside the repo and are ignored by `.gitignore`.

## Useful sanity checks

Import check:

```bash
python -c "from radio_gs.config import load_config; print('config ok')"
python -c "from radio_gs.models.hcd_codec import HCDCodec; print('codec ok')"
python -c "from radio_gs.rendering.feature_renderer import FeatureFieldRenderer; print('renderer ok')"
```

Smoke test:

```bash
python radio_gs/scripts/smoke_test_radio_gs.py
```

## Status

This repository snapshot is meant to be the clean starting point for the standalone scene-understanding line. It already contains the feature-index repair, fresh-feature rerun tooling, and the current `V13/V14`-era training / evaluation scripts.
