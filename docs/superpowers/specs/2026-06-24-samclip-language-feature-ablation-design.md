# SAM-CLIP Language-Feature Ablation Design

## Goal

Add a reproducible ablation that replaces RADIO feature supervision with the
precomputed LangSplat/OpenGaussian `language_features` caches while keeping the
RADIO-GS hybrid feature-field machinery as fixed as possible. The experiment
answers whether the hybrid field gives stable gains when the base supervision is
SAM-segmented CLIP/OpenCLIP rather than RADIO.

## Scope

The primary ablation uses cached `*_s.npy` and `*_f.npy` language features from:

- `/mnt/pool/sqy/3d_understanding/lerf_ovs`
- `/root/RADIO-GS/dataset/scannet_og`

The first runnable target is a smoke run on one LERF scene and one prepared
ScanNet scene. The paper-facing target is all four LERF OVS scenes and the
VALA8 ScanNet scenes:

`scene0000_00`, `scene0062_00`, `scene0070_00`, `scene0097_00`,
`scene0140_00`, `scene0347_00`, `scene0400_00`, `scene0590_00`.

## Non-Goals

- Do not re-extract pure dense CLIP patch features.
- Do not add a new test-time SAM proposal/refinement stage to the main
  ablation.
- Do not use existing RADIO teacher caches, RADIO adaptor losses, SigLIP
  projection heads, or SigLIP text embeddings for this ablation.
- Do not compare a full RADIO+DINO/SAM helper stack against a stripped CLIP
  stack as the main conclusion. Any such comparison must be labeled as an
  auxiliary benchmark-matched variant.

## Data Format

LangSplat language caches are segment-prototype features:

- `*_f.npy`: `N x 512` CLIP/OpenCLIP segment prototypes.
- `*_s.npy`: `4 x H x W` segment id maps, with invalid pixels marked as `-1`.

RADIO-GS training loaders already expect dense torch tensors under
`backbone/rgb_*.pt`. The converter will therefore materialize dense
`[512, Hf, Wf]` tensors by selecting a mask level and assigning:

`dense[y, x] = feature_map[seg_map[level, y, x]]`

Invalid pixels are stored as zeros and must not index `feature_map[-1]`. The
initial implementation writes only the dense tensor cache, matching the current
RADIO-GS loader contract. Valid pixels are L2-normalized per pixel after
resizing, while zero invalid pixels remain zero. Output roots use explicit level
names:

- `output/samclip_features_lerf_l1/<scene>/backbone/rgb_<frame>.pt`
- `output/samclip_features_scannet_og_l1/<scene>/backbone/rgb_<frame>.pt`

The converter supports levels `1`, `2`, and `3`. Level `1` is the default smoke
path because LERF-style baselines commonly use a small-object level for rendered
feature maps. The formal sweep should include all three levels and optionally an
ensemble readout.

## Components

1. `radio_gs/scripts/convert_langsplat_language_features.py`
   Converts `*_s.npy`/`*_f.npy` caches into RADIO-GS-compatible dense tensor
   caches. It supports LERF and ScanNet filename conventions, validates index
   ranges, writes a manifest, and can run in dry-run mode.

2. Config generation helpers
   Clone existing LERF and ScanNet hybrid configs into SAM-CLIP variants. The
   generated configs set `radio_feature_dim: 512`, point `feature_dir` and
   `val_feature_dir` at the converted caches, and disable all RADIO/SigLIP
   helper losses and checkpoints.

3. CLIP text readout
   Add or reuse an OpenCLIP text scorer for LERF 2D, LERF direct 3D, and
   ScanNet point segmentation. The scorer must match the cached feature space
   used by the LangSplat caches, defaulting to `ViT-B-16` with
   `laion2b_s34b_b88k` unless the cache metadata proves otherwise.

4. Evaluation entry points
   LERF 2D can reuse the pre-rendered feature evaluator after rendering
   SAM-CLIP field outputs as dense maps. LERF 3D and ScanNet point segmentation
   need a `--feature-space openclip` or equivalent path that skips SigLIP/RADIO
   projection and directly scores normalized 512-D features against OpenCLIP
   text embeddings.

## Configuration Policy

The clean main ablation keeps the hybrid field but disables RADIO-specific
modules:

- `radio_feature_dim: 512`
- `siglip_alignment_weight: 0`
- `siglip_summary_alignment_weight: 0`
- `text_heatmap_distill_weight: 0`
- all `radio_adaptor_*_weight: 0`
- no `siglip_projection_weights`
- no `summary_head_weights`
- no `direct_point_teacher_cache` that contains RADIO/SigLIP features
- no `direct_point_text_embeddings` from SigLIP2 checkpoints

HCD, feature reconstruction, visibility/depth geometry machinery, and RGB/3DGS
geometry assets remain unchanged when they are not tied to RADIO dimensions.

## Data Flow

1. Convert cached language features into dense 512-D tensor caches.
2. Generate SAM-CLIP configs from the matched RADIO-GS configs.
3. Train the hybrid field from scratch using the converted 512-D targets.
4. Render/evaluate LERF 2D with OpenCLIP text scoring.
5. Evaluate direct 3D LERF and ScanNet point features with the same OpenCLIP
   text scoring policy.
6. Aggregate results by scene and feature level.

## Validation

Converter validation:

- Unit test a small synthetic `*_s.npy`/`*_f.npy` pair with invalid pixels and
  out-of-range segment ids.
- Dry-run one LERF scene and one ScanNet scene.
- Convert one frame and verify tensor shape, dtype, finite values, and per-pixel
  normalization.

Training validation:

- Run one short LERF training job with `radio_feature_dim: 512`.
- Run one short ScanNet training job with `radio_feature_dim: 512`.
- Confirm no SigLIP summary/projection checkpoint is loaded.

Evaluation validation:

- Run LERF 2D OpenCLIP readout on one rendered scene.
- Run direct 3D LERF OpenCLIP readout on one checkpoint.
- Run ScanNet point segmentation OpenCLIP readout on one checkpoint.
- Confirm result JSON records dataset, scene, feature level, OpenCLIP model,
  and disabled helper modules.

## Risks

- The LangSplat cache may have been generated with a different CLIP/OpenCLIP
  checkpoint than the evaluator default. The converter manifest should record
  the assumed model and the evaluator should make the model explicit.
- Level choice affects object scale. Results must be reported per level instead
  of silently selecting the best level.
- Some existing ScanNet training configs rely on RADIO teacher point caches.
  A matched clean comparison must disable those modules for both SAM-CLIP and
  the corresponding RADIO-lite control, or clearly label the comparison as
  auxiliary.
- Dense materialization can be large. The converter should support `float16`
  output and per-scene incremental conversion.

## Acceptance Criteria

- The converter produces RADIO-GS-compatible dense feature caches for at least
  one LERF and one ScanNet scene.
- A generated SAM-CLIP config trains without 1280-D RADIO/SigLIP assumptions.
- LERF 2D, LERF 3D, and ScanNet point segmentation have an OpenCLIP readout path
  that accepts 512-D rendered/decoded features.
- Smoke results are written to auditable JSON files with scene, level, and
  feature-space metadata.
