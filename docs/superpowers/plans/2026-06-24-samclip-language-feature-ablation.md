# SAM-CLIP Language-Feature Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and smoke-test a RADIO-GS ablation that trains on LangSplat cached SAM-CLIP `language_features` instead of RADIO feature targets.

**Architecture:** Add a small data-conversion CLI that materializes LangSplat segment-prototype caches as dense `[512,H,W]` `rgb_*.pt` tensors, then add config-generation helpers that clone matched RADIO-GS configs with RADIO/SigLIP helper modules disabled. Reuse the existing rendered-feature and pre-rendered LERF OpenCLIP evaluator first, then add explicit OpenCLIP feature-space branches for direct-3D LERF and ScanNet point segmentation after the training smoke path is working.

**Tech Stack:** Python, PyTorch, NumPy, PyYAML, pytest, OpenCLIP, existing RADIO-GS `train_feature_field.py`, `render_codec_features.py`, and LERF/ScanNet evaluators. GPU experiments must use `CUDA_VISIBLE_DEVICES=4`.

---

## File Structure

- Create `radio_gs/scripts/convert_langsplat_language_features.py`
  Converts `*_s.npy`/`*_f.npy` to dense torch tensor caches and writes a JSON manifest.
- Create `tests/test_convert_langsplat_language_features.py`
  Unit tests for materialization, invalid pixels, frame-id mapping, discovery, and dry-run conversion.
- Create `radio_gs/scripts/generate_samclip_ablation_configs.py`
  Clones existing YAML configs into SAM-CLIP variants with `radio_feature_dim: 512` and disabled RADIO/SigLIP helper modules.
- Create `tests/test_generate_samclip_ablation_configs.py`
  Unit tests for LERF and ScanNet config cloning.
- Create `radio_gs/evaluation/openclip_readout.py`
  Shared OpenCLIP text scorer and pure tensor scoring helpers.
- Modify `radio_gs/scripts/eval_prerendered_lerf_features.py`
  Import the shared scorer without changing the existing CLI behavior.
- Create `tests/test_openclip_readout.py`
  Unit tests for embedding normalization and relevance scoring using explicit fake embeddings.
- Modify `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
  Add `--feature_space openclip` path that skips SigLIP summary projection for 512-D decoded features.
- Modify `radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py`
  Add `--feature_space openclip` path that directly scores decoded 512-D point features against OpenCLIP text embeddings.

## Task 1: Converter Unit Tests

**Files:**
- Create: `tests/test_convert_langsplat_language_features.py`

- [ ] **Step 1: Write converter tests**

Add tests with these cases:

```python
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from radio_gs.scripts import convert_langsplat_language_features as conv


def test_materialize_dense_feature_handles_invalid_pixels():
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    seg = np.array(
        [
            [[0, 1], [2, -1]],
            [[1, 1], [0, 0]],
            [[2, 2], [1, 1]],
            [[0, 0], [2, 2]],
        ],
        dtype=np.int32,
    )

    dense = conv.materialize_dense_feature(features, seg, level=0, output_size=None, dtype=torch.float32)

    assert dense.shape == (2, 2, 2)
    assert torch.allclose(dense[:, 0, 0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(dense[:, 0, 1], torch.tensor([0.0, 1.0]))
    assert torch.allclose(dense[:, 1, 0], torch.tensor([0.6, 0.8]), atol=1e-6)
    assert torch.allclose(dense[:, 1, 1], torch.zeros(2))


def test_materialize_dense_feature_rejects_out_of_range_ids():
    features = np.eye(2, dtype=np.float32)
    seg = np.zeros((4, 2, 2), dtype=np.int32)
    seg[1, 0, 0] = 2

    with pytest.raises(ValueError, match="out of range"):
        conv.materialize_dense_feature(features, seg, level=1, output_size=None, dtype=torch.float32)


def test_feature_stem_to_frame_id_lerf_and_scannet():
    assert conv.feature_stem_to_frame_id("frame_00016") == 16
    assert conv.feature_stem_to_frame_id("0") == 0
    assert conv.output_feature_name("frame_00016") == "rgb_16.pt"
    assert conv.output_feature_name("0") == "rgb_0.pt"


def test_convert_scene_writes_manifest_and_tensor(tmp_path):
    source = tmp_path / "language_features"
    source.mkdir()
    np.save(source / "frame_00001_f.npy", np.eye(3, dtype=np.float32))
    np.save(source / "frame_00001_s.npy", np.zeros((4, 2, 2), dtype=np.int32))
    output = tmp_path / "converted"

    manifest = conv.convert_scene(
        source,
        output,
        levels=[1],
        output_size=(1, 1),
        dtype=torch.float32,
        dry_run=False,
    )

    tensor_path = output / "l1" / "backbone" / "rgb_1.pt"
    assert tensor_path.exists()
    tensor = torch.load(tensor_path, map_location="cpu")
    assert tensor.shape == (3, 1, 1)
    assert manifest["levels"] == [1]
    assert manifest["frames_converted"] == 1
    assert (output / "l1" / "samclip_manifest.json").exists()
    saved = json.loads((output / "l1" / "samclip_manifest.json").read_text(encoding="utf-8"))
    assert saved["feature_dim"] == 3
```

- [ ] **Step 2: Run tests and confirm they fail before implementation**

Run:

```bash
pytest tests/test_convert_langsplat_language_features.py -q
```

Expected: fails with `ModuleNotFoundError` or missing attributes because the converter does not exist yet.

## Task 2: Converter Implementation

**Files:**
- Create: `radio_gs/scripts/convert_langsplat_language_features.py`
- Test: `tests/test_convert_langsplat_language_features.py`

- [ ] **Step 1: Implement converter**

Create a CLI with these public functions and exact signatures:

```python
def feature_stem_to_frame_id(stem: str) -> int:
    """Return the integer frame id from a LERF or ScanNet language-feature stem."""


def output_feature_name(stem: str) -> str:
    """Return the RADIO-GS tensor-cache filename for a language-feature stem."""


def materialize_dense_feature(
    feature_map: np.ndarray,
    seg_map: np.ndarray,
    *,
    level: int,
    output_size: tuple[int, int] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert one segment-prototype map into a dense [C,H,W] tensor."""


def discover_feature_pairs(language_feature_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return sorted (stem, f_path, s_path) tuples with both cache files present."""


def convert_scene(
    language_feature_dir: Path,
    output_root: Path,
    *,
    levels: Sequence[int],
    output_size: tuple[int, int] | None,
    dtype: torch.dtype,
    dry_run: bool,
) -> dict[str, object]:
    """Convert all cache pairs in a scene and return the manifest payload."""
```

Implementation requirements:

- Validate `seg_map.ndim == 3`, `feature_map.ndim == 2`, and `level` is in range.
- Preserve `-1` invalid pixels as zero vectors.
- Reject valid segment ids outside `[0, feature_map.shape[0])`.
- Normalize valid vectors with `torch.nn.functional.normalize`.
- Resize only after constructing `[1,C,H,W]`; use bilinear interpolation and renormalize valid nonzero pixels.
- Write tensors to `<output_root>/l<level>/backbone/rgb_<frame_id>.pt`.
- Write `<output_root>/l<level>/samclip_manifest.json`.
- CLI arguments: `--language-feature-dir`, `--output-root`, `--levels`, `--output-size H W`, `--dtype fp16|fp32`, `--dry-run`.

- [ ] **Step 2: Run converter unit tests**

Run:

```bash
pytest tests/test_convert_langsplat_language_features.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Dry-run real LERF and ScanNet sources**

Run:

```bash
python -m radio_gs.scripts.convert_langsplat_language_features \
  --language-feature-dir /mnt/pool/sqy/3d_understanding/lerf_ovs/figurines/langsplat/language_features \
  --output-root output/samclip_features_lerf/figurines \
  --levels 1 \
  --output-size 90 123 \
  --dry-run

python -m radio_gs.scripts.convert_langsplat_language_features \
  --language-feature-dir /root/RADIO-GS/dataset/scannet_og/scene0000_00/language_features \
  --output-root output/samclip_features_scannet_og/scene0000_00 \
  --levels 1 \
  --output-size 60 80 \
  --dry-run
```

Expected: both commands print frame counts and feature dimension `512`.

## Task 3: SAM-CLIP Config Generator

**Files:**
- Create: `tests/test_generate_samclip_ablation_configs.py`
- Create: `radio_gs/scripts/generate_samclip_ablation_configs.py`

- [ ] **Step 1: Write config cloning tests**

Add tests that create a small template YAML and assert the generated YAML has:

```python
radio_feature_dim: 512
feature_dir: <samclip root>/<scene>/l1
val_feature_dir: <samclip root>/<scene>/l1
siglip_alignment_weight: 0.0
siglip_summary_alignment_weight: 0.0
text_heatmap_distill_weight: 0.0
radio_adaptor_cross_view_weight: 0.0
direct_point_teacher_cache: ""
direct_point_text_embeddings: ""
siglip_projection_weights: ""
```

Also assert `output_dir` and `exp_name` include `samclip_l1`.

- [ ] **Step 2: Implement config generator**

Create these functions:

```python
RADIO_HELPER_ZERO_KEYS = (
    "siglip_alignment_weight",
    "siglip_summary_alignment_weight",
    "text_heatmap_distill_weight",
    "radio_adaptor_alignment_weight",
    "radio_adaptor_relation_weight",
    "radio_adaptor_local_affinity_weight",
    "radio_adaptor_token_contrast_weight",
    "radio_adaptor_peak_background_weight",
    "radio_adaptor_region_weight",
    "radio_adaptor_mask_logit_weight",
    "radio_adaptor_cross_view_weight",
    "radio_adaptor_cross_view_propagation_weight",
    "radio_adaptor_cross_view_mask_propagation_weight",
    "direct_point_loss_weight",
    "direct_point_summary_alignment_weight",
    "direct_point_summary_adapter_weight",
    "direct_point_text_loss_weight",
    "direct_point_adapter_text_loss_weight",
    "direct_point_adapter_text_distill_weight",
    "direct_point_text_pseudo_ce_weight",
    "direct_point_adapter_text_pseudo_ce_weight",
    "direct_point_adapter_decoder_anchor_weight",
    "direct_point_text_distill_weight",
)
RADIO_HELPER_EMPTY_KEYS = (
    "siglip_projection_weights",
    "siglip_summary_head_weights",
    "direct_point_teacher_cache",
    "direct_point_text_embeddings",
    "text_heatmap_distill_embeddings",
)
```

The CLI accepts `--templates`, `--scene`, `--samclip-root`, `--level`, `--output-root`, `--variant`, and `--repo-root`.

- [ ] **Step 3: Run config tests**

Run:

```bash
pytest tests/test_generate_samclip_ablation_configs.py -q
```

Expected: all tests pass.

## Task 4: Shared OpenCLIP Readout Helper

**Files:**
- Create: `radio_gs/evaluation/openclip_readout.py`
- Create: `tests/test_openclip_readout.py`
- Modify: `radio_gs/scripts/eval_prerendered_lerf_features.py`

- [ ] **Step 1: Write readout tests**

Test pure tensor behavior without importing OpenCLIP:

```python
import torch

from radio_gs.evaluation.openclip_readout import cosine_logits, normalized_embeddings


def test_normalized_embeddings_leaves_zero_rows_zero():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    y = normalized_embeddings(x)
    assert torch.allclose(y[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(y[1], torch.zeros(2))


def test_cosine_logits_scores_features_against_text():
    features = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = cosine_logits(features, text)
    assert torch.allclose(logits, torch.eye(2), atol=1e-6)
```

- [ ] **Step 2: Implement helper and refactor evaluator import**

Move reusable scoring into `openclip_readout.py` with these public objects:

```python
NEGATIVE_PROMPTS = ("object", "things", "stuff", "texture")


def normalized_embeddings(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize rows or channel vectors while keeping all-zero vectors zero."""


def cosine_logits(features: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    """Return normalized feature/text cosine logits."""


class OpenCLIPTextScorer:
    """OpenCLIP text encoder plus LERF relevance-map scoring."""
```

Keep `eval_prerendered_lerf_features.py` CLI output unchanged.

- [ ] **Step 3: Run readout and existing LERF tests**

Run:

```bash
pytest tests/test_openclip_readout.py tests/test_eval_prerendered_lerf_features.py -q
```

Expected: all tests pass.

## Task 5: Real Cache Conversion Smoke

**Files:**
- No code files unless earlier tasks reveal a converter bug.

- [ ] **Step 1: Convert one LERF scene and one ScanNet scene**

Run CPU conversion:

```bash
python -m radio_gs.scripts.convert_langsplat_language_features \
  --language-feature-dir /mnt/pool/sqy/3d_understanding/lerf_ovs/figurines/langsplat/language_features \
  --output-root output/samclip_features_lerf/figurines \
  --levels 1 \
  --output-size 90 123 \
  --dtype fp16

python -m radio_gs.scripts.convert_langsplat_language_features \
  --language-feature-dir /root/RADIO-GS/dataset/scannet_og/scene0000_00/language_features \
  --output-root output/samclip_features_scannet_og/scene0000_00 \
  --levels 1 \
  --output-size 60 80 \
  --dtype fp16
```

Expected: manifests are written and sample tensors have shape `[512,90,123]` and `[512,60,80]`.

- [ ] **Step 2: Inspect sample tensors**

Run:

```bash
python - <<'PY'
import torch
for p in [
    "output/samclip_features_lerf/figurines/l1/backbone/rgb_1.pt",
    "output/samclip_features_scannet_og/scene0000_00/l1/backbone/rgb_0.pt",
]:
    t = torch.load(p, map_location="cpu")
    print(p, tuple(t.shape), t.dtype, torch.isfinite(t).all().item(), float(t.abs().sum()))
PY
```

Expected: finite tensors with nonzero sums.

## Task 6: Generate Smoke Configs

**Files:**
- Generated YAML only under `radio_gs/configs/generated/samclip_ablation/`

- [ ] **Step 1: Generate one LERF and one ScanNet smoke config**

Run:

```bash
python -m radio_gs.scripts.generate_samclip_ablation_configs \
  --templates radio_gs/configs/generated/ablation/lerf_figurines_component_direct_codec_seed7.yaml \
  --scene figurines \
  --samclip-root output/samclip_features_lerf \
  --level 1 \
  --output-root radio_gs/configs/generated/samclip_ablation \
  --variant samclip_l1_smoke_e1 \
  --epochs 1

python -m radio_gs.scripts.generate_samclip_ablation_configs \
  --templates radio_gs/configs/generated/scannet_og/scannet_og_hybrid_v14_scene0000_00.yaml \
  --scene scene0000_00 \
  --samclip-root output/samclip_features_scannet_og \
  --level 1 \
  --output-root radio_gs/configs/generated/samclip_ablation \
  --variant samclip_l1_smoke_e1 \
  --epochs 1
```

Expected: generated configs load with `radio_gs.config.load_config` and `radio_feature_dim == 512`.

## Task 7: GPU4 Training Smoke

**Files:**
- Output artifacts under `output/radio_gs/*samclip*`

- [ ] **Step 1: Train LERF smoke on GPU4**

Run:

```bash
CUDA_VISIBLE_DEVICES=4 python -m radio_gs.scripts.train_feature_field \
  --config radio_gs/configs/generated/samclip_ablation/lerf_figurines_samclip_l1_smoke_e1.yaml
```

Expected: training reaches the end of epoch 1 and writes a checkpoint without loading SigLIP/RADIO helper checkpoints.

- [ ] **Step 2: Train ScanNet smoke on GPU4**

Run:

```bash
CUDA_VISIBLE_DEVICES=4 python -m radio_gs.scripts.train_feature_field \
  --config radio_gs/configs/generated/samclip_ablation/scene0000_00_samclip_l1_smoke_e1.yaml
```

Expected: training reaches the end of epoch 1 and writes a checkpoint without loading SigLIP/RADIO helper checkpoints.

## Task 8: LERF 2D OpenCLIP Smoke Evaluation

**Files:**
- Output artifacts under `output/samclip_ablation_eval/`

- [ ] **Step 1: Render decoded features from the LERF smoke checkpoint**

Run:

```bash
CUDA_VISIBLE_DEVICES=4 python -m radio_gs.scripts.render_codec_features \
  --config radio_gs/configs/generated/samclip_ablation/lerf_figurines_samclip_l1_smoke_e1.yaml \
  --checkpoint output/radio_gs/lerf_figurines_samclip_l1_smoke_e1/checkpoints/best.pth \
  --output_dir output/samclip_ablation_eval/figurines_l1_smoke_rendered \
  --split val \
  --gpu 0
```

Expected: rendered tensors are written under `output/samclip_ablation_eval/figurines_l1_smoke_rendered/backbone`.

- [ ] **Step 2: Convert rendered `.pt` tensors to pre-rendered `.npy` maps**

Add `radio_gs/scripts/convert_rendered_pt_features_to_npy.py`. It reads
`backbone/rgb_*.pt`, writes `frame_XXXXX.npy` with shape `[H,W,512]`, and
accepts `--input-dir`, `--output-dir`, and `--frame-prefix frame_`.

- [ ] **Step 3: Run LERF 2D OpenCLIP evaluation on GPU4**

Run:

```bash
CUDA_VISIBLE_DEVICES=4 python -m radio_gs.scripts.eval_prerendered_lerf_features \
  --label-root /mnt/pool/sqy/3d_understanding/lerf_ovs/label/figurines \
  --feature-dirs output/samclip_ablation_eval/figurines_l1_smoke_rendered_npy \
  --scene figurines \
  --device cuda \
  --output-json output/samclip_ablation_eval/figurines_l1_smoke_lerf2d.json
```

Expected: JSON with `macro.loc_acc`, `macro.miou`, and `macro.objects`.

## Task 9: Direct-3D LERF and ScanNet OpenCLIP Branches

**Files:**
- Modify: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- Modify: `radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py`
- Tests: extend `tests/test_lerf_direct_3d_selection.py` and `tests/test_scannet_pointcloud_eval.py`

- [ ] **Step 1: Add CLI flags**

Add to both scripts:

```python
parser.add_argument("--feature_space", choices=("siglip_summary", "openclip"), default="siglip_summary")
parser.add_argument("--openclip_model", default="ViT-B-16")
parser.add_argument("--openclip_pretrained", default="laion2b_s34b_b88k")
```

- [ ] **Step 2: Skip summary projection in OpenCLIP mode**

When `feature_space == "openclip"`, decoded features must satisfy `features.shape[-1] == 512` or rendered maps must satisfy `C == 512`. Normalize them and compute cosine logits directly against OpenCLIP class/query text embeddings.

- [ ] **Step 3: Run targeted tests**

Run:

```bash
pytest tests/test_lerf_direct_3d_selection.py tests/test_scannet_pointcloud_eval.py -q
```

Expected: existing SigLIP behavior remains unchanged and new OpenCLIP unit tests pass with fake embeddings.

## Task 10: Experiment Launch Policy on GPU4

**Files:**
- Generated result JSON under `output/samclip_ablation_eval/`

- [ ] **Step 1: Confirm GPU4 is still idle**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Expected: GPU index `4` has enough free memory.

- [ ] **Step 2: Use GPU4 for all CUDA work**

Every CUDA command must set:

```bash
CUDA_VISIBLE_DEVICES=4
```

Inside scripts that also take `--gpu`, pass `--gpu 0` because the process-visible GPU 0 maps to physical GPU4.

- [ ] **Step 3: Keep GPU4 saturated during formal runs**

After smoke passes, run one training process at the largest stable batch size first. If memory utilization remains low, run conversion/CPU preprocessing in parallel but avoid two training processes unless a single process leaves large unused memory and dataloader stalls dominate.

## Task 11: Final Verification

**Files:**
- All touched code and generated smoke artifacts.

- [ ] **Step 1: Run unit tests**

Run:

```bash
pytest \
  tests/test_convert_langsplat_language_features.py \
  tests/test_generate_samclip_ablation_configs.py \
  tests/test_openclip_readout.py \
  tests/test_eval_prerendered_lerf_features.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Capture smoke results**

Record:

- Converter manifest paths.
- Generated config paths.
- LERF smoke checkpoint path.
- ScanNet smoke checkpoint path.
- LERF 2D smoke result JSON.
- Any direct-3D/ScanNet OpenCLIP smoke result JSON that completes.

- [ ] **Step 3: Commit implementation**

Run:

```bash
git status --short
git add radio_gs/evaluation/openclip_readout.py \
  radio_gs/scripts/convert_langsplat_language_features.py \
  radio_gs/scripts/generate_samclip_ablation_configs.py \
  radio_gs/scripts/eval_prerendered_lerf_features.py \
  radio_gs/scripts/eval_lerf_direct_3d_selection.py \
  radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py \
  tests/test_convert_langsplat_language_features.py \
  tests/test_generate_samclip_ablation_configs.py \
  tests/test_openclip_readout.py
git commit -m "feat: add samclip language feature ablation pipeline"
```

Expected: commit succeeds. If direct-3D or ScanNet evaluator changes are not completed in this first implementation pass, omit those two files from the commit and clearly report them as remaining work.
