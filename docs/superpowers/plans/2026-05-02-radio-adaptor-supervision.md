# RADIO Adaptor Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add low-risk C-RADIOv4 adaptor supervision for `dino_v3` and `sam3`, plus a paper-facing SAM3/FMGS design note.

**Architecture:** Keep RADIO-GS output as reconstructed 1280d RADIO backbone features. Add frozen adaptor losses by projecting both decoded RADIO-GS features and RADIO teacher features through frozen RADIO adaptor heads, then matching them in adaptor space. Treat FMGS-style DINO/SAM alignment as auxiliary supervision rather than a replacement for the main 1280d reconstruction objective.

**Tech Stack:** PyTorch, RADIO checkpoint state dict, existing `RadioGSConfig`, existing `train_feature_field.py`, pytest.

---

### Task 1: Generic RADIO Adaptor Module

**Files:**
- Create: `radio_gs/models/radio_adaptors.py`
- Test: `tests/test_radio_adaptors.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import torch

from radio_gs.models.radio_adaptors import (
    RadioMLPAdaptor,
    load_radio_adaptor_from_checkpoint,
    project_feature_map_with_adaptor,
)


def _state(prefix: str, input_dim: int = 4, hidden_dim: int = 6, output_dim: int = 3):
    return {
        f"{prefix}.fc1.weight": torch.randn(hidden_dim, input_dim),
        f"{prefix}.fc1.bias": torch.randn(hidden_dim),
        f"{prefix}.blocks.0.0.weight": torch.ones(hidden_dim),
        f"{prefix}.blocks.0.0.bias": torch.zeros(hidden_dim),
        f"{prefix}.blocks.0.2.weight": torch.randn(hidden_dim, hidden_dim),
        f"{prefix}.blocks.0.2.bias": torch.randn(hidden_dim),
        f"{prefix}.blocks.1.0.weight": torch.ones(hidden_dim),
        f"{prefix}.blocks.1.0.bias": torch.zeros(hidden_dim),
        f"{prefix}.blocks.1.2.weight": torch.randn(hidden_dim, hidden_dim),
        f"{prefix}.blocks.1.2.bias": torch.randn(hidden_dim),
        f"{prefix}.final.0.weight": torch.ones(hidden_dim),
        f"{prefix}.final.0.bias": torch.zeros(hidden_dim),
        f"{prefix}.final.2.weight": torch.randn(output_dim, hidden_dim),
        f"{prefix}.final.2.bias": torch.randn(output_dim),
    }


def test_load_radio_adaptor_from_checkpoint_supports_dino_v3_alias(tmp_path: Path):
    ckpt = {"state_dict": _state("_feature_projections.dino_v3_7b")}
    path = tmp_path / "radio.pth"
    torch.save(ckpt, path)

    adaptor = load_radio_adaptor_from_checkpoint(path, "dino_v3", kind="feature_projection")

    assert isinstance(adaptor, RadioMLPAdaptor)
    assert adaptor.input_dim == 4
    assert adaptor.output_dim == 3


def test_project_feature_map_with_adaptor_preserves_spatial_shape():
    adaptor = RadioMLPAdaptor(input_dim=4, hidden_dim=6, output_dim=3, num_blocks=2)
    features = torch.randn(2, 4, 5, 7)

    projected = project_feature_map_with_adaptor(features, adaptor)

    assert projected.shape == (2, 3, 5, 7)
    norms = projected.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
```

- [ ] **Step 2: Run red test**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_radio_adaptors.py -q`

Expected: import failure because `radio_gs.models.radio_adaptors` does not exist.

- [ ] **Step 3: Implement module**

Create `RadioMLPAdaptor`, checkpoint-prefix alias resolution for `dino_v3 -> dino_v3_7b`, and feature-map projection.

- [ ] **Step 4: Run green test**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_radio_adaptors.py -q`

Expected: all tests pass.

### Task 2: Adaptor Consistency Loss

**Files:**
- Create: `radio_gs/losses/radio_adaptor_loss.py`
- Test: `tests/test_radio_adaptor_loss.py`

- [ ] **Step 1: Write failing tests**

```python
import torch

from radio_gs.losses.radio_adaptor_loss import compute_radio_adaptor_alignment_loss


class ScaleAdaptor(torch.nn.Module):
    def forward(self, x):
        return x * 2.0


def test_compute_radio_adaptor_alignment_loss_returns_zero_without_adaptors():
    decoded = torch.randn(1, 4, 2, 2)
    target = decoded.clone()

    loss, stats = compute_radio_adaptor_alignment_loss(decoded, target, {})

    assert loss.item() == 0.0
    assert stats == {}


def test_compute_radio_adaptor_alignment_loss_matches_identical_features():
    decoded = torch.randn(1, 4, 2, 2)
    target = decoded.clone()

    loss, stats = compute_radio_adaptor_alignment_loss(
        decoded,
        target,
        {"sam3": ScaleAdaptor()},
    )

    assert loss.item() < 1e-6
    assert "sam3" in stats
```

- [ ] **Step 2: Run red test**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_radio_adaptor_loss.py -q`

Expected: import failure because loss module does not exist.

- [ ] **Step 3: Implement loss helper**

Use frozen adaptors to project decoded and target feature maps, normalize in adaptor space, and average cosine distance across configured adaptor names.

- [ ] **Step 4: Run green test**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_radio_adaptor_loss.py -q`

Expected: all tests pass.

### Task 3: Trainer Integration

**Files:**
- Modify: `radio_gs/config.py`
- Modify: `radio_gs/scripts/train_feature_field.py`
- Test: `tests/test_radio_adaptor_trainer_config.py`

- [ ] **Step 1: Write failing tests**

```python
from radio_gs.config import RadioGSConfig
from radio_gs.scripts.train_feature_field import parse_radio_adaptor_names


def test_radio_adaptor_config_defaults_are_disabled():
    cfg = RadioGSConfig()

    assert cfg.radio_adaptor_alignment_weight == 0.0
    assert cfg.radio_adaptor_alignment_names == ""


def test_parse_radio_adaptor_names_deduplicates_and_strips():
    assert parse_radio_adaptor_names("dino_v3, sam3, dino_v3") == ["dino_v3", "sam3"]
```

- [ ] **Step 2: Run red test**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_radio_adaptor_trainer_config.py -q`

Expected: missing config fields or helper.

- [ ] **Step 3: Add config fields and trainer hooks**

Add disabled-by-default config fields:

```python
radio_adaptor_alignment_names: str = ""
radio_adaptor_alignment_weight: float = 0.0
radio_adaptor_alignment_kind: str = "feature_projection"
radio_adaptor_alignment_checkpoint: str = "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
```

Trainer loads frozen adaptors only when names and weight are both set, computes `l_radio_adaptors`, adds it to training loss and validation metrics.

- [ ] **Step 4: Run green tests**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_radio_adaptor_trainer_config.py tests/test_radio_adaptor_loss.py tests/test_radio_adaptors.py -q`

Expected: all tests pass.

### Task 4: SAM3/FMGS Design Note

**Files:**
- Create: `docs/radio_adaptor_sam3_fmgs_strategy.md`

- [ ] **Step 1: Document source-grounded design**

Summarize:
- FMGS uses DINO features and dot-product pixel-alignment to sharpen CLIP-like feature fields.
- RADIO-GS can map the same idea to DINOv3 adaptor consistency and SAM3 mask-region consistency.
- Stage 1 implements adaptor feature consistency.
- Stage 2 should add SAM3 mask prior caches and mask-aware region/boundary losses.

- [ ] **Step 2: Verify docs mention exact enabled fields**

Run: `rg -n "radio_adaptor_alignment|SAM3|FMGS|DINOv3" docs/radio_adaptor_sam3_fmgs_strategy.md`

Expected: all terms are present.

### Task 5: Verification and Commit

- [ ] **Step 1: Run targeted tests**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_radio_adaptors.py tests/test_radio_adaptor_loss.py tests/test_radio_adaptor_trainer_config.py -q`

Expected: pass.

- [ ] **Step 2: Run smoke import**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh - <<'PY'\nfrom radio_gs.config import RadioGSConfig\nfrom radio_gs.models.radio_adaptors import load_radio_adaptor_from_checkpoint\ncfg = RadioGSConfig(radio_adaptor_alignment_names='dino_v3,sam3', radio_adaptor_alignment_weight=0.05)\nprint(cfg.radio_adaptor_alignment_names, cfg.radio_adaptor_alignment_weight)\nPY`

Expected: prints `dino_v3,sam3 0.05`.

- [ ] **Step 3: Diff check**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 4: Commit only this task's files**

Run:

```bash
git add docs/superpowers/plans/2026-05-02-radio-adaptor-supervision.md docs/radio_adaptor_sam3_fmgs_strategy.md radio_gs/config.py radio_gs/models/radio_adaptors.py radio_gs/losses/radio_adaptor_loss.py radio_gs/scripts/train_feature_field.py tests/test_radio_adaptors.py tests/test_radio_adaptor_loss.py tests/test_radio_adaptor_trainer_config.py
git commit -m "feat: add RADIO adaptor consistency supervision"
```

