# Journal Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the project toward a journal-grade version by strengthening the persistent direct field, adding boundary-quality evidence, and introducing optional official foundation-head cache supervision while preserving existing protocols.

**Architecture:** The upgrade is split into three independent tracks. Direct-field v2 improves VPR-to-field distillation without touching the evaluator. Boundary evaluation adds GT-free mask refinement diagnostics and boundary metrics without changing primitive selection. Foundation cache supervision adds optional official DINO/SAM/SigLIP cache readers and losses while keeping current RADIO-adaptor fallback behavior unchanged.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, pytest, existing RADIO-GS training/evaluation scripts.

---

### Task 1: Direct-Field VPR Distillation v2

**Files:**
- Modify: `radio_gs/scripts/train_scannet_point_summary_adapter.py`
- Modify: `tests/test_point_summary_adapter_training.py`

- [ ] **Step 1: Write failing tests**

Add tests covering:

```python
def test_teacher_sample_weights_support_clipped_log_mode():
    cache = {
        "view_counts": torch.tensor([0.0, 1.0, 10.0, 100.0]),
        "valid": torch.tensor([False, True, True, True]),
    }
    weights = _build_teacher_sample_weights(
        cache,
        mode="clipped_log",
        min_weight=0.25,
        percentile_low=0.0,
        percentile_high=75.0,
    )
    assert weights[0].item() == 0.0
    assert weights[1:].min().item() >= 0.25
    assert weights.max().item() <= 1.0
```

```python
def test_adapter_text_rank_loss_matches_teacher_ordering():
    pred = torch.tensor([[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]])
    teacher = torch.tensor([[2.0, 0.0, -2.0], [0.0, 1.0, -1.0]])
    loss, stats = compute_text_rank_distillation_loss(
        pred,
        teacher,
        sample_weights=torch.tensor([1.0, 0.5]),
        margin=0.1,
        topk=2,
    )
    assert loss.item() >= 0.0
    assert stats["rank_pairs"].item() > 0
```

- [ ] **Step 2: Run RED**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_point_summary_adapter_training.py::test_teacher_sample_weights_support_clipped_log_mode tests/test_point_summary_adapter_training.py::test_adapter_text_rank_loss_matches_teacher_ordering
```

Expected: fail because `clipped_log` and `compute_text_rank_distillation_loss` are not implemented.

- [ ] **Step 3: Implement minimal direct-field v2**

Add:

```text
--teacher_sample_weight_mode clipped_log
--teacher_sample_weight_percentile_low
--teacher_sample_weight_percentile_high
--text_rank_distill_weight
--text_rank_distill_margin
--text_rank_distill_topk
```

Use the teacher cache `view_counts` only; do not use LERF masks. Apply the new rank loss only when teacher and student text-score tensors are already available.

- [ ] **Step 4: Run GREEN**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_point_summary_adapter_training.py
bash radio_gs/scripts/run_repo_python.sh -m py_compile radio_gs/scripts/train_scannet_point_summary_adapter.py
```

Expected: tests pass and script compiles.

### Task 2: Boundary Metrics and Refined-Mask Reporting

**Files:**
- Modify: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- Modify: `tests/test_lerf_direct_3d_selection.py`

- [ ] **Step 1: Write failing tests**

Add tests covering:

```python
def test_boundary_f_score_prefers_aligned_edges():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:24, 8:24] = 1
    aligned = gt.copy()
    shifted = np.zeros_like(gt)
    shifted[10:26, 10:26] = 1
    assert boundary_f_score(aligned, gt, dilation_ratio=0.02) > boundary_f_score(shifted, gt, dilation_ratio=0.02)
```

```python
def test_trimap_iou_ignores_far_background():
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:24, 8:24] = 1
    pred = gt.copy()
    pred[0:2, 0:2] = 1
    score = trimap_iou(pred, gt, dilation_pixels=2)
    assert 0.95 <= score <= 1.0
```

- [ ] **Step 2: Run RED**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_lerf_direct_3d_selection.py::test_boundary_f_score_prefers_aligned_edges tests/test_lerf_direct_3d_selection.py::test_trimap_iou_ignores_far_background
```

Expected: fail because `boundary_f_score` and `trimap_iou` are not implemented.

- [ ] **Step 3: Implement metrics**

Add deterministic binary boundary extraction using OpenCV morphology. Include `boundary_f`, `trimap_iou`, and existing overlap stats in per-query JSON output. Keep raw mIoU/Acc unchanged.

- [ ] **Step 4: Run GREEN**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_lerf_direct_3d_selection.py
bash radio_gs/scripts/run_repo_python.sh -m py_compile radio_gs/scripts/eval_lerf_direct_3d_selection.py
```

Expected: tests pass and script compiles.

### Task 3: Optional Official Foundation-Cache Supervision

**Files:**
- Create: `radio_gs/models/foundation_cache.py`
- Modify: `radio_gs/config.py`
- Modify: `radio_gs/scripts/train_feature_field.py`
- Create or modify: `tests/test_foundation_cache.py`

- [ ] **Step 1: Write failing tests**

Add tests covering:

```python
def test_load_foundation_cache_accepts_mask_logits_and_tokens(tmp_path):
    path = tmp_path / "cache.pt"
    torch.save(
        {
            "version": 1,
            "frame_id": 7,
            "heads": {
                "sam3": {"mask_logits": torch.randn(2, 8, 8)},
                "dino_v3": {"tokens": torch.randn(4, 16)},
                "siglip2": {"tokens": torch.randn(4, 16)},
            },
        },
        path,
    )
    cache = load_foundation_cache(path)
    assert cache.heads["sam3"].mask_logits.shape == (2, 8, 8)
    assert cache.heads["dino_v3"].tokens.shape == (4, 16)
```

```python
def test_foundation_cache_loss_is_zero_without_cache():
    loss, stats = compute_foundation_cache_supervision_loss(
        decoded_features=torch.randn(1, 1280, 4, 4),
        cache=None,
        projectors={},
    )
    assert loss.item() == 0.0
    assert stats["enabled"] == 0
```

- [ ] **Step 2: Run RED**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_foundation_cache.py
```

Expected: fail because the cache module does not exist.

- [ ] **Step 3: Implement cache types and loss helper**

Implement a small typed loader for external caches:

```text
version: 1
frame_id: int or str
heads:
  sam3:
    mask_logits: [M,H,W]
  dino_v3:
    tokens: [N,C] or feature_map: [C,H,W]
  siglip2:
    tokens: [N,C] or feature_map: [C,H,W]
```

Keep this optional. If no cache is configured, behavior must be bit-for-bit equivalent except for extra zero-valued stats.

- [ ] **Step 4: Wire optional config**

Add config keys:

```text
foundation_cache_root: ""
foundation_cache_weight: 0.0
foundation_cache_heads: ""
foundation_cache_mask_logit_weight: 0.0
foundation_cache_token_weight: 0.0
```

The first implementation may only load and compute explicit auxiliary losses when matching cache files exist. Missing cache files should log a warning and skip the loss, not crash training.

- [ ] **Step 5: Run GREEN**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_foundation_cache.py tests/test_radio_adaptor_loss.py
bash radio_gs/scripts/run_repo_python.sh -m py_compile radio_gs/models/foundation_cache.py radio_gs/scripts/train_feature_field.py
```

Expected: tests pass and scripts compile.

### Task 4: Integration and Experiment Gate

**Files:**
- Modify: `docs/PROJECT_MAINLINE.md`
- Modify: `docs/submission_status.md`
- Modify as needed: `paper/radio_gs_draft.tex`

- [ ] **Step 1: Run focused regression**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_point_summary_adapter_training.py tests/test_lerf_direct_3d_selection.py tests/test_foundation_cache.py tests/test_radio_adaptor_loss.py
```

Expected: all tests pass.

- [ ] **Step 2: Run compile checks**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m py_compile radio_gs/scripts/train_scannet_point_summary_adapter.py radio_gs/scripts/eval_lerf_direct_3d_selection.py radio_gs/models/foundation_cache.py radio_gs/scripts/train_feature_field.py
```

Expected: all files compile.

- [ ] **Step 3: Launch experiments only after code passes**

Run direct-field v2 experiments first on cached LERF VPR features. Promote only if the direct field beats the registered VPR row under the same fixed protocol or produces a clear positive table that can be honestly reported as a journal extension.

- [ ] **Step 4: Update paper wording**

Use conservative claims:

```text
official foundation-cache supervision is optional;
SAM3-adaptor fallback is not official SAM3 instance segmentation;
raw OpenGaussian-style metrics and refined-boundary diagnostics are separated.
```

