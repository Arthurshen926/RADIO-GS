# SAM3 Object-Boundary Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a low-risk official-SAM3 object-boundary supervision branch that can improve mask boundaries and localization without replacing the RADIO-GS direct field.

**Architecture:** Extend the existing official foundation cache path to preserve SAM3 proposal metadata and add feature-region compactness, inter-proposal separation, and feature-boundary alignment losses. Keep all weights defaulted to zero so existing checkpoints and experiments are unchanged unless configs opt in.

**Tech Stack:** PyTorch, existing `radio_gs.models.foundation_cache`, existing `train_feature_field.py` foundation-cache training hook, pytest.

---

### Task 1: Preserve SAM3 Proposal Metadata

**Files:**
- Modify: `radio_gs/models/foundation_cache.py`
- Test: `tests/test_foundation_cache.py`

- [x] Add `queries`, `scores`, and `boxes_xyxy` to `FoundationHeadCache`.
- [x] Parse optional `scores` as `[M]` and `boxes_xyxy` as `[M,4]`.
- [x] Add a unit test that loads a SAM3 cache payload and checks these fields are preserved.

### Task 2: Add SAM3 Region And Boundary Losses

**Files:**
- Modify: `radio_gs/models/foundation_cache.py`
- Test: `tests/test_foundation_cache.py`

- [x] Add feature-region compactness loss: pixels inside each SAM3 mask are pulled toward their proposal feature mean.
- [x] Add disjoint-proposal separation loss: proposal means for low-overlap masks are pushed below a cosine margin.
- [x] Add feature-boundary alignment loss: decoded feature gradients are encouraged to align with SAM3 mask-logit boundaries.
- [x] Add tests showing compact features inside masks and sharp feature boundaries have lower loss than noisy/blurry alternatives.

### Task 3: Expose Config And Training Hook

**Files:**
- Modify: `radio_gs/config.py`
- Modify: `radio_gs/scripts/train_feature_field.py`
- Test: `tests/test_radio_adaptor_trainer_config.py`

- [x] Add config fields with zero defaults:
  - `foundation_cache_region_consistency_weight`
  - `foundation_cache_region_separation_weight`
  - `foundation_cache_feature_boundary_weight`
  - `foundation_cache_region_score_threshold`
  - `foundation_cache_region_max_masks`
  - `foundation_cache_region_separation_margin`
- [x] Pass these fields into `compute_foundation_cache_supervision_loss`.
- [x] Extend config tests so defaults are accepted and nonzero settings propagate.

### Task 4: Verification And First Experiment Command

**Files:**
- Test: `tests/test_foundation_cache.py`
- Test: `tests/test_build_sam3_foundation_cache.py`
- Test: `tests/test_radio_adaptor_trainer_config.py`

- [x] Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest \
  tests/test_foundation_cache.py \
  tests/test_build_sam3_foundation_cache.py \
  tests/test_radio_adaptor_trainer_config.py
```

- [x] Add opt-in four-scene configs:
  - `radio_gs/configs/lerf_hybrid_v14_figurines_sam3_object_boundary_ft.yaml`
  - `radio_gs/configs/lerf_hybrid_v14_ramen_sam3_object_boundary_ft.yaml`
  - `radio_gs/configs/lerf_hybrid_v14_teatime_sam3_object_boundary_ft.yaml`
  - `radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_sam3_object_boundary_ft.yaml`
- [ ] Once GPU4/5 are genuinely free, launch the opt-in object-boundary fine-tune experiments. Do not replace the mainline unless direct 3D mIoU and boundary metrics improve together.
