# SAM3 Training-View Proposal Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate label-free SAM3 training-view object proposal registration for LERF Direct 3D Object Selection.

**Architecture:** Add a focused proposal-registration module under `radio_gs/models/`, then wire it into `eval_lerf_direct_3d_selection.py` as an optional score-fusion stage after primitive text scores are computed and before selection. Keep the existing Direct3D protocol and make the feature optional through CLI flags.

**Tech Stack:** PyTorch, existing `FoundationCache` loader, existing LERF camera poses/intrinsics, pytest, RADIO-GS evaluator scripts.

---

### Task 1: Proposal Registration Module

**Files:**
- Create: `radio_gs/models/sam3_proposal_registration.py`
- Test: `tests/test_sam3_proposal_registration.py`

- [ ] **Step 1: Write tests**

Cover three behaviors:

```python
def test_build_memberships_from_logits_keeps_confident_pairs():
    logits = torch.tensor([[[4.0, -4.0], [-4.0, 4.0]]])
    pixels_xy = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    scores = torch.tensor([0.8])
    memberships = build_sam3_mask_memberships(
        logits, pixels_xy, scores=scores, min_probability=0.5
    )
    assert memberships.row_indices.tolist() == [0, 1]
    assert memberships.proposal_indices.tolist() == [0, 0]
    assert torch.all(memberships.weights > 0.5)
```

```python
def test_fuse_scores_uses_proposal_pooled_scores_for_low_margin_rows():
    scores = torch.tensor([[0.55, 0.50], [0.90, 0.10]])
    row = torch.tensor([0, 1])
    prop = torch.tensor([0, 0])
    weights = torch.tensor([1.0, 1.0])
    fused, stats = fuse_scores_with_sam3_proposals(
        scores, row, prop, weights, alpha=0.5, gate="low_margin", margin_threshold=0.1
    )
    assert fused[0, 0] > scores[0, 0]
    assert torch.allclose(fused[1], scores[1])
    assert stats["num_proposals"] == 1
```

```python
def test_empty_memberships_return_original_scores():
    scores = torch.randn(3, 2)
    fused, stats = fuse_scores_with_sam3_proposals(
        scores,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        alpha=0.5,
    )
    assert torch.allclose(fused, scores)
    assert stats["num_memberships"] == 0
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_sam3_proposal_registration.py
```

Expected: import failure because the module is not implemented.

- [ ] **Step 3: Implement module**

Implement:

- `Sam3ProposalMemberships` dataclass with `row_indices`, `proposal_indices`, `weights`, `num_rows`, `num_proposals`.
- `build_sam3_mask_memberships(mask_logits, pixels_xy, scores=None, visibility=None, min_probability=0.5, max_masks=None, proposal_offset=0)`.
- `fuse_scores_with_sam3_proposals(scores, row_indices, proposal_indices, weights, alpha=0.35, gate="low_margin", margin_threshold=0.05, min_weight_sum=1e-6)`.

- [ ] **Step 4: Run tests and confirm pass**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_sam3_proposal_registration.py
```

Expected: pass.

### Task 2: Direct3D Evaluator Integration

**Files:**
- Modify: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- Modify: `tests/test_lerf_direct_3d_selection.py`

- [ ] **Step 1: Add CLI parser test**

Add a parser test that checks:

- `--sam3_proposal_registration_dir`
- `--sam3_proposal_registration_alpha`
- `--sam3_proposal_registration_min_probability`
- `--sam3_proposal_registration_gate low_margin`
- `--sam3_proposal_registration_margin_threshold`

- [ ] **Step 2: Implement helpers**

Add small helper functions near the existing proposal smoothing helpers:

- `_resolve_sam3_proposal_cache_paths(cache_root, scene)`.
- `_project_points_to_image(xyz, pose_w2c, intrinsics, image_width, image_height)`.
- `_build_sam3_training_view_memberships(...)`.

Use `load_foundation_cache(..., require_official=True)`. If no cache is found, return original scores and record `enabled=false`.

- [ ] **Step 3: Fuse scores**

After primitive text scores and existing aggregation are available, but before threshold/top-k selection, call `fuse_scores_with_sam3_proposals` when `--sam3_proposal_registration_dir` is set. Record stats in `protocol`.

- [ ] **Step 4: Verify tests**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_lerf_direct_3d_selection.py tests/test_sam3_proposal_registration.py
```

Expected: pass.

### Task 3: Smoke Experiment

**Files:**
- Create: `paper/artifacts/lerf_direct3d_sam3_training_view_proposal_registration_20260525.md`
- Create: `paper/artifacts/lerf_direct3d_sam3_training_view_proposal_registration_20260525.json`

- [ ] **Step 1: Run Figurines smoke**

Run on a free GPU:

```bash
CUDA_VISIBLE_DEVICES=4 bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.scripts.eval_lerf_direct_3d_selection \
  --scene figurines \
  --output_dir output/radio_gs/lerf_direct3d_sam3_trainview_propreg_smoke_20260525 \
  --sam3_proposal_registration_dir output/radio_gs/foundation_cache_sam3_modelscope_mapped_trainviews \
  --sam3_proposal_registration_alpha 0.35 \
  --sam3_proposal_registration_min_probability 0.55 \
  --sam3_proposal_registration_gate low_margin \
  --sam3_proposal_registration_margin_threshold 0.05
```

- [ ] **Step 2: Compare against strict baseline**

Compare to:

`output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260517_geometry_pad16_full/figurines/lerf_direct_3d_selection_results.json`

- [ ] **Step 3: Decide promotion**

Promote only if mean mIoU, Acc@0.25, or boundary metrics improve without using query RGB SAM3 readout. Otherwise record as an ablation/failure analysis and keep the current main row.

### Task 4: Verification and Paper Artifacts

**Files:**
- Modify: `paper/artifacts/README.md`
- Modify: `docs/PROJECT_MAINLINE.md`
- Modify: `docs/paper_draft_current.md`
- Optionally modify: `paper/artifacts/final_rows.yaml`

- [ ] **Step 1: Run validation**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q \
  tests/test_sam3_proposal_registration.py \
  tests/test_lerf_direct_3d_selection.py \
  tests/test_validate_final_rows_registry.py \
  tests/test_validate_paper_claims.py
```

- [ ] **Step 2: Update paper docs**

Document exact protocol, cache provenance, whether the row is promoted, and why.

- [ ] **Step 3: Final sanity**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m radio_gs.scripts.validate_final_rows_registry paper/artifacts/final_rows.yaml
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m radio_gs.scripts.validate_paper_claims
```
