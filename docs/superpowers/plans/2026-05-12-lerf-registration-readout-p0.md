# LERF Registration Readout P0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and validate a Dr. Splat-style, GT-free rendered-feature-to-primitive registration readout for LERF direct 3D object selection.

**Architecture:** Keep the existing HGCF/CTR/VFA/FGC model unchanged. At evaluation time, render post-refiner RADIO features from selected posed views, project them to SigLIP2 space, sample those text-aligned features back to Gaussian centers with depth/alpha visibility checks, then query the registered primitive embeddings directly with text.

**Tech Stack:** PyTorch, gsplat renderer, existing RADIO-GS LERF evaluators, pytest.

---

### Task 1: Unit-Test Registration Helpers

**Files:**
- Modify: `tests/test_lerf_direct_3d_selection.py`
- Modify: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`

- [ ] **Step 1: Write failing tests**

```python
def test_select_registration_frame_ids_uses_official_available_frames():
    frames = select_registration_frame_ids(
        available_pose_ids=[1, 2, 3, 4, 5],
        annotated_frame_ids=[2, 3, 4],
        official_frame_ids=[3, 5, 9],
        mode="official",
        max_frames=0,
    )
    assert frames == [3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_lerf_direct_3d_selection.py -q`
Expected: import or missing-function failure.

- [ ] **Step 3: Implement helper functions**

Add `select_registration_frame_ids`, `score_text_aligned_embeddings`, and `merge_registered_scores` to the direct evaluator.

- [ ] **Step 4: Run tests**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_lerf_direct_3d_selection.py -q`
Expected: all tests in the new file pass.

### Task 2: Add Registered-View Score Source

**Files:**
- Modify: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`

- [ ] **Step 1: Add CLI knobs**

Add `--score_source direct|registered_view`, `--registration_frame_mode official|annotated|all_poses`, `--registration_max_frames`, visibility thresholds, and fallback selection.

- [ ] **Step 2: Implement P0 registration**

Render current model features per registration view, project to SigLIP2, sample back to Gaussian centers with `sample_multiview_radio_targets`, average valid embeddings, score text queries, and optionally fall back to direct Gaussian scores for unseen primitives.

- [ ] **Step 3: Smoke test one scene on GPU 4**

Run a small ratio sweep on `figurines` with `CUDA_VISIBLE_DEVICES=4` and `--gpu 0`.

### Task 3: Report Results and Decide Next Branch

**Files:**
- Modify: `docs/submission_status.md`
- Modify: `output/radio_gs/reports/lerf_direct_3d_selection.md`

- [ ] **Step 1: If P0 improves direct selection, document it as an OPR diagnostic**

Record the exact command, result JSON path, and macro metrics.

- [ ] **Step 2: If P0 remains weak, document why**

Use the result to justify a trained OPR/object proposal branch rather than threshold tuning.
