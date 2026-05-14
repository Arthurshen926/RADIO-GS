# Qualitative-to-Method Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert two qualitative weaknesses into auditable, GT-free method/analysis upgrades without changing the main OpenGaussian-style direct-3D protocol by stealth.

**Architecture:** Add optional post-render boundary snapping for VPR masks and cycle-consistent filtering for DINO matching. Keep raw protocol metrics as the main rows; report refined masks and filtered matching as separate ablations/visualizations with explicit labels.

**Tech Stack:** Python, NumPy, OpenCV, PyTorch, existing LERF evaluation scripts and pytest tests.

---

### Task 1: VPR RGB Boundary Snapping

**Files:**
- Modify: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- Test: `tests/test_lerf_direct_3d_selection.py`

- [ ] **Step 1: Write failing tests**

Add tests that call `refine_mask_with_rgb_edges(rgb, mask, ...)` on synthetic RGB rectangles and verify that the refinement is deterministic, keeps empty masks empty, and removes boundary spill outside an RGB edge.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_lerf_direct_3d_selection.py::test_refine_mask_with_rgb_edges_snaps_to_color_boundary -q
```

Expected: fails because `refine_mask_with_rgb_edges` does not exist.

- [ ] **Step 3: Implement minimal refinement**

Add a GT-free OpenCV GrabCut-style function initialized from the rendered VPR mask. Use only RGB and the predicted mask; never read GT inside the function.

- [ ] **Step 4: Add CLI plumbing**

Add `--mask_refinement none|rgb_grabcut`, `--mask_refinement_iters`, `--mask_refinement_dilate`, and `--mask_refinement_erode`. Store the choice in JSON summaries and save refined masks under the same selection tag when enabled.

- [ ] **Step 5: Validate on saved/cached VPR runs**

Run cache-backed evaluation on four LERF scenes under the promoted selector and compare raw vs refined mIoU/Acc. Promote only if macro improves without hiding the protocol distinction.

### Task 2: DINO Cycle-Consistent Matching and Propagation Figure

**Files:**
- Modify: `radio_gs/scripts/eval_lerf_sam_dino_tasks.py`
- Test: `tests/test_lerf_sam_dino_tasks.py`

- [ ] **Step 1: Write failing tests**

Add tests for `dense_match_points(..., mutual_check=True, cycle_max_distance=...)` showing that a match is kept only when target-to-source nearest-neighbor maps back near the original source token.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_lerf_sam_dino_tasks.py::test_dense_match_points_mutual_filter_removes_cycle_inconsistent_match -q
```

Expected: fails because the new keyword arguments are unsupported.

- [ ] **Step 3: Implement mutual filtering**

Extend `dense_match_points` with optional `mutual_check`, `cycle_max_distance`, and `min_score`. Keep default behavior unchanged.

- [ ] **Step 4: Update visualization semantics**

Add a DINO propagation visual that shows source RGB/mask, teacher propagated mask, rendered propagated mask, and error/heatmap panels. Keep raw line matching available but stop using it as the main paper figure.

- [ ] **Step 5: Validate and refresh paper assets**

Run the relevant tests, regenerate the SAM/DINO qualitative figure, update captions to say DINO line matching is diagnostic and robust mask propagation is the promoted evidence.

### Task 3: Reporting

**Files:**
- Modify: `paper/radio_gs_draft.tex`
- Modify: `docs/PROJECT_MAINLINE.md`
- Modify: `docs/submission_status.md`

- [ ] **Step 1: Update wording**

State that boundary snapping is an optional 2D readout refinement, not the OpenGaussian-style primitive-selection score unless explicitly reported as a separate ablation.

- [ ] **Step 2: Run verification**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_lerf_direct_3d_selection.py tests/test_lerf_sam_dino_tasks.py -q
bash radio_gs/scripts/run_repo_python.sh -m py_compile radio_gs/scripts/eval_lerf_direct_3d_selection.py radio_gs/scripts/eval_lerf_sam_dino_tasks.py
latexmk -pdf -interaction=nonstopmode -halt-on-error radio_gs_draft.tex
git diff --check
```

Expected: tests pass, scripts compile, PDF builds, and whitespace check is clean.
