# Object Proposal Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a label-free object/proposal memory readout that improves RADIO-GS boundary consistency and direct field usability across LERF rendered grounding, LERF direct 3D object selection, and ScanNet direct point querying.

**Architecture:** Introduce a small proposal-memory utility that pools logits/features over GT-free spatial proposals, then integrate it as a readout-time consistency module. The first implementation uses spatial connected proposals and high-confidence residual blending, so it remains compatible with the existing compact foundation-feature field and does not depend on GT masks.

**Tech Stack:** Python, PyTorch, NumPy, SciPy cKDTree/connected components, existing RADIO-GS evaluators, pytest via `radio_gs/scripts/run_repo_python.sh`.

---

### Task 1: Proposal Memory Utility

**Files:**
- Create: `radio_gs/models/proposal_memory.py`
- Test: `tests/test_proposal_memory.py`

- [ ] **Step 1: Write failing tests**

Add tests that require:
- pooling point logits by integer proposal labels with confidence weights;
- residual proposal-logit propagation that improves noisy points while preserving the original tensor shape;
- rejecting label/logit length mismatches.

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_proposal_memory.py
```

Expected: FAIL because `radio_gs.models.proposal_memory` does not exist.

- [ ] **Step 2: Implement minimal utility**

Create `ProposalMemory`, `build_proposal_memory_from_labels`, and `propagate_logits_with_proposals`. Keep the API tensor-based and label-free; labels with negative ids are ignored as background/unassigned.

- [ ] **Step 3: Verify utility tests**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_proposal_memory.py
```

Expected: PASS.

### Task 2: ScanNet Proposal Readout

**Files:**
- Modify: `radio_gs/scripts/eval_scannet_pointcloud_radio_gs.py`
- Modify: `tests/test_scannet_pointcloud_eval.py`

- [ ] **Step 1: Write failing parser/unit tests**

Add tests for `--proposal_smoothing`, `--proposal_voxel_size`, and a helper that builds voxel proposal labels from xyz. Verify two nearby points share a proposal and distant points do not.

- [ ] **Step 2: Implement ScanNet proposal smoothing**

Add `PROPOSAL_SMOOTHING_MODES = ("none", "voxel")`. After all per-point logits are collected, build voxel proposals from xyz and blend proposal-pooled logits with the original logits. Record the protocol in the JSON report.

- [ ] **Step 3: Run focused tests**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_proposal_memory.py tests/test_scannet_pointcloud_eval.py
```

Expected: PASS.

### Task 3: LERF Direct 3D Proposal Readout

**Files:**
- Modify: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- Modify: `tests/test_lerf_direct_3d_selection.py`

- [ ] **Step 1: Add failing direct-selection tests**

Add a helper-level test showing that proposal-pooled scores recover a compact object component when one primitive has a noisy low score but its proposal peers are confident.

- [ ] **Step 2: Implement proposal-score smoothing**

Add a direct 3D readout option that builds voxel proposal ids from Gaussian centers, pools text scores per proposal, and blends them before selection. Keep the existing score-source and SAM/adaptor mask-refinement paths unchanged.

- [ ] **Step 3: Run focused tests**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_proposal_memory.py tests/test_lerf_direct_3d_selection.py
```

Expected: PASS.

### Task 4: GPU Validation

**Files:**
- Generate artifacts under `output/` and `paper/artifacts/`

- [ ] **Step 1: Run LERF direct3D smoke sweeps on GPU4/GPU5**

Use the strongest current direct-field score caches and compare no proposal smoothing versus proposal smoothing on at least `waldo_kitchen` and one easier scene.

- [ ] **Step 2: Run ScanNet VALA8 proposal smoothing on GPU4/GPU5**

Run the same VALA-aligned 8-scene split as the current paper row. Compare the macro split19/split15/split10 rows against the current spatial-KNN result.

- [ ] **Step 3: Promote only positive evidence**

If proposal smoothing improves the macro row without introducing protocol leakage, update `paper/artifacts/final_rows.yaml` and related tables. If it is negative, keep it as an ablation/failure analysis and do not promote.

### Task 5: Paper Integration

**Files:**
- Modify: `paper/radio_gs_draft.tex`
- Modify: `paper/artifacts/final_rows.yaml`
- Modify: `docs/PROJECT_MAINLINE.md`
- Modify: `docs/paper_draft_current.md`

- [ ] **Step 1: Update method wording**

Describe proposal memory as a label-free object-aware readout over the same compact foundation-feature field, not as a separate task-specific model.

- [ ] **Step 2: Update quantitative tables**

Keep tables concise and only promote validated rows. Add a short failure/stability note if the proposal readout helps boundaries but not every category.

- [ ] **Step 3: Final verification**

Run focused tests, paper validators, and LaTeX build before claiming completion.
