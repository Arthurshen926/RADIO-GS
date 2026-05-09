# Submission Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the paper-facing gaps identified from `ChatGPT-项目总结与优化.md`: storage evidence, failure analysis, baseline-safe framing, CTF-GS naming, and GPU experiment launch discipline.

**Architecture:** Add small report-generation scripts with focused tests, then wire their outputs into the LaTeX draft and project status documents. GPU experiments are launched through explicit generated configs and queue scripts that write to new output directories, never overwriting frozen mainline artifacts.

**Tech Stack:** Python standard library, PyTorch checkpoint metadata reads, pytest, LaTeX draft updates, bash queue scripts.

---

### Task 1: Storage Footprint Evidence

**Files:**
- Create: `radio_gs/scripts/build_storage_footprint_report.py`
- Create: `tests/test_build_storage_footprint_report.py`
- Generate: `output/radio_gs/reports/storage_footprint_report.md`
- Generate: `paper/storage_footprint_table.tex`

- [ ] **Step 1: Write the failing test**

Create a test that writes a tiny ASCII PLY with `element vertex 10`, a mock checkpoint with `model_state_dict`, `codec_state_dict`, and `refiner_state_dict`, then asserts the report computes direct 1280d fp16 bytes and compact checkpoint bytes.

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_storage_footprint_report.py -q`

Expected: FAIL because `build_storage_footprint_report.py` does not exist.

- [ ] **Step 3: Implement the script**

Implement functions:
- `parse_ply_vertex_count(path: Path) -> int`
- `checkpoint_tensor_bytes(path: Path) -> dict[str, int]`
- `build_rows() -> list[dict[str, object]]`
- `write_markdown(rows, path)`
- `write_latex(rows, path)`

Default rows must cover Figurines, Ramen, Teatime, and Waldo Kitchen mainline checkpoints.

- [ ] **Step 4: Run focused tests**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_storage_footprint_report.py -q`

Expected: PASS.

- [ ] **Step 5: Generate storage report/table**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_storage_footprint_report.py`

Expected outputs:
- `output/radio_gs/reports/storage_footprint_report.md`
- `paper/storage_footprint_table.tex`

### Task 2: Failure Analysis Evidence

**Files:**
- Create: `radio_gs/scripts/build_lerf_failure_analysis.py`
- Create: `tests/test_build_lerf_failure_analysis.py`
- Generate: `output/radio_gs/reports/lerf_failure_analysis.md`
- Generate: `paper/lerf_failure_analysis_table.tex`

- [ ] **Step 1: Write the failing test**

Use a tiny synthetic `lerf_ovs_results.json` with two categories and assert that the script ranks failed/fragile categories by LocAcc and mIoU.

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_lerf_failure_analysis.py -q`

Expected: FAIL because `build_lerf_failure_analysis.py` does not exist.

- [ ] **Step 3: Implement the script**

Implement extraction from the current selected rendered LERF JSON files listed in `output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv`. Output worst categories and paper-safe interpretation.

- [ ] **Step 4: Run focused tests**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_lerf_failure_analysis.py -q`

Expected: PASS.

- [ ] **Step 5: Generate failure report/table**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_lerf_failure_analysis.py`

Expected outputs:
- `output/radio_gs/reports/lerf_failure_analysis.md`
- `paper/lerf_failure_analysis_table.tex`

### Task 3: Paper Draft Closure

**Files:**
- Modify: `paper/radio_gs_draft.tex`
- Modify: `docs/PROJECT_MAINLINE.md`
- Modify: `docs/submission_status.md`

- [ ] **Step 1: Update method naming**

Use `CTF-GS` as the paper method name while preserving `RADIO-GS` as the repository/project implementation name. Rename module prose to `HGCF`, `CTR`, `VFA`, and `FGC`.

- [ ] **Step 2: Add storage table**

Input `paper/storage_footprint_table.tex` in the experiments section and explain direct 1280d fp16 vs compact checkpoint footprint.

- [ ] **Step 3: Add failure analysis section**

Input `paper/lerf_failure_analysis_table.tex` and write the small-object / peak-region tradeoff analysis.

- [ ] **Step 4: Strengthen baseline-safe framing**

Make the paper explicitly state that external baselines are either reproduced under a unified protocol or reported as protocol-misaligned official-source context; do not claim unresolved hardcoded rows as final.

- [ ] **Step 5: Verify text references**

Run: `rg -n "HCD|FDH|RADIO-GS|storage_footprint|failure" paper/radio_gs_draft.tex docs/PROJECT_MAINLINE.md docs/submission_status.md`

Expected: legacy names only appear as implementation aliases or project names.

### Task 4: GPU Experiment Launch Discipline

**Files:**
- Create: `radio_gs/scripts/generate_scannet_dino_cv_configs.py`
- Create: `tests/test_generate_scannet_dino_cv_configs.py`
- Create: `radio_gs/scripts/launch_scannet_dino_cv_queue.sh`
- Create: `tests/test_scannet_dino_cv_queue_launcher.py`
- Generate: `output/radio_gs/reports/scannet_dino_cv_launch_plan.md`

- [ ] **Step 1: Write config-generator test**

Assert generated configs preserve v67 fair settings, use new output dirs, set `batch_size=4`, and set conservative DINO cross-view weights.

- [ ] **Step 2: Implement config generator**

Generate full 10-scene configs by copying active v67 configs and adding DINO cross-view options. Use `0.001` as the conservative default weight unless overridden.

- [ ] **Step 3: Write queue launcher test**

Assert launcher prints preserved prompt templates and maps scenes to generated config paths.

- [ ] **Step 4: Implement queue launcher**

Run train+eval per scene in foreground for one GPU. The caller can split scenes across GPU4 and GPU5 with two background processes.

- [ ] **Step 5: Dry-run queues**

Run dry-run or print commands for GPU4/GPU5 scene splits before launching actual work.

- [ ] **Step 6: Launch GPU4/GPU5 jobs**

Launch two independent queues with `setsid`/background output logs under `output/radio_gs/scannet_dino_cv_queue_20260509/`.

### Task 5: Verification and Commit

**Files:**
- All modified tracked files.

- [ ] **Step 1: Run focused tests**

Run storage, failure-analysis, and DINO queue tests.

- [ ] **Step 2: Run full test suite**

Run: `CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Run diff check**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 4: Commit**

Commit paper closure tooling and docs in one commit, leaving the user-provided expert file untracked.
