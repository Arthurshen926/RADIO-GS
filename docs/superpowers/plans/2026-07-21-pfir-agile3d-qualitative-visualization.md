# PFIR and AGILE3D Qualitative Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate auditable, paper-ready static qualitative panels for the frozen PFIR and AGILE3D formal evaluations.

**Architecture:** A new standalone report generator reads only formal result artifacts. It has isolated helpers for deterministic case selection, aligned 3-D rendering, and AGILE trajectory replay; the final writer emits PNGs, an audit JSON, and Markdown without changing the evaluator outputs.

**Tech Stack:** Python, NumPy, PyTorch, Matplotlib, Pillow, existing PFIR/AGILE3D evaluators.

---

### Task 1: Define immutable visualization records

**Files:**
- Create: `radio_gs/scripts/build_pfir_agile3d_qualitative_report.py`
- Create: `tests/test_build_pfir_agile3d_qualitative_report.py`

- [ ] **Step 1: Write failing selection tests**

```python
def test_select_pfir_cases_uses_fixed_predicates_and_query_id_ties():
    selected = select_pfir_cases(ranking_rows, selection_rows)
    assert [row["kind"] for row in selected] == ["success", "rank_mask_gap", "same_class_confusion"]

def test_select_agile_cases_is_deterministic_under_input_permutation():
    assert select_agile_cases(rows, coverage) == select_agile_cases(list(reversed(rows)), coverage)
```

- [ ] **Step 2: Run the tests and verify they fail because the module is absent**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_build_pfir_agile3d_qualitative_report.py`

Expected: import failure for `build_pfir_agile3d_qualitative_report`.

- [ ] **Step 3: Implement typed case selection helpers**

```python
def select_pfir_cases(ranking_rows, selection_rows):
    """Return the lexically tie-broken success, rank/mask gap, and confusion rows."""

def select_agile_cases(object_rows, scene_coverage):
    """Return high/median/low coverage examples from formal rows only."""
```

- [ ] **Step 4: Run the selection tests**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_build_pfir_agile3d_qualitative_report.py`

Expected: PASS.

### Task 2: Build exact 3-D mask renderers

**Files:**
- Modify: `radio_gs/scripts/build_pfir_agile3d_qualitative_report.py`
- Modify: `tests/test_build_pfir_agile3d_qualitative_report.py`

- [ ] **Step 1: Write failing mask-overlay tests**

```python
def test_error_colors_distinguish_true_positive_false_positive_and_false_negative():
    colors = mask_error_colors(np.array([True, False]), np.array([True, True]))
    assert tuple(colors[0]) == TP_COLOR
    assert tuple(colors[1]) == FN_COLOR
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_build_pfir_agile3d_qualitative_report.py`

Expected: missing `mask_error_colors`.

- [ ] **Step 3: Implement deterministic camera projection and 3-D error overlay rendering**

```python
def render_mask_view(xyz, ground_truth, prediction, *, clicks=(), title=""):
    """Render a geometry-only fixed camera view; no labels are used to place the camera."""
```

- [ ] **Step 4: Run renderer tests**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_build_pfir_agile3d_qualitative_report.py`

Expected: PASS.

### Task 3: Implement formal-artifact loading and AGILE replay validation

**Files:**
- Modify: `radio_gs/scripts/build_pfir_agile3d_qualitative_report.py`
- Modify: `tests/test_build_pfir_agile3d_qualitative_report.py`

- [ ] **Step 1: Write a failing trajectory comparator test**

```python
def test_assert_trajectory_matches_rejects_metric_drift():
    with pytest.raises(AssertionError):
        assert_trajectory_matches([0.1, 0.2], [0.1, 0.3])
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_build_pfir_agile3d_qualitative_report.py`

Expected: missing `assert_trajectory_matches`.

- [ ] **Step 3: Load PFIR prediction arrays and replay AGILE selected examples with formal configuration**

```python
def replay_agile_case(...):
    """Return masks and official clicks at 1, 5, and 15 while preserving the formal predictor contract."""
```

- [ ] **Step 4: Run unit tests and one real-case trajectory check**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_build_pfir_agile3d_qualitative_report.py`

Expected: PASS; the real replay check is exact to the stored formal trajectory.

### Task 4: Produce and validate report artifacts

**Files:**
- Modify: `radio_gs/scripts/build_pfir_agile3d_qualitative_report.py`
- Create: `output/benchmark_qualitative_report/README.md`
- Create: `output/benchmark_qualitative_report/case_selection_audit.json`
- Create: `output/benchmark_qualitative_report/pfir_qualitative.png`
- Create: `output/benchmark_qualitative_report/agile3d_qualitative.png`

- [ ] **Step 1: Add a CLI with explicit formal-result roots and output directory**

```bash
bash radio_gs/scripts/run_repo_python.sh -m radio_gs.scripts.build_pfir_agile3d_qualitative_report \
  --pfir-root output/scannet_pfir_small_v1/test_v1_final/reconstruction_v1 \
  --agile-results output/agile3d_scannet40/formal_v1/results.json \
  --output-dir output/benchmark_qualitative_report
```

- [ ] **Step 2: Generate the report and assert its audit checks pass**

Expected: both PNGs, the Markdown report, and audit JSON exist; selected AGILE replays match formal stored trajectories.

- [ ] **Step 3: Inspect the rendered PNGs**

Use image inspection to verify that labels fit, all panels are nonblank, crop/GT/prediction/error are visibly distinct, and click steps progress from 1 to 5 to 15.

- [ ] **Step 4: Run focused regression tests**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_build_pfir_agile3d_qualitative_report.py tests/test_agile3d_scannet40_protocol.py tests/test_scannet_pfir_protocol.py`

Expected: PASS.
