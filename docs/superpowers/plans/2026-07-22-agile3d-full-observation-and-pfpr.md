# AGILE3D Full-Observation and PFPR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AGILE3D use the shared continuous canonical world-query route and add a held-out pose-free patch-to-3D retrieval benchmark.

**Architecture:** The legacy feature-cache evaluator remains frozen as an ablation. A new evaluator reads canonical primitive banks and Gaussian geometry directly, compiles standard world queries, and continuously reads support at the released 5 cm points. PFPR is a separate protocol package so its point-retrieval targets cannot be confused with PFIR instance ranking.

**Tech Stack:** Python, PyTorch, SciPy KD-tree, existing canonical capability cache, shared query compiler/engine, released AGILE3D protocol.

---

### Task 1: Add a reusable continuous Gaussian field readout

**Files:**
- Modify: `radio_gs/querying/query_compilers.py`
- Modify: `tests/test_query_compilers.py`

- [ ] **Step 1: Write a failing unit test**

```python
def test_continuous_gaussian_readout_normalizes_local_support():
    values, support = continuous_gaussian_readout(...)
    assert values.tolist() == pytest.approx([0.0, 1.0])
    assert support.tolist() == pytest.approx([1.0, 1.0])
```

- [ ] **Step 2: Run the test and verify it fails because the helper is absent**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_query_compilers.py::test_continuous_gaussian_readout_normalizes_local_support`

Expected: import failure for `continuous_gaussian_readout`.

- [ ] **Step 3: Implement the covariance-aware normalized readout**

The helper accepts Gaussian centers, covariance/precision, scalar primitive
values, official points, optional opacities, and a fixed nearest-candidate
limit. It returns both the normalized value and the unnormalized support mass.

- [ ] **Step 4: Run the focused unit test**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_query_compilers.py::test_continuous_gaussian_readout_normalizes_local_support`

Expected: PASS.

### Task 2: Implement the direct AGILE3D canonical predictor

**Files:**
- Create: `radio_gs/benchmarks/agile3d_scannet40/evaluate_canonical_field.py`
- Modify: `tests/test_agile3d_scannet40_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

```python
def test_canonical_field_predictor_uses_standard_world_query_without_observation_lift():
    predictor = CanonicalFieldPointPredictor(...)
    assert predictor.protocol_report()["observation_lift"] == "none"

def test_canonical_field_predictor_reports_continuous_support_before_labels():
    report = predictor.support_report(official_points)
    assert report["labels_opened"] is False
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_agile3d_scannet40_protocol.py -k canonical_field_predictor`

Expected: import failure for the new evaluator module.

- [ ] **Step 3: Implement one direct canonical predictor**

Load a fail-closed capability bank, support graph, and matching Gaussian
geometry. Compile all accumulated released clicks through
`compile_world_3d_query`, run `CanonicalQueryEngine`, then read probability
and selected support continuously at official 5 cm points. The only result
mask is their thresholded intersection.

- [ ] **Step 4: Add full-observation/source and support-gate audit fields**

The CLI records `source_observation_root`, its declared contract, continuous
support mass, and support fraction before labels are used. `--require-support-gate`
rejects incomplete fields; the dense overlap pilot may report but cannot claim
the full-observation main setting.

- [ ] **Step 5: Run focused protocol tests**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_agile3d_scannet40_protocol.py -k 'canonical_field_predictor or continuous'`

Expected: PASS.

### Task 3: Run an immutable dense-overlap promotion

**Files:**
- Create: `radio_gs/scripts/run_agile3d_canonical_dense20_promotion.sh`
- Create: `output/agile3d_scannet40/canonical_dense20_promotion_v1/`

- [ ] **Step 1: Partition the frozen 20 dense-overlap scenes across idle GPUs**

Each worker receives a disjoint scene list and the same fixed direct canonical
configuration. It writes one shard result and never changes click order,
thresholds, or fields.

- [ ] **Step 2: Run continuous support audit and all three evaluation shards**

Run the direct evaluator with the released 5 cm points and `test_set_calibration=false`.

- [ ] **Step 3: Merge only complete, compatible shards**

The merge script verifies identical protocol hashes and the exact 20-scene
source list before reporting aggregate metrics.

- [ ] **Step 4: Compare against the frozen legacy dense20 result**

Report the direct interface and legacy observation-lift rows side by side as
a promotion diagnostic, never as a 312-scene replacement.

### Task 4: Add PFPR-Small v1 protocol and evaluator

**Files:**
- Create: `radio_gs/benchmarks/scannet_pfpr/__init__.py`
- Create: `radio_gs/benchmarks/scannet_pfpr/protocol.py`
- Create: `radio_gs/benchmarks/scannet_pfpr/build_benchmark.py`
- Create: `radio_gs/benchmarks/scannet_pfpr/evaluate_predictions.py`
- Create: `tests/test_scannet_pfpr_protocol.py`

- [ ] **Step 1: Write failing tests for held-out anchor construction and spatial NMS**

```python
def test_anchor_patch_never_exposes_pose_or_depth_to_method_manifest(): ...
def test_fixed_radius_nms_returns_spatially_distinct_hypotheses(): ...
def test_recall_uses_3d_anchor_distance_not_instance_identity(): ...
```

- [ ] **Step 2: Run the tests and observe the missing-package failure**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_scannet_pfpr_protocol.py`

Expected: import failure for `radio_gs.benchmarks.scannet_pfpr`.

- [ ] **Step 3: Implement manifest-only query construction**

Sample eligible held-out RGB-D pixels, compute private 3-D anchors, write RGB
patches and a method-visible manifest without pose/depth/anchor coordinates,
and keep evaluator target data in a private file.

- [ ] **Step 4: Implement point retrieval scoring and evaluation**

Use center DINO descriptors, fixed 10 cm candidate NMS, and frozen distance
thresholds. Write scene-macro and query-micro R@K, MRR, and error statistics.

- [ ] **Step 5: Run PFPR tests**

Run: `bash radio_gs/scripts/run_repo_python.sh -m pytest -q tests/test_scannet_pfpr_protocol.py`

Expected: PASS.
