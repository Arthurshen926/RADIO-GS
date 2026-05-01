# Submission Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first paper-freeze automation layer and use it to refresh the RADIO-GS submission package.

**Architecture:** Add one focused report builder that reads existing artifacts, computes paper-facing aggregates, and writes both markdown and JSON outputs. Keep the builder independent from training/evaluation scripts so it can run CPU-only and be tested with synthetic fixtures.

**Tech Stack:** Python 3.9, standard library, pytest, existing RADIO-GS artifact layout.

---

## File Map

- Create `radio_gs/scripts/build_submission_freeze_report.py`: CPU-only artifact parser and markdown/JSON report generator.
- Create `tests/test_build_submission_freeze_report.py`: TDD coverage for LERF row parsing, ScanNet v67 aggregation, warning generation, and output files.
- Modify `docs/submission_status.md`: point readers to the generated freeze report and update the completion estimate after the report is generated.
- Generate `output/radio_gs/reports/submission_freeze_report.md`: paper-facing report built from current artifacts.
- Generate `output/radio_gs/reports/submission_freeze_manifest.json`: machine-readable source-of-truth manifest.

## Task 1: Add ScanNet v67 Aggregation Test

**Files:**
- Create: `tests/test_build_submission_freeze_report.py`
- Create: `radio_gs/scripts/build_submission_freeze_report.py`

- [ ] **Step 1: Write the failing test**

Add a test that creates two synthetic ScanNet result JSON files and asserts the builder computes the macro mIoU.

```python
def test_collect_scannet_v67_results_computes_macro(tmp_path):
    root = tmp_path / "scannet"
    for scene, vals in {
        "scene0000_00": (0.2, 0.3, 0.4),
        "scene0062_00": (0.4, 0.5, 0.6),
    }.items():
        out = root / f"{scene}_v67_teacherbalanced_fromv63_best_gidx_labelpoint"
        out.mkdir(parents=True)
        (out / "scannet_pointcloud_radio_gs_results.json").write_text(json.dumps({
            "macro": {
                "19": {"miou": vals[0], "macc": 0.7},
                "15": {"miou": vals[1], "macc": 0.8},
                "10": {"miou": vals[2], "macc": 0.9},
            },
            "args": {
                "query_mode": "gaussian_index",
                "opacity_filter_mode": "label_index",
                "gaussian_index_position_mode": "label_point",
            },
        }))

    summary = report.collect_scannet_v67(root)

    assert summary["scene_count"] == 2
    assert summary["macro_miou"] == {"19": 0.3, "15": 0.4, "10": 0.5}
    assert not summary["warnings"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_submission_freeze_report.py::test_collect_scannet_v67_results_computes_macro -q
```

Expected: FAIL because `radio_gs.scripts.build_submission_freeze_report` does not exist.

- [ ] **Step 3: Implement minimal ScanNet collector**

Create `radio_gs/scripts/build_submission_freeze_report.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCAN_SPLITS = ("19", "15", "10")


def _round4(value: float) -> float:
    return round(float(value), 4)


def collect_scannet_v67(eval_root: str | Path) -> dict[str, Any]:
    root = Path(eval_root)
    pattern = "scene*_v67_teacherbalanced_fromv63_best_gidx_labelpoint/scannet_pointcloud_radio_gs_results.json"
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in sorted(root.glob(pattern)):
        payload = json.loads(path.read_text())
        scene = path.parent.name.split("_v67_")[0]
        args = payload.get("args", {})
        if args.get("query_mode") != "gaussian_index":
            warnings.append(f"{scene}: query_mode is {args.get('query_mode')}")
        if args.get("opacity_filter_mode") != "label_index":
            warnings.append(f"{scene}: opacity_filter_mode is {args.get('opacity_filter_mode')}")
        if args.get("gaussian_index_position_mode") != "label_point":
            warnings.append(f"{scene}: gaussian_index_position_mode is {args.get('gaussian_index_position_mode')}")
        macro = payload["macro"]
        rows.append({
            "scene": scene,
            "path": str(path),
            "miou": {split: float(macro[split]["miou"]) for split in SCAN_SPLITS},
            "macc": {split: float(macro[split]["macc"]) for split in SCAN_SPLITS},
        })
    macro_miou = {
        split: _round4(sum(row["miou"][split] for row in rows) / len(rows))
        for split in SCAN_SPLITS
    } if rows else {split: 0.0 for split in SCAN_SPLITS}
    macro_macc = {
        split: _round4(sum(row["macc"][split] for row in rows) / len(rows))
        for split in SCAN_SPLITS
    } if rows else {split: 0.0 for split in SCAN_SPLITS}
    return {
        "scene_count": len(rows),
        "rows": rows,
        "macro_miou": macro_miou,
        "macro_macc": macro_macc,
        "warnings": warnings,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_submission_freeze_report.py::test_collect_scannet_v67_results_computes_macro -q
```

Expected: PASS.

## Task 2: Add LERF Best-Row Parsing

**Files:**
- Modify: `radio_gs/scripts/build_submission_freeze_report.py`
- Modify: `tests/test_build_submission_freeze_report.py`

- [ ] **Step 1: Write failing test**

Add:

```python
def test_collect_lerf_best_row_reads_macro(tmp_path):
    csv_path = tmp_path / "current_best_lerf_ovs_per_scene.csv"
    csv_path.write_text(
        "scene,loc_acc,miou,temp,checkpoint,config,output_dir,summary\n"
        "figurines,0.8,0.4,50,ckpt,cfg,out,sum\n"
        "ramen,0.9,0.6,40,ckpt,cfg,out,sum\n"
        "macro,0.85,0.5,,,,,\n"
    )

    summary = report.collect_lerf_best(csv_path)

    assert summary["macro_loc_acc"] == 0.85
    assert summary["macro_miou"] == 0.5
    assert len(summary["rows"]) == 2
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_submission_freeze_report.py::test_collect_lerf_best_row_reads_macro -q
```

Expected: FAIL because `collect_lerf_best` is missing.

- [ ] **Step 3: Implement CSV parser**

Add:

```python
import csv


def collect_lerf_best(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    rows: list[dict[str, Any]] = []
    macro_loc_acc = 0.0
    macro_miou = 0.0
    if not path.exists():
        return {"rows": rows, "macro_loc_acc": 0.0, "macro_miou": 0.0, "warnings": [f"missing {path}"]}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["scene"] == "macro":
                macro_loc_acc = _round4(float(row["loc_acc"]))
                macro_miou = _round4(float(row["miou"]))
                continue
            rows.append(row)
    return {"rows": rows, "macro_loc_acc": macro_loc_acc, "macro_miou": macro_miou, "warnings": []}
```

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_submission_freeze_report.py::test_collect_lerf_best_row_reads_macro -q
```

Expected: PASS.

## Task 3: Generate Markdown and Manifest

**Files:**
- Modify: `radio_gs/scripts/build_submission_freeze_report.py`
- Modify: `tests/test_build_submission_freeze_report.py`

- [ ] **Step 1: Write failing test**

Add:

```python
def test_write_report_outputs_markdown_and_manifest(tmp_path):
    output_dir = tmp_path / "reports"
    lerf = {"macro_loc_acc": 0.85, "macro_miou": 0.5, "rows": [], "warnings": []}
    scannet = {
        "scene_count": 1,
        "macro_miou": {"19": 0.3, "15": 0.4, "10": 0.5},
        "macro_macc": {"19": 0.6, "15": 0.7, "10": 0.8},
        "rows": [{"scene": "scene0000_00", "path": "result.json", "miou": {"19": 0.3, "15": 0.4, "10": 0.5}}],
        "warnings": ["external baselines unresolved"],
    }

    paths = report.write_freeze_outputs(output_dir, lerf, scannet)

    markdown = paths["markdown"].read_text()
    manifest = json.loads(paths["manifest"].read_text())
    assert "Submission Freeze Report" in markdown
    assert "LERF-OVS" in markdown
    assert "ScanNet" in markdown
    assert manifest["lerf"]["macro_loc_acc"] == 0.85
    assert manifest["scannet"]["macro_miou"]["10"] == 0.5
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_submission_freeze_report.py::test_write_report_outputs_markdown_and_manifest -q
```

Expected: FAIL because `write_freeze_outputs` is missing.

- [ ] **Step 3: Implement output writer**

Add functions to write markdown and JSON with clear protocol labels and warnings.

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_submission_freeze_report.py::test_write_report_outputs_markdown_and_manifest -q
```

Expected: PASS.

## Task 4: Add CLI and Generate Current Report

**Files:**
- Modify: `radio_gs/scripts/build_submission_freeze_report.py`
- Generate: `output/radio_gs/reports/submission_freeze_report.md`
- Generate: `output/radio_gs/reports/submission_freeze_manifest.json`

- [ ] **Step 1: Add CLI test**

Add a test that calls `main([...])` with temp paths and verifies outputs exist.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_submission_freeze_report.py::test_main_writes_outputs -q
```

Expected: FAIL because `main` is missing or incomplete.

- [ ] **Step 3: Implement argparse `main`**

Implement options:

- `--lerf_csv`
- `--scannet_eval_root`
- `--output_dir`

- [ ] **Step 4: Run all new tests**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_submission_freeze_report.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Generate current report**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_submission_freeze_report.py \
  --lerf_csv output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv \
  --scannet_eval_root output/scannet_pointcloud_eval \
  --output_dir output/radio_gs/reports
```

Expected: report and manifest files are written.

## Task 5: Refresh Submission Status Doc

**Files:**
- Modify: `docs/submission_status.md`

- [ ] **Step 1: Update text**

Add a short “Current source of truth” section pointing to:

- `output/radio_gs/reports/submission_freeze_report.md`
- `output/radio_gs/reports/submission_freeze_manifest.json`

Update the estimated completion from 55-60% to a conditional estimate based on the generated report.

- [ ] **Step 2: Run markdown/source checks**

Run:

```bash
git diff --check docs/submission_status.md radio_gs/scripts/build_submission_freeze_report.py tests/test_build_submission_freeze_report.py
```

Expected: no whitespace errors.

## Task 6: GPU Queue Triage

**Files:**
- No required file changes.

- [ ] **Step 1: Check GPU4/GPU5**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Expected: GPU4 and GPU5 are available before launching jobs.

- [ ] **Step 2: Decide launch set from manifest warnings**

If the freeze manifest reports missing ScanNet scenes, launch only the missing fair v67 eval jobs. If ScanNet is complete, launch one LERF fixed-protocol visualization/profile job on GPU4 and one efficiency/profile job on GPU5.

- [ ] **Step 3: Record launched commands**

Append the exact commands to `output/radio_gs/reports/submission_freeze_gpu_queue.md` so paper provenance includes queue state.

## Task 7: Verification

**Files:**
- Existing generated and modified files.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_submission_freeze_report.py tests/test_generate_visualizations_v2_grounding.py tests/test_lerf_prompt_sweep.py tests/test_scannet_pointcloud_eval.py tests/test_scannet_og_config_generator.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run diff check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 3: Review generated report**

Open:

```bash
sed -n '1,220p' output/radio_gs/reports/submission_freeze_report.md
```

Expected: report lists LERF macro, ScanNet v67 macro, warnings, and source artifact paths.

