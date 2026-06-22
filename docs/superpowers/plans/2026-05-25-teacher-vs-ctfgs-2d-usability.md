# Frame-wise RADIO vs GaussFM 2D Usability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 2D frame-wise-RADIO-vs-GaussFM feature-usability report and connect it to the paper-facing artifact set.

**Architecture:** Add one CPU-only report builder that reads existing JSON artifacts, emits JSON/Markdown/LaTeX summaries, and keeps claim language conservative when the teacher still wins on some metrics. Unit tests drive the parser and writer behavior before production code is added.

**Tech Stack:** Python standard library, pytest, existing RADIO-GS artifact conventions, Markdown and LaTeX table output.

---

## File Structure

- Create `tests/test_build_teacher_vs_ctfgs_2d_usability_report.py`: unit tests with temporary controlled-evidence and SAM3/DINO aggregate fixtures.
- Create `radio_gs/scripts/build_teacher_vs_ctfgs_2d_usability_report.py`: parser, table builder, Markdown writer, LaTeX writer, and CLI.
- Modify `paper/artifacts/README.md`: add the new report to the T1 artifact list.
- Modify `docs/PROJECT_MAINLINE.md`: reference the consolidated report in the downstream usability section.
- Modify `docs/paper_draft_current.md`: add a short paper-facing 2D frame-wise-RADIO-vs-GaussFM usability subsection.

## Task 1: Tests First

**Files:**
- Create: `tests/test_build_teacher_vs_ctfgs_2d_usability_report.py`

- [ ] **Step 1: Write fixture-driven tests**

```python
from pathlib import Path
import json

from radio_gs.scripts import build_teacher_vs_ctfgs_2d_usability_report as report


def test_build_report_marks_selected_task_wins_and_caveats(tmp_path: Path) -> None:
    controlled = tmp_path / "controlled.json"
    controlled.write_text(json.dumps({"rows": [
        {"method": "Frame-wise RADIO", "lerf_loc_acc": 0.80, "lerf_miou": 0.46},
        {"method": "Nearest-view RADIO cache", "lerf_loc_acc": 0.27, "lerf_miou": 0.15},
        {"method": "Per-Gaussian 1280-D RADIO memory", "lerf_loc_acc": 0.56, "lerf_miou": 0.32},
        {"method": "Full GaussFM", "lerf_loc_acc": 0.87, "lerf_miou": 0.52},
    ]}), encoding="utf-8")
    sam_dino = tmp_path / "sam_dino.json"
    sam_dino.write_text(json.dumps({"macro": {
        "sam3": {
            "point_prompt_segmentation": {"teacher": {"loc_acc": 1.0, "miou": 0.37, "n_samples": 10}, "rendered": {"loc_acc": 1.0, "miou": 0.42, "n_samples": 10}},
            "box_prompt_segmentation": {"teacher": {"loc_acc": 0.87, "miou": 0.66, "n_samples": 10}, "rendered": {"loc_acc": 0.82, "miou": 0.67, "n_samples": 10}},
            "mask_prompt_propagation": {"teacher": {"loc_acc": 0.79, "miou": 0.36, "n_samples": 10}, "rendered": {"loc_acc": 0.67, "miou": 0.38, "n_samples": 10}},
        },
        "dino_v3": {
            "dense_matching": {"teacher": {"hit_rate": 0.57, "mean_score": 0.85, "n_matches": 30}, "rendered": {"hit_rate": 0.54, "mean_score": 0.90, "n_matches": 30}},
            "mask_propagation": {"teacher": {"loc_acc": 0.76, "miou": 0.51, "n_samples": 10}, "rendered": {"loc_acc": 0.79, "miou": 0.48, "n_samples": 10}},
        },
    }}), encoding="utf-8")

    built = report.build_report(controlled, sam_dino)

    assert built["summary"]["primary_rendered_wins"] == 5
    assert built["summary"]["primary_total"] == 6
    assert built["summary"]["universal_superiority"] is False
    assert any("DINOv3 mask propagation mIoU" in item for item in built["summary"]["caveats"])


def test_write_report_outputs_markdown_json_and_latex(tmp_path: Path) -> None:
    built = {
        "text_grounding_rows": [
            {"method": "Frame-wise RADIO", "loc_acc": 0.8, "miou": 0.46, "delta_loc_acc": 0.0, "delta_miou": 0.0},
            {"method": "Full GaussFM", "loc_acc": 0.87, "miou": 0.52, "delta_loc_acc": 0.07, "delta_miou": 0.06},
        ],
        "frozen_head_rows": [
            {"task": "SAM3 point prompt", "primary_metric": "mIoU", "teacher_primary": 0.37, "rendered_primary": 0.42, "delta_primary": 0.05, "secondary_metric": "LocAcc", "teacher_secondary": 1.0, "rendered_secondary": 1.0, "delta_secondary": 0.0, "n": 10, "winner": "rendered"},
            {"task": "DINOv3 mask propagation", "primary_metric": "mIoU", "teacher_primary": 0.51, "rendered_primary": 0.48, "delta_primary": -0.03, "secondary_metric": "LocAcc", "teacher_secondary": 0.76, "rendered_secondary": 0.79, "delta_secondary": 0.03, "n": 10, "winner": "teacher"},
        ],
        "summary": {"primary_rendered_wins": 1, "primary_total": 2, "universal_superiority": False, "caveats": ["DINOv3 mask propagation mIoU remains frame-wise-RADIO-stronger."]},
        "sources": {},
    }

    report.write_outputs(built, tmp_path / "out.json", tmp_path / "out.md", tmp_path / "out.tex")

    assert "selected downstream tasks" in (tmp_path / "out.md").read_text(encoding="utf-8")
    assert "DINOv3 mask propagation" in (tmp_path / "out.tex").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["summary"]["universal_superiority"] is False
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_teacher_vs_ctfgs_2d_usability_report.py -q
```

Expected: FAIL because `radio_gs.scripts.build_teacher_vs_ctfgs_2d_usability_report` does not exist.

## Task 2: Report Builder

**Files:**
- Create: `radio_gs/scripts/build_teacher_vs_ctfgs_2d_usability_report.py`
- Test: `tests/test_build_teacher_vs_ctfgs_2d_usability_report.py`

- [ ] **Step 1: Implement dataclass-free parsing helpers**

Implement:

```python
def build_report(controlled_path: Path, sam_dino_path: Path, *, ransac_path: Path | None = None, prototype_path: Path | None = None) -> dict[str, object]:
    ...

def write_outputs(report: Mapping[str, object], json_path: Path, markdown_path: Path, latex_path: Path) -> None:
    ...
```

The report must compute deltas against the frame-wise RADIO reference row for text grounding, use mIoU as the primary metric for segmentation/propagation tasks, use mean score as the primary DINO dense matching metric, and store caveats for metrics where the teacher remains stronger.

- [ ] **Step 2: Run tests to verify GREEN**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_teacher_vs_ctfgs_2d_usability_report.py -q
```

Expected: PASS.

## Task 3: Real Artifact Build

**Files:**
- Generate: `paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.json`
- Generate: `paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md`
- Generate: `paper/tables/teacher_vs_ctfgs_2d_usability_20260525.tex`

- [ ] **Step 1: Run the builder on real artifacts**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_teacher_vs_ctfgs_2d_usability_report.py
```

Expected: three output paths printed.

- [ ] **Step 2: Inspect output**

Run:

```bash
sed -n '1,220p' paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md
```

Expected: the report lists the LERF controlled evidence table, the frozen-head SAM3/DINO table, and caveats for DINO metrics where the teacher remains stronger.

## Task 4: Documentation Hookup

**Files:**
- Modify: `paper/artifacts/README.md`
- Modify: `docs/PROJECT_MAINLINE.md`
- Modify: `docs/paper_draft_current.md`

- [ ] **Step 1: Update paper-facing docs**

Add short references to the new artifact. The text must say that the new table supports selected downstream feature-usability improvements and explicitly avoids a universal superiority claim.

- [ ] **Step 2: Verify docs and claims**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest tests/test_build_teacher_vs_ctfgs_2d_usability_report.py tests/test_validate_paper_claims.py -q
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m radio_gs.scripts.validate_paper_claims
```

Expected: both commands exit 0.

## Self-Review Checklist

- Every task maps to a requirement in the design spec.
- The plan keeps artifact generation separate from method-training changes.
- The report has a conservative caveat path for frame-wise-RADIO-stronger DINO metrics.
- No step requires GPU or modifies running experiments.
