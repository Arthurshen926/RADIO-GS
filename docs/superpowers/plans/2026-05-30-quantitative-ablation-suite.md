# Quantitative Ablation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified quantitative ablation suite that ranks the largest GaussFM contributions across LERF rendered grounding, LERF direct 3D selection, ScanNet point-query transfer, and 2D frozen-head feature usability.

**Architecture:** Add one focused report builder that reads the current paper-facing artifacts and emits JSON, Markdown, and LaTeX tables. The generated outputs separate same-protocol contribution ranking from diagnostic or protocol-mixed rows, so the paper can explain which modules matter without overstating incomparable experiments.

**Tech Stack:** Python 3.9, standard library JSON/regex/pathlib, existing repository artifacts, LaTeX tabular output.

---

### Task 1: Add Unified Quantitative Ablation Builder

**Files:**
- Create: `radio_gs/scripts/build_quantitative_ablation_suite.py`
- Output: `paper/artifacts/quantitative_ablation_suite.json`
- Output: `paper/artifacts/quantitative_ablation_suite.md`
- Output: `paper/quantitative_ablation_summary_table.tex`

- [x] **Step 1: Implement report data model and artifact extraction**

Create a small script with:
- typed dictionaries for contribution rows;
- source validation for the artifacts currently used in the paper;
- explicit, auditable rows for core architecture, rendered readout, direct-3D readout, ScanNet readout, and downstream frozen-head tasks.

- [x] **Step 2: Generate ranking and outputs**

The builder writes:
- JSON with all rows and missing follow-up runs;
- Markdown with detailed tables and interpretation;
- LaTeX with a compact top-contribution table.

- [x] **Step 3: Run generation**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_quantitative_ablation_suite.py
```

Expected: three output paths printed and files created.

### Task 2: Connect Paper Table and Manifest

**Files:**
- Modify: `paper/radio_gs_draft.tex`
- Modify: `radio_gs/scripts/build_paper_assets_manifest.py`

- [x] **Step 1: Add the compact contribution table to the core ablation section**

Insert `\input{quantitative_ablation_summary_table}` near the existing core ablation discussion, and describe that the table ranks same-protocol deltas while keeping diagnostics marked.

- [x] **Step 2: Register the new artifacts in the asset manifest builder**

Add the JSON/Markdown/LaTeX outputs to `build_paper_assets_manifest.py`.

- [x] **Step 3: Regenerate the manifest**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_paper_assets_manifest.py --output paper/artifacts/paper_assets_manifest.json
```

Expected: manifest includes the new quantitative ablation artifacts.

### Task 3: Verify

**Files:**
- Test compile: `radio_gs/scripts/build_quantitative_ablation_suite.py`
- Test compile: `paper/radio_gs_draft.tex`

- [x] **Step 1: Static and output verification**

Run:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m py_compile radio_gs/scripts/build_quantitative_ablation_suite.py
git diff --check
```

Expected: both commands exit 0.

- [x] **Step 2: LaTeX verification**

### Task 4: Add Direct3D Compact-Readout Factorial Summary

**Files:**
- Modify: `radio_gs/scripts/build_quantitative_ablation_suite.py`
- Modify: `paper/radio_gs_draft.tex`
- Modify: `radio_gs/scripts/build_paper_assets_manifest.py`
- Output: `paper/lerf_direct3d_compact_readout_ablation_table.tex`
- Output: `paper/artifacts/direct3d_compact_readout_factorial_summary.md`

- [x] **Step 1: Generate compact-readout split from machine-readable Direct3D artifact**

Read `paper/artifacts/lerf_direct3d_compact_readout_ablation_20260528.json` and emit a compact LaTeX table plus Markdown summary separating strict pure one-map, prompt ensemble, RGB component guard, and RGB/score-component guard rows.

- [x] **Step 2: Connect the table to the Direct3D section**

Insert `\input{lerf_direct3d_compact_readout_ablation_table}` near the LERF Direct 3D result and explicitly reference the table in the surrounding protocol discussion.

- [x] **Step 3: Register the generated Direct3D split artifacts**

Add the LaTeX table and Markdown summary to `build_paper_assets_manifest.py`.

Run:

```bash
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error radio_gs_draft.tex
```

Expected: PDF build exits 0.
