# GaussFM Academic Short Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a concise 11-slide academic PPTX for the GaussFM paper.

**Architecture:** Add one focused Python builder script that uses `python-pptx`, existing paper figures, and hard-coded paper-facing result rows. The script writes both the PPTX and a Markdown outline.

**Tech Stack:** Python 3.9, `python-pptx`, Pillow, existing `paper/figures/*.png` assets.

---

### Task 1: Add Short Presentation Builder

**Files:**
- Create: `radio_gs/scripts/build_academic_short_presentation.py`
- Create: `paper/gaussfm_academic_short_presentation.pptx`
- Create: `paper/gaussfm_academic_short_presentation_outline.md`

- [ ] **Step 1: Implement a single-purpose PPTX builder**

Use `python-pptx` to create an 11-slide deck. Include reusable helpers for title, bullet, card, table, and image placement. Use existing PNG assets from `paper/figures`.

- [ ] **Step 2: Generate the deck**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_academic_short_presentation.py
```

Expected: the script writes `paper/gaussfm_academic_short_presentation.pptx` and `paper/gaussfm_academic_short_presentation_outline.md`.

- [ ] **Step 3: Verify visible terminology**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile
p = Path("paper/gaussfm_academic_short_presentation.pptx")
with ZipFile(p) as zf:
    text = "\n".join(zf.read(n).decode("utf-8", "ignore") for n in zf.namelist() if n.endswith(".xml"))
for banned in ["OpenGaFF", "CTF-GS", "CTFGS"]:
    assert banned not in text, banned
assert "GaussFM" in text
print("pptx terminology ok")
PY
```

Expected: `pptx terminology ok`.

- [ ] **Step 4: Verify repository hygiene**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.
