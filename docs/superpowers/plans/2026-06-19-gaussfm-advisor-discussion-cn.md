# GaussFM 中文导师讨论版 PPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Chinese advisor-discussion PPT deck with fuller experiments and ablations.

**Architecture:** Add one independent Python builder script that produces a new PPTX and Markdown outline. The script uses current paper-facing numbers from existing artifacts and does not overwrite the English short deck.

**Tech Stack:** Python 3.9, `python-pptx`, Pillow, existing paper figure assets.

---

### Task 1: Add Chinese Advisor Deck Builder

**Files:**
- Create: `radio_gs/scripts/build_advisor_discussion_cn_presentation.py`
- Create: `paper/gaussfm_advisor_discussion_cn.pptx`
- Create: `paper/gaussfm_advisor_discussion_cn_outline.md`

- [ ] **Step 1: Implement the builder script**

Create a 20-slide Chinese deck with reusable helpers for text, cards, tables, and figures. Use current terminology and claim boundaries.

- [ ] **Step 2: Generate the deck**

Run:

```bash
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/build_advisor_discussion_cn_presentation.py
```

Expected: script writes the PPTX and outline under `paper/`.

- [ ] **Step 3: Verify structure and terminology**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile
p = Path("paper/gaussfm_advisor_discussion_cn.pptx")
with ZipFile(p) as zf:
    names = zf.namelist()
    slides = [n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
    text = "\n".join(zf.read(n).decode("utf-8", "ignore") for n in names if n.endswith(".xml"))
assert len(slides) == 20, len(slides)
for token in ["GaussFM", "frame-wise RADIO", "VALA-aligned"]:
    assert token in text, token
for banned in ["OpenGaFF", "CTF-GS", "CTFGS"]:
    assert banned not in text, banned
print("advisor deck structure and terminology ok")
PY
```

Expected: `advisor deck structure and terminology ok`.

- [ ] **Step 4: Run hygiene check**

Run:

```bash
git diff --check -- \
  docs/superpowers/specs/2026-06-19-gaussfm-advisor-discussion-cn-design.md \
  docs/superpowers/plans/2026-06-19-gaussfm-advisor-discussion-cn.md \
  radio_gs/scripts/build_advisor_discussion_cn_presentation.py \
  paper/gaussfm_advisor_discussion_cn_outline.md
```

Expected: no output and exit code 0.
