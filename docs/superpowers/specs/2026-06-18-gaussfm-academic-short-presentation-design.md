# GaussFM Academic Short Presentation Design

Date: 2026-06-18

## Goal

Create a concise 10-minute academic presentation for the GaussFM paper. The deck should be understandable to both close peers and broader computer-vision researchers.

## Scope

- Output a new PPTX, not a replacement for the existing long project decks.
- Use the approved problem-method-evidence structure.
- Keep the deck to 11 slides.
- Use the current paper terminology: `GaussFM`, `foundation-feature Gaussian memory`, `VALA-aligned`, `frame-wise RADIO`.
- Do not expose `OpenGaFF` or `CTF-GS` in visible slide text.

## Slide Story

1. Thesis: GaussFM is a compact foundation-feature Gaussian memory.
2. Gap: 2D foundation features are view-local, high-dimensional, and weak for direct 3D queries.
3. Research question: turn high-dimensional multiview features into compact, queryable 3D scene memory.
4. Method overview: show training and inference flow.
5. Core components: compact Gaussian codes, spatial context, foundation-space decoder/refiner, reliability/visibility.
6. Contributions: compact memory, unified query interfaces, evidence of downstream usability and transfer.
7. Result 1: LERF rendered-view open-vocabulary query.
8. Result 2: LERF direct 3D object selection.
9. Result 3: VALA-aligned ScanNet point query.
10. Why it works: ablation and feature usability summary.
11. Takeaway: one compact memory supports 2D, 3D, and point-level open-vocabulary understanding.

## Visual Style

Use a restrained academic style: white or very light gray background, dark text, one teal accent, one blue accent, thin separators, and sparse cards/tables. Each slide should have one headline claim and at most three short supporting bullets.

## Deliverables

- `paper/gaussfm_academic_short_presentation.pptx`
- `paper/gaussfm_academic_short_presentation_outline.md`
- `radio_gs/scripts/build_academic_short_presentation.py`

## Verification

- Run the builder script.
- Confirm the PPTX opens structurally as a valid zip package.
- Scan PPTX XML for banned visible terms: `OpenGaFF`, `CTF-GS`, `CTFGS`.
- Confirm expected method name and title are present.
