# RADIO-GS LaTeX Draft

Active manuscript entry point:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error radio_gs_draft.tex
bibtex radio_gs_draft
pdflatex -interaction=nonstopmode -halt-on-error radio_gs_draft.tex
pdflatex -interaction=nonstopmode -halt-on-error radio_gs_draft.tex
```

The draft is intentionally conservative:

- RADIO-GS frozen LERF, ScanNet, and profile numbers are included.
- External LERF baseline rows are discussed as unresolved until provenance is
  closed in `output/radio_gs/reports/baseline_source_verification.md`.
- Figure paths point to the freeze shortlist rather than older visualization
  folders.
