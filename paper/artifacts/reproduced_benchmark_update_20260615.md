# Paper Result Provenance Correction and Contribution Wording Update

Date: 2026-06-15

This note has been corrected to record the locally auditable GaussFM outputs.
External rows in the associated tables are published source-paper context and
must not be described as local same-protocol reproductions.

## Contribution wording

The compact RADIO feature field is part of the first major contribution, not a separate minor implementation detail. The updated wording is:

> We introduce a compact reconstructive RADIO Gaussian feature field. Each Gaussian stores a low-dimensional compact latent rather than a full 1280-D RADIO feature; spatial context, visibility/reliability cues, and HCD/CTR decoding reconstruct RADIO-compatible features on demand for rendered dense maps and primitive/point queries.

This wording is now reflected in:

- `paper/artifacts/project_midterm_report_cn_20260615.md`
- `paper/artifacts/tpami_storyline_outline_20260609.md`

## Paper-facing quantitative results

The main LERF and ScanNet tables use local GaussFM outputs alongside explicitly
labelled published context. A strict unified rerun is pending.

### LERF-OVS 2D open-vocabulary query

Ours: 58.89 mIoU / 85.98 Acc.

Per scene:

- Figurines: 52.43 / 82.14
- Teatime: 65.15 / 89.83
- Ramen: 63.25 / 90.14
- Waldo Kitchen: 54.75 / 81.82

### LERF-OVS 3D direct open-vocabulary query

Ours: 50.14 mIoU / 70.44 Acc@0.25.

Per scene:

- Figurines: 51.04 / 67.86
- Teatime: 56.40 / 76.27
- Ramen: 59.99 / 83.10
- Waldo Kitchen: 33.12 / 54.55

### VALA-aligned ScanNet-v2 protocol

Ours:

- 19 classes: 38.06 mIoU / 61.29 mAcc
- 15 classes: 38.71 mIoU / 63.15 mAcc
- 10 classes: 47.11 mIoU / 72.00 mAcc

## Updated files

- `paper/artifacts/final_rows.yaml`
- `paper/artifacts/project_midterm_report_cn_20260615.md`
- `paper/artifacts/tpami_storyline_outline_20260609.md`
- `paper/lerf_ovs_main_table.tex`
- `paper/lerf_direct_3d_selection_table.tex`
- `paper/scannet_published_context_table.tex`
- `paper/radio_gs_tpami.tex`
- `paper/radio_gs_draft.tex`

Historical audit artifacts are intentionally left unchanged unless they are regenerated, because they record prior intermediate experiments.
