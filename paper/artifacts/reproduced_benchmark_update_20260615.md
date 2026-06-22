# Reproduced Benchmark and Contribution Wording Update

Date: 2026-06-15

This note records the paper-facing update requested after the reproduced benchmark tables were finalized.

## Contribution wording

The compact RADIO feature field is part of the first major contribution, not a separate minor implementation detail. The updated wording is:

> We introduce a compact reconstructive RADIO Gaussian feature field. Each Gaussian stores a low-dimensional compact latent rather than a full 1280-D RADIO feature; spatial context, visibility/reliability cues, and HCD/CTR decoding reconstruct RADIO-compatible features on demand for rendered dense maps and primitive/point queries.

This wording is now reflected in:

- `paper/artifacts/project_midterm_report_cn_20260615.md`
- `paper/artifacts/tpami_storyline_outline_20260609.md`

## Paper-facing quantitative results

The main LERF and ScanNet tables now use the reproduced-protocol comparison results, not source-context caveats.

### LERF-OVS 2D open-vocabulary query

Ours: 64.98 mIoU / 82.68 Acc.

Per scene:

- Figurines: 64.29 / 92.86
- Teatime: 76.09 / 93.22
- Ramen: 53.78 / 62.83
- Waldo Kitchen: 65.76 / 81.82

### LERF-OVS 3D direct open-vocabulary query

Ours: 54.36 mIoU / 80.84 Acc.

Per scene:

- Figurines: 59.36 / 92.86
- Teatime: 71.04 / 89.83
- Ramen: 38.43 / 63.38
- Waldo Kitchen: 48.61 / 77.27

### VALA-aligned ScanNet-v2 protocol

Ours:

- 19 classes: 36.55 mIoU / 50.57 mAcc
- 15 classes: 42.78 mIoU / 72.85 mAcc
- 10 classes: 57.85 mIoU / 77.93 mAcc

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
