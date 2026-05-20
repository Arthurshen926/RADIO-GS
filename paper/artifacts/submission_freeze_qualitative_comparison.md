# Submission-Freeze Qualitative Comparison

Generated on 2026-05-02 from frozen LERF overlay assets.

## Main Figure

- Figure: `output/radio_gs/paper_figures/submission_freeze_lerf_qualitative_comparison.png`
- Script: `radio_gs/scripts/compose_lerf_qualitative_comparison.py`
- LaTeX include: `paper/radio_gs_draft.tex`

The main figure uses one representative query from each frozen LERF scene and
shows `RGB`, `GT`, `Teacher RADIO`, and `RADIO-GS (Ours)` columns. It does not
fabricate external baseline visualizations because protocol-matched LERF,
LangSplat, or LEGaussians per-frame outputs are not present in the local
submission-freeze package.

## Rows

| Scene | Frame | Query | Teacher source | RADIO-GS source |
|---|---:|---|---|---|
| Figurines | 00152 | green apple | `output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/visualisations/figurines/lerf_grounding_frame_00152_teacher_green_apple.png` | `output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/visualisations/figurines/lerf_grounding_frame_00152_rendered_green_apple.png` |
| Ramen | 00024 | wavy noodles | `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00024_teacher_wavy_noodles.png` | `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00024_rendered_wavy_noodles.png` |
| Teatime | 00140 | coffee mug | `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00140_teacher_coffee_mug.png` | `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00140_rendered_coffee_mug.png` |
| Waldo Kitchen | 00089 | knife | `output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/visualisations/waldo_kitchen/lerf_grounding_frame_00089_teacher_knife.png` | `output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/visualisations/waldo_kitchen/lerf_grounding_frame_00089_rendered_knife.png` |

## Appendix Candidate

- Full debug-grid figure:
  `output/radio_gs/paper_figures/submission_freeze_lerf_qualitative_debug_grid.png`

This version keeps the complete per-scene query grids and is better suited for
appendix/error analysis than for the main paper.

## External Baseline Replacement Plan

If protocol-matched external outputs become available, add columns between `GT`
and `RADIO-GS (Ours)` in the same visual style:

- `LERF`
- `LangSplat`
- `LEGaussians`

The replacement columns should use the same frame, query, image crop, colormap,
and overlay alpha as the RADIO-GS figure.

## Figure-Style References

The figure format follows the qualitative-comparison pattern used by LERF,
LangSplat, and LEGaussians: rows are scene/query examples, columns compare RGB,
annotation/reference, and method heatmap overlays.

- LERF: https://openaccess.thecvf.com/content/ICCV2023/html/Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.html
- LangSplat: https://langsplat.github.io/
- LEGaussians: https://openaccess.thecvf.com/content/CVPR2024/html/Shi_Language_Embedded_3D_Gaussians_for_Open-Vocabulary_Scene_Understanding_CVPR_2024_paper.html

