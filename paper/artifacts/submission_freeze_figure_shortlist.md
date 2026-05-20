# Submission Freeze Figure Shortlist

Date: 2026-05-02

These qualitative assets come from the formal LERF overlay/profile runs listed
in `submission_freeze_gpu_queue.md`. Use these paths when assembling the paper
qualitative figure so that every image is tied to the frozen evaluation package.

## Recommended Main Qualitative Grid

Assembled main-paper figure:
`output/radio_gs/paper_figures/submission_freeze_lerf_qualitative_comparison.png`

Appendix/debug-grid candidate:
`output/radio_gs/paper_figures/submission_freeze_lerf_qualitative_debug_grid.png`

| Scene | Candidate Frame | Rendered Grid | Teacher Grid | Why use it |
|---|---:|---|---|---|
| Figurines | 00152 | `output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/visualisations/figurines/lerf_grounding_frame_00152_rendered.png` | `output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/visualisations/figurines/lerf_grounding_frame_00152_teacher.png` | Small-object stress case with multiple fine-grained objects. |
| Ramen | 00024 | `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00024_rendered.png` | `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00024_teacher.png` | Strong food-object scene with several repeated semantic targets. |
| Teatime | 00140 | `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00140_rendered.png` | `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00140_teacher.png` | Clear scene-level layout and diverse object categories. |
| Waldo Kitchen | 00089 | `output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/visualisations/waldo_kitchen/lerf_grounding_frame_00089_rendered.png` | `output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/visualisations/waldo_kitchen/lerf_grounding_frame_00089_teacher.png` | Cluttered kitchen grounding case with practical failure/success examples. |

## All Frozen Rendered Grids

### Figurines

- `output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/visualisations/figurines/lerf_grounding_frame_00041_rendered.png`
- `output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/visualisations/figurines/lerf_grounding_frame_00105_rendered.png`
- `output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/visualisations/figurines/lerf_grounding_frame_00152_rendered.png`
- `output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/visualisations/figurines/lerf_grounding_frame_00195_rendered.png`

### Ramen

- `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00006_rendered.png`
- `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00024_rendered.png`
- `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00060_rendered.png`
- `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00065_rendered.png`
- `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00081_rendered.png`
- `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00119_rendered.png`
- `output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/visualisations/ramen/lerf_grounding_frame_00128_rendered.png`

### Teatime

- `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00002_rendered.png`
- `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00025_rendered.png`
- `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00043_rendered.png`
- `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00107_rendered.png`
- `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00129_rendered.png`
- `output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/visualisations/teatime/lerf_grounding_frame_00140_rendered.png`

### Waldo Kitchen

- `output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/visualisations/waldo_kitchen/lerf_grounding_frame_00053_rendered.png`
- `output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/visualisations/waldo_kitchen/lerf_grounding_frame_00066_rendered.png`
- `output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/visualisations/waldo_kitchen/lerf_grounding_frame_00089_rendered.png`
- `output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/visualisations/waldo_kitchen/lerf_grounding_frame_00140_rendered.png`
- `output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/visualisations/waldo_kitchen/lerf_grounding_frame_00154_rendered.png`

## Notes

- The full freeze overlay set contains 358 PNG files, including per-query
  teacher/rendered overlays.
- Use rendered grids for the main paper and teacher grids for appendix or error
  analysis.
- Do not mix these with older `lerf_eval_*` visualizations unless the caption
  clearly marks the protocol difference.
