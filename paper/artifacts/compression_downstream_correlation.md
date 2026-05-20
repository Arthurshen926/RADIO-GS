# Compression vs Downstream Correlation

- Storage source: `/root/RADIO-GS/output/radio_gs/reports/storage_footprint_report.md`
- Rendered source: `/root/RADIO-GS/output/radio_gs/reports/lerf_rendered_grounding_paper_ckpt_threshold_sweep.json` at variant `0.60`
- Direct3D root: `/root/RADIO-GS/output/radio_gs/lerf_direct_3d_selection_threshold_grabcut_20260515` with selection `thr0p25`

| Scene | Saving ratio | Rendered mIoU | Direct3D mIoU | Direct3D Acc@0.25 |
|---|---:|---:|---:|---:|
| Figurines | 1.74x | 0.4244 | 0.5309 | 0.7857 |
| Ramen | 3.00x | 0.6201 | 0.5805 | 0.7465 |
| Teatime | 3.32x | 0.5760 | 0.5662 | 0.7627 |
| Waldo Kitchen | 4.04x | 0.4769 | 0.2429 | 0.4091 |

## Correlations

| Pair | Pearson r |
|---|---:|
| saving ratio vs rendered mIoU | 0.3606 |
| saving ratio vs Direct3D mIoU | -0.6158 |

## Interpretation

- The saving ratio is a compact-storage accounting ratio, not an accuracy predictor.
- A weak or negative correlation means higher compression on a larger scene should not be framed as causing stronger downstream mIoU.
- This table supports separating the compactness claim from the direct-query robustness claim.
