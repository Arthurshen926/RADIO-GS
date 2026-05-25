# LERF Direct-3D Feature-only SAM Boundary Readout (2026-05-25)

Protocol: no query/evaluation RGB SAM3. Prompt-conditioned feature-only SAM mask head uses direct-3D rendered coarse masks, score-heatmap guards, and geometry gates. Baseline is the existing no-RGB compact direct-field row.

## feature_sam3_heatguard

| Scene | Best tag | mIoU | Δ vs no-RGB baseline | Boundary-F | SAM accept | thr0p25 mIoU |
|---|---:|---:|---:|---:|---:|---:|
| figurines | thr0p11 | 0.4200 | -0.0061 | 0.5800 | 0.232 | 0.3660 |
| teatime | thr0p5 | 0.5497 | +0.0271 | 0.7005 | 0.254 | 0.5246 |
| waldo_kitchen | thr0p3 | 0.2641 | -0.0116 | 0.2950 | 0.091 | 0.2635 |

Macro best mIoU: 0.4113; macro Δ: +0.0031.

## feature_sam3_strictgate

| Scene | Best tag | mIoU | Δ vs no-RGB baseline | Boundary-F | SAM accept | thr0p25 mIoU |
|---|---:|---:|---:|---:|---:|---:|
| ramen | thr0p45 | 0.5078 | -0.0633 | 0.6406 | 0.000 | 0.4828 |
| waldo_kitchen | thr0p3 | 0.2621 | -0.0136 | 0.2937 | 0.000 | 0.2538 |

Macro best mIoU: 0.3849; macro Δ: -0.0385.

## feature_sam3_ramen_weightedlog

| Scene | Best tag | mIoU | Δ vs no-RGB baseline | Boundary-F | SAM accept | thr0p25 mIoU |
|---|---:|---:|---:|---:|---:|---:|
| ramen | thr0p45 | 0.5115 | -0.0596 | 0.6372 | 0.113 | 0.4849 |

Macro best mIoU: 0.5115; macro Δ: -0.0596.

Conclusion: implemented and debugged, but not promoted as the main LERF direct-3D row yet. It improves Teatime and provides positive per-query boundary corrections, but hurts Ramen/Waldo and slightly trails Figurines under the no-RGB baseline. Keep as an ablation/failure analysis unless a later gate improves macro mIoU.