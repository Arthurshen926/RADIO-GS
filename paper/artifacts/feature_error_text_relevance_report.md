# Feature Error vs Text Relevance Error

- Rendered source: `/root/RADIO-GS/output/radio_gs/reports/lerf_rendered_grounding_paper_ckpt_threshold_sweep.json` at variant `0.60`
- Feature error proxy: `1 - best validation cos_decoded` from the frozen scene training log.
- Text relevance error proxies: `1 - rendered mIoU` and `1 - rendered LocAcc` under the frozen LERF evaluator.

| Scene | Best val cosine | Feature error | Rendered mIoU | mIoU error | Rendered LocAcc | Loc error |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.7618 | 0.2382 | 0.4244 | 0.5756 | 0.8214 | 0.1786 |
| Ramen | 0.8376 | 0.1624 | 0.6201 | 0.3799 | 0.9014 | 0.0986 |
| Teatime | 0.7995 | 0.2005 | 0.5760 | 0.4240 | 0.8983 | 0.1017 |
| Waldo Kitchen | 0.7791 | 0.2209 | 0.4769 | 0.5231 | 0.8636 | 0.1364 |

## Correlations

| Pair | Pearson r |
|---|---:|
| feature error vs mIoU error | 0.9568 |
| feature error vs LocAcc error | 0.8713 |

## Interpretation

- This is a scene-level mechanism audit, not a per-query causal proof.
- If correlations are weak, the paper should avoid claiming that lower global reconstruction error alone explains text grounding.
- Stronger future evidence would require per-view or per-query feature residuals aligned with text heatmap failures.
