# LERF Failure Analysis

This report is generated from the current best rendered LERF-OVS JSON files. Rows are ranked by localization failure first and mIoU second, exposing the small-object and peak-placement failure modes discussed in the paper.

## Worst / Fragile Categories

| Scene | Category | LocAcc | mIoU | Samples |
|---|---|---:|---:|---:|
| Figurines | bag | 0.000 | 0.000 | 1 |
| Figurines | miffy | 0.000 | 0.006 | 2 |
| Teatime | dall-e brand | 0.000 | 0.010 | 2 |
| Figurines | jake | 0.000 | 0.017 | 3 |
| Waldo Kitchen | ketchup | 0.000 | 0.027 | 1 |
| Waldo Kitchen | spatula | 0.000 | 0.029 | 1 |
| Ramen | corn | 0.000 | 0.095 | 5 |
| Figurines | red toy chair | 0.000 | 0.126 | 1 |
| Figurines | pirate hat | 0.500 | 0.119 | 4 |
| Waldo Kitchen | sink | 0.500 | 0.837 | 2 |
| Ramen | napkin | 0.600 | 0.123 | 5 |
| Teatime | bag of cookies | 0.600 | 0.556 | 5 |

## Per-Scene Summary

| Scene | Categories | Mean LocAcc | Mean mIoU |
|---|---:|---:|---:|
| Figurines | 21 | 0.770 | 0.421 |
| Ramen | 14 | 0.900 | 0.559 |
| Teatime | 14 | 0.868 | 0.521 |
| Waldo Kitchen | 18 | 0.861 | 0.375 |

## Interpretation

- Most hard cases are categories where one feature-cell peak shift is enough to fail the LERF LocAcc metric.
- Figurines contributes many fragile small-object categories, supporting a targeted small-object/feature-resolution analysis.
- mIoU and LocAcc should be interpreted together: broader object regions can raise overlap while moving the argmax outside a small annotation mask.
