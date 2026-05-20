# Alpha/Depth Boundary Alignment Report

- Sweep source: `output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260517_geometry.json`
- Run: `pad16_geometry`; selection: `thr0p25`
- Query records: `208`
- Query records with alpha/depth geometry metrics: `208`
- Query overlay artifacts: `208`
- Status: `available`

## Scene Coverage

| Scene | N | Query records | Geometry records | mIoU | Boundary-F |
|---|---:|---:|---:|---:|---:|
| Figurines | 56 | 56 | 56 | 0.6136 | 0.7116 |
| Ramen | 71 | 71 | 71 | 0.6409 | 0.7713 |
| Teatime | 59 | 59 | 59 | 0.6130 | 0.7255 |
| Waldo Kitchen | 22 | 22 | 22 | 0.4142 | 0.4638 |

## Query Correlations

| Pair | Pearson r |
|---|---:|
| boundary error vs alpha edge gt boundary mean | 0.1515 |
| boundary error vs depth edge gt boundary mean | 0.1178 |
| boundary error vs discontinuity gt boundary mean | 0.1438 |
| boundary error vs discontinuity error boundary mean | 0.1506 |

## Discontinuity Buckets

| Bucket | Count | Mean IoU | Mean boundary error | Mean disc. on GT boundary | Mean disc. on error boundary |
|---|---:|---:|---:|---:|---:|
| low | 70 | 0.6361 | 0.2473 | 0.1938 | 0.1776 |
| mid | 69 | 0.6290 | 0.2543 | 0.4960 | 0.4473 |
| high | 69 | 0.5394 | 0.3699 | 0.7370 | 0.6437 |

## Worst Geometry-Aligned Cases

| Scene | Frame | Category | IoU | Boundary error | Disc. GT boundary | Disc. error boundary | Overlay |
|---|---|---|---:|---:|---:|---:|---|
| Figurines | `frame_00041` | pikachu | 0.0000 | 1.0000 | 0.7483 | 0.8463 | `geometry_overlays/thr0p25/figurines/frame_00041_pikachu.png` |
| Figurines | `frame_00105` | pirate hat | 0.0000 | 1.0000 | 0.8349 | 0.8349 | `geometry_overlays/thr0p25/figurines/frame_00105_pirate hat.png` |
| Ramen | `frame_00128` | napkin | 0.0000 | 1.0000 | 0.8289 | 0.8289 | `geometry_overlays/thr0p25/ramen/frame_00128_napkin.png` |
| Teatime | `frame_00129` | yellow pouf | 0.0000 | 1.0000 | 0.8070 | 0.8070 | `geometry_overlays/thr0p25/teatime/frame_00129_yellow pouf.png` |
| Figurines | `frame_00152` | pirate hat | 0.0000 | 1.0000 | 0.7858 | 0.7858 | `geometry_overlays/thr0p25/figurines/frame_00152_pirate hat.png` |
| Figurines | `frame_00041` | miffy | 0.0000 | 1.0000 | 0.5872 | 0.7775 | `geometry_overlays/thr0p25/figurines/frame_00041_miffy.png` |
| Figurines | `frame_00195` | pirate hat | 0.0000 | 1.0000 | 0.7685 | 0.7685 | `geometry_overlays/thr0p25/figurines/frame_00195_pirate hat.png` |
| Teatime | `frame_00043` | hooves | 0.0000 | 1.0000 | 0.7141 | 0.7141 | `geometry_overlays/thr0p25/teatime/frame_00043_hooves.png` |
| Waldo Kitchen | `frame_00154` | knife | 0.0000 | 1.0000 | 0.7053 | 0.7053 | `geometry_overlays/thr0p25/waldo_kitchen/frame_00154_knife.png` |
| Waldo Kitchen | `frame_00066` | plastic ladle | 0.0000 | 1.0000 | 0.6961 | 0.6901 | `geometry_overlays/thr0p25/waldo_kitchen/frame_00066_plastic ladle.png` |
