# Boundary Error Readout

- Sweep source: `/root/RADIO-GS/output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260516.json`
- Run: `pad16`; selection: `thr0p25`
- Query count with per-query boundary details: `208`
- Boundary error is `1 - boundary_f`; trimap error is `1 - trimap_iou`.

## Strict Readout

| Macro mIoU | Boundary-F | Boundary error | Trimap IoU | Trimap error |
|---:|---:|---:|---:|---:|
| 0.5705 | 0.6681 | 0.3319 | 0.3958 | 0.6042 |

## Scene Rows

| Scene | mIoU | Boundary-F | Boundary error | Trimap IoU | Trimap error | N |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.6136 | 0.7116 | 0.2884 | 0.3986 | 0.6014 | 56 |
| Ramen | 0.6409 | 0.7713 | 0.2287 | 0.4400 | 0.5600 | 71 |
| Teatime | 0.6130 | 0.7255 | 0.2745 | 0.4626 | 0.5374 | 59 |
| Waldo Kitchen | 0.4142 | 0.4638 | 0.5362 | 0.2820 | 0.7180 | 22 |

## Query Correlations

| Pair | Pearson r |
|---|---:|
| IoU vs boundary-F | 0.9148 |
| IoU vs trimap IoU | 0.8412 |
| IoU vs abs(log overselect ratio) | -0.6217 |

## Overselect Buckets

| Bucket | Count | Mean IoU | Mean boundary-F | Mean boundary error | Mean trimap IoU | Mean overselect |
|---|---:|---:|---:|---:|---:|---:|
| under | 52 | 0.1420 | 0.2958 | 0.7042 | 0.0903 | 0.1891 |
| balanced | 134 | 0.8559 | 0.9363 | 0.0637 | 0.5665 | 1.0056 |
| over | 22 | 0.1400 | 0.3081 | 0.6919 | 0.2935 | 5.1616 |

## GT-Area Buckets

| Bucket | Count | Mean IoU | Mean boundary-F | Mean boundary error | Mean trimap IoU | Mean overselect |
|---|---:|---:|---:|---:|---:|---:|
| small | 70 | 0.5304 | 0.6837 | 0.3163 | 0.4334 | 1.8821 |
| mid | 69 | 0.7028 | 0.7995 | 0.2005 | 0.4827 | 1.1069 |
| large | 69 | 0.5728 | 0.6463 | 0.3537 | 0.3393 | 0.7249 |

## Worst Boundary Cases

| Scene | Frame | Category | IoU | Boundary-F | Trimap IoU | GT px | Pred px | Overselect |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Figurines | `frame_00041` | miffy | 0.0000 | 0.0000 | 0.0000 | 2252 | 3791 | 1.6834 |
| Figurines | `frame_00041` | pikachu | 0.0000 | 0.0000 | 0.0000 | 12082 | 7249 | 0.6000 |
| Figurines | `frame_00041` | pirate hat | 0.0000 | 0.0000 | 0.0000 | 1835 | 0 | 0.0000 |
| Figurines | `frame_00041` | tesla door handle | 0.0000 | 0.0000 | 0.0000 | 6872 | 4227 | 0.6151 |
| Figurines | `frame_00105` | pirate hat | 0.0000 | 0.0000 | 0.0000 | 899 | 0 | 0.0000 |
| Figurines | `frame_00105` | tesla door handle | 0.0000 | 0.0000 | 0.0000 | 10582 | 0 | 0.0000 |
| Figurines | `frame_00152` | jake | 0.0000 | 0.0000 | 0.0000 | 10803 | 55885 | 5.1731 |
| Figurines | `frame_00152` | pirate hat | 0.0000 | 0.0000 | 0.0000 | 2080 | 0 | 0.0000 |
| Figurines | `frame_00195` | bag | 0.0000 | 0.0000 | 0.0000 | 6112 | 0 | 0.0000 |
| Figurines | `frame_00195` | miffy | 0.0000 | 0.0000 | 0.0000 | 974 | 8078 | 8.2936 |

## Alpha/Depth Discontinuity Status

- Alpha/depth discontinuity maps are not present in the frozen SAM3-box result JSONs.
- The report therefore supports a measured boundary-error readout, not a causal occlusion/discontinuity claim.
- The current occlusion-adjacent evidence remains protocol metadata and the negative alpha/alpha-depth registration-weight ablation; a stronger audit needs saved per-query alpha/depth edges aligned to the official masks.
