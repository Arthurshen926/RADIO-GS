# Direct3D Confidence and Coverage Analysis

- Source root: `/root/RADIO-GS/output/radio_gs/lerf_direct_3d_selection_threshold_grabcut_20260515`
- Selection: `thr0p25`
- Teacher-score proxy: mean of top 1.00% primitive scores per category

## Scene View-Coverage

| Scene | Mean valid views | Registered fraction | mIoU | Acc@0.25 | Zero-pred rate | Mean top-score |
|---|---:|---:|---:|---:|---:|---:|
| figurines | 9.5911 | 0.2516 | 0.5309 | 0.7857 | 0.0714 | 0.3258 |
| ramen | 20.8716 | 0.6784 | 0.5805 | 0.7465 | 0.0282 | 0.4103 |
| teatime | 13.0006 | 0.5474 | 0.5662 | 0.7627 | 0.0169 | 0.4840 |
| waldo_kitchen | 5.7542 | 0.5049 | 0.2429 | 0.4091 | 0.1818 | 0.3588 |

## Scene Correlations

| Pair | Pearson r |
|---|---:|
| mean valid views vs mIoU | 0.7588 |
| registered fraction vs mIoU | 0.0961 |
| mean top-score vs mIoU | 0.4326 |
| mean text margin vs mIoU | 0.2355 |

## Teacher-Score Confidence Buckets

| Bucket | Queries | Score range | Mean IoU | Acc@0.25 | Zero-pred rate | Mean GT pixels |
|---|---:|---:|---:|---:|---:|---:|
| low | 70 | 0.0443-0.2816 | 0.4345 | 0.6143 | 0.1000 | 12647.0 |
| mid | 69 | 0.2816-0.5296 | 0.5132 | 0.7101 | 0.0435 | 18025.6 |
| high | 69 | 0.5749-0.9363 | 0.6358 | 0.8551 | 0.0145 | 41830.9 |

## Text-Ambiguity Buckets

| Bucket | Queries | Margin range | Mean IoU | Acc@0.25 | Zero-pred rate | Mean GT pixels |
|---|---:|---:|---:|---:|---:|---:|
| ambiguous | 70 | -0.4152-0.0859 | 0.4329 | 0.6286 | 0.1000 | 9771.0 |
| mixed | 69 | 0.0913-0.3955 | 0.5303 | 0.7246 | 0.0290 | 20254.5 |
| distinct | 69 | 0.3955-0.8882 | 0.6202 | 0.8261 | 0.0290 | 42519.5 |

## Low-Confidence Failure Examples

| Scene | Frame | Category | IoU | Pred px | GT px | Top-score |
|---|---|---|---:|---:|---:|---:|
| figurines | frame_00041 | pirate hat | 0.0000 | 0 | 1835 | 0.0448 |
| figurines | frame_00105 | pirate hat | 0.0000 | 0 | 899 | 0.0448 |
| figurines | frame_00152 | pirate hat | 0.0000 | 0 | 2080 | 0.0448 |
| figurines | frame_00195 | pirate hat | 0.0000 | 0 | 1873 | 0.0448 |
| waldo_kitchen | frame_00053 | yellow desk | 0.0000 | 1959 | 66361 | 0.0584 |
| waldo_kitchen | frame_00066 | plastic ladle | 0.0000 | 0 | 3558 | 0.0652 |
| ramen | frame_00128 | corn | 0.0768 | 3708 | 723 | 0.1082 |
| ramen | frame_00119 | corn | 0.0803 | 8347 | 775 | 0.1082 |
| ramen | frame_00065 | corn | 0.0903 | 6931 | 1199 | 0.1082 |
| ramen | frame_00024 | corn | 0.0956 | 13815 | 1837 | 0.1082 |
| ramen | frame_00060 | corn | 0.1045 | 10210 | 1412 | 0.1082 |
| ramen | frame_00128 | hand | 0.0000 | 0 | 9381 | 0.1087 |

## Ambiguous-Text Failure Examples

| Scene | Frame | Category | IoU | Pred px | GT px | Text margin |
|---|---|---|---:|---:|---:|---:|
| teatime | frame_00107 | bear nose | 0.0926 | 2336 | 22933 | -0.4152 |
| teatime | frame_00002 | bear nose | 0.0943 | 2852 | 9906 | -0.4152 |
| teatime | frame_00025 | bear nose | 0.1355 | 3399 | 7639 | -0.4152 |
| teatime | frame_00025 | hooves | 0.0000 | 1096 | 4678 | -0.3102 |
| teatime | frame_00043 | hooves | 0.0577 | 3274 | 55160 | -0.3102 |
| waldo_kitchen | frame_00066 | plastic ladle | 0.0000 | 0 | 3558 | -0.2782 |
| figurines | frame_00152 | red apple | 0.2371 | 4473 | 3010 | -0.1873 |
| figurines | frame_00041 | pirate hat | 0.0000 | 0 | 1835 | -0.1745 |
| figurines | frame_00105 | pirate hat | 0.0000 | 0 | 899 | -0.1745 |
| figurines | frame_00152 | pirate hat | 0.0000 | 0 | 2080 | -0.1745 |
| figurines | frame_00195 | pirate hat | 0.0000 | 0 | 1873 | -0.1745 |
| waldo_kitchen | frame_00053 | yellow desk | 0.0000 | 1959 | 66361 | -0.1561 |

## Interpretation Notes

- View coverage is scene-level registration support from the VPR score cache; it is a GT-free mechanism proxy, not a causal ablation.
- Teacher-score confidence is computed only from primitive text-score caches and is independent of LERF masks.
- Text ambiguity uses the margin between a query score and the strongest competing scene category on the same top-scoring primitives.
- Query rows are evaluation instances, so repeated categories across frames receive the same category-level score proxy.
