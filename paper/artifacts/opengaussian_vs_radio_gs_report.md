# OpenGaussian vs RADIO-GS Baseline Report

Baseline selected: OpenGaussian, because its official release covers both ScanNet open-vocabulary point-cloud understanding and LeRF object selection, and the repository provides reproducible training/evaluation code.

Sources: OpenGaussian project page `https://3d-aigc.github.io/OpenGaussian/`, official code `https://github.com/yanmin-wu/OpenGaussian`, and arXiv `https://arxiv.org/abs/2406.02058`.

## ScanNet 3D Segmentation

| Method | Source | split19 mIoU | split19 mAcc | split15 mIoU | split15 mAcc | split10 mIoU | split10 mAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| OpenGaussian | paper Table 2 | 0.2473 | 0.4154 | 0.3013 | 0.4825 | 0.3829 | 0.5519 |
| OpenGaussian | local reproduction | 0.2978 | 0.4507 | 0.3275 | 0.4964 | 0.4186 | 0.5847 |
| RADIO-GS | local v67 direct point-query | 0.3538 | 0.6076 | 0.3573 | 0.6203 | 0.4293 | 0.7051 |

### Per-Scene Local Reproduction

| Scene | OpenGaussian 19 mIoU/mAcc | OpenGaussian 15 mIoU/mAcc | OpenGaussian 10 mIoU/mAcc | RADIO-GS 19 mIoU/mAcc | RADIO-GS 15 mIoU/mAcc | RADIO-GS 10 mIoU/mAcc |
|---|---:|---:|---:|---:|---:|---:|
| scene0000_00 | 0.3335/0.5001 | 0.3306/0.5462 | 0.3434/0.5897 | 0.2939/0.5972 | 0.2739/0.5896 | 0.3065/0.6854 |
| scene0062_00 | 0.3410/0.5544 | 0.3435/0.5598 | 0.4877/0.6938 | 0.3794/0.7303 | 0.3794/0.7303 | 0.5266/0.8295 |
| scene0070_00 | 0.2119/0.3259 | 0.2693/0.3241 | 0.3683/0.4929 | 0.2297/0.3311 | 0.2405/0.3525 | 0.3238/0.4918 |
| scene0097_00 | 0.3333/0.4853 | 0.3097/0.5282 | 0.4787/0.6555 | 0.4533/0.7681 | 0.4311/0.7537 | 0.4851/0.7114 |
| scene0140_00 | 0.2297/0.4066 | 0.3010/0.4079 | 0.4265/0.5665 | 0.3194/0.5175 | 0.3671/0.5922 | 0.3888/0.6614 |
| scene0200_00 | 0.3752/0.5244 | 0.3787/0.5330 | 0.3462/0.4830 | 0.4335/0.6872 | 0.4335/0.6872 | 0.5132/0.7855 |
| scene0347_00 | 0.3295/0.4441 | 0.3280/0.4899 | 0.3832/0.4903 | 0.5938/0.7618 | 0.5754/0.7479 | 0.7067/0.8010 |
| scene0400_00 | 0.2369/0.3972 | 0.3020/0.5363 | 0.4108/0.6100 | 0.3299/0.6603 | 0.3299/0.6603 | 0.3908/0.7832 |
| scene0590_00 | 0.2945/0.4098 | 0.3936/0.5287 | 0.5449/0.6822 | 0.2664/0.4387 | 0.2963/0.4949 | 0.3643/0.6352 |
| scene0645_00 | 0.2923/0.4594 | 0.3181/0.5095 | 0.3959/0.5830 | 0.2381/0.5842 | 0.2458/0.5943 | 0.2875/0.6666 |

## LERF-OVS

OpenGaussian reports LeRF as 3D object selection mIoU and mAcc@0.25. RADIO-GS reports rendered-feature 2D grounding and, when available, VPR-backed direct 3D primitive selection. The direct 3D rows follow the same query-select-render metric family, while rendered-feature rows are a different protocol.

| Method | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | paper object-selection mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| OpenGaussian | paper object-selection mAcc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| RADIO-GS | rendered-feature LocAcc | 0.8214 | 0.9014 | 0.8983 | 0.8636 | 0.8712 |
| RADIO-GS | rendered-feature heatmap mIoU | 0.4244 | 0.6201 | 0.5760 | 0.4769 | 0.5243 |
| RADIO-GS/CTF-GS | VPR direct 3D selection mIoU | 0.5309 | 0.5805 | 0.5662 | 0.2429 | 0.4801 |
| RADIO-GS/CTF-GS | VPR direct 3D selection Acc@0.25 | 0.7857 | 0.7465 | 0.7627 | 0.4091 | 0.6760 |

### Local OpenGaussian LeRF Asset Check

| Scene | Images | Language masks | Language feats | Labels | Ready |
|---|---:|---:|---:|---:|---|
| figurines | 299 | 0 | 0 | 4 | False |
| ramen | 131 | 0 | 0 | 7 | False |
| teatime | 177 | 0 | 0 | 6 | False |
| waldo_kitchen | 187 | 0 | 0 | 5 | False |

Local OpenGaussian LeRF reproduction is blocked: OpenGaussian's LeRF recipe requires per-frame `language_features/*_s.npy` SAM masks and `language_features/*_f.npy` CLIP features. The inspected local LERF folders have images/COLMAP/labels but no complete `language_features/` assets.

## Qualitative Artifacts

- ScanNet GT/RADIO-GS/OpenGaussian montage: `output/baselines/opengaussian/scannet_qualitative_comparison.png`
- Per-scene OpenGaussian PLY/PNG files: `output/baselines/opengaussian/scannet_eval/visualizations/{scene}/`
- RADIO-GS v67 per-scene PLY files: `output/scannet_pointcloud_eval/{scene}_v67_teacherbalanced_fromv63_best_gidx_labelpoint/visualizations/{scene}/`
