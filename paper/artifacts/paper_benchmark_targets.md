# RADIO-GS Published Benchmark Targets

This sheet lists the published methods that should anchor the paper's comparison section. The first block is the primary main-table target set.

## Primary main-table targets

| Method | Paper | Venue | Why it belongs in the main table | Source |
|---|---|---|---|---|
| LERF | LERF: Language Embedded Radiance Fields | ICCV 2023 | Official ICCV 2023 Table 1 LocAcc row; paper reports percentages and the macro here is recomputed over the four LERF-OVS scenes. | https://openaccess.thecvf.com/content/ICCV2023/html/Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.html |
| LangSplat | LangSplat: 3D Language Gaussian Splatting | CVPR 2024 | Official CVPR 2024 Table 1 LangSplat LocAcc row; paper reports percentages and the values are converted to decimals. | https://openaccess.thecvf.com/content/CVPR2024/html/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.html |
| LEGaussians | Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding | CVPR 2024 | Official CVPR 2024 supplementary Table 5 LA row; the supplement labels the last scene as kitchen, so this is an official-source context row rather than a reproduced local-protocol row. | https://openaccess.thecvf.com/content/CVPR2024/supplemental/Shi_Language_Embedded_3D_CVPR_2024_supplemental.pdf |

## Supplementary published methods

| Method | Paper | Venue | Best use in the paper | Source |
|---|---|---|---|---|
| Gaussian Grouping | Gaussian Grouping: Segment and Edit Anything in 3D Scenes | ECCV 2024 | Supplementary comparison for open-world 3D segmentation / editing | https://ymq2017.github.io/gaussian-grouping/ |
| 3D Gaussian Splatting | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | SIGGRAPH 2023 | Geometry / rendering efficiency upper-bound, not a grounding baseline | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ |

## Important exclusions

- `Feature3DGS-style` should stay an internal reproduced baseline or ablation label. It should not be presented as a published external SOTA method.
- Replica room_0 depth results are currently strong supporting evidence, but they do not yet replace the need for published open-vocabulary 3D grounding comparisons.

## Next benchmark actions

1. Keep the official-source four-scene LERF-OVS main table frozen unless new reproduced baselines are added.
2. If making a strict SOTA claim, reproduce LERF/LangSplat/LEGaussians under the local evaluator instead of mixing paper protocols.
3. Keep ScanNet v67 as direct-query transfer evidence rather than a full leaderboard claim.
