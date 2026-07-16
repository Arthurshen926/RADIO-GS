# Official SigLIP2 semantic-oracle decision — 2026-07-16

The frozen four-scene LERF-OVS teacher oracle uses exact annotation category
strings, the official SigLIP2-G text embeddings, no mask refinement, a fixed
protocol threshold, and no test-set calibration.

| teacher | protocol-aligned mIoU | localization accuracy |
|---|---:|---:|
| official SigLIP2 spatial adaptor | 0.015317 | 0.043269 |
| official multiscale crop visual summary | 0.150193 | 0.283654 |

The level-1 spatial adaptor is not sufficiently language aligned even before
3-D reconstruction.  The level-2 official crop-summary teacher improves mIoU
by 9.81x and localization by 6.56x, so it is the selected semantic supervision
route.  It still falls short of a strong final text system; the next mainline
step is lifting these official summaries into the same canonical primitives,
not introducing a scene-specific text head.  The global learned region bridge
remains level 3 and optional.

Artifacts:

- `output/optimization_20260716/semantic_oracles/siglip2_spatial_teacher_lerf.json`
- `output/optimization_20260716/semantic_oracles/siglip2_crop_summary_teacher_lerf.json`
- `output/optimization_20260716/semantic_oracles/official_crop_summary_lerf/manifest.json`
