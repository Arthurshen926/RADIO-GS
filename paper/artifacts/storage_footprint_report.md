# Storage Footprint Report

This report compares direct per-Gaussian 1280-D fp16 teacher-feature storage against the stored compact CTF-GS checkpoint footprint. The compact footprint includes the Gaussian feature-field state dict, CTR/HCD codec, and VFA/screen refiner tensors. VPR is an inference-time readout; the registered SigLIP2 primitive cache and per-query voxel-score cache are reported separately because they are not persistent checkpoint state.

| Scene | #Gaussians | Direct 1280-D fp16 | Compact ckpt | Saving | Optional VPR emb. cache | Voxel score cache | Compact + optional VPR | Saving w/ optional VPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Figurines | 168,791 | 412.1 MiB | 237.0 MiB | 1.74x | 494.5 MiB | 6.8 MiB | 738.2 MiB | 0.56x |
| Ramen | 382,687 | 934.3 MiB | 311.2 MiB | 3.00x | 1121.2 MiB | 10.2 MiB | 1442.6 MiB | 0.65x |
| Teatime | 460,157 | 1123.4 MiB | 338.1 MiB | 3.32x | 1348.1 MiB | 12.3 MiB | 1698.5 MiB | 0.66x |
| Waldo Kitchen | 691,728 | 1688.8 MiB | 418.5 MiB | 4.04x | 2026.5 MiB | 23.7 MiB | 2468.8 MiB | 0.68x |

## Notes

- Direct storage assumes storing only the 1280-D teacher feature as fp16 per Gaussian; it excludes ordinary 3DGS RGB/geometry attributes.
- Compact total is conservative because it counts the whole feature-field model state dict, including existing 3DGS geometry/RGB tensors carried inside the checkpoint, plus decoder and refiner tensors.
- VPR does not add persistent trained parameters. If one persists the registered 1536-D SigLIP2 primitive embeddings and scene-query voxel scores instead of streaming them during evaluation, that optional cache is larger than direct 1280-D feature storage on these scenes; the paper therefore separates stored compact checkpoint footprint from optional VPR inference cache footprint.
- The paper should describe this as a footprint accounting table rather than a pure per-primitive compression ratio.

## Sources

- Figurines: `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/latest.pth`
- Ramen: `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth`
- Teatime: `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth`
- Waldo Kitchen: `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth`
