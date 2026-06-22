# Storage Footprint Report

This report separates three storage accounting levels so that the compact feature-memory claim is not hidden by carried 3DGS geometry/RGB tensors. Direct storage assumes one 1280-D fp16 RADIO reference feature per Gaussian. Latent payload counts only the stored compact per-Gaussian semantic code. Feature-memory package counts the latent payload plus global field heads, CTR/HCD codec, and VFA/refiner tensors. Full checkpoint is a conservative deployable accounting that also includes ordinary 3DGS geometry/RGB and appearance tensors carried inside the model state dict. VPR caches are reported separately because they are optional inference artifacts, not persistent trained state.

| Scene | #Gaussians | Direct 1280-D fp16 | Latent payload | Latent saving | Feature-memory package | Package saving | Full checkpoint | Full saving | Optional VPR emb. cache | Voxel score cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Figurines | 168,791 | 412.1 MiB | 20.6 MiB | 20.00x | 199.0 MiB | 2.07x | 237.0 MiB | 1.74x | 494.5 MiB | 6.8 MiB |
| Ramen | 382,687 | 934.3 MiB | 46.7 MiB | 20.00x | 225.1 MiB | 4.15x | 311.2 MiB | 3.00x | 1121.2 MiB | 10.2 MiB |
| Teatime | 460,157 | 1123.4 MiB | 56.2 MiB | 20.00x | 234.5 MiB | 4.79x | 338.1 MiB | 3.32x | 1348.1 MiB | 12.3 MiB |
| Waldo Kitchen | 691,728 | 1688.8 MiB | 84.4 MiB | 20.00x | 262.8 MiB | 6.43x | 418.5 MiB | 4.04x | 2026.5 MiB | 23.7 MiB |

## Notes

- Direct storage assumes storing only the 1280-D RADIO reference feature as fp16 per Gaussian; it excludes ordinary 3DGS RGB/geometry attributes.
- The latent payload is the clean per-Gaussian semantic storage number. It is approximately the 64-D compact code and gives the expected 20x reduction relative to 1280-D fp16 RADIO reference features.
- The feature-memory package adds scene-global heads and decoders. Those fixed tensors do not scale with the number of Gaussians, so their overhead is amortized more strongly on larger indoor/outdoor scenes; the per-scene growing term remains the compact latent payload.
- Full checkpoint is intentionally conservative because it counts the whole feature-field model state dict, including existing 3DGS geometry/RGB tensors carried inside the checkpoint, plus decoder and refiner tensors. Use it for deployable footprint, not for pure feature-memory compression.
- VPR does not add persistent trained parameters. If one persists the registered 1536-D SigLIP2 primitive embeddings and scene-query voxel scores instead of streaming them during evaluation, that optional cache is larger than direct 1280-D feature storage on these scenes; the paper therefore separates stored compact checkpoint footprint from optional VPR inference cache footprint.
- The paper should report latent/package/full-checkpoint columns rather than only the full checkpoint; otherwise the compact feature-memory advantage is underestimated.

## Sources

- Figurines: `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/latest.pth`
- Ramen: `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth`
- Teatime: `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth`
- Waldo Kitchen: `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth`
