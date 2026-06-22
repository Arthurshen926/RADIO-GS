# LERF Per-Gaussian 1280-D Explicit Baseline

Protocol: Per-Gaussian 1280-D explicit RADIO memory. Cached frame-wise RADIO feature maps are registered to visible Gaussian centers, stored as fp16 1280-D vectors, rendered back to LERF views, and evaluated with the same frozen SigLIP2 text scorer.

| Scene | LocAcc | mIoU | N | Registered | Fraction | Storage MiB |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.6429 | 0.2729 | 56 | 18961/168791 | 0.1123 | 412.1 |
| Ramen | 0.6056 | 0.4333 | 71 | 108491/382687 | 0.2835 | 934.3 |
| Teatime | 0.5085 | 0.3541 | 59 | 121795/460157 | 0.2647 | 1123.4 |
| Waldo Kitchen | 0.5000 | 0.2125 | 22 | 101996/691728 | 0.1475 | 1688.8 |

## Aggregate

| Aggregate | LocAcc | mIoU | Registered fraction | Storage MiB |
|---|---:|---:|---:|---:|
| Macro | 0.5642 | 0.3182 | 0.2020 | 1039.7 |
| Query-weighted | 0.5769 | 0.3443 | -- | -- |

## Interpretation

- This row is not compact: it stores fp16 1280-D RADIO features per Gaussian.
- It is a 3D scene-memory baseline because features are attached to Gaussian primitives and rendered to novel views.
- Invalid or never-visible Gaussians receive zero features; registered fraction is reported per scene.
