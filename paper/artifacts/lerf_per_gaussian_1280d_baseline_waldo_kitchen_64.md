# LERF Per-Gaussian 1280-D Explicit Baseline

Protocol: Per-Gaussian 1280-D explicit RADIO memory. Cached RADIO teacher feature maps are registered to visible Gaussian centers, stored as fp16 1280-D vectors, rendered back to LERF views, and evaluated with the same frozen SigLIP2 text scorer.

| Scene | LocAcc | mIoU | N | Registered | Fraction | Storage MiB |
|---|---:|---:|---:|---:|---:|---:|
| Waldo Kitchen | 0.5000 | 0.2125 | 22 | 101996/691728 | 0.1475 | 1688.8 |

## Aggregate

| Aggregate | LocAcc | mIoU | Registered fraction | Storage MiB |
|---|---:|---:|---:|---:|
| Macro | 0.5000 | 0.2125 | 0.1475 | 1688.8 |
| Query-weighted | 0.5000 | 0.2125 | -- | -- |

## Interpretation

- This row is not compact: it stores fp16 1280-D RADIO features per Gaussian.
- It is a 3D scene-memory baseline because features are attached to Gaussian primitives and rendered to novel views.
- Invalid or never-visible Gaussians receive zero features; registered fraction is reported per scene.
