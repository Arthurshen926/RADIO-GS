# LERF Per-Gaussian 1280-D Explicit Baseline

Protocol: Per-Gaussian 1280-D explicit RADIO memory. Cached RADIO teacher feature maps are registered to visible Gaussian centers, stored as fp16 1280-D vectors, rendered back to LERF views, and evaluated with the same frozen SigLIP2 text scorer.

| Scene | LocAcc | mIoU | N | Registered | Fraction | Storage MiB |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.6429 | 0.2729 | 56 | 18961/168791 | 0.1123 | 412.1 |

## Aggregate

| Aggregate | LocAcc | mIoU | Registered fraction | Storage MiB |
|---|---:|---:|---:|---:|
| Macro | 0.6429 | 0.2729 | 0.1123 | 412.1 |
| Query-weighted | 0.6429 | 0.2729 | -- | -- |

## Interpretation

- This row is not compact: it stores fp16 1280-D RADIO features per Gaussian.
- It is a 3D scene-memory baseline because features are attached to Gaussian primitives and rendered to novel views.
- Invalid or never-visible Gaussians receive zero features; registered fraction is reported per scene.
