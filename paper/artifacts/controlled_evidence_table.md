# Controlled Evidence Table

This table consolidates existing frozen-protocol evidence without inventing unmeasured rows. `not evaluated` means that the artifact set does not contain that measurement for the variant.

| Method | Compact | 3D memory | Novel-view feature | Direct 3D query | LERF LocAcc | LERF mIoU | Direct 3D mIoU/Acc@0.25 | Storage | Runtime | Source |
|---|---|---|---|---|---:|---:|---|---|---|---|
| Frame-wise RADIO | no | no | no | no | 0.7985 | 0.4634 | not applicable | per-frame feature cache | not profiled here | paper/radio_gs_draft.tex frame-wise-RADIO-vs-rendered table |
| Nearest-view RADIO cache | no | no | cache-only | no | 0.2722 | 0.1545 | not applicable | per-frame feature cache | mean nearest distance 0.4582 | lerf_nearest_view_cache_baseline.json |
| Per-Gaussian 1280-D RADIO memory | no | yes | yes | partial | 0.5642 | 0.3182 | not evaluated | 1039.7 MiB mean fp16 feature storage | registered fraction 0.2020 | lerf_per_gaussian_1280d_baseline.json |
| Full GaussFM | yes | yes | yes | yes | 0.8598 | 0.5707 | pure compact 0.4570/0.6851; score-guard compact 0.5014/0.7044; SAM3-box diag. 0.5705/0.6835 | 3.03x mean compact checkpoint saving | 31.2s mean LERF overlay | peak-component LERF artifact + direct-3D registry |
| w/o FDH warm-start | yes | yes | yes | not evaluated | 0.8018 | 0.4236 | not evaluated | not separately measured | not separately profiled | lerf_component_ablation.json |
| w/o refiner | yes | yes | yes | not evaluated | 0.8401 | 0.4796 | not evaluated | not separately measured | not separately profiled | lerf_component_ablation.json |
| w/o hybrid | yes | yes | yes | not evaluated | 0.8394 | 0.5069 | not evaluated | not separately measured | not separately profiled | lerf_component_ablation.json |
| w/o HCD | yes | yes | yes | not evaluated | 0.5306 | 0.2596 | not evaluated | not separately measured | not separately profiled | lerf_component_ablation.json |
