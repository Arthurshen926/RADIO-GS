# LERF Direct3D Proposal-Memory Ablation

Proposal-memory score smoothing is retained as an ablation only for LERF direct 3D: the uniform and gated variants did not beat the existing fixed-threshold SAM3-box row on figurines.

| Row | mIoU | Acc@0.25 | Boundary-F | Proposal |
|---|---:|---:|---:|---|
| direct3d_baseline_figurines_thr0p25 | 0.6136 | 0.6964 | 0.7116 | none |
| direct3d_propmem_all_figurines_thr0p25 | 0.5797 | 0.6607 | 0.6715 | voxel alpha=0.4 voxel=0.08 gate= |
| direct3d_propmem_gated_figurines_thr0p25 | 0.6013 | 0.6786 | 0.6957 | voxel alpha=0.5 voxel=0.04 gate=low_margin |
