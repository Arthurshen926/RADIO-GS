# LERF-OVS Direct 3D Object Selection

Protocol: OpenGaussian-style direct 3D primitive selection with View-to-Primitive Registration (VPR). Query scores are computed on Gaussian primitives from rendered-view SigLIP2 features registered back to 3D with depth/alpha visibility checks (all_poses, max_frames=128, scoring=softmax_scene; GT-free voxel_max context aggregation is applied (res=80, blend=0.5); GT-free selection floor=0.005, selection cap=0.02); selected primitives are rendered only for mask evaluation.

Input root: `output/radio_gs/lerf_direct_3d_selection_max128_finemstd_20260514`
Paper-facing fixed selection: `meanstd2p5`. The complete ratio sweep below is diagnostic and should be reported separately from rendered-view grounding.

## Selector Sweep

| Selection | Figurines mIoU | Ramen mIoU | Teatime mIoU | Waldo mIoU | Macro mIoU | Macro Acc@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| meanstd2p2 | 0.4687 | 0.4545 | 0.5202 | 0.2146 | 0.4145 | 0.6810 |
| meanstd2p3 | 0.4738 | 0.4546 | 0.5208 | 0.2208 | 0.4175 | 0.6854 |
| meanstd2p4 | 0.4755 | 0.4555 | 0.5198 | 0.2206 | 0.4178 | 0.6899 |
| meanstd2p5 | 0.4818 | 0.4555 | 0.5150 | 0.2217 | 0.4185 | 0.6899 |
| meanstd2p6 | 0.4848 | 0.4552 | 0.5082 | 0.2222 | 0.4176 | 0.6769 |
| meanstd2p7 | 0.4797 | 0.4549 | 0.5010 | 0.2221 | 0.4144 | 0.6769 |
| meanstd2p8 | 0.4770 | 0.4550 | 0.4975 | 0.2231 | 0.4132 | 0.6769 |

## Paper-Facing Direct-Selection Context

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| CTF-GS | SigLIP2 | fixed meanstd2p5 mIoU | 0.4818 | 0.4555 | 0.5150 | 0.2217 | 0.4185 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene mIoU | 0.4848 | 0.4555 | 0.5208 | 0.2231 | 0.4210 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| CTF-GS | SigLIP2 | fixed meanstd2p5 Acc@0.25 | 0.8214 | 0.7324 | 0.7966 | 0.4091 | 0.6899 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene Acc@0.25 | 0.8036 | 0.7324 | 0.7966 | 0.4091 | 0.6854 |

Interpretation: the registration readout substantially closes the primitive-level gap versus the original Gaussian-center readout while keeping the OpenGaussian-style query-select-render-evaluate protocol. GT-free voxel context aggregation further improves fixed-ratio direct selection by reducing primitive-level fragmentation, though Waldo Kitchen remains the weakest scene and should be discussed as a remaining object-fragmentation/registration-coverage limitation.

