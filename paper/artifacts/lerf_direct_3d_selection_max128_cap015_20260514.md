# LERF-OVS Direct 3D Object Selection

Protocol: OpenGaussian-style direct 3D primitive selection with View-to-Primitive Registration (VPR). Query scores are computed on Gaussian primitives from rendered-view SigLIP2 features registered back to 3D with depth/alpha visibility checks (all_poses, max_frames=128, scoring=softmax_scene; GT-free voxel_max context aggregation is applied (res=80, blend=0.5); GT-free selection floor=0.005, selection cap=0.015); selected primitives are rendered only for mask evaluation.

Input root: `output/radio_gs/lerf_direct_3d_selection_max128_cap015_20260514`
Paper-facing fixed selection: `meanstd2p5`. The complete ratio sweep below is diagnostic and should be reported separately from rendered-view grounding.

## Selector Sweep

| Selection | Figurines mIoU | Ramen mIoU | Teatime mIoU | Waldo mIoU | Macro mIoU | Macro Acc@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| meanstd2p3 | 0.4755 | 0.4606 | 0.4978 | 0.2317 | 0.4164 | 0.7013 |
| meanstd2p5 | 0.4829 | 0.4615 | 0.4965 | 0.2328 | 0.4184 | 0.7013 |
| meanstd2p7 | 0.4835 | 0.4616 | 0.4900 | 0.2323 | 0.4168 | 0.6884 |

## Paper-Facing Direct-Selection Context

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| CTF-GS | SigLIP2 | fixed meanstd2p5 mIoU | 0.4829 | 0.4615 | 0.4965 | 0.2328 | 0.4184 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene mIoU | 0.4835 | 0.4616 | 0.4978 | 0.2328 | 0.4189 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| CTF-GS | SigLIP2 | fixed meanstd2p5 Acc@0.25 | 0.8214 | 0.7042 | 0.7797 | 0.5000 | 0.7013 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene Acc@0.25 | 0.8036 | 0.7042 | 0.7797 | 0.5000 | 0.6969 |

Interpretation: the registration readout substantially closes the primitive-level gap versus the original Gaussian-center readout while keeping the OpenGaussian-style query-select-render-evaluate protocol. GT-free voxel context aggregation further improves fixed-ratio direct selection by reducing primitive-level fragmentation, though Waldo Kitchen remains the weakest scene and should be discussed as a remaining object-fragmentation/registration-coverage limitation.

