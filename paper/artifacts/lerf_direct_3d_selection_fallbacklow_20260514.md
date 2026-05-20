# LERF-OVS Direct 3D Object Selection

Protocol: OpenGaussian-style direct 3D primitive selection with View-to-Primitive Registration (VPR). Query scores are computed on Gaussian primitives from rendered-view SigLIP2 features registered back to 3D with depth/alpha visibility checks (all_poses, max_frames=96, scoring=softmax_scene; GT-free voxel_max context aggregation is applied (res=80, blend=0.5); GT-free selection floor=0.005, selection cap=0.02); selected primitives are rendered only for mask evaluation.

Input root: `output/radio_gs/lerf_direct_3d_selection_fallbacklow_20260514`
Paper-facing fixed selection: `meanstd2p5`. The complete ratio sweep below is diagnostic and should be reported separately from rendered-view grounding.

## Selector Sweep

| Selection | Figurines mIoU | Ramen mIoU | Teatime mIoU | Waldo mIoU | Macro mIoU | Macro Acc@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| meanstd2 | 0.4436 | 0.2683 | 0.2292 | 0.1356 | 0.2692 | 0.4229 |
| meanstd2p5 | 0.4436 | 0.2683 | 0.2292 | 0.1356 | 0.2692 | 0.4229 |
| meanstd3 | 0.4436 | 0.2683 | 0.2292 | 0.1356 | 0.2692 | 0.4229 |

## Paper-Facing Direct-Selection Context

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| CTF-GS | SigLIP2 | fixed meanstd2p5 mIoU | 0.4436 | 0.2683 | 0.2292 | 0.1356 | 0.2692 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene mIoU | 0.4436 | 0.2683 | 0.2292 | 0.1356 | 0.2692 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| CTF-GS | SigLIP2 | fixed meanstd2p5 Acc@0.25 | 0.7143 | 0.4225 | 0.3729 | 0.1818 | 0.4229 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene Acc@0.25 | 0.7143 | 0.4225 | 0.3729 | 0.1818 | 0.4229 |

Interpretation: the registration readout substantially closes the primitive-level gap versus the original Gaussian-center readout while keeping the OpenGaussian-style query-select-render-evaluate protocol. GT-free voxel context aggregation further improves fixed-ratio direct selection by reducing primitive-level fragmentation, though Waldo Kitchen remains the weakest scene and should be discussed as a remaining object-fragmentation/registration-coverage limitation.

