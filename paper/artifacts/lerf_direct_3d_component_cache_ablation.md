# LERF-OVS Direct 3D Object Selection

Protocol: OpenGaussian-style direct 3D primitive selection with View-to-Primitive Registration (VPR). Query scores are computed on Gaussian primitives from rendered-view SigLIP2 features registered back to 3D with depth/alpha visibility checks (all_poses, max_frames=128, scoring=softmax_scene; GT-free voxel_max context aggregation is applied (res=80, blend=0.5); GT-free selection floor=0.005, selection cap=0.0175); selected primitives are rendered only for mask evaluation.

Input root: `output/radio_gs/lerf_direct_3d_selection_max128_cap0175_cache_softmax_20260514`
Paper-facing fixed selection: `meanstd2p5`. The complete ratio sweep below is diagnostic and should be reported separately from rendered-view grounding.

## Selector Sweep

| Selection | Figurines mIoU | Ramen mIoU | Teatime mIoU | Waldo mIoU | Macro mIoU | Macro Acc@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| meanstd2p5 | 0.4834 | 0.4681 | 0.5012 | 0.2379 | 0.4226 | 0.6864 |

## Paper-Facing Direct-Selection Context

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| CTF-GS | SigLIP2 | fixed meanstd2p5 mIoU | 0.4834 | 0.4681 | 0.5012 | 0.2379 | 0.4226 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene mIoU | 0.4834 | 0.4681 | 0.5012 | 0.2379 | 0.4226 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| CTF-GS | SigLIP2 | fixed meanstd2p5 Acc@0.25 | 0.8214 | 0.7183 | 0.7966 | 0.4091 | 0.6864 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene Acc@0.25 | 0.8214 | 0.7183 | 0.7966 | 0.4091 | 0.6864 |

Interpretation: the registration readout substantially closes the primitive-level gap versus the original Gaussian-center readout while keeping the OpenGaussian-style query-select-render-evaluate protocol. GT-free voxel context aggregation further improves fixed-ratio direct selection by reducing primitive-level fragmentation, though Waldo Kitchen remains the weakest scene and should be discussed as a remaining object-fragmentation/registration-coverage limitation.

## Direct-Readout Diagnostics

These variants do not use GT masks for scoring. They test whether the direct-selection gap is caused by raw Gaussian-center readout, VPR scoring, VFA, view coverage, visibility checks, or GT-free spatial aggregation.

| Variant | Best fixed selection | Fixed macro mIoU | Fixed macro Acc@0.25 | Best-by-scene macro mIoU | Best-by-scene macro Acc@0.25 |
|---|---|---:|---:|---:|---:|
| topcc2_mean | meanstd2p5 | 0.4190 | 0.6821 | 0.4190 | 0.6821 |
| seedexpand1_sum | meanstd2p5 | 0.3386 | 0.5876 | 0.3386 | 0.5876 |

Diagnostic takeaway: VPR is the main factor that improves direct 3D object selection. View coverage, VFA, and GT-free voxel context control the precision/coverage tradeoff, while Waldo Kitchen remains the hardest fragmented scene.

