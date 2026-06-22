# LERF-OVS Direct 3D Object Selection

Protocol: OpenGaussian-style direct 3D primitive selection with View-to-Primitive Registration (VPR). Query scores are computed on Gaussian primitives from rendered-view SigLIP2 features registered back to 3D with depth/alpha visibility checks (all_poses, max_frames=128, scoring=softmax_scene; GT-free selection floor=0.005, selection cap=0.018; GT-free rgb_grabcut mask refinement); selected primitives are rendered only for mask evaluation.

Input root: `output/radio_gs/lerf_direct_3d_selection_threshold_grabcut_20260515`
Paper-facing fixed selection: `thr0p25`. The complete selector sweep below is diagnostic and should be reported separately from rendered-view grounding.

## Selector Sweep

| Selection | Figurines mIoU | Ramen mIoU | Teatime mIoU | Waldo mIoU | Macro mIoU | Macro Acc@0.25 |
|---|---:|---:|---:|---:|---:|---:|
| thr0p1 | 0.4652 | 0.5147 | 0.5572 | 0.2263 | 0.4408 | 0.6534 |
| thr0p12 | 0.4797 | 0.5219 | 0.5610 | 0.2277 | 0.4476 | 0.6579 |
| thr0p15 | 0.4889 | 0.5435 | 0.5599 | 0.2261 | 0.4546 | 0.6684 |
| thr0p18 | 0.5076 | 0.5804 | 0.5587 | 0.2281 | 0.4687 | 0.6644 |
| thr0p2 | 0.5222 | 0.5822 | 0.5608 | 0.2330 | 0.4746 | 0.6644 |
| thr0p25 | 0.5309 | 0.5805 | 0.5662 | 0.2429 | 0.4801 | 0.6760 |
| thr0p3 | 0.5324 | 0.5813 | 0.5474 | 0.2463 | 0.4769 | 0.6640 |
| thr0p35 | 0.5326 | 0.5777 | 0.5358 | 0.2323 | 0.4696 | 0.6553 |
| thr0p4 | 0.5327 | 0.5712 | 0.4970 | 0.2278 | 0.4572 | 0.6426 |

## Paper-Facing Direct-Selection Context

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| GaussFM | SigLIP2 | fixed thr0p25 mIoU | 0.5309 | 0.5805 | 0.5662 | 0.2429 | 0.4801 |
| GaussFM | SigLIP2 | diagnostic best-by-scene mIoU | 0.5327 | 0.5822 | 0.5662 | 0.2463 | 0.4819 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| GaussFM | SigLIP2 | fixed thr0p25 Acc@0.25 | 0.7857 | 0.7465 | 0.7627 | 0.4091 | 0.6760 |
| GaussFM | SigLIP2 | diagnostic best-by-scene Acc@0.25 | 0.7679 | 0.7465 | 0.7627 | 0.4091 | 0.6715 |

Interpretation: the registration readout substantially closes the primitive-level gap versus the original Gaussian-center readout while keeping the OpenGaussian-style query-select-render-evaluate protocol. The promoted fixed-threshold selector reduces primitive-level clutter under the same global rule; Waldo Kitchen remains the weakest scene and should be discussed as a remaining object-fragmentation/registration-coverage limitation.

## Direct-Readout Diagnostics

These variants do not use GT masks for scoring. They test whether the direct-selection gap is caused by raw Gaussian-center readout, VPR scoring, VFA, view coverage, visibility checks, or GT-free spatial aggregation.

| Variant | Best fixed selection | Fixed macro mIoU | Fixed macro Acc@0.25 | Best-by-scene macro mIoU | Best-by-scene macro Acc@0.25 |
|---|---|---:|---:|---:|---:|
| meanstd_current | meanstd2 | 0.4461 | 0.6611 | 0.4525 | 0.6682 |

Diagnostic takeaway: VPR is the main factor that improves direct 3D object selection. View coverage, VFA, selection calibration, and optional GT-free projection cleanup control the precision/coverage tradeoff, while Waldo Kitchen remains the hardest fragmented scene.

