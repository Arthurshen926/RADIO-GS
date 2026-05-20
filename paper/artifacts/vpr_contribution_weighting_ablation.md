# Contribution-Weighted VPR Ablation

Protocol: registered_view + softmax_scene + 96 all poses + voxel_max(res=80, blend=0.50) + fixed top0p02. The only change is registration_weight_mode.

| Variant | Figurines | Ramen | Teatime | Waldo | Macro mIoU | Macro Acc@0.25 | Conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| uniform main | 0.4055 | 0.4491 | 0.4862 | 0.1991 | 0.3850 | 0.6428 | promoted mainline |
| alpha weighted | 0.1973 | 0.3841 | 0.4244 | 0.1853 | 0.2978 | 0.5389 | not promoted; hurts small-object and macro mIoU |
| alpha_depth weighted | 0.1979 | 0.3825 | 0.4239 | 0.1827 | 0.2967 | 0.5345 | not promoted; hurts small-object and macro mIoU |

Interpretation: the Dr. Splat-inspired alpha/alpha-depth weighting option is implemented and tested, but it is not effective under the current center-sampling VPR approximation. It lowers registered coverage for small objects and reduces macro mIoU. The paper should keep uniform VPR as the main readout and mention contribution-weighted registration as a negative ablation/future route requiring true rasterization-contribution assignment rather than center-sampled alpha weighting.
