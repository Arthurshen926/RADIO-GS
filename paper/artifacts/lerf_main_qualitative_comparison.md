# LERF Main Qualitative Comparison

- Figure: `paper/figures/lerf_main_qualitative_comparison.png`
- Baseline: `Dr. Splat (repro.)`
- Ours: `CTF-GS compact`
- Ours source root: `output/radio_gs/lerf_direct3d_prompt_ensemble_policy_masks_20260528`

| Scene | Frame | Query | Baseline IoU | Ours IoU |
| --- | --- | --- | ---: | ---: |
| figurines | 00041 | `old camera` | 0.0340 | 0.9092 |
| ramen | 00024 | `onion segments` | 0.0549 | 0.7712 |
| teatime | 00140 | `bag of cookies` | 0.3696 | 0.9743 |
| waldo_kitchen | 00140 | `plate` | 0.0768 | 0.6830 |

Protocol note: the Ours panels use compact direct-field primitive scores with a frozen SigLIP2 prompt ensemble, opacity-gated point-adapter blending, and support-aware component cleanup. No VPR cache or official RGB SAM3 readout is used for these panels.
