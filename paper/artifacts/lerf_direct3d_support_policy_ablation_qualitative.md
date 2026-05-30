# LERF Direct3D Support-Policy Ablation Qualitative

Qualitative ablation for the compact direct-field support policy. The base column uses the prior compact direct mask; the final column uses the current prompt-ensemble + component-support policy. No VPR cache or official RGB SAM3 readout is used by either row.

Figure: `paper/figures/lerf_direct3d_support_policy_ablation_qualitative.png`

| Scene | Frame | Query | Base IoU | Support-policy IoU | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| waldo_kitchen | 00053 | knife | 0.1400 | 0.4135 | +0.2736 |
| waldo_kitchen | 00140 | spoon | 0.0680 | 0.3373 | +0.2693 |
| ramen | 00024 | wavy noodles | 0.0000 | 0.4981 | +0.4981 |
| teatime | 00140 | plate | 0.3524 | 0.5072 | +0.1549 |
