# LERF Direct3D Support-Policy Ablation Qualitative

Qualitative ablation for the compact direct-field support policy. The base column uses the prior compact direct mask; the final column uses the current prompt-ensemble + component-support policy. No VPR cache or official RGB SAM3 readout is used by either row.

Figure: `paper/figures/lerf_direct3d_support_policy_ablation_qualitative.png`

| Scene | Frame | Query | Base IoU | Support-policy IoU | Delta |
| --- | --- | --- | ---: | ---: | ---: |
| ramen | 00006 | nori | 0.6614 | 0.8227 | +0.1613 |
| ramen | 00081 | bowl | 0.4408 | 0.6665 | +0.2257 |
| teatime | 00025 | plate | 0.5192 | 0.6152 | +0.0960 |
| waldo_kitchen | 00140 | dark cup | 0.7229 | 0.8073 | +0.0845 |
