# LERF Direct 3D Published Context

This table positions the local CTF-GS + VPR result against recent
primitive-/instance-level open-vocabulary 3DGS papers. It is not a
same-evaluator leaderboard: external rows are published context values, while
CTF-GS is the local fixed mean+2.5std+floor0.005+cap0.018 OpenGaussian-style VPR row.

| Method | Source type | mIoU (%) | Acc@0.25 (%) | Paper use |
|---|---|---:|---:|---|
| OpenGaussian | official paper | 38.36 | 51.43 | main official-source anchor |
| Dr. Splat | published context | 43.29 | 64.30 | related-work context |
| CAGS | published context | 50.79 | 69.62 | related-work context |
| InstanceGaussian | published context | 45.30 | 58.44 | related-work context |
| OpenGaFF | arXiv context | 54.36 | 80.84 | recent arXiv context only |
| CTF-GS + VPR | local SigLIP2 mean+2.5std+floor0.005+cap0.018 | 42.27 | 69.06 | local result |

Conclusion: CTF-GS + VPR slightly exceeds the OpenGaussian official macro mIoU
reference and strongly improves Acc@0.25 relative to OpenGaussian, but it is not
a global direct-3D SOTA claim against newer instance-/context-aware methods
without local same-evaluator reruns.
