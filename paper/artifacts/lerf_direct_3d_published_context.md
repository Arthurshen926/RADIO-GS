# LERF Direct 3D Published Context

This table positions the local CTF-GS result against the OpenGaFF-aligned
published context for primitive-/instance-level open-vocabulary 3DGS papers. It
is not a same-evaluator leaderboard: external rows are published context values,
while CTF-GS rows are local OpenGaussian-style query-select-render evaluations.
CAGS is intentionally excluded from the comparison set for the current
submission route.

| Method | Source type | mIoU (%) | Acc@0.25 (%) | Paper use |
|---|---|---:|---:|---|
| OpenGaussian | official paper | 38.36 | 51.43 | main official-source anchor |
| Dr. Splat | published context | 43.29 | 64.30 | related-work context |
| InstanceGaussian | published context | 45.30 | 58.44 | related-work context |
| OpenGaFF | arXiv context | 54.36 | 80.84 | recent arXiv context only |
| CTF-GS + VPR | local SigLIP2 fixed `thr0p25` + RGB snap | 48.01 | 67.60 | local strict primitive row |
| CTF-GS + direct field + SAM3 box | local fixed global threshold, frozen official SAM3 readout | 57.05 | 68.35 | local strict boundary-readout row |

Conclusion: the fixed direct-field SAM3-box row exceeds the OpenGaFF published
macro mIoU context value (57.05 vs. 54.36) but remains below OpenGaFF on
Acc@0.25 (68.35 vs. 80.84). The correct claim is therefore mIoU-competitive or
stronger under an OpenGaFF-aligned published context, not universal direct-3D
dominance.
