# LERF Direct 3D Published Context

This table positions the local CTF-GS result against the published context for
primitive-/instance-level open-vocabulary 3DGS papers. It is not a
same-evaluator leaderboard: external rows are published context values, while
CTF-GS rows are local OpenGaussian-style query-select-render evaluations. The
row set follows the recent OpenGaFF comparison set, but the OpenGaFF method row
itself is intentionally omitted for the current submission route. CAGS is also
excluded from the main comparison set.

| Method | Source type | mIoU (%) | Acc@0.25 (%) | Paper use |
|---|---|---:|---:|---|
| OpenGaussian | official paper | 38.36 | 51.43 | main official-source anchor |
| Dr. Splat | published context | 43.29 | 64.30 | related-work context |
| InstanceGaussian | published context | 45.30 | 58.44 | related-work context |
| CTF-GS + VPR | local SigLIP2 fixed `thr0p25` + RGB snap | 48.01 | 67.60 | local strict primitive row |
| CTF-GS + direct field + SAM3 box | local fixed global threshold, frozen official SAM3 readout | 57.05 | 68.35 | local strict boundary-readout row |

Conclusion: the fixed direct-field SAM3-box row is stronger than the listed
published baseline rows on mIoU and competitive on Acc@0.25, but the external
rows are not local same-evaluator reruns. The correct claim is therefore strong
published-context direct-3D evidence, not universal direct-3D dominance.
