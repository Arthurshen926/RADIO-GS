# Direct native-coordinate capacity control

This additional complete-cohort control addresses an a priori limitation of the
shared affine residual: it cannot separately correct spatially different
projection mixtures. No protected-native benchmark scores were read in choosing
this control. It does not change the earlier three-arm protocol or its results.

Warm-start every scene from `protected_relations_v1/protected_training`, never
from a scene- or task-selected winner. Use all 30 fields and the same five
benchmark cohorts, all four UQIS inputs, same v2 query program and standard-mode
source-free inference. Geometry, grouping, shared acceptance parameters, observed
and anchor support remain exact. Only the existing full-dimensional Gaussian
semantic and local coordinates can change; no compression, codebook, source bank,
new teacher or extra 3D label is introduced.

Use the same FunctionalSource regions, teacher pseudoqueries, 64 sampled pixel
pairs, seed 20261009, two complete epochs, preference temperature .1, directional
trust coefficient 1 and equal channel weights as the shared-residual control.
Optimize full native coordinates directly. Sparse row-local Adam uses .001,
betas (.9,.999), epsilon 1e-8, no weight decay and gradient norm clip 1. Each row
has its own update count; absent rows do not move from stale momentum. These
optimizer/parameterization differences are explicit: this is a capacity and
optimization control, not an isolated parameter-count causal estimate. Report
row exposure, including never-updated observed rows, rather than claiming that
complete region exposure guarantees complete Gaussian optimization.

Coordinates and full-precision moments live on CPU and active rows execute the
objective on GPU. This changes working memory placement, not model dimensions.
Keep native coordinates in FP32 during optimization and bake normalized observed
rows into the original export dtype. Source development follows the same fixed
post-training audit (seed 20261010); it never selects checkpoints. Training,
buffer/parameter protection and independent copied-export replay are verified.

Construction of all 30 exports precedes prediction; all predictions precede
scoring. Compare against the protected field and shared-residual field on every
task and scene. Retain failures, source loss, protected point-output invariance,
and all original v10 contract limitations. Single-vector coordinates, even when
fully optimized, are not automatically a sufficient multimodal capacity upper
bound. No default promotion, per-task winner selection or claim of complete
performance recovery is authorized by this experiment's design alone.
