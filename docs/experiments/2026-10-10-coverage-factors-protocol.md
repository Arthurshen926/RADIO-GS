# Joint opacity function migration, independent of benchmark queries

Freeze before first real-scene validation. Same four LERF fields and 72 training
views. No benchmark query, teacher detection, evaluation label, cached query
prediction or fitting answer is a compiler input. Construction cameras and
existing masks are permitted offline. Default member decoder stays unchanged.

Compile 16x16 deterministic cell-centre samples per view, including empty/unknown
locations (no mask-dependent selection). Store depth-ordered Gaussian incidences,
unclipped footprint alpha and existing constraint-atom IDs in the SAME field.pt.
No pixel coordinates, image dimensions, cameras, RGB or source raster is exported.
This is a source-derived sampled geometric factor graph, NOT full-raster coverage
or a source-independent geometric prior. It is an experimental uncompressed
functional reference, not an efficiency contribution or a finished decoder.

Crucial: enumerate the full tile intersection list, not the old visible hit list.
Retain hidden Gaussian footprints. Reapply member-scaled alpha cutoff1/255,
alpha ceiling.999 and exclusive termination at transmittance1e-4 to match gsplat.
Observe the same field.observed domain as relaxed_subset_alpha.

For EVERY view, compare compiled factor outputs to new renderer calls on seven
query-independent conditions: empty, full, seed20261010+view Bernoulli.15 and.5,
back-depth half, x-coordinate half, continuous uniform members. Compare at the
fixed256 construction samples. Frozen maximum alpha absolute error1e-4; actual
silhouette decisions at10.5/255 must match exactly. Compare all Gaussian gradients
of mean sampled alpha for the continuous condition with atol2e-4. No tolerance
adaptation from observed results. Validate atom binding from the full incidence
partition and preserve every parent field tensor exactly.

Fresh-process reload replays all conditions without opening observations or any
source sidecar. Fixtures hold reference outputs and membership-generation seeds,
not inputs used by the field operator; no sampled pixel IDs needed at runtime.
Python file-read guard is an audit aid, not an OS security sandbox.

Report pass/fail, full/hidden/soft cases, storage, runtime and limitations. A pass
means joint opacity function at sampled construction factors transferred. It does
NOT mean region acceptance, member decoder, held-out view generalization, full
five benchmarks or v10 performance recovered. Next task is shared member decoding
trained on independent construction conditions, no per-query source fitting.
