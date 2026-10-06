# NVOS: source-anchored component inference and structural graph control

This fixed bounded experiment extends the positive spatial-constraint candidate
(68.6122% IoU) on exactly the same frozen maps, cameras and input rights. It tests
whether unanchored components explain residual false positives. It does not
assume that SAM-complement pixels are negatives, and it does not equate lack of
connectivity with a verified semantic exclusion label.

## Registered hypothesis and variants

A prompted instance may be inferred as candidate components connected to explicit
positive anchors. This is a potentially false hypothesis: occluded, fragmented
or multi-part objects may legitimately occupy several disconnected components.
The experiment therefore remains diagnostic unless full-cohort evidence supports
promotion.

- Baseline: unchanged 80-step positive-plus-SAM-positive query-time optimization.
- Spatial component: above-threshold candidate nodes plus explicit anchors,
  connected by spatial edges; retain components with positive anchors.
- Structure component: same operation, but remove edges whose distilled 32D
  structural suffix cosine is below 0.5.
- Structure smooth component: apply the existing `continuous_convex_v2` graph
  solver, then the same structural component selection.

The geometry graph uses 16 neighbors, with edge distance capped at three times
the full observed-domain median nearest-neighbor distance. The structural graph
uses cosine >=0.5. Continuous smoothing uses unary confidence0.5, lambda1 and
64 CG iterations. Positive anchors have >=0.1 of normalized reference rendering
adjoint mass contributed by the authorized positive scribble. No target mask or
RGB participates in these choices.

All component variants finish with the existing 80-step positive-only reference
repair. This can reintroduce removed elements, so **final connectivity is not a
hard guarantee**. Reports separately record filtered and repaired selection size,
reintroduced elements, removed SAM-majority elements and SAM-positive evidence
mass, and reference SAM recall before/after repair. No supported anchor means
abstention to baseline rather than an invented all-negative result.

## Reproducibility and preserved failed attempts

The initial registration was cancelled before any target report or GT scoring to
add the requested source-side diagnostics. Graph parameters did not change.
The next attempt failed closed before GT scoring because exact floating-point
cold replay of an 80-step GPU optimization was too strict: flower maximum score
difference was2.98e-7, mean6.49e-9, with zero binary-mask changes.

The final v3 preserves both attempts and all algorithm parameters. It requires
maximum floating error <=1e-6, exact prediction-grid binary masks, and exact
native scoring-grid binary masks. It reports actual floating errors instead of
claiming bitwise optimizer repeatability. All four predictions are persisted
before target GT becomes readable. Reference negative scribbles, target RGB,
and training supervision feature reads remain denied by the evaluator audit hook.

Authority: `output/nvos_system_refinement_20260919_v3/registration.json`.

## Complete eight-task results

| Task | Baseline | Spatial component | Structural component | Smooth structural component |
|---|---:|---:|---:|---:|
| fern | 73.98% | 65.62% | 65.38% | 65.31% |
| flower | 83.03% | 82.71% | 74.19% | 73.53% |
| fortress | 93.96% | 93.22% | 93.51% | 93.60% |
| horns_center | 68.45% | 91.77% | 89.10% | 89.62% |
| horns_left | 60.14% | 89.07% | 89.54% | 89.76% |
| leaves | 23.74% | 69.38% | 69.43% | 69.45% |
| orchids | 82.14% | 65.12% | 65.13% | 65.34% |
| trex | 63.47% | 78.20% | 77.88% | 77.81% |
| **Macro** | **68.61%** | **79.39%** | **78.02%** | **78.05%** |

Macro AP: 86.99% / 90.11% / 89.64% / 88.83%.

The spatial arm improves mean IoU by **10.7742 percentage points**, with four
task gains and four regressions. Its largest regression is orchids−17.02points,
then fern−8.36points. No per-task winner is substituted. This is a useful opt-in
candidate, not an unconditional default replacement. Learned structural edges
do not improve the aggregate beyond the purely spatial arm in this control.

All eight baseline IoU and pixel-accuracy values exactly match the earlier
68.6122% candidate. Prediction and scoring-grid masks replay exactly. Maximum
continuous cold-optimization difference across tasks is 3.58e-07.

The complete four-arm eight-task run took 452.6seconds, 
52.7–59.7seconds per task. These include cold loads, 
repeated base optimization, graph construction, repair, persistence and scoring;
they do not measure isolated deployed-query latency.

## Source-side risk distribution for the spatial arm

These are reference teacher/geometry consistency diagnostics, not target labels.

| Task | Removed SAM-positive mass | Remaining reference SAM recall | Reintroduced elements after repair |
|---|---:|---:|---:|
| flower | 0.010% | 99.956% | 0 |
| fortress | 0.018% | 99.776% | 0 |
| fern | 20.598% | 78.437% | 1 |
| horns_center | 0.810% | 97.055% | 0 |
| horns_left | 0.004% | 99.848% | 0 |
| leaves | 0.000% | 100.000% | 0 |
| orchids | 8.203% | 91.287% | 0 |
| trex | 17.236% | 77.543% | 0 |

Every arm keeps the original positive-prompt agreement (all supported positives
satisfied; fern retains its unsupported-pixel failures). Nevertheless, satisfying
sparse positive prompts does not preserve full object extent. Fern and orchids
lose source SAM-positive support and target recall. A final repair reintroduces
one fern element, so final results must not be described as strictly connected.

Conversely, trex loses substantial SAM-positive evidence while target IoU improves.
Therefore a threshold on removed SAM mass alone cannot establish whether removed
regions are erroneous teacher expansion or legitimate object parts. No such gate
is fitted here. This requires independent cross-view identity and range evidence.

Leaves illustrates the main success: positive-only correction could not change
its already-satisfied prompt, while anchoring the spatial components strongly
reduces unrelated selected range. This supports instance-aware spatial inference,
not the universal claim that disconnected components are false objects.

Seal: `paper/artifacts/nvos_system_refinement_eight_20260919.json`.
Source-risk metrics for all three arms are in its per-task reference constraints.

## Opt-in runtime and cold replay

`nvos_system_refinement_20260919_runtime.py` exposes the registered spatial arm
without any target RGB/GT argument. A fresh flower execution through this adapter
reproduces both prediction-grid and final scoring-grid spatial masks exactly,
with continuous max error2.98e-7 and identical IoU82.7129%.

The accompanying cold baseline optimization changes one final bilinear-grid
threshold pixel despite zero prediction-grid binary differences (max floating
error2.38e-7). The combined evaluator correctly fails closed at that strict
check. This failed attempt is retained. It does not invalidate the completed
eight-task authority, where every baseline check passed, but demonstrates that
GPU query optimization is not universally bitwise/binary deterministic at
threshold boundaries. No threshold is adjusted to hide the difference.

The runtime-only comparison uses the sealed scoring shape and saved prediction
arrays, not target GT. Seal: `paper/artifacts/nvos_system_refinement_runtime_20260919.json`.
