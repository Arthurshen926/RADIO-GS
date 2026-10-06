# Compact selected-set inference with boundary evidence (next96)

2026-10-02. All12scene controls complete. High-resolution evidence plus an
unoccluded-footprint prior is a higher-mean LERF research candidate, not a new
default:61.478% selected projection IoU versus next93 60.613%. Retain next93 as
comparison/rollback and next80 as stable reference. Important identity/recall
regressions remain; no per-query winner assembly.

## Mechanism

The masked subset-silhouette objective does not directly discourage redundant
selected elements. Full-scene visibility is insufficient for this purpose:
primitives hidden by nonmembers can reappear after subset selection. Add an
explicit compact-explanation prior on membership, retaining the source data loss:

L = L_known_source + 0.1 * sum_i(c_i * sigmoid(theta_i)) /
                              sum_i(c_i * initial_membership_i).

Only semantic-observed elements can change. Coefficients are fixed before fitting:
- **count:** confidence-weighted number of active source camera frusta containing
  the Gaussian. This is view-weighted count, not uniform global cardinality.
- **footprint:** sum over those views of confidence * opacity * 2*pi*sqrt(det C2D),
  using the same gsplat projection and source-camera near/far planes as rendering.
  C2D includes the renderer's default2D covariance epsilon. This is unoccluded
  integrated Gaussian alpha, not union area, clipped silhouette area, or physical
  truth. Frustum-edge footprints are not analytically cropped to the viewport.

The regularizer uses smooth membership so annealing does not erase its gradient.
Normalize by the source initializer's cost (therefore1 at initialization), not GT
area or a benchmark-calibrated target. Use weight0.1 for every query/scene/arm; no
coefficient sweep. Source masks/unknown permissions and confidence are unchanged.

This is a prior, not newly observed negative evidence. It can penalize genuine
unseen or incompletely observed object parts; that bias is central to interpreting
the regressions. It does not prove a removed Gaussian was semantically wrong.

## Matched controls

Reuse next95 paired low/high caches, exact same identities/seed admission,13refit
query columns, geometry/opacity/native map, source frames and solver budget.
Other54 query selections retained. No retracking or teacher additions.

1. Low96x128 + count prior.
2. Low96x128 + footprint prior.
3. High192x256 + footprint prior, same next95 interleaved96x128 samples/view.

High and low retain their respective next95 ray positions/one-pixel boundary
bands. High changes evidence resolution and physical boundary width jointly.
There is no claim of equal rasterization FLOPs. All293/122/169/180source frames
remain eligible; steps are max(240, source frame count). The evaluator and target
permissions are frozen. No target labels enter optimization or prior construction.

## Results

Selected Gaussian silhouette projection IoU (%):

| Scene | next93 / low no-prior | Low count | Low footprint | High footprint |
|---|---:|---:|---:|---:|
| figurines | 67.065 | 64.794 | 65.224 | 65.982 |
| ramen | 54.234 | 53.758 | 53.920 | 55.303 |
| teatime | 62.098 | 62.961 | 62.262 | 63.673 |
| waldo_kitchen | 59.054 | 59.097 | 59.434 | 60.953 |
| Macro | 60.613 | 60.153 | 60.210 | **61.478** |

High alone in next95 was60.256%; adding the footprint prior gives+1.222points
on the matched high control, and+0.865points over retained next93. Three scenes
improve versus next93; figurines regresses67.065→65.982. Low priors alone fail to
improve the overall mean. This supports complementarity of better boundary
support and compact selection for this development set, not universal improvement.

Selected queries, next93→high footprint:

| Query | Parent | Candidate |
|---|---:|---:|
| pikachu | 93.20 | 81.63 |
| tesla door handle | 51.82 | 65.51 |
| waldo | 78.22 | 56.59 |
| kamaboko | 72.80 | 78.38 |
| onion segments | 0.00 | 0.00 |
| sake cup | 60.89 | 67.03 |
| dall-e brand | 55.49 | 49.59 |
| paper napkin | 58.92 | 79.86 |
| ketchup | 49.92 | 69.54 |
| ottolenghi | 68.83 | 63.45 |
| plastic ladle | 0.02 | 0.00 |
| pour-over vessel | 69.31 | 86.68 |
| spatula | 28.71 | 38.90 |

No per-query branch assembly. In particular, waldo78.22→56.59, pikachu93.20→81.63,
DALL-E55.49→49.59 and Ottolenghi68.83→63.45 remain material regressions. Onion0
and ladle0 remain identity/scope failures.

## What changed in the error

Posthoc mean pixel precision/recall, next93→high footprint:
- Spatula: precision30.1→42.3%, recall86.3→82.9%, IoU28.71→38.90.
- Pour-over: precision77.5→99.5%, recall86.8→87.1%, IoU69.31→86.68.
- Kamaboko: precision74.7→99.1%, recall96.8→79.0%, IoU72.80→78.38.
- Waldo: precision94.2→98.3%, recall82.8→57.2%, IoU78.22→56.59.

The prior successfully removes projected false extent in some cases but can
shrink to incomplete object support in others. Source reconstruction alone does
not identify complete target extent. It is unjustified to conclude that all
remaining failures are geometry, or that sparsity always recovers identity.

Across the13refit queries, high-prior selected memberships total8549 versus25547
in the matched high-no-prior control (66.5% fewer). These are summed query-element
memberships, not unique map primitives, scene compression, or a measured runtime
speedup. Some removed elements are not visible in scored views. All12outputs
preserve native semantic scores and the other54 selection columns exactly.
Semantic IoU66.930/AP84.146 unchanged by construction.

## Decision and next work

Keep high footprint as an OPTIONAL higher-mean candidate, without changing the
decoder default (support-prior none) or stable recipe. Preserve both regressions
and successes in reporting; do not replace a complete-object claim with the mean.

The next mechanism should address completeness under partial supervision:
- distinguish reliable evidence that an element is outside the target from a
  compactness preference when its identity/scope is unresolved;
- verify complete-scope hypotheses in other source views before making them
  constraints, including whole/part and multiple-instance identity;
- use source-based selection preservation/completeness measurements, not target
  GT, to determine whether regularization deletes established object support.
These remain unimplemented research questions; do not claim the Waldo loss fixed.

Boundary-carrier refinement remains separate. The gain here establishes removable
false extent on the existing carrier, not an irreducible geometry bound or proof
that mixed Gaussian boundaries are solved. The multimodal native map is unchanged;
this is still optional source-assisted query-time extent fitting, not an improved
compact student or a validated unified four-query method.

No new NVOS/ScanNet/UQIS, independent3DGT, held-out-test or SOTA result. Repeated
LERF development evaluation cannot establish publication-ready generalization.

## Reproduction/artifacts

run_compact_scope_next96.py --scene SCENE --gpu GPU
  --arms low_count low_footprint high_footprint

summarize_compact_next96.py; measure_compact_errors_next96.py;
measure_compact_sets_next96.py.

- output/next96/summary.json (baseline matched to each raster).
- output/next96/error_decomposition.json, selected_sets.json.
- output/compact_scope_next96/<arm>/<key>/fit.json, typed_readout.pt,
  anchored/pipeline_calibrated/evaluation.json.
- configs/recipes/next96_research.json.
