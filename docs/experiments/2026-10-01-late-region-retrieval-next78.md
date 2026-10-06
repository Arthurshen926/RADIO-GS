# Unmerged regional evidence and late native-query pooling, next78

## Implementation and controlled comparison

Previous controls changed weights or area slots but still lost individual source
regions before queries arrived. This iteration stores an auxiliary regional bank
on the SAME frozen next66 Gaussian geometry. Every cached native region in every
legal source view retains its1536D descriptor, original region index, source
provenance and sparse Gaussian memberships. No top-k region or Gaussian cap,
instance merging, teacher/query encoder change, target RGB or annotation input.

For each source view, membership is its regional alpha-adjoint contribution
divided by that Gaussian's total visible contribution. Store all positive entries
with float32 weights and lossless int32 local indices. The initial figurines and
ramen banks occupy approximately2.1GiB and1.4GiB on disk. This is additional
map state, not a compact-field memory or latency claim.

`radio_gs/querying/region_bank.py::query_region_bank` accepts arbitrary native-space
query vectors. Let w_gr be a region's per-view Gaussian membership and c_rq the
native region/query cosine. Two matched readouts are compared:

- Mean: sum(w_gr*c_rq)/sum(w_gr).
- Log-mean-exp: log(sum(w_gr*exp(20*c_rq))/sum(w_gr))/20.

The fixed inverse temperature20 was set before evaluation; no scene/query
threshold selection. It emphasizes high-matching regional evidence without
choosing a scale using labels. Each view is normalized separately, then integrated
using the Gaussian's visible alpha mass. Views with no regional support contribute
no evidence. No query-conditioned map artifacts are stored in the bank; evaluation
score tables remain transient outputs.

This is per-Gaussian regional pooling AFTER projection, whereas next75 averages
normalized region mixtures at pixels BEFORE projection and normalizes final
Gaussian features. Thus comparison with next75 changes normalization and
aggregation order; the matched mean control is needed to isolate the nonlinear
readout. The control does not hold every arithmetic operation equal to next75.

Preserve the original source scalar calibration, source scope refinement and
historical confidence-distribution transport. First freeze next75's selected
membership to measure semantic response. Any joint decoder uses the unchanged
source masks, steps and selected-silhouette scorer. These retained downstream
steps still depend on historical query-specific source evidence; an arbitrary
native-query bank API does not establish end-to-end unseen-query calibration.

## What this does and does not establish

This implements preservation and query-time access to individual region evidence.
It does NOT establish region identity, directional containment or a compatible
object/part selection rule. Log-mean-exp can still emphasize a semantically
incorrect coarse region. No explicit relation graph, multimodal compact student,
or same-map four-query benchmark completion is claimed. No previous failed
next76/77 policy is silently added to the new readout.

All results are development evidence, under the existing frozen evaluation.
Selected projection is not independent3D-GT evaluation. NVOS/ScanNet/UQIS are not
rerun in this iteration.

Artifacts: `output/region_bank_next78/{scene}/manifest.json`, regional shards,
`output/late_region_next78/{pooling}`, and `output/next78/summary.json`.
Scripts: `build_region_bank_next78.py`, `read_region_bank_next78.py`,
`run_late_region_next78.py`, `decode_late_region_next78.py`, `summarize_next78.py`.

## Follow-up: preserve absolute scope support

Both normalized-pooling pilots fell below next75. A further mechanism control
uses `scope_max`: within each source view, response(g,q) is the maximum of
membership(g,r)*cosine(r,q) across regions, with zero representing no positive
regional evidence. Keep the same visibility-weighted cross-view mean, source
calibration, refinement and evaluator. This preserves fractional region coverage
rather than dividing it away. No scene/query-specific region choice or tuning.

This control was added after the initial pilot results, so it is a sequential
development experiment, not a predeclared independent test. It also changes
pooling from a normalized average to maximum; any gain cannot be attributed to
coverage preservation alone without an additional max-only ablation. It does
not establish cross-view instance equivalence or complete object boundaries.

## Completed pilot results

| Scene | Readout | 2D IoU | AP | Selected projection IoU |
|---|---|---:|---:|---:|
| figurines | baseline | 68.517 | 87.626 | 59.432 |
| figurines | mean | 67.421 | 86.314 | 59.432 |
| figurines | logmeanexp | 67.723 | 86.524 | 59.432 |
| figurines | scope_max | 67.498 | 84.491 | 59.432 |
| figurines | joint | 67.498 | 84.491 | 56.668 |
| ramen | baseline | 62.609 | 78.300 | 44.018 |
| ramen | mean | 62.415 | 77.208 | 44.018 |
| ramen | logmeanexp | 62.526 | 77.644 | 44.018 |
| ramen | scope_max | 60.623 | 75.231 | 44.018 |
| ramen | joint | 60.623 | 75.231 | 44.657 |

All8pilot evaluations completed: three semantic readouts on two scenes with
fixed selected membership, plus two scope_max extent decodes. No full-four-scene
expansion and no promotion. Retain next75's complete LERF candidate.
The small ramen selected gain does not justify combining per-scene winners;
figurines selected projection loses2.76pp. No standalone nonlinear-temperature
sweep, score-threshold search or unit-test scaffolding was introduced.

## Scientific conclusion and next step

Individual native observations and their geometric support can now be retrieved
without rebuilding the scene or losing intermediate regions. This is a reusable
implementation advance, NOT demonstrated benchmark or method superiority.
The current nonlinear pooling is slightly better than its matched mean control
but below next75; preserving raw observations alone is insufficient. These
experiments do not prove that the bank contains no useful information, nor that
teacher recognition is the sole bottleneck. Aggregation, source-region correctness
and downstream query-specific refinement remain intertwined.

The next mechanism must infer compatible region identities and scope rather than
reduce all observations immediately to a pointwise mean or maximum. The bank
provides original region IDs, source views, descriptors and fractional support
for that work, but the correspondence/containment solver and its independent
validation are still OPEN. Native-space support for arbitrary query vectors is
implemented; end-to-end unseen-query calibration, image-query benchmarks and
prompt-query integration are not completed by this text-only development run.

No deployed/default map, retained recipe, NVOS result or ScanNet result changed.
This iteration did not achieve a significant performance improvement or a SOTA
result, and does not close the project's larger research objective.
