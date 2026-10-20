# TMM response paper: retained evidence, common core and full-source replay

Date: 2026-10-16. Branch: `codex/tmm-query-response`.
Worktree: `/root/RADIO-GS-query-field`. No training or new accuracy experiment.

## Scope and implementation

Following the user's paper scope, freeze representations and preserve strong
providers. The mechanism is query-conditioned admission of partial evidence
for response refinement. Teacher dimensions, compact-field training and storage
compression are not research variables. ADR0022 records the change.

`response_refinement.py` exposes (1) the extracted retained visibility fusion,
and (2) a directional residual prototype with separate positive/exclusion
permission, query compatibility, unknown fallback and legitimate negative
correction of high priors. The prototype needs validated per-input permission
providers and matched cross-task accuracy experiments before promotion.

The existing LERF fitting entry point calls the extracted fusion. Observation
construction was factored into an iterator without changing its mask, confidence,
pixel phase, camera, native-positive protection or default solver rules.

## Retained evidence to paper tables

`build_tmm_evidence.py` reads 42 source JSON files, records their SHA256 hashes,
verifies complete v10 job identity and saved job results, and recomputes every
macro column. It verifies 31 complete jobs and 67 UQIS targets / 268 queries.
Generated outputs: `paper/tmm_response/evidence.json` and three LaTeX tables.

The two direct mechanism precursors are query-conditioned identity admission
(LERF selected projection64.1205→66.6617) and admitted response integration
(semantic50.5885→70.3775). Frozen provider evidence includes image local matching
(30.7392→34.8762), complete membership reuse(34.8762→39.8980), real-observation
point membership(2D/3D both+9.62pp), and dual-native category probabilities
(three held-out rooms55.2606→57.6083). These are different controlled records;
their effects cannot be added or assigned wholesale to a new common operator.

The complete v10 table remains the system performance reference, not the new
prototype's accuracy. All LERF response variants are retained, including the
inverse solver's higher IoU and the native floor's higher AP. NVOS scope controls
and compact pooled negative results remain in the evidence ledger.

## Numerical extraction replay

`scripts/research/replay_tmm_response.py` recomputes each original source operator
and admitted observation. On these same observations it runs both the retained
formula and extracted core, then compares the latter to the historical full
response. The existing selected-set tensors are checked for equality. It reads
no target ground truth and saves no tensors.

| Scene | Source views | Query columns | Max error: fresh legacy/core | Max error: historical response | 0.5 crossings |
|---|---:|---:|---:|---:|---:|
| figurines | 293 | 21 | 2.38e-7 | 3.58e-7 | 0 |
| ramen | 122 | 15 | 3.58e-7 | 4.17e-7 | 0 |
| teatime | 169 | 14 | 2.98e-7 | 4.77e-7 | 0 |
| waldo_kitchen | 180 | 18 | 2.98e-7 | 4.17e-7 | 0 |

Total:764 source views,68 query columns(67 labelled texts and1 unlabelled probe).
Selected-set and observed-support tensors match exactly in all scenes. Threshold
crossings refer to Gaussian response values, not newly scored rendered masks.
Responses are numerically close, not bit-identical. Fresh error tolerance1e-6;
historical tolerance2e-5; actual errors are much smaller.

Initial run `legacy_core_replay` failed an absolute-only1e-6 test on accumulated
mass. The identical CUDA sparse products have parallel accumulation roundoff;
mass can be much larger than a bounded response. Revised replay records raw mass
errors(up to9.77e-4), checks both relative and absolute1e-6 tolerances, and still
checks normalized response absolute1e-6. It passes. The initial failure receipt
is retained; no historical results or answers were overwritten.

Successful receipts:
`output/tmm_response_20261016/study/legacy_core_replay_v2/{manifest,summary}.json`
and four per-scene reports. The managed study budget is2GiB; only small JSON
reports were written(16,040bytes including the initial failure), without duplicated geometry, teacher, operator or response
tensors. GPU peak allocated memory across these replays was about1.28GB.

## Verification and next scientific step

- 13 targeted behavioral tests pass.
- Changed Python modules compile; `git diff --check` passes.
- TMM method/evidence working draft compiles with generated tables.
- Full-source numerical compatibility replay passes; no new accuracy claimed.

Next: validate legal positive/exclusion permission providers for every input
type, preserve the frozen native identity and complete-membership capabilities,
then compare original response, existing fusion and common refinement across the
entire five-benchmark cohort. Identity removal, unknown-as-negative and matched
random filtering are the focused controls. Do not restart feature-field design
or tune a single benchmark in isolation.
