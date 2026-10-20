# ADR0023: Learn when a complete query response should be replaced

Date: 2026-10-16. Status: accepted research direction; empirical promotion pending.
Supersedes ADR0022's next-experiment prescription, retaining its frozen-provider,
input-access, metric and storage contracts.

## Problem and decision

The user requests substantial progress rather than further local anchor,
confidence or threshold patches. Q01 changes anchors without reliably improving
complete answers. Q02 supplies better complete candidates for 27/67 image queries,
but its verification rule causes more damage than repair. A feature score or a
conditional member probability does not estimate the value of replacing an answer.

Use one **learned complete-response replacement** mechanism:

1. Frozen providers produce an incumbent and a bounded pool of intact responses.
2. Query/observation agreement and visibility produce a fixed-dimensional response
   signature. All applicable teacher scores participate, independently of their
   feature dimensions. Teacher order has no meaning.
3. Learn a pairwise difference in actual response utility from separate labelled
   scenes. Fit all ordered pairs of complete answers for the same query, with equal
   scene/query weight. Neither representation nor element decoder is retrained.
4. Compare alternatives to the incumbent using an antisymmetric gain predictor.
   Accept one whole answer only above a globally calibrated margin. Retain the
   incumbent on ties, missing evidence or reject-all calibration.

There is one learned decision mechanism, not a chain of new identity, geometry
and membership networks. The partial-evidence contract remains relevant to its
inputs; the new learned decision replaces the hand-designed verification rule.
Do not present the old text-admission gains as evidence for this new estimator.

## Concrete first implementation

`response_replacement.py` pools query-rank/response overlap within a view, then
across views. It excludes the union of all proposal views. View count is an
observability indicator, not a guarantee of independent evidence: candidates
themselves still consume the map's observations. Outside-response association
does not automatically become negative membership authority.

Four regional identity proposals are decoded by both unchanged providers:
`DirectAnchorMembership` and `InstanceObservationScope`. The incumbent is always
present, and selected probabilities AND actual selections are returned intact.
Actual user points retain their frozen point semantics in the first transfer.
An ExtraTrees regressor estimates the signed IoU difference. Its antisymmetric
evaluation cancels order-independent offsets and gives an exact zero self-gain.

The training manifest fixes 22 fit rooms (up to 32 deterministically sampled
eligible expressions per room), three calibration rooms and four development
rooms. Calibration/development use all retained expressions. These rooms are
disjoint from the current benchmark scene names. The membership provider was
also fitted on the 22 fit rooms; disclose this inherited overlap. The development
rooms were previously opened and are not an untouched final test.

One margin maximizes calibration scene-macro IoU subject to nonnegative mean
change in each calibration room. Reject-all is a valid outcome. This is an
empirical guard, not a statistical safety or domain-transfer guarantee. The
calibration incumbent is the direct provider. Transfer to a scope incumbent,
image queries and a second teacher must be measured without fitting a new margin.

Controls are the same learner without query features, and inference with query
scores jointly permuted within each view. The latter preserves teacher score
multisets, their correlations, complete candidates and geometry. Report both
repair and harm, candidate headroom, and unconstrained versus calibrated decisions.

## Cross-task completion contract

The algorithm accepts complete responses, but its current regional signature and
independent supervision cover instance answers. A common function signature is
not completed source-task transfer. The full-five gate remains mandatory:

| Output family | Correct decision unit | Required next transfer evidence |
|---|---|---|
| LERF semantic 2D | complete query response and original renderer | source-region signature, unchanged legal source access, AP and threshold IoU |
| LERF projected set | actual set plus its original projection | separately preserve/re-score the independent set; never call it true 3D IoU |
| NVOS | complete reference-conditioned response | reference-authority signature; reference and propagated partial observations distinguished |
| ScanNet OVS | complete category menu and original argmax | joint category competition; no independently stitched class winners |
| UQIS | complete instance score and actual set | shared text/image policy, all applicable teachers, point preservation and all 268 scores |

Source-assisted adapters require suitable independent supervision and cannot be
silently represented as validated by natural-text instance training. Reference
metrics from those families remain labelled **transfer pending**. Do not tune on
opened benchmarks, promote a task-specific best arm, or claim five-task success
from unchanged references. The performance default remains v10 until the joint
gate is met.

## Reproduction and storage

`configs/research/response_replacement.json` fixes the experiment before fitting.
Training and transfer runners seal source/input hashes and the model before
benchmark evaluation. Training stores compact signatures/targets; dense responses
use a bounded 64-entry per-worker cache. Benchmark predictions remain in bounded
RAM and are released after scoring. Preserve the small fitted model; do not copy
teachers, native fields, maps or raw observations. Existing study budgets apply.

The initial `response_replacement_train_v1` compilation was stopped before model
fit/calibration to include both frozen providers. It is an incomplete preparation,
not a failed scientific comparison or a searched model. Its source snapshot and
stop receipt are retained. `response_replacement_train_v2` is the first fitted
experiment for the two-provider architecture.

## Joint supervision phase (D01/L03/L04)

The first policy transfers poorly from text to image queries despite useful
complete image candidates. The next fixed protocol is
`configs/research/response_replacement_joint.json`. It retains the SAME signature,
candidate providers and learner capacity, and supplies independently constructed
real image queries with both original SigLIP and DINO local matching teachers.
All reference views and their temporal/pose neighbors are excluded from every
provider's observation carrier. Original regional masks/descriptors are sliced
by legal views; view-balanced transfer is recomputed exactly from that subset.
The official full-scan mesh is retained, so geometry itself is not held out.

Fit all ordered response pairs with equal room/modality/query weight. Text
calibration uses its deployed scope incumbent, image calibration the direct
incumbent. One shared margin must yield nonnegative mean change in all six
calibration room/modality cells. The estimator never receives a modality label.
This is not per-modality routing or separate benchmark thresholds. The protocol
does not certify generalization; reject-all remains a legitimate empirical result.

This joint experiment changes supervision, balancing and incumbent calibration
together. Its difference from L02 does not isolate any one factor causally.
Query-blind and matched query-permutation controls remain necessary. Source-task
transfer and whole-menu class competition remain explicit unfinished work; a
completed D01 corpus or a common API cannot close those scientific gaps.

## Set-consistent utility control (L05/L06)

L03's largest predicted image gain on independent calibration is harmful; the
shared protection rule consequently rejects every replacement. L05 tests an
architectural hypothesis before inspecting new L04 benchmark feedback: assign
one potential to each candidate in the shared candidate-set context instead of
regressing ordered pairs independently. Inputs are the original signature, its
set mean, and their difference; targets are actual candidate IoU minus the query
pool's mean IoU. Gain is the difference from the incumbent potential.

This makes ordering candidate-equivariant and relative comparisons consistent
around cycles. It does not guarantee accurate utility or prove inconsistency
caused L03's error. The teacher/provider candidates, supervision, split, tree
capacity and global six-cell calibration rule remain the same. No new field
representation, modality routing or benchmark threshold is introduced. This
controlled architecture result must be reported alongside the joint pairwise
model, including query-blind/permuted controls and the declared development gate.

## Cross-task evidence and supervision boundary (A08/A09)

The 22-job source audit chooses intact E02 answers from legal source agreement,
without fitted thresholds or benchmark labels in decisions. The source fitting
score fails as a utility estimator: ScanNet replaces ten menus and harms seven;
LERF IoU gain survives correspondence permutation. Historical pool capacity is
small when every reported metric must be preserved. These are diagnostics, not
a trained source transfer or a five-task success.

Official whole-menu/union target manifests are ready on 29 separate rooms, with
exact mesh alignment, 19-column menus and three chronological view blocks. Six
classes are absent from calibration. Original Mask-Adapter and literal-category
source evidence remain to compile. Target availability is not complete utility
supervision. The independent LERF set needs explicit candidates, and menu utility
must account for its multiple outputs. Evidence already used by an incumbent
must not be called independently held out. The historical label oracle cannot
become an inference policy.

## Full-teacher category supervision and joint utility (D02/D03/L07)

The category branch preserves original SigLIP2, Mask-Adapter and literal SAM3
teachers. D02 stores only 19-class native cosine responses on original regions.
D03 streams SAM3 masks, computes original foreground/context identity and
detector-mass matching, and produces three complete menus: dual-native/source
incumbent, dual-native and source readout. It preserves original no-source
fallback and entropy weighting. This independent training carrier uses mesh
vertices and depth-tested normalized pixel contributions; Gaussian renderer
equivalence is explicitly NOT established.

Fit uses one full-map and three chronological candidate/evidence view folds per
room. Category signatures average all 19 classes without label-based filtering.
The utility target is mean complete-menu mIoU19/15/10; one answer serves every
submenu. L07 uses the same set-conditioned scalar estimator and capacity, with
equal room/task weight across text, image and category supervision. Calibration
and development use only full-map menu episodes. One margin must protect all 15
calibration room/output cells, with no category or task-specific thresholds.

These are concrete protocol decisions, not results. Fit requires complete D03
input rather than silently dropping rooms/teachers. Model and margin are sealed
before development is opened. Query-blind, query-permuted and zero-margin
controls remain. Six calibration classes are absent; AP and independent LERF-set
utility are not yet supervised by this phase. Neither compiled data nor a passed
local guard closes the five-benchmark transfer gate.

## Fixed-model category/instance transfer (D04/L08/A10)

Keep the benchmark's original Gaussian carrier, cached SigLIP/Mask-Adapter query
encodings and source SAM3 evidence. Reconstruct three complete menus and verify
the original primary and full fusion numerically and by exact 19/15/10 argmax.
Retain original incumbent bytes after replay. All ten scenes pass in D04 v2;
no benchmark labels or learned policy are used during this preparation.

L08 uses the same L07 checkpoint and full-policy margin for category menus and
UQIS text/image answers. One category choice serves all submenus; actual point
answers remain intact. Seal every prediction for both families before fresh
scoring. Query-blind, correspondence-permuted and unrestricted controls use the
same fixed candidate pools. No family-specific refit, margin or winner assembly.

Source signatures use binary positive native transfer support and its per-view
union. This is region-covered visibility, not full geometric visibility, and
differs from independent mesh supervision. Record the difference explicitly.
Shared dimensions alone do not establish equivalent input distributions.

After scoring, A10 measures capacity and decision regret separately. Its menu
oracle chooses one intact candidate under either mean-utility or all-output
protection; it never combines submenu winners. UQIS historical pool utilities
are admissible for diagnosis only after exact candidate-set/order and target/
coordinate hash checks, verified against fresh selected-output scores. Oracle
gain cannot be reported as method gain. LERF/NVOS remain pending in this transfer;
neither the prepared menu pool nor a successful partial transfer closes all five.

## Candidate-relative context control (L09/A11)

Before L07 training or L08 benchmark results existed, a label-free inspection
found coverage and regional-density signatures outside the independent mesh
fit range on retained Gaussian menus. In particular, the mean selected fraction
over all 19 classes equals observed-carrier fraction divided by 19, and is shared
by every complete menu. Such context can condition utility without directly
distinguishing candidates. Range differences alone do not prove harmful reliance.

L09 fixes one architecture control: train the same scalar utility only on
`x - mean(X)`, removing the absolute candidate and set-mean input blocks. Keep all
teachers, complete pools, 39-dimensional signatures, data splits, room/task
weighting, estimator capacity/seed, and the shared 15-cell calibration guard.
Candidate-common offsets cannot change this estimator's input; candidate order
remains equivariant. It may also discard useful query-difficulty context, so
benefit must be measured, not assumed. Its design and source snapshots are sealed
before parent training results; target signature inspection is disclosed.

A11 audits fixed predicted versus actual gains on the already opened independent
calibration/development rooms, and source-only feature ranges on benchmark
signatures. Impurity importance is descriptive, not causal evidence. The audit
does not read benchmark labels, refit models, or alter margins. Neither L09
development results nor input invariance establish all-five performance.

NVOS currently has one cached target region per source view in all eight jobs.
The present regional query signature requires at least two regions per view, so
it yields no query verification there. Reusing it would force abstention rather
than test transferable query evidence. Covering NVOS requires a justified
observation interface; unchanged outputs must be labeled compatibility only.

## Query-evidence residual utility (L10/L11)

L07 leaves large instance candidate capacity unused; structural-only learning
can outperform its full model on development. L09 shows that removing absolute
context alone is not a solution. Test one utility decomposition, keeping all
teachers and intact candidates: a structural prior plus a query-conditioned
correction. The prior uses the original nine structural signature coordinates;
the correction uses the original full set-context signature. Their sum remains
one scalar utility potential, with one retain/replace decision and global margin.

Construct correction targets from five room-grouped out-of-fold prior fits on
the 22 fit rooms. Every task and chronological category episode of a room stays
in the same fold. Fit the final prior on all fit rooms. Train corrections on
actual utility minus the out-of-room prior, paired with matched within-view
query-permuted signatures whose correction target is zero. Zero means no query
correction credit, not zero actual candidate utility; it is a regularizer whose
assumption must be tested. Each pair shares the original episode weight equally.

Use 64 trees for each component (128 total), explicitly more than L07's 64-tree
model. A matched 128-tree no-null control shares the same prior and substitutes
duplicate aligned samples for null samples, preserving counts and weight mass.
Keep the query-blind prior, inference permutation and zero-margin controls.
The same 15 calibration cells and 22/3/4 split apply. Both model and control are
sealed before development; earlier development/benchmark feedback informed this
design and is disclosed. No untouched-test or new risk guarantee is claimed.

L11 predeclares all five arms on ScanNet and UQIS together. Every arm uses one
checkpoint bundle and one independently calibrated margin for all applicable
outputs; the no-null arm is an ablation, never a taskwise winning substitute.
All predictions precede target scoring, with the existing 2 GiB transient and
2 GiB persistent-study limits. LERF/NVOS applicability remains explicitly open.
