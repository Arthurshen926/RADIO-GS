# Q02: verify identity while preserving complete response candidates

Protocol fixed before prediction/scoring. Original v10 and every successful
provider remain unchanged. The prior Q01 pool, even selected using labels, could
gain only 0.9121 IoU points over v10 (finite-pool diagnosis, never a method).
Q02 changes the identity-proposal interface rather than further tuning Q01.

## Fixed hypotheses and verification

For each of all 67 image queries in nine rooms, use three existing identity
proposals: the original joint SigLIP/local-DINO winner, the native SigLIP winner,
and the local-DINO winner. Duplicate seed regions share a decode. Each proposed
region uses v10's unchanged geometric median anchor, complete membership
classifier and .35 selection threshold. The original full response remains
candidate zero. No additional observations, teachers, weights or supervision.

Legal cached native query similarities and joint percentile scores are reused.
Since joint = (rank_SigLIP + rank_DINO)/(2*regional_count), DINO ranks are recovered
algebraically. Their half-integer rank lattice, average-tie ranks and forward
joint-score reconstruction are checked. This recovers ordinal information only;
it does not invent DINO cosine values. Candidate zero must replay all retained
image scores within 1e-7 and its selections exactly before scoring is permitted.

For every candidate selected set H, compute its co-visible IoU t_hr with each
map region r. Within each view, compute z_r as the equal mean of the two teachers'
within-view percentile ranks. The verification contrast in view f is

    D_hf = sum_r t_hr z_r / sum_r t_hr
           - sum_r (1-t_hr) z_r / sum_r (1-t_hr).

The second term describes an alternative-hypothesis contrast; it supplies no
negative labels to the complete membership decoder. Exclude the UNION of all
three proposal views for every candidate. Use only views where all candidates
have positive mass in both terms. Thus candidates are compared on the same
evidence domain, with equal weight per view. Select the candidate with largest
mean contrast; ties or no common evidence retain candidate zero. Entire output
responses and selected sets are returned unchanged. Scores are not calibrated
identity probabilities; shared observation use remains statistically dependent.

The fixed negative control permutes z within each common view (seed 20261016),
preserving all proposals, complete outputs, geometry, evidence domains and the
per-view query-score multiset. This is one permutation, not a random-seed sweep.
No thresholds, teacher weights, proposal counts or verification variants are
searched after scoring. A label-selected best result in this finite three-answer
pool is computed only after prediction sealing, to separate inadequate proposals
from inadequate verification. It is not an automatic result or a universal bound.

## Cohort and storage

Seal both arms and all proposal responses for all 67 image queries before any
evaluator access. Score all 268 UQIS queries for AP and actual selected-set IoU;
text and actual points are exact controls. Report all nine rooms, paired scene
differences, descriptive bootstrap intervals and baseline-correct/incorrect
diagnoses. No oracle choice enters the verification rule or inference outputs.

The other four benchmark outputs remain explicit v10 historical references;
this targeted study does not constitute fresh inference on all 31 jobs. New
arrays exist in bounded volatile RAM only, released after scoring. Code/input
hashes, prediction seal, all metrics and finite-pool diagnosis are durable under
the original 2 GiB study budget. Main inference defaults are unchanged.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/research/run_tmm_complete_response_verification.py --run complete_response_verification_replay
```

## Complete results and attribution

All 67 original image scores and selections replay exactly. All candidate and
automatic predictions seal before scoring all 268 queries. Rank recovery has
maximum pre-rounding lattice error 0.0003813; every recovered rank population
and forward fusion passes the stated checks.

| UQIS image, scene macro (%) | v10 | Verified | Permuted |
|---|---:|---:|---:|
| AP | 59.3088 | 59.5690 | 56.1093 |
| Actual selected-set IoU | 39.8980 | 38.2163 | 36.4864 |
| Correct anchors / 67 | 44 | 42 | 40 |

AP alone would give the wrong acceptance decision. Actual IoU drops **1.6816 pp**
versus v10 (descriptive nine-scene 95% interval [-5.2721,+2.1612]); six rooms
decline. Versus the fixed permutation, IoU is +1.7300 pp (interval
[-3.9031,+6.6658]) and AP is +3.4597 pp ([-3.0699,+9.9813]). These intervals do
not establish a reliable query-specific benefit. No arm is promoted.

The verifier chooses the joint answer 30 times, SigLIP 14 times and DINO 23 times;
37 anchors change. Seven queries lack common verification views and retain the
original answer. Of the 44 originally correct anchors, five become incorrect;
three of the original 23 errors are repaired. On those fixed baseline strata,
query-mean IoU changes 55.0459->50.7856% and 9.8872->13.5745%, respectively.
These are post-seal conditional descriptions, not new causal contrasts. They
demonstrate why retaining the old answer in the pool is insufficient: the
admission decision can still discard it incorrectly.

The finite-pool, label-selected best actual IoU is **45.0975%**. There is a better
proposal for 27 of 67 queries. The scene-macro decomposition, rounded, is

    38.2163 ≈ 39.8980 + 5.1995 - 6.8811  (percentage points).

The +5.1995 term is available gain in these three frozen answers; 6.8811 is
selection regret relative to their label-selected best. Neither is an automatic
gain. Taking the SigLIP winner globally gives 33.5887% and taking DINO globally
31.1768%, both below the joint baseline. Thus replacing joint fusion with a
single teacher is not supported either. The candidate pool has useful diversity,
but the current common-view rank contrast cannot reliably exploit it.

Text AP/IoU remain 26.5959/16.3865%, 2D point 75.7901/49.7357%, and 3D point
75.3253/49.0470%. Source-family references remain LERF semantic IoU 70.4007%,
LERF selected-set 2D projection IoU 66.9633%, NVOS IoU 86.5134%, and ScanNet
19-class mIoU 61.0115%; these were not rerun. The complete reference metric
menus, AP, all query and scene records remain in the sealed summary and scorer
reports. No cross-benchmark winner stitching or new common-system claim.

## Convergence decision

Retire this uncalibrated rank-contrast selector as a default candidate. Keep the
complete-response proposal interface and its preservation checks as experimental
infrastructure, not a demonstrated accuracy contribution. Q01's limited pool
had only 0.91 pp available gain; Q02's 5.20 pp isolates a more useful research
variable: whether an alternative complete response should replace the original.
Do not tune margins or thresholds against these 67 queries. Any learned or
calibrated replacement rule must use declared separate development/calibration
data and measure repair and damage together, before untouched validation.
No need to retrain the complete membership field for that question.

The run retains **2,179,809 bytes** of reports and source snapshots, uses
**294,524,940 bytes** of volatile prediction RAM, then releases it. Nine-scene
prediction time totals 157.88 seconds; hashing and scoring are additional.
No new weights, persistent dense predictions or teacher caches were created.
