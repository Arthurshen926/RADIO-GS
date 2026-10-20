# Query-conditioned witnesses before complete membership (Q01)

Protocol fixed before target scoring on 2026-10-16. Run:
`output/tmm_response_20261016/study/query_witness_anchor_v1`.
This is a targeted mechanism study, not E05 or an all-31 new system evaluation.

## Question and fixed variables

A06 showed that deleting the forced-positive bit does not change any selected
element. The next question is whether the query's regional evidence can better
locate the uncertain conditioning anchor **without replacing the effective
complete-carrier classifier**. The original SigLIP/local-DINO identity seed,
all teachers, map, native identity responses, classifier and .35 threshold stay
fixed. Retained legal query encodings are hashed and reused; they are not freshly
re-encoded. There is no training or source RGB/depth/mask access at query time.

This is different from historical query-score argmax within a seed (which failed
its four-room text gate) and from replacing the complete classifier with a
query-feature learner. It also differs from choosing whole-object scope by the
mean identity of a cross-view track, which previously degraded UQIS.

## One intervention and a matched negative control

Exclude the selected seed's entire view. On all remaining supported regions,
reuse the exact `native_identity_posterior` operator from the LERF local positive
control, with its original >=.5 admission rule. The channel for this extra
witness is native SigLIP cosine, not the spacing of fused ordinal ranks. The
original joint SigLIP/DINO seed selection remains intact. The candidate domain
here differs from detector-conditioned LERF proposals; transfer is a hypothesis.

For region r in view f, let a_r be the admitted relative identity weight and
normalize w_r = a_r / sum_{s in f} a_s. Inside the original seed only, compute

    u_i = sum_r M_ir w_r / sum_f V_if 1[sum_{r in f} a_r > 0].

Zero known mass gives zero weight. Choose the actual seed member nearest the
u-weighted coordinate median. If no seed member has positive witness mass,
fall back to the exact original geometric median rule. The original classifier
then decodes all carrier elements from that anchor, with its unchanged override
and threshold. Witnesses locate a candidate; they do not establish its identity,
make all seed members positive, or introduce downstream negative labels.

The second arm permutes the same weights among supported regions **within each
view**, with fixed seed 20261016. It preserves each view's weight multiset,
admitted count and normalized budget while breaking query-to-region association.
Both arms use the same seed, geometry, center rule and complete decoder. This is
one fixed random control, not an estimate over random-control seeds. The final
spatial influence and chosen-anchor displacement are deliberately not matched.

## Evaluation and storage

All 67 image-query predictions in all nine rooms and both arms are fixed and
hashed before evaluator access. Original complete-decoder replay must match all
retained image selections and scores (maximum tolerance 1e-7). All 268 independent
queries are scored for AP and actual-set IoU. Text and actual-point outputs are
exact retained controls. Each result is a nine-scene macro average; paired
bootstrap intervals are descriptive on this already opened development cohort.

LERF semantic 2D, LERF selected-set 2D projection, NVOS and ScanNet retain the
complete v10 reference metrics with **fresh_inference=false**. No per-task winner
is promoted or stitched into a common method. A successful local intervention
would still need wider applicability and untouched validation scenes.

All new dense predictions are volatile RAM only. The persistent run keeps code
snapshots, input identities, prediction seals, per-query/scene metrics and
post-seal attribution. Existing study/run/artifact storage ceilings are retained.
Reproduction command (use a unique run name):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/research/run_tmm_query_witness_anchor.py --run query_witness_anchor_replay
```

## Results

All 67 original image responses replay exactly (maximum error 0, identical
selections). Both candidate arms complete, then all 268 queries are scored.

| UQIS image, scene macro (%) | v10 | Query witness | Permuted witness |
|---|---:|---:|---:|
| AP | 59.3088 | 58.7739 | 58.4641 |
| Actual-set IoU | 39.8980 | 38.7674 | 38.7483 |
| Correct anchors / 67 | 44 | 44 | 43 |

The query arm changes 55 anchors. Its image IoU difference is **-1.1306 pp**,
descriptive paired nine-scene 95% interval **[-2.4180, -0.1772] pp**; seven rooms
decline and two improve. Against the weight-permutation control, its advantage
is only **+0.0191 pp**, interval **[-0.3648, +0.4363] pp**. AP relative to v10
changes -0.5349 pp, interval [-2.1267, +0.5196] pp. No candidate is promoted.
One permutation cannot establish equivalence to all random controls.

Text AP/IoU remain 26.5959/16.3865%; 2D point 75.7901/49.7357%; 3D point
75.3253/49.0470%. Other frozen benchmark references remain LERF semantic IoU
70.4007%, selected-set 2D projection IoU 66.9633%, NVOS IoU 86.5134%, and
ScanNet 19-class mIoU 61.0115%. These source-family values were **not rerun**.
The complete AP/category-menu/reference metrics remain in `summary.json`.

## Attribution and convergence decision

Post-seal diagnosis finds one originally wrong anchor becomes correct, while one
originally correct anchor becomes wrong. Among the 43 queries whose anchor stays
on the target in both arms, query-mean IoU falls from 55.8545% to 54.3421%.
These conditional subsets are diagnostics, not independent causal comparisons.
They show why anchor correctness alone is insufficient to assess preservation:
different points on the same object can condition different observation and
complete-member responses. The original correct-to-wrong case even gains IoU,
so binary anchor correctness cannot replace full output evaluation.

The identity gate from detector-conditioned LERF candidates has **not** transferred
successfully to this global native-score/weighted-center construction. The
controlled change supplies little demonstrated query-specific benefit, and
moving the anchor damages some of the capability we meant to preserve. Retire
this construction as a default candidate; do not scan its thresholds, mix its
per-scene winners, or call it a shared mechanism gain.

The next unresolved interface is identity-conditioned complete response, with
two simultaneous requirements: repair wrong referents and retain the original
complete extent when the referent is right. A replacement must preserve the
existing response as an explicit reference and test identity association
separately from geometric displacement. Neither a weighted coordinate center
nor an uncalibrated self-member probability satisfies that interface.

The persistent run is **2,123,520 bytes**; dense predictions used **117,809,976
bytes** of volatile RAM and were released. No new weights or persistent dense
predictions were created. Prediction time summed over nine scenes is 186.79 s
(scoring and hash checks are additional). Source, evaluator assets, input and
prediction hashes plus full metrics are retained for reproduction.
