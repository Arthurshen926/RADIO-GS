# A06 — isolate estimated-anchor authority from query identity

## Scientific decision

Removing the single-vertex positive override does not materially recover accuracy
on this cohort. Preserve the existing classifier and concentrate future admission
work on whether a proposed identity actually answers the query. Confidence in
membership conditional on an anchor is a different quantity.

This is a completed causal audit, not a fifth universal-method experiment. All
67 image interventions were sealed before loading evaluator labels, then all 268
UQIS queries were freshly scored. LERF2D, selected-set projection, NVOS and ScanNet
remain explicit retained compatibility references; they were not freshly inferred.
No new common rule or default was selected.

## Exact intervention

`estimated_anchor_patch` evaluates the same frozen member classifier at the same
estimated anchor and omits only `probability[anchor] = 1`. It preserves selected
region, conditioning anchor, every other vertex, teacher/map inputs and threshold
0.35. Actual points and text cannot enter this control. Image boolean selections
are recomputed only at that vertex using the same strict threshold. No training,
threshold search, label-based routing or additional teacher calls take place.

Every original array is referenced by hash. Only 67 sparse patches, code snapshots,
small reports and metrics are persisted; zero new dense arrays or weights are saved.
The complete audit adds approximately 1.45 MB under the unchanged 2 GiB budget.

## Full feedback

| Metric (%) | Frozen v10 | Remove override | Scope |
|---|---:|---:|---|
| LERF2D IoU | 70.400676 | unchanged | Retained compatibility |
| LERF selected-set 2D projection IoU | 66.963274 | unchanged | Retained compatibility; not 3D GT |
| NVOS IoU | 86.513417 | unchanged | Retained compatibility |
| ScanNet mIoU19 | 61.011491 | unchanged | Retained compatibility |
| UQIS text AP | 26.595870 | 26.595870 | Fresh scoring, identical arrays |
| UQIS image AP | 59.308782 | 59.308926 | Fresh sparse intervention |
| UQIS 2D point AP | 75.790097 | 75.790097 | Fresh scoring, identical arrays |
| UQIS 3D point AP | 75.325251 | 75.325251 | Fresh scoring, identical arrays |
| UQIS image actual-set IoU | 39.897960 | 39.897960 | Fresh scoring, identical selections |

All 67 anchor score entries change from 1 to values in [0.914032, 0.939022].
All remain above 0.35, hence **zero selected elements change**. Image AP effect is
+0.000144 pp, with descriptive paired nine-scene 95% interval
[-0.000759, +0.001192] pp. One scene improves and one worsens; this is not a useful
performance gain. All text/actual-point responses and every actual instance set
remain exact.

## Attribution after sealing

44 of 67 image anchors lie on their target; 23 do not. Correct-anchor image IoU
has query mean 55.05% and conditional scene mean 55.09% (9 rooms). Incorrect-anchor
IoU has query mean 9.89% and conditional scene mean 10.75% (8 rooms). These are
conditional associations, not the causal effect of correcting the anchor.

Among the 23 wrong anchors, 16 selected regions contain target elements and seven
contain none. Thus a region-to-anchor transition may lose identity in some cases,
while other cases already fail at region selection. Four correct-anchor queries
still have IoU below 0.25, so identity is not a sufficient explanation of all
extent errors either. No oracle anchors or oracle predictions were created.

Raw self-member confidence has identity-correctness AUROC 0.5316 on this opened
cohort. Correct/incorrect groups have almost equal mean scores, 0.9253/0.9243.
This is a parameter-free diagnostic, not a calibrated rejection rule.

A historical four-room development control already replaces the geometric median
with the highest view-balanced native query score inside the chosen region. Its
mean IoU changes 7.008791% to 7.022610%, only +0.013818 pp, and fails its prespecified
gate. Do not rerun this simple heuristic as if it were a new promising method.
Its evidence is `output/typed_referent_binding_20261006/query_supported_anchor/summary.json`;
it is a different text-development cohort, not a UQIS image transfer result.

## Consequence for the one mechanism

Continue separating query applicability `g(h, q)` from conditional extent
`m(i | h, E)`. The inherited member model remains useful when the identity is
correct. A new admission rule must preserve that ability while handling ambiguous
identity conditions; weakening all member responses would repeat the earlier
loss of correct cases. Avoid solving this by retraining another generic local
membership head or by treating its high self-score as identity confidence.

The next mechanism intervention should retain query evidence through the
observation-to-identity transition, with a matched attenuation/rejection control.
Its scope must include actual source-family admission and legal map-only queries
before claiming transfer. Preserve actual clicks, typed category/union outputs,
effective numerical backends and all teacher providers. Evaluate the full cohort
together; this audit does not justify selecting an image-only best branch.

## Reproduction

From the active worktree, with the existing `cybersim_agent` environment:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -u scripts/research/audit_tmm_estimated_anchor.py --run estimated_anchor_authority_replay
python -m radio_gs.response_study verify
python -m radio_gs.response_study refresh
python -m radio_gs.response_study paper
```

Use a new output name. Original complete receipt:
`output/tmm_response_20261016/study/estimated_anchor_authority_v1/summary.json`.
`patches_complete.json` binds all nine prediction reports; `evaluation_manifest.json`
binds the seal. `anchor_diagnosis.json` keeps the evaluator-only strata. Current
entry-point verification checks the seal and UQIS per-scene aggregates without
loading large arrays. Paper builders export the result and self-score diagnosis
to `paper/tmm_response/`; the audit stays separate from E01–E04 metric rows.
